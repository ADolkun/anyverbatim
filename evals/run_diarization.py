"""Diarization selection eval (E1): ``uv run python -m evals.run_diarization run --diarizer ID --corpus ami ...``.

Runs one diarizer over prepared recordings (``evals.recordings_data``), then scores each at collar 0 and collar 0.25
with overlap scored, WDER with and without smoothing, speaker-count error, real-time factor and peak footprint. Every
recording is resumable and everything is written under ``$ANYVERBATIM_EVAL_DIR/recordings/runs/diarization/<id>/``.

Two backends:

- ``ProductionBackend`` drives ``anyverbatim.diarize``, so the Nemotron ids run through the same worker the pipeline
  uses;
- ``ExternalBackend`` runs any command that writes an RTTM, for pyannote in its throwaway uv project.

WDER needs the words of an ASR run per recording: ``words.json`` (a list of ``{"text", "start", "end"}``) next to
each recording's audio; ``python -m evals.run_recordings_asr words`` writes it. Without it WDER is skipped.

The ``owner`` corpus lists the owner's own recordings in a private index. They have no reference RTTM, so each is scored
by its speaker count only (against the owner's count when the index gives one) and compared to other diarizers through
``python -m evals.disputes``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import click
from pydantic import BaseModel, ConfigDict

from anyverbatim import atomic, proc
from anyverbatim.contract.models import Word
from anyverbatim.diarize import registry as diarize_registry
from anyverbatim.diarize import runner as diarize_runner
from anyverbatim.diarize.attribution import assign
from anyverbatim.diarize.protocol import Diarization as StoredDiarization
from anyverbatim.diarize.protocol import DiarizeRequest, read_file_diarization
from anyverbatim.diarize.protocol import FileJob as DiarizeFileJob
from anyverbatim.diarize.protocol import load as load_diarization
from anyverbatim.diarize.runner import DiarizeResult
from anyverbatim.errors import AnyverbatimError, ConfigError, EngineError
from evals.diarization import (
    COLLARS_S,
    DerScore,
    MacroBootstrap,
    Turn,
    WderScore,
    der,
    macro_rate,
    paired_macro_bootstrap,
    pooled_rate,
    read_rttm,
    speaker_count,
    speaker_count_error,
    wder,
    write_rttm,
)
from evals.recordings_data import OWNER, WORDS_FILE, CorpusIndex, load_index, load_owner, recordings_dir

PRIMARY_CORPORA = ("ami", "alimeeting")
CORPORA = (*PRIMARY_CORPORA, "ascend", OWNER)
# The owner corpus runs only when named, so a missing owner index never stops the public run.
DEFAULT_CORPORA = (*PRIMARY_CORPORA, "ascend")
SCORE_FILE = "score.json"
SUMMARY_FILE = "summary.json"
PYANNOTE_ENV = {"PYANNOTE_METRICS_ENABLED": "0"}
GAP_S = 0.5
EXTERNAL_TIMEOUT_FACTOR = 3.0
EXTERNAL_TIMEOUT_FLOOR_S = 30 * 60.0


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True, slots=True)
class Recording:
    corpus: str
    id: str
    wav: Path
    # None for an owner recording, which has no reference.
    reference: Path | None
    duration_s: float
    expected_speakers: int | None = None

    @property
    def words(self) -> Path:
        return self.wav.parent / WORDS_FILE


@dataclass(frozen=True, slots=True)
class Run:
    """What a backend hands back for one recording."""

    turns: tuple[Turn, ...]
    diarize_s: float
    load_s: float
    peak_footprint_bytes: int | None
    # Whatever ``Backend.attribute`` needs again, for example the stored frame probabilities.
    handle: object


class Backend(Protocol):
    def diarize(self, recording: Recording, out_dir: Path) -> Run: ...

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str | None]: ...


class DerRecord(_Frozen):
    collar_s: float
    total: float
    missed: float
    false_alarm: float
    confusion: float
    # None when nothing was scored (a NaN would not survive the JSON round trip).
    rate: float | None


class WderRecord(_Frozen):
    errors: int
    eligible: int
    total_words: int
    rate: float | None


class RecordingScore(_Frozen):
    corpus: str
    recording: str
    duration_s: float
    der: tuple[DerRecord, ...]
    reference_speakers: int
    hypothesis_speakers: int
    speaker_count_error: int
    diarize_s: float
    load_s: float
    rtf: float
    peak_footprint_bytes: int | None
    # Digest of everything the score was computed from besides the diarizer; a changed input rescores.
    inputs: str
    # None when the recording has no words.json.
    wder_raw: WderRecord | None
    wder_smoothed: WderRecord | None
    # Distinct speakers among the attributed words, before and after smoothing.
    attributed_speakers_raw: int | None = None
    attributed_speakers_smoothed: int | None = None


class OwnerScore(_Frozen):
    """An owner recording has no reference: only the speaker count and the cost are scored."""

    corpus: str
    recording: str
    duration_s: float
    hypothesis_speakers: int
    # The owner's own count, when the index gives it.
    expected_speakers: int | None
    speaker_count_error: int | None
    diarize_s: float
    load_s: float
    rtf: float
    peak_footprint_bytes: int | None
    inputs: str
    attributed_speakers_raw: int | None = None
    attributed_speakers_smoothed: int | None = None


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _der_record(collar_s: float, score: DerScore) -> DerRecord:
    return DerRecord(
        collar_s=collar_s,
        total=score.total,
        missed=score.missed,
        false_alarm=score.false_alarm,
        confusion=score.confusion,
        rate=_finite(score.rate),
    )


def _wder_record(score: WderScore) -> WderRecord:
    return WderRecord(
        errors=score.errors, eligible=score.eligible, total_words=score.total_words, rate=_finite(score.rate)
    )


def load_words(path: Path) -> list[Word] | None:
    if not path.is_file():
        return None
    return [Word.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def input_digest(recording: Recording) -> str:
    """Hash of the reference, the words and the collars: what a cached score depends on besides the diarizer."""
    digest = hashlib.sha256()
    digest.update(recording.reference.read_bytes() if recording.reference else b"")
    digest.update(b"\0")
    if recording.expected_speakers is not None:
        digest.update(str(recording.expected_speakers).encode())
        digest.update(b"\0")
    digest.update(recording.words.read_bytes() if recording.words.is_file() else b"")
    digest.update(b"\0")
    digest.update(repr(COLLARS_S).encode())
    return digest.hexdigest()


def score_recording(backend: Backend, recording: Recording, out_dir: Path) -> RecordingScore:
    """Diarize one recording and score it; the hypothesis RTTM is kept next to the score."""
    if recording.reference is None:
        raise ConfigError(f"{recording.id} has no reference; score it with score_owner_recording")
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = read_rttm(recording.reference)[recording.id]
    run = backend.diarize(recording, out_dir / "worker")
    write_rttm(out_dir / "hypothesis.rttm", recording.id, run.turns)
    words = load_words(recording.words)
    attributed = (
        None if words is None else [backend.attribute(words, run, smoothing=smoothing) for smoothing in (False, True)]
    )
    records = (
        None if words is None or attributed is None else [_wder_record(wder(reference, words, a)) for a in attributed]
    )
    return RecordingScore(
        corpus=recording.corpus,
        recording=recording.id,
        duration_s=recording.duration_s,
        der=tuple(_der_record(c, der(reference, run.turns, collar_s=c)) for c in COLLARS_S),
        reference_speakers=speaker_count(reference),
        hypothesis_speakers=speaker_count(run.turns),
        speaker_count_error=speaker_count_error(reference, run.turns),
        diarize_s=run.diarize_s,
        load_s=run.load_s,
        rtf=run.diarize_s / recording.duration_s,
        peak_footprint_bytes=run.peak_footprint_bytes,
        inputs=input_digest(recording),
        wder_raw=None if records is None else records[0],
        wder_smoothed=None if records is None else records[1],
        attributed_speakers_raw=None if attributed is None else len({x for x in attributed[0] if x is not None}),
        attributed_speakers_smoothed=None if attributed is None else len({x for x in attributed[1] if x is not None}),
    )


def score_owner_recording(backend: Backend, recording: Recording, out_dir: Path) -> OwnerScore:
    """Diarize one owner recording and count its speakers; the hypothesis RTTM is kept for the dispute builder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run = backend.diarize(recording, out_dir / "worker")
    write_rttm(out_dir / "hypothesis.rttm", recording.id, run.turns)
    words = load_words(recording.words)
    attributed = (
        None if words is None else [backend.attribute(words, run, smoothing=smoothing) for smoothing in (False, True)]
    )
    count = speaker_count(run.turns)
    expected = recording.expected_speakers
    return OwnerScore(
        corpus=recording.corpus,
        recording=recording.id,
        duration_s=recording.duration_s,
        hypothesis_speakers=count,
        expected_speakers=expected,
        speaker_count_error=None if expected is None else count - expected,
        diarize_s=run.diarize_s,
        load_s=run.load_s,
        rtf=run.diarize_s / recording.duration_s,
        peak_footprint_bytes=run.peak_footprint_bytes,
        inputs=input_digest(recording),
        attributed_speakers_raw=None if attributed is None else len({x for x in attributed[0] if x is not None}),
        attributed_speakers_smoothed=None if attributed is None else len({x for x in attributed[1] if x is not None}),
    )


def _der_score(record: DerRecord) -> DerScore:
    return DerScore(record.total, record.missed, record.false_alarm, record.confusion)


def summarize(scores: Sequence[RecordingScore | OwnerScore]) -> dict[str, object]:
    """Aggregate per corpus and over the primary corpora.

    Every figure of a public corpus is a rate and no recording id appears. The owner block lists speaker counts by the
    generic label the owner index requires (``rec-01``).
    """
    if not scores:
        raise ConfigError("no recordings were scored")
    owner = [s for s in scores if isinstance(s, OwnerScore)]
    referenced = [s for s in scores if isinstance(s, RecordingScore)]

    def der_block(items: Sequence[RecordingScore]) -> dict[str, dict[str, float]]:
        return {
            f"collar_{collar}": {
                "macro": macro_rate([_der_score(s.der[i]) for s in items]),
                "pooled": pooled_rate([_der_score(s.der[i]) for s in items]),
            }
            for i, collar in enumerate(COLLARS_S)
        }

    def wder_block(items: Sequence[RecordingScore], smoothed: bool) -> float | None:
        records = [s.wder_smoothed if smoothed else s.wder_raw for s in items]
        if any(r is None for r in records):
            return None
        done = [r for r in records if r is not None]
        eligible = sum(r.eligible for r in done)
        return sum(r.errors for r in done) / eligible if eligible else None

    def smoothed_count_error(items: Sequence[RecordingScore]) -> float | None:
        counts = [s.attributed_speakers_smoothed for s in items]
        if any(c is None for c in counts):
            return None
        return sum(abs((c or 0) - s.reference_speakers) for c, s in zip(counts, items, strict=True)) / len(items)

    def block(items: Sequence[RecordingScore]) -> dict[str, object]:
        footprints = [s.peak_footprint_bytes for s in items if s.peak_footprint_bytes is not None]
        return {
            "recordings": len(items),
            "audio_hours": sum(s.duration_s for s in items) / 3600,
            "der": der_block(items),
            "wder_raw": wder_block(items, False),
            "wder_smoothed": wder_block(items, True),
            "mean_abs_speaker_count_error": sum(abs(s.speaker_count_error) for s in items) / len(items),
            "mean_signed_speaker_count_error": sum(s.speaker_count_error for s in items) / len(items),
            "mean_abs_smoothed_speaker_count_error": smoothed_count_error(items),
            "rtf": sum(s.diarize_s for s in items) / sum(s.duration_s for s in items),
            "peak_footprint_gib": max(footprints) / 2**30 if footprints else None,
        }

    def owner_block(items: Sequence[OwnerScore]) -> dict[str, object]:
        footprints = [s.peak_footprint_bytes for s in items if s.peak_footprint_bytes is not None]
        errors = [abs(s.speaker_count_error) for s in items if s.speaker_count_error is not None]
        return {
            "recordings": len(items),
            "audio_hours": sum(s.duration_s for s in items) / 3600,
            "hypothesis_speakers": {s.recording: s.hypothesis_speakers for s in items},
            "smoothed_attributed_speakers": {
                s.recording: s.attributed_speakers_smoothed for s in items if s.attributed_speakers_smoothed is not None
            },
            "mean_abs_speaker_count_error": sum(errors) / len(errors) if errors else None,
            "rtf": sum(s.diarize_s for s in items) / sum(s.duration_s for s in items),
            "peak_footprint_gib": max(footprints) / 2**30 if footprints else None,
        }

    corpora = sorted({s.corpus for s in referenced})
    primary = [s for s in referenced if s.corpus in PRIMARY_CORPORA]
    has_both = all(any(s.corpus == corpus for s in referenced) for corpus in PRIMARY_CORPORA)
    blocks: dict[str, object] = {corpus: block([s for s in referenced if s.corpus == corpus]) for corpus in corpora}
    if owner:
        blocks[OWNER] = owner_block(owner)
    return {
        "corpora": blocks,
        "primary_macro_der_collar_0": macro_rate([_der_score(s.der[0]) for s in primary]) if has_both else None,
    }


def recordings_of(corpus: str, only: Sequence[str] = (), root: Path | None = None) -> list[Recording]:
    corpus_dir = (root or recordings_dir()) / corpus
    if corpus == OWNER:
        found = [
            Recording(corpus, o.entry.id, o.wav, None, o.duration_s, o.entry.speakers)
            for o in load_owner(corpus_dir)
            if not only or o.entry.id in only
        ]
    else:
        index: CorpusIndex = load_index(corpus_dir)
        found = [
            Recording(corpus, e.id, corpus_dir / e.audio, corpus_dir / e.reference_rttm, e.duration_s)
            for e in index.recordings
            if not only or e.id in only
        ]
    missing = set(only) - {r.id for r in found}
    if missing:
        raise ConfigError(f"{corpus} has no recording(s) {', '.join(sorted(missing))}")
    return found


def run_corpora(
    backend: Backend,
    diarizer: str,
    recordings: Sequence[Recording],
    runs_dir: Path,
) -> list[RecordingScore]:
    return _resume(RecordingScore, score_recording, backend, diarizer, recordings, runs_dir)


def run_owner(
    backend: Backend,
    diarizer: str,
    recordings: Sequence[Recording],
    runs_dir: Path,
) -> list[OwnerScore]:
    return _resume(OwnerScore, score_owner_recording, backend, diarizer, recordings, runs_dir)


def _resume[S: (RecordingScore, OwnerScore)](
    model: type[S],
    scorer: Callable[[Backend, Recording, Path], S],
    backend: Backend,
    diarizer: str,
    recordings: Sequence[Recording],
    runs_dir: Path,
) -> list[S]:
    scores: list[S] = []
    for recording in recordings:
        out_dir = runs_dir / diarizer / recording.corpus / recording.id
        path = out_dir / SCORE_FILE
        cached = model.model_validate_json(path.read_text(encoding="utf-8")) if path.is_file() else None
        if cached is not None and cached.inputs == input_digest(recording):
            scores.append(cached)
            continue
        score = scorer(backend, recording, out_dir)
        if cached is not None:
            # Only the scoring inputs changed. A resumed worker reports no work, so the original timing stands.
            score = score.model_copy(
                update={
                    "diarize_s": cached.diarize_s,
                    "load_s": cached.load_s,
                    "rtf": cached.rtf,
                    "peak_footprint_bytes": cached.peak_footprint_bytes,
                }
            )
        atomic.write_text(path, score.model_dump_json(indent=2) + "\n")
        scores.append(score)
    return scores


class ProductionBackend:
    """Run a registered diarizer through the worker of ``anyverbatim.diarize``, one file per request."""

    def __init__(
        self, diarizer_id: str, worker: Callable[[DiarizeRequest], DiarizeResult] = diarize_runner.run_worker
    ) -> None:
        self._id = diarizer_id
        self._worker = worker

    def diarize(self, recording: Recording, out_dir: Path) -> Run:
        definition = diarize_registry.get(self._id)
        diarize_registry.check_dependencies(definition)
        job = DiarizeFileJob(id=recording.id, wav=str(recording.wav))
        result = self._worker(DiarizeRequest(diarizer=definition, out_dir=str(out_dir), files=(job,)))
        stored = read_file_diarization(out_dir, recording.id, self._id)
        turns = tuple(Turn(t.start, t.end, f"c{t.speaker}") for t in stored.turns)
        metrics = result.metrics
        handle = load_diarization(out_dir, recording.id, self._id)
        return Run(turns, metrics.diarize_s, metrics.load_s, metrics.peak_footprint_bytes, handle)

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str | None]:
        assigned = assign(list(words), cast(StoredDiarization, run.handle), smoothing=smoothing)
        return list(assigned.speakers)


def attribute_by_turns(words: Sequence[Word], turns: Sequence[Turn]) -> list[str]:
    """Attribute words as WhisperX does: largest overlap, else the nearest turn within 0.5 s, else the previous word."""
    labels: list[str] = []
    for word in words:
        overlap: dict[str, float] = {}
        for turn in turns:
            seconds = min(word.end, turn.end) - max(word.start, turn.start)
            if seconds > 0:
                overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + seconds
        if overlap:
            labels.append(max(overlap, key=lambda s: (overlap[s], s)))
            continue
        near = [(max(turn.start - word.end, word.start - turn.end, 0.0), turn.speaker) for turn in turns]
        distance, speaker = min(near, default=(GAP_S + 1, "?"))
        labels.append(speaker if distance <= GAP_S else (labels[-1] if labels else speaker))
    return labels


def _turns_for(path: Path, recording_id: str) -> tuple[Turn, ...]:
    """Read the turns of ``recording_id`` from an RTTM an external command wrote.

    Every prepared recording is named ``audio.wav``, so a script that takes the id from the file name writes the wrong
    one. A file with a single id is taken as this recording's; one with several ids but not this one is an error,
    never silently "no speech".
    """
    parsed = read_rttm(path)
    if recording_id in parsed:
        return tuple(parsed[recording_id])
    if len(parsed) == 1:
        return tuple(next(iter(parsed.values())))
    if parsed:
        raise EngineError(f"{path} has records for {sorted(parsed)} but none for {recording_id}")
    return ()


class ExternalBackend:
    """Run ``<command> WAV OUT.rttm`` (pyannote in its throwaway project) and read the RTTM it writes.

    A sibling ``OUT.exclusive.rttm``, when written, is what words are attributed from; DER always uses the regular
    output. ``PYANNOTE_METRICS_ENABLED=0`` is set for the child. There is no model load to separate out, so the time
    is the child's wall time.
    """

    def __init__(self, command: Sequence[str]) -> None:
        if not command:
            raise ConfigError("an external diarizer needs a command; pass --external-command")
        self._command = list(command)

    def diarize(self, recording: Recording, out_dir: Path) -> Run:
        from anyverbatim.mlx_worker import default_memory_cap_bytes, mlx_lock

        out_dir.mkdir(parents=True, exist_ok=True)
        rttm = out_dir / "output.rttm"
        argv = [*self._command, str(recording.wav), str(rttm)]
        started = time.monotonic()
        deadline = started + EXTERNAL_TIMEOUT_FACTOR * recording.duration_s + EXTERNAL_TIMEOUT_FLOOR_S
        env = {**os.environ, **PYANNOTE_ENV}
        with (
            mlx_lock(),
            proc.background(
                argv, log=out_dir / "worker.log", memory_watch=proc.MemoryWatch(default_memory_cap_bytes()), env=env
            ) as child,
        ):
            while child.alive():
                if time.monotonic() > deadline:
                    raise EngineError(f"{argv[0]} did not finish within {deadline - started:.0f} s")
                time.sleep(0.5)
            code, peak = child.returncode, child.peak_footprint_bytes
        elapsed = time.monotonic() - started
        if code != 0 or not rttm.is_file():
            raise EngineError(f"{argv[0]} exited with {code} for {recording.id}; see {out_dir / 'worker.log'}")
        turns = _turns_for(rttm, recording.id)
        exclusive = rttm.with_name("output.exclusive.rttm")
        attributed = _turns_for(exclusive, recording.id) if exclusive.is_file() else turns
        return Run(turns, elapsed, 0.0, peak, attributed)

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str | None]:
        return list(attribute_by_turns(words, cast(Sequence[Turn], run.handle)))


def make_backend(diarizer: str, external_command: str | None) -> Backend:
    if diarizer.startswith("d-pyannote"):
        return ExternalBackend(shlex.split(external_command or ""))
    return ProductionBackend(diarizer)


@click.group()
def main() -> None:
    """Diarization selection eval (data under $ANYVERBATIM_EVAL_DIR/recordings)."""


@main.command()
@click.option("--diarizer", required=True, help="Registry id (d-nemotron-8bit, d-nemotron-bf16) or d-pyannote-*.")
@click.option(
    "--corpus", "corpora", multiple=True, type=click.Choice(CORPORA), default=DEFAULT_CORPORA, show_default=True
)
@click.option("--recordings", help="Comma-separated recording ids (default: every one in the corpus).")
@click.option("--external-command", help="For d-pyannote-*: command taking WAV and OUT.rttm arguments.")
def run(diarizer: str, corpora: tuple[str, ...], recordings: str | None, external_command: str | None) -> None:
    """Diarize and score; resumes per recording."""
    only = [r for r in (recordings or "").split(",") if r]
    selected = [r for corpus in corpora for r in recordings_of(corpus, only if len(corpora) == 1 else ())]
    runs_dir = recordings_dir() / "runs" / "diarization"
    backend = make_backend(diarizer, external_command)
    scores: list[RecordingScore | OwnerScore] = [
        *run_corpora(backend, diarizer, [r for r in selected if r.reference is not None], runs_dir),
        *run_owner(backend, diarizer, [r for r in selected if r.reference is None], runs_dir),
    ]
    summary = summarize(scores)
    atomic.write_text(runs_dir / diarizer / SUMMARY_FILE, json.dumps(summary, indent=2) + "\n")
    click.echo(json.dumps(summary, indent=2))


def der_rates(scores: Sequence[RecordingScore], collar_index: int = 0) -> dict[str, float]:
    """Per-recording DER keyed ``corpus/recording``, for pairing two diarizers."""
    return {
        f"{s.corpus}/{s.recording}": _der_score(s.der[collar_index]).rate for s in scores if s.corpus in PRIMARY_CORPORA
    }


def compare(challenger: Sequence[RecordingScore], incumbent: Sequence[RecordingScore], seed: int = 0) -> MacroBootstrap:
    """Bootstrap the macro DER difference, challenger minus incumbent, at collar 0 over the primary corpora."""
    a, b = der_rates(challenger), der_rates(incumbent)
    if set(a) != set(b):
        raise ConfigError("the two runs scored different primary recordings; finish both first")
    keys = sorted(a)
    return paired_macro_bootstrap([a[k] for k in keys], [b[k] for k in keys], seed)


def _stored_scores(diarizer: str) -> list[RecordingScore]:
    base = recordings_dir() / "runs" / "diarization" / diarizer
    return [
        RecordingScore.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(base.glob(f"*/*/{SCORE_FILE}"))
        if p.parent.parent.name != OWNER
    ]


def _stored_owner_scores(diarizer: str) -> list[OwnerScore]:
    base = recordings_dir() / "runs" / "diarization" / diarizer / OWNER
    return [OwnerScore.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(base.glob(f"*/{SCORE_FILE}"))]


@main.command(name="compare")
@click.option("--challenger", required=True)
@click.option("--incumbent", default="d-nemotron-8bit", show_default=True)
@click.option("--seed", default=0, show_default=True)
def compare_command(challenger: str, incumbent: str, seed: int) -> None:
    """Paired macro-DER bootstrap (collar 0, AMI and AliMeeting) between two finished runs."""
    result = compare(_stored_scores(challenger), _stored_scores(incumbent), seed)
    click.echo(
        f"{challenger} minus {incumbent}: {result.difference * 100:+.2f} points, "
        f"95% CI [{result.low * 100:+.2f}, {result.high * 100:+.2f}], excludes 0: {result.excludes_zero}"
    )


@main.command()
@click.option("--diarizer", required=True)
def summary(diarizer: str) -> None:
    """Re-aggregate the per-recording scores already on disk."""
    scores: list[RecordingScore | OwnerScore] = [*_stored_scores(diarizer), *_stored_owner_scores(diarizer)]
    click.echo(json.dumps(summarize(scores), indent=2))


def run_cli() -> None:
    try:
        main(standalone_mode=False)
    except AnyverbatimError as exc:
        raise SystemExit(f"error: {exc}") from None
    except click.exceptions.Abort:
        raise SystemExit(130) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    run_cli()
