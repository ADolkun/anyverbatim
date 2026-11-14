"""Diarization selection eval (E1): ``uv run python -m evals.run_diarization run --diarizer ID --corpus ami ...``.

Runs one diarizer over prepared recordings (``evals.recordings_data``), then scores each at collar 0 and collar 0.25
with overlap scored, WDER with and without smoothing, speaker-count error, real-time factor and peak footprint. Every
recording is resumable and everything is written under ``$YTVERBATIM_EVAL_DIR/recordings/runs/diarization/<id>/``.

Two seams keep the production worker and pyannote out of this module's imports:

- ``ProductionBackend`` drives ``ytverbatim.diarize`` (unit U6a), loaded lazily, so the Nemotron ids run through the
  same worker the pipeline uses;
- ``ExternalBackend`` runs any command that writes an RTTM, for pyannote in its throwaway uv project.

WDER needs the words of an ASR run per recording: ``words.json`` (a list of ``{"text", "start", "end"}``) next to
each recording's audio. Without it WDER is skipped.
"""

from __future__ import annotations

import importlib
import json
import os
import shlex
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import click
from pydantic import BaseModel, ConfigDict

from evals.diarization import (
    COLLARS_S,
    DerScore,
    Turn,
    WderScore,
    der,
    macro_rate,
    pooled_rate,
    read_rttm,
    speaker_count,
    speaker_count_error,
    wder,
    write_rttm,
)
from evals.recordings_data import CorpusIndex, load_index, recordings_dir
from ytverbatim import atomic, proc
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError, EngineError, YtverbatimError

PRIMARY_CORPORA = ("ami", "alimeeting")
CORPORA = (*PRIMARY_CORPORA, "ascend")
WORDS_FILE = "words.json"
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
    reference: Path
    duration_s: float

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

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str]: ...


class DerRecord(_Frozen):
    collar_s: float
    total: float
    missed: float
    false_alarm: float
    confusion: float
    rate: float


class WderRecord(_Frozen):
    errors: int
    eligible: int
    total_words: int
    rate: float


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
    # None when the recording has no words.json.
    wder_raw: WderRecord | None
    wder_smoothed: WderRecord | None


def _der_record(collar_s: float, score: DerScore) -> DerRecord:
    return DerRecord(
        collar_s=collar_s,
        total=score.total,
        missed=score.missed,
        false_alarm=score.false_alarm,
        confusion=score.confusion,
        rate=score.rate,
    )


def _wder_record(score: WderScore) -> WderRecord:
    return WderRecord(errors=score.errors, eligible=score.eligible, total_words=score.total_words, rate=score.rate)


def load_words(path: Path) -> list[Word] | None:
    if not path.is_file():
        return None
    return [Word.model_validate(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def score_recording(backend: Backend, recording: Recording, out_dir: Path) -> RecordingScore:
    """Diarize one recording and score it; the hypothesis RTTM is kept next to the score."""
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = read_rttm(recording.reference)[recording.id]
    run = backend.diarize(recording, out_dir / "worker")
    write_rttm(out_dir / "hypothesis.rttm", recording.id, run.turns)
    words = load_words(recording.words)
    wders = (
        (None, None)
        if words is None
        else tuple(
            _wder_record(wder(reference, words, backend.attribute(words, run, smoothing=s))) for s in (False, True)
        )
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
        wder_raw=wders[0],
        wder_smoothed=wders[1],
    )


def _der_score(record: DerRecord) -> DerScore:
    return DerScore(record.total, record.missed, record.false_alarm, record.confusion)


def summarize(scores: Sequence[RecordingScore]) -> dict[str, object]:
    """Aggregate per corpus and over the primary corpora. Every figure is a rate; no recording ids appear."""
    if not scores:
        raise ConfigError("no recordings were scored")

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
            "rtf": sum(s.diarize_s for s in items) / sum(s.duration_s for s in items),
            "peak_footprint_gib": max(footprints) / 2**30 if footprints else None,
        }

    corpora = sorted({s.corpus for s in scores})
    primary = [s for s in scores if s.corpus in PRIMARY_CORPORA]
    return {
        "corpora": {corpus: block([s for s in scores if s.corpus == corpus]) for corpus in corpora},
        "primary_macro_der_collar_0": macro_rate([_der_score(s.der[0]) for s in primary]) if primary else None,
    }


def recordings_of(corpus: str, only: Sequence[str] = (), root: Path | None = None) -> list[Recording]:
    corpus_dir = (root or recordings_dir()) / corpus
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
    scores: list[RecordingScore] = []
    for recording in recordings:
        out_dir = runs_dir / diarizer / recording.corpus / recording.id
        path = out_dir / SCORE_FILE
        if path.is_file():
            scores.append(RecordingScore.model_validate_json(path.read_text(encoding="utf-8")))
            continue
        score = score_recording(backend, recording, out_dir)
        atomic.write_text(path, score.model_dump_json(indent=2) + "\n")
        scores.append(score)
    return scores


class ProductionBackend:
    """Run a registered diarizer through the worker of ``ytverbatim.diarize``, one file per request."""

    def __init__(self, diarizer_id: str) -> None:
        self._id = diarizer_id
        self._module = {
            name: importlib.import_module(f"ytverbatim.diarize.{name}")
            for name in ("registry", "protocol", "runner", "attribution")
        }

    def diarize(self, recording: Recording, out_dir: Path) -> Run:
        registry, protocol, runner = (self._module[n] for n in ("registry", "protocol", "runner"))
        definition = registry.get(self._id)
        registry.check_dependencies(definition)
        job = protocol.FileJob(id=recording.id, wav=str(recording.wav))
        request = protocol.DiarizeRequest(diarizer=definition, out_dir=str(out_dir), files=(job,))
        result = runner.run_worker(request)
        stored = protocol.read_file_diarization(out_dir, recording.id)
        turns = tuple(Turn(t.start, t.end, f"c{t.speaker}") for t in stored.turns)
        metrics = result.metrics
        return Run(
            turns, metrics.diarize_s, metrics.load_s, metrics.peak_footprint_bytes, protocol.load(out_dir, recording.id)
        )

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str]:
        assigned = self._module["attribution"].assign(list(words), run.handle, smoothing=smoothing)
        return [speaker or "?" for speaker in cast(Sequence[str | None], assigned.speakers)]


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
        from ytverbatim.asr.runner import default_memory_cap_bytes, worker_lock

        out_dir.mkdir(parents=True, exist_ok=True)
        rttm = out_dir / "output.rttm"
        argv = [*self._command, str(recording.wav), str(rttm)]
        started = time.monotonic()
        deadline = started + EXTERNAL_TIMEOUT_FACTOR * recording.duration_s + EXTERNAL_TIMEOUT_FLOOR_S
        env = {**os.environ, **PYANNOTE_ENV}
        with (
            worker_lock(),
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
        turns = tuple(read_rttm(rttm).get(recording.id, []))
        exclusive = rttm.with_name("output.exclusive.rttm")
        attributed = tuple(read_rttm(exclusive).get(recording.id, [])) if exclusive.is_file() else turns
        return Run(turns, elapsed, 0.0, peak, attributed)

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str]:
        return attribute_by_turns(words, cast(Sequence[Turn], run.handle))


def make_backend(diarizer: str, external_command: str | None) -> Backend:
    if diarizer.startswith("d-pyannote"):
        return ExternalBackend(shlex.split(external_command or ""))
    return ProductionBackend(diarizer)


@click.group()
def main() -> None:
    """Diarization selection eval (data under $YTVERBATIM_EVAL_DIR/recordings)."""


@main.command()
@click.option("--diarizer", required=True, help="Registry id (d-nemotron-8bit, d-nemotron-bf16) or d-pyannote-*.")
@click.option("--corpus", "corpora", multiple=True, type=click.Choice(CORPORA), default=CORPORA, show_default=True)
@click.option("--recordings", help="Comma-separated recording ids (default: every one in the corpus).")
@click.option("--external-command", help="For d-pyannote-*: command taking WAV and OUT.rttm arguments.")
def run(diarizer: str, corpora: tuple[str, ...], recordings: str | None, external_command: str | None) -> None:
    """Diarize and score; resumes per recording."""
    only = [r for r in (recordings or "").split(",") if r]
    selected = [r for corpus in corpora for r in recordings_of(corpus, only if len(corpora) == 1 else ())]
    runs_dir = recordings_dir() / "runs" / "diarization"
    scores = run_corpora(make_backend(diarizer, external_command), diarizer, selected, runs_dir)
    summary = summarize(scores)
    atomic.write_text(runs_dir / diarizer / SUMMARY_FILE, json.dumps(summary, indent=2) + "\n")
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.option("--diarizer", required=True)
def summary(diarizer: str) -> None:
    """Re-aggregate the per-recording scores already on disk."""
    base = recordings_dir() / "runs" / "diarization" / diarizer
    scores = [
        RecordingScore.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(base.glob(f"*/*/{SCORE_FILE}"))
    ]
    click.echo(json.dumps(summarize(scores), indent=2))


def run_cli() -> None:
    try:
        main(standalone_mode=False)
    except YtverbatimError as exc:
        raise SystemExit(f"error: {exc}") from None
    except click.exceptions.Abort:
        raise SystemExit(130) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    run_cli()
