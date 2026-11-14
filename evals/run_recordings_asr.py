"""Language-policy and meeting ASR eval (E2/E3): ``uv run python -m evals.run_recordings_asr``.

One hypothesis at a time: ``run --hyp H`` decodes AMI, AliMeeting, the ASCEND sessions and clips with it (the owner's
recordings only with ``--corpus owner``), resuming per chunk and per finished worker folder. Each worker folder keeps
a digest of its audio, so replaced audio under the same id is refused, not scored with the old rows.

Then ``score --hyp H`` (error rate and translation incidents per corpus), ``compare --a H --b H`` (paired bootstrap
per corpus and over the two meeting corpora; the recording-clustered ``decision`` interval is the one to decide on)
and ``words --hyp H`` (the ``words.json`` that ``evals.run_diarization`` scores WDER with). Everything prints
aggregates only; the owner's disputes come from ``python -m evals.disputes``.

No hypothesis gets a language hint. The two language-ID hypotheses choose within ``run_asr.ALLOWED_LANGUAGES``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import click

from evals.normalize import View
from evals.recordings_corpus import (
    ALIMEETING,
    AMI,
    ASCEND_UTTERANCES,
    ASR_CORPORA,
    FOLDER_CORPORA,
    METRICS,
    PUBLIC_CORPORA,
    Item,
    asr_layout,
    batch_key,
    finished_folders,
    finished_rows,
    load_items,
    recording_job,
    stamp_inputs,
)
from evals.recordings_data import OWNER
from evals.recordings_scoring import (
    MEETINGS,
    Scored,
    WindowScore,
    language_seconds,
    paired,
    score_items,
    summarize_corpus,
    write_words_for,
)
from evals.store import Layout
from ytverbatim import config
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis, RunMetrics, WorkerRequest
from ytverbatim.asr.runner import WorkerResult, memory_cap_bytes, run_worker
from ytverbatim.errors import ConfigError, YtverbatimError

E2_HYPOTHESES = ("w-prod", "w-vad", "w-vad-lid", "w-runs-lid")
UTTERANCE_BATCH = 50
MEETING_CORPORA = (AMI, ALIMEETING)
SCORED_CORPORA = tuple(c for c in ASR_CORPORA if c != OWNER)

Worker = Callable[[WorkerRequest], WorkerResult]


@dataclass(frozen=True, slots=True)
class RunSummary:
    files_decoded: int
    files_resumed: int
    metrics: tuple[RunMetrics, ...]


def batches(items: Sequence[Item], size: int) -> list[list[Item]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def run_corpus(
    layout: Layout,
    hypothesis: str,
    items: Sequence[Item],
    *,
    batch_size: int = 1,
    worker: Worker = run_worker,
    memory_cap: int | None = None,
) -> RunSummary:
    """Decode ``items`` with one hypothesis, one worker per batch; what a finished worker holds is not decoded again.

    Only the unfinished items are batched, so a different ``batch_size`` or a subset never decodes an item twice.
    """
    definition = registry.get(hypothesis)
    done = finished_folders(layout, hypothesis, items)
    pending = [i for i in items if i.id not in done]
    metrics: list[RunMetrics] = []
    for batch in batches(pending, batch_size):
        stamp_inputs(layout.run_dir(hypothesis) / batch_key([i.id for i in batch]), batch)
        request = WorkerRequest(
            engine=definition,
            out_dir=str(layout.run_dir(hypothesis) / batch_key([i.id for i in batch])),
            files=tuple(recording_job(i, definition) for i in batch),
            memory_cap_bytes=memory_cap,
        )
        metrics.append(worker(request).metrics)
    return RunSummary(len(pending), len(done), tuple(metrics))


def scored(corpus: str, hypothesis: str, view: View, items: Sequence[Item], root: Path | None = None) -> Scored:
    if corpus not in METRICS:
        raise ConfigError(f"{corpus} has no reference to score against; it is judged by disputes only")
    rows = finished_rows(asr_layout(corpus, root), hypothesis, items)
    missing = [i.id for i in items if i.id not in rows]
    if missing:
        raise ConfigError(
            f"{hypothesis} is not finished for {len(missing)} of {len(items)} {corpus} recordings; run it first"
        )
    return score_items(items, rows, METRICS[corpus], view)


def _only(corpora: Sequence[str], recordings: str | None) -> list[str]:
    ids = [r for r in (recordings or "").split(",") if r]
    if ids and len(corpora) != 1:
        raise click.UsageError("--recordings needs exactly one --corpus")
    return ids


def _load_all(corpora: Sequence[str], only: Sequence[str]) -> dict[str, list[Item]]:
    """Load every selected corpus first, so a missing one stops the command before any decoding or scoring."""
    return {corpus: load_items(corpus, only) for corpus in corpora}


_corpus_option = click.option(
    "--corpus",
    "corpora",
    multiple=True,
    type=click.Choice(ASR_CORPORA),
    default=PUBLIC_CORPORA,
    show_default=True,
    help="The owner corpus runs only when named.",
)
_recordings_option = click.option("--recordings", help="Comma-separated recording ids (needs one --corpus).")
_hyp_option = click.option("--hyp", "hypothesis", required=True, type=click.Choice(E2_HYPOTHESES))
_view_option = click.option(
    "--view",
    type=click.Choice(["raw", "itn"]),
    default="itn",
    show_default=True,
    help="itn maps quantities and spelled-out numbers to digits on both sides; raw keeps surface form.",
)


@click.group()
def main() -> None:
    """Language policy and meeting ASR eval (data under $YTVERBATIM_EVAL_DIR/recordings)."""


@main.command()
@_hyp_option
@_corpus_option
@_recordings_option
@click.option("--batch-size", default=UTTERANCE_BATCH, show_default=True, help="Clips per worker (ASCEND utterances).")
def run(hypothesis: str, corpora: tuple[str, ...], recordings: str | None, batch_size: int) -> None:
    """Decode with one hypothesis; resumes per chunk and per finished worker."""
    only = _only(corpora, recordings)
    registry.check_dependencies(registry.get(hypothesis))
    cap = memory_cap_bytes(config.load_options().asr.memory_cap_gib)
    loaded = _load_all(corpora, only)
    for corpus, items in loaded.items():
        finished_folders(asr_layout(corpus), hypothesis, items)
    for corpus, items in loaded.items():
        size = batch_size if corpus == ASCEND_UTTERANCES else 1
        summary = run_corpus(asr_layout(corpus), hypothesis, items, batch_size=size, worker=run_worker, memory_cap=cap)
        decode = sum(m.decode_s for m in summary.metrics)
        audio = sum(m.audio_s for m in summary.metrics)
        click.echo(
            f"{hypothesis} {corpus}: {summary.files_decoded} decoded, {summary.files_resumed} resumed, "
            f"{sum(m.chunks_decoded for m in summary.metrics)} chunks, RTF {decode / audio if audio else 0:.3f}"
        )


@main.command()
@_hyp_option
@_corpus_option
@_recordings_option
@_view_option
def score(hypothesis: str, corpora: tuple[str, ...], recordings: str | None, view: View) -> None:
    """Error rate and translation incidents per corpus; for the owner corpus, the languages the run chose."""
    report: dict[str, object] = {}
    for corpus, items in _load_all(corpora, _only(corpora, recordings)).items():
        if corpus == OWNER:
            rows = finished_rows(asr_layout(corpus), hypothesis, items)
            report[corpus] = {
                "recordings": len(items),
                "audio_hours": sum(i.duration_s for i in items) / 3600,
                "decoded_recordings": len(rows),
                "language_seconds": language_seconds([r for rs in rows.values() for r in rs]),
            }
            continue
        scores = scored(corpus, hypothesis, view, items)
        report[corpus] = summarize_corpus(corpus, METRICS[corpus], view, items, scores).model_dump()
    click.echo(json.dumps({"hypothesis": hypothesis, "corpora": report}, indent=2))


@main.command(name="compare")
@click.option("--a", "first", required=True, type=click.Choice(E2_HYPOTHESES))
@click.option("--b", "second", required=True, type=click.Choice(E2_HYPOTHESES))
@click.option("--corpus", "corpora", multiple=True, type=click.Choice(SCORED_CORPORA), default=PUBLIC_CORPORA)
@_view_option
@click.option("--seed", default=0, show_default=True)
def compare_command(first: str, second: str, corpora: tuple[str, ...], view: View, seed: int) -> None:
    """Paired bootstrap of the pooled error-rate difference A minus B, per corpus.

    ``decision`` is the interval to decide on: it resamples recordings, because the windows of one recording are
    correlated. ``by_window`` resamples windows and is narrower; it is there for comparison.

    ``meetings`` pools the AMI and AliMeeting windows by reference unit, so AliMeeting's characters outnumber AMI's
    words and dominate the figure; ``macro_difference`` averages the two corpora's differences with equal weight.
    """
    left: dict[str, list[WindowScore]] = {}
    right: dict[str, list[WindowScore]] = {}
    for corpus, items in _load_all(corpora, ()).items():
        left[corpus] = scored(corpus, first, view, items).windows
        right[corpus] = scored(corpus, second, view, items).windows
    results = {corpus: paired(left[corpus], right[corpus], seed).model_dump() for corpus in corpora}
    if set(MEETING_CORPORA) <= set(corpora):
        results[MEETINGS] = {
            **paired(_pool(left, MEETING_CORPORA), _pool(right, MEETING_CORPORA), seed).model_dump(),
            "macro_difference": sum(results[c]["difference"] for c in MEETING_CORPORA) / len(MEETING_CORPORA),
            "note": "pooled over reference units: AliMeeting characters outnumber AMI words and dominate",
        }
    click.echo(
        json.dumps(
            {
                "a": first,
                "b": second,
                "difference_is_a_minus_b": True,
                "decision_interval": "decision",
                "results": results,
            },
            indent=2,
        )
    )


def _pool(windows: Mapping[str, Sequence[WindowScore]], corpora: Sequence[str]) -> list[WindowScore]:
    """Prefix each window's item with its corpus, so recordings of different corpora never collide when pooled."""
    return [replace(w, item=f"{c}/{w.item}") for c in corpora for w in windows[c]]


@main.command()
@_hyp_option
@click.option(
    "--corpus",
    "corpora",
    multiple=True,
    type=click.Choice(FOLDER_CORPORA),
    default=tuple(c for c in FOLDER_CORPORA if c != OWNER),
    help="The owner corpus runs only when named.",
)
@_recordings_option
def words(hypothesis: str, corpora: tuple[str, ...], recordings: str | None) -> None:
    """Write words.json next to each recording's audio from this hypothesis's rows (for WDER)."""
    ready: dict[str, tuple[list[Item], dict[str, list[ChunkHypothesis]]]] = {}
    for corpus, items in _load_all(corpora, _only(corpora, recordings)).items():
        rows = finished_rows(asr_layout(corpus), hypothesis, items)
        unfinished = [i.id for i in items if i.id not in rows]
        if unfinished:
            raise ConfigError(f"{hypothesis} is not finished for {len(unfinished)} {corpus} recordings; run it first")
        ready[corpus] = (items, rows)
    for corpus, (items, rows) in ready.items():
        click.echo(f"{corpus}: wrote words.json for {write_words_for(items, rows)}")


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
