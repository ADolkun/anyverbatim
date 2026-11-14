"""Language-policy and meeting ASR eval (E2/E3): ``uv run python -m evals.run_recordings_asr``.

One hypothesis at a time: ``run --hyp H`` decodes AMI, AliMeeting, the ASCEND sessions and clips and the owner's
recordings with it, resuming per chunk and per finished worker folder. Then ``score --hyp H`` (error rate and
translation incidents per corpus), ``compare --a H --b H`` (paired bootstrap per corpus and over the two meeting
corpora) and ``words --hyp H`` (the ``words.json`` that ``evals.run_diarization`` scores WDER with). Everything prints
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
    Item,
    asr_layout,
    batch_key,
    finished_ids,
    load_items,
)
from evals.recordings_data import OWNER
from evals.recordings_scoring import (
    MEETINGS,
    WindowScore,
    language_seconds,
    paired,
    score_items,
    summarize_corpus,
    write_words_for,
)
from evals.run_asr import ALLOWED_LANGUAGES
from evals.store import Layout, rows_by_file
from ytverbatim import config
from ytverbatim.asr import registry
from ytverbatim.asr.models import FileJob, RunMetrics, WorkerRequest
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.asr.runner import WorkerResult, finished, memory_cap_bytes, run_worker
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


def recording_job(item: Item, definition: EngineDef) -> FileJob:
    """Build a job with no language hint; a language-ID hypothesis gets the allowed set instead."""
    detects = definition.language_policy() == "detect"
    return FileJob(id=item.id, wav=str(item.wav), languages=ALLOWED_LANGUAGES if detects else ())


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
    """Decode ``items`` with one hypothesis, one worker per batch; a batch finished before is not decoded again."""
    definition = registry.get(hypothesis)
    decoded = resumed = 0
    metrics: list[RunMetrics] = []
    for batch in batches(items, batch_size):
        request = WorkerRequest(
            engine=definition,
            out_dir=str(layout.run_dir(hypothesis) / batch_key([i.id for i in batch])),
            files=tuple(recording_job(i, definition) for i in batch),
            memory_cap_bytes=memory_cap,
        )
        if finished(request) is not None:
            resumed += len(batch)
            continue
        metrics.append(worker(request).metrics)
        decoded += len(batch)
    return RunSummary(decoded, resumed, tuple(metrics))


def scored_windows(
    corpus: str, hypothesis: str, view: View, items: Sequence[Item], root: Path | None = None
) -> list[WindowScore]:
    if corpus not in METRICS:
        raise ConfigError(f"{corpus} has no reference to score against; it is judged by disputes only")
    layout = asr_layout(corpus, root)
    missing = [i.id for i in items if i.id not in finished_ids(layout, hypothesis)]
    if missing:
        raise ConfigError(
            f"{hypothesis} is not finished for {len(missing)} of {len(items)} {corpus} recordings; run it first"
        )
    return score_items(items, rows_by_file(layout, hypothesis), METRICS[corpus], view)


def _only(corpora: Sequence[str], recordings: str | None) -> list[str]:
    ids = [r for r in (recordings or "").split(",") if r]
    if ids and len(corpora) != 1:
        raise click.UsageError("--recordings needs exactly one --corpus")
    return ids


_corpus_option = click.option(
    "--corpus", "corpora", multiple=True, type=click.Choice(ASR_CORPORA), default=ASR_CORPORA, show_default=True
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
    for corpus in corpora:
        items = load_items(corpus, only)
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
    only = _only(corpora, recordings)
    report: dict[str, object] = {}
    for corpus in corpora:
        items = load_items(corpus, only)
        if corpus == OWNER:
            layout = asr_layout(corpus)
            rows = rows_by_file(layout, hypothesis)
            report[corpus] = {
                "recordings": len(items),
                "audio_hours": sum(i.duration_s for i in items) / 3600,
                "decoded_recordings": len(finished_ids(layout, hypothesis) & {i.id for i in items}),
                "language_seconds": language_seconds([r for rs in rows.values() for r in rs]),
            }
            continue
        windows = scored_windows(corpus, hypothesis, view, items)
        report[corpus] = summarize_corpus(corpus, METRICS[corpus], view, items, windows).model_dump()
    click.echo(json.dumps({"hypothesis": hypothesis, "corpora": report}, indent=2))


@main.command(name="compare")
@click.option("--a", "first", required=True, type=click.Choice(E2_HYPOTHESES))
@click.option("--b", "second", required=True, type=click.Choice(E2_HYPOTHESES))
@click.option("--corpus", "corpora", multiple=True, type=click.Choice(SCORED_CORPORA), default=SCORED_CORPORA)
@_view_option
@click.option("--seed", default=0, show_default=True)
def compare_command(first: str, second: str, corpora: tuple[str, ...], view: View, seed: int) -> None:
    """Paired bootstrap of the pooled error-rate difference A minus B over windows, per corpus.

    ``meetings`` pools the AMI and AliMeeting windows, so a recording's weight is its reference length whichever of
    WER or CER the corpus uses.
    """
    left: dict[str, list[WindowScore]] = {}
    right: dict[str, list[WindowScore]] = {}
    for corpus in corpora:
        items = load_items(corpus)
        left[corpus] = scored_windows(corpus, first, view, items)
        right[corpus] = scored_windows(corpus, second, view, items)
    results = {corpus: paired(left[corpus], right[corpus], seed).model_dump() for corpus in corpora}
    if set(MEETING_CORPORA) <= set(corpora):
        results[MEETINGS] = paired(_pool(left, MEETING_CORPORA), _pool(right, MEETING_CORPORA), seed).model_dump()
    click.echo(json.dumps({"a": first, "b": second, "difference_is_a_minus_b": True, "results": results}, indent=2))


def _pool(windows: Mapping[str, Sequence[WindowScore]], corpora: Sequence[str]) -> list[WindowScore]:
    """Prefix each window's item with its corpus, so recordings of different corpora never collide when pooled."""
    return [replace(w, item=f"{c}/{w.item}") for c in corpora for w in windows[c]]


@main.command()
@_hyp_option
@click.option("--corpus", "corpora", multiple=True, type=click.Choice(FOLDER_CORPORA), default=FOLDER_CORPORA)
@_recordings_option
def words(hypothesis: str, corpora: tuple[str, ...], recordings: str | None) -> None:
    """Write words.json next to each recording's audio from this hypothesis's rows (for WDER)."""
    only = _only(corpora, recordings)
    for corpus in corpora:
        items = load_items(corpus, only)
        layout = asr_layout(corpus)
        unfinished = [i.id for i in items if i.id not in finished_ids(layout, hypothesis)]
        if unfinished:
            raise ConfigError(f"{hypothesis} is not finished for {len(unfinished)} {corpus} recordings; run it first")
        click.echo(f"{corpus}: wrote words.json for {write_words_for(items, rows_by_file(layout, hypothesis))}")


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
