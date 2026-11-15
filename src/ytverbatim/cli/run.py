from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import click

from ytverbatim import config, paths, pipeline, targets
from ytverbatim.acquire.session import REPORT_FILE, BatchReport, VideoStatus, stop_fix
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.preflight import fix_context
from ytverbatim.auth.source import resolve_source
from ytverbatim.cli import interactive
from ytverbatim.cli.fetch import EXIT_AUTH, EXIT_FAILED, EXIT_STOPPED
from ytverbatim.contract import LLMUsage, RecordingSource
from ytverbatim.errors import AuthError, BatchStopError, EngineMemoryError, YtverbatimError
from ytverbatim.index import INDEX_MD
from ytverbatim.llm import apply_spec
from ytverbatim.ytdlp.runner import redact


@click.command(name="run")
@click.argument("raw_targets", metavar="TARGETS...", nargs=-1, required=True)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output root, one folder per video or recording (default: the videos or recordings folder).",
)
@click.option(
    "--auth", "auth_spec", metavar="SPEC", help="YouTube auth source for this run: BROWSER[:DIR], cookie-file:PATH."
)
@click.option("--engine", "engine", metavar="ID", help="ASR hypothesis to run, for every source (default w-prod).")
@click.option(
    "--language",
    metavar="CODE",
    help="Force one spoken language (zh, en, ...) instead of YouTube's declared one or detection.",
)
@click.option(
    "--source",
    "transcript_source",
    type=click.Choice(["asr", "captions"]),
    help="YouTube: transcribe with ASR (default), or trust the manual caption track and fall back to ASR without one.",
)
@click.option(
    "--correct/--no-correct",
    default=None,
    help="LLM correction of misheard words. Default: on whenever there is an LLM; --correct without one is an error.",
)
@click.option(
    "--chapters", type=click.Choice(["auto", "none"]), help="Chapter headings: auto uses YouTube's chapters (default)."
)
@click.option(
    "--llm",
    "llm_spec",
    metavar="SPEC",
    help=(
        "LLM for punctuation, correction and numeric verdicts: PROVIDER:MODEL, or none for no LLM. "
        "Default: chatgpt:gpt-6.1-sol at medium effort (needs `ytverbatim login`); "
        "alternatives: claude-cli:claude-sonnet-5-5, openai-compat:MODEL."
    ),
)
@click.option("--captions/--no-captions", default=None, help="YouTube: download caption tracks (default on).")
@click.option(
    "--force",
    is_flag=True,
    help="Recompute every stage from scratch, including paid LLM calls, discarding all stored results.",
)
@click.option(
    "--ignore-cooldown", is_flag=True, help="YouTube: start even though a bot or rate-limit cooldown is active."
)
def run_command(  # noqa: PLR0913
    raw_targets: tuple[str, ...],
    out: Path | None,
    auth_spec: str | None,
    engine: str | None,
    language: str | None,
    transcript_source: str | None,
    chapters: str | None,
    llm_spec: str | None,
    *,
    correct: bool | None,
    captions: bool | None,
    force: bool,
    ignore_cooldown: bool,
) -> None:
    """Turn YouTube videos, playlists or channels, or local recordings, into context.json and context.md.

    TARGETS are YouTube URLs or video ids, or local audio or video files and folders (searched recursively for media
    files); one run takes one kind. Each source is transcribed locally, its punctuation restored and its numbers
    verified. Videos go to one folder per video id; recordings to one folder per id derived from their audio, so a
    renamed or re-saved file is still recognised, with an index.md listing them all. Every stage is stored, so a
    rerun redoes only what changed options reach; a source whose context.json is current is skipped unless --force
    is given.
    Exit codes: 0 ok, 1 some sources failed, 2 auth error, 3 batch stopped (cooldown).
    """
    try:
        found = [targets.classify(target) for target in raw_targets]
    except ValueError as exc:
        _fail(str(exc))
    files = [target for target in found if isinstance(target, Path)]
    if any(isinstance(target, targets.MemoSelector) for target in found):
        _fail("memos: targets are not supported yet; share the memo out of Voice Memos and pass the file")
    if files and len(files) != len(found):
        _fail("run YouTube and local targets separately")
    overrides: dict[str, dict[str, object]] = {
        "asr": {"engine": engine, "source": transcript_source, "language": language},
        "text": {"correct": correct, "chapters": chapters},
    }
    if files:
        youtube_only = {
            "--auth": auth_spec,
            "--source": transcript_source,
            "--captions/--no-captions": captions,
            "--ignore-cooldown": ignore_cooldown or None,
        }
        given = [flag for flag, value in youtube_only.items() if value is not None]
        if given:
            _fail(f"{', '.join(given)} only apply to YouTube targets")
        _run_files(files, out, overrides, llm_spec, force=force)
    source = None
    try:
        source = resolve_source(auth_spec)
        options = _options(overrides, llm_spec)
        root = out if out is not None else paths.default_output_root()
        report = pipeline.process_many(
            raw_targets,
            options,
            out=root,
            source=source,
            captions=True if captions is None else captions,
            force=force,
            ignore_cooldown=ignore_cooldown,
            interactive=interactive(),
            progress=_print_progress,
            on_video=_print_result,
        )
    except BatchStopError as exc:
        _stopped(exc)
    except AuthError as exc:
        click.echo(f"✗ {exc.state.value}: {redact(exc.detail)}" if exc.detail else f"✗ {exc.state.value}", err=True)
        click.echo(f"  Fix: {fix_for(exc.state, **fix_context(source))}", err=True)
        raise SystemExit(EXIT_AUTH) from None
    except EngineMemoryError as exc:
        click.echo(f"■ {redact(exc.detail)}", err=True)
        click.echo(f"  Fix: {exc.fix}", err=True)
        raise SystemExit(EXIT_FAILED) from None
    except YtverbatimError as exc:
        click.echo(f"error: {redact(str(exc))}", err=True)
        raise SystemExit(EXIT_FAILED) from None
    _print_summary(report, root)
    raise SystemExit(report.exit_code)


def _fail(message: str, fix: str = "") -> NoReturn:
    click.echo(f"error: {redact(message)}", err=True)
    if fix:
        click.echo(f"  Fix: {fix}", err=True)
    raise SystemExit(EXIT_FAILED)


def _options(overrides: config.Overrides, llm_spec: str | None) -> config.Options:
    options = config.load_options(overrides=overrides)
    if llm_spec is not None:
        options = replace(options, llm=apply_spec(options.llm, llm_spec))
    return options


def _run_files(
    files: list[Path], out: Path | None, overrides: config.Overrides, llm_spec: str | None, *, force: bool
) -> NoReturn:
    root = out if out is not None else paths.recordings_root()
    try:
        options = _options(overrides, llm_spec)
        media = targets.expand(files, exclude=root)
        if not media:
            _fail("no audio or video files found in the folders given")
        report = pipeline.process_files(
            media,
            options,
            out=root,
            force=force,
            interactive=interactive(),
            progress=_print_progress,
            on_video=_print_result,
        )
    except EngineMemoryError as exc:
        click.echo(f"■ {redact(exc.detail)}", err=True)
        click.echo(f"  Fix: {exc.fix}", err=True)
        raise SystemExit(EXIT_FAILED) from None
    except YtverbatimError as exc:
        _fail(str(exc), exc.fix)
    _print_summary(report, root)
    click.echo(f"index: {redact(str(root / INDEX_MD))}")
    raise SystemExit(report.exit_code)


def _print_progress(event: pipeline.ProgressEvent) -> None:
    detail = f" {event.detail}" if event.detail else ""
    click.echo(f"… {event.video_id}  {event.stage}{detail}", err=True)


def _print_result(result: pipeline.VideoResult) -> None:
    outcome = result.outcome
    name = result.name or outcome.video_id
    if outcome.status is not VideoStatus.OK:
        detail = f": {outcome.detail}" if outcome.detail else ""
        click.echo(f"✗ {name}  {outcome.error}{detail}")
        if outcome.fix:
            click.echo(f"    Fix: {outcome.fix}")
        return
    if pipeline.DUPLICATE in outcome.reused:
        click.echo(f"= {name}  {outcome.detail}")
        return
    document, markdown = result.document, result.markdown
    if document is None or markdown is None:
        return
    flags = sum(len(segment.flags) for segment in document.segments)
    facts = [f"{len(document.segments)} segments", f"{len(document.paragraphs)} paragraphs"]
    if not isinstance(document.source, RecordingSource):
        facts.append(f"{len(document.chapters)} chapters")
    facts += [f"{flags} flags", f"punctuation: {document.transcript.punctuation}"]
    if outcome.reused and "context" in outcome.reused:
        facts.append("already done")
    else:
        facts.append(f"{result.elapsed_s:.0f}s")
    click.echo(f"✓ {name}  {redact(str(markdown))}  ({' · '.join(facts)})")
    for notice in result.notices:
        click.echo(f"  ! {notice}")
    if result.timings:
        click.echo(f"    stages: {' · '.join(_timing(t) for t in result.timings)}")
    if document.llm_usage:
        # What the document cost in total, batches reused from earlier runs included; not what this run spent.
        click.echo(f"    document LLM tokens: {' · '.join(_usage(u) for u in document.llm_usage)}")


def _usage(usage: LLMUsage) -> str:
    def tokens(count: int | None) -> str:
        return "?" if count is None else str(count)

    return f"{usage.stage} {usage.calls} call(s), {tokens(usage.input_tokens)} in / {tokens(usage.output_tokens)} out"


def _timing(timing: pipeline.StageTiming) -> str:
    return f"{timing.stage} stored" if timing.reused else f"{timing.stage} {timing.seconds:.0f}s"


def _print_summary(report: BatchReport, root: Path) -> None:
    duplicates = sum(pipeline.DUPLICATE in v.reused for v in report.videos)
    ok = sum(v.status is VideoStatus.OK for v in report.videos) - duplicates
    failed = sum(v.status is VideoStatus.ERROR for v in report.videos)
    skipped = sum(v.status is VideoStatus.NOT_ATTEMPTED for v in report.videos)
    parts = [f"{ok} ok", f"{failed} failed"]
    if duplicates:
        parts.append(f"{duplicates} duplicate{'s' if duplicates > 1 else ''}")
    if skipped:
        parts.append(f"{skipped} not attempted")
    click.echo(f"{', '.join(parts)} · report: {redact(str(root / REPORT_FILE))}")
    if report.capped:
        click.echo("! the per-run video cap was reached; raise ytdlp.max_videos_per_run to process more")
    if report.stopped is not None:
        click.echo(f"■ batch stopped by {report.stopped.signal.value}", err=True)
        click.echo(f"  Fix: {stop_fix(report.stopped.signal, report.stopped.until)}", err=True)


def _stopped(exc: BatchStopError) -> NoReturn:
    click.echo(f"■ {exc}", err=True)
    click.echo(f"  Fix: {stop_fix(exc.signal, exc.until)}", err=True)
    click.echo("  (--ignore-cooldown starts anyway, at the account's risk.)", err=True)
    raise SystemExit(EXIT_STOPPED)
