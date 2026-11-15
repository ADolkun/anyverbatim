import math
from pathlib import Path
from typing import NoReturn

import click

from ytverbatim import ids, index, paths, pipeline, speakers, store
from ytverbatim.cli.fetch import EXIT_FAILED
from ytverbatim.contract import ContextDocument, RecordingSource, load_document
from ytverbatim.errors import YtverbatimError
from ytverbatim.ytdlp.runner import redact

_HOUR = 3600


@click.command(name="speakers")
@click.argument("target")
@click.argument("assignments", metavar="[ID=NAME]...", nargs=-1)
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    help="Output root where a recording or video id is looked up (default: the recordings or videos folder).",
)
def speakers_command(target: str, assignments: tuple[str, ...], out: Path | None) -> None:
    """List or name the speakers of a diarized recording, without transcribing anything again.

    TARGET is the recording's folder, its context.json, or its rec-... id. Without ID=NAME arguments it lists each
    speaker's id, name, talk time and first line. With them it names the speakers: `S1=Alice S2=Bob`; `S2=` clears a
    name, and giving two ids one name merges them into one person (one label, summed talk time, consecutive
    paragraphs joined). Then context.json, context.md and the recordings index.md are rewritten; no ASR worker or
    LLM runs. The names are kept in speakers.json beside context.json and apply only to the speaker attribution they
    were given for: re-diarizing, or a change to attribution or ASR, drops them with a warning, never mislabelling a
    speaker. Names are never sent to an LLM. A later `run` with other options than the config file's rebuilds the
    document once from its stored stages.
    """
    try:
        directory = _resolve(target, out)
        if assignments:
            document = _rename(directory, assignments)
        else:
            document = _read(directory)
            if speakers.bound(directory, _assignment(directory)).stale:
                click.echo("note: " + speakers.STALE_WARNING, err=True)
    except YtverbatimError as exc:
        _fail(str(exc), exc.fix)
    for line in _table(document):
        click.echo(line)
    if assignments:
        click.echo(redact(str(directory / pipeline.CONTEXT_MD)))


def _fail(message: str, fix: str = "") -> NoReturn:
    click.echo(f"error: {redact(message)}", err=True)
    if fix:
        click.echo(f"  Fix: {fix}", err=True)
    raise SystemExit(EXIT_FAILED)


def _resolve(target: str, out: Path | None) -> Path:
    path = Path(target)
    if path.is_dir():
        return path
    if path.is_file() and path.name == pipeline.CONTEXT_JSON:
        return path.parent
    if ids.is_video_id(target) or ids.is_recording_id(target):
        return paths.source_dir(target, out)
    raise click.UsageError(f"{target!r} is not a folder, a context.json or a recording id")


def _read(directory: Path) -> ContextDocument:
    context = directory / pipeline.CONTEXT_JSON
    if not context.is_file():
        raise click.UsageError(f"{redact(str(context))} does not exist; transcribe the recording first")
    document = load_document(context)
    if not document.speakers:
        raise click.UsageError(f"{directory.name} has no speakers: it was not diarized")
    return document


def _assignment(directory: Path) -> str | None:
    try:
        return speakers.assignment_of(directory)
    except YtverbatimError:
        # Listing needs no attribution; only the stale-names note does, and without stages there is nothing to compare.
        return None


def _rename(directory: Path, assignments: tuple[str, ...]) -> ContextDocument:
    changes = speakers.parse(assignments)
    with store.lock(directory):
        document = _read(directory)
        assignment = speakers.assignment_of(directory)
        names = speakers.changed(document, speakers.bound(directory, assignment).names, changes)
        renamed = speakers.named(document, names)
        speakers.write(directory, assignment, names)
        pipeline.save_context(directory, renamed)
        if isinstance(renamed.source, RecordingSource) and ids.is_recording_id(directory.name):
            index.write_index(directory.parent)
    return renamed


def _table(document: ContextDocument) -> list[str]:
    lines = speakers.lines(document)
    width = max(len(line.id) for line in lines)
    named = [line.name or "-" for line in lines]
    name_width = max(len(name) for name in named)
    return [
        f"{line.id:<{width}}  {name:<{name_width}}  {_clock(line.speech_s):>7}  "
        f"[{_clock(line.start)}] {line.first_line}"
        for line, name in zip(lines, named, strict=True)
    ]


def _clock(seconds: float) -> str:
    total = math.floor(seconds)
    hours, rest = divmod(total, _HOUR)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
