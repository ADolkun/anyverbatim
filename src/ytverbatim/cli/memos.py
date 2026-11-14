import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import click

from ytverbatim import paths
from ytverbatim.acquire.voicememos import Availability, Memo, MemoCache, read_library
from ytverbatim.cli.fetch import EXIT_FAILED
from ytverbatim.errors import ConfigError
from ytverbatim.pipeline import CONTEXT_JSON
from ytverbatim.ytdlp.runner import redact

DEFAULT_LIMIT = 20
_TITLE_WIDTH = 40
_AVAILABILITY_LABEL: dict[Availability, str] = {
    "not_downloaded": "not downloaded",
    "unsupported": "unsupported",
}


@dataclass(frozen=True)
class _Entry:
    memo: Memo
    status: str
    recording_id: str | None


@click.command(name="memos")
@click.option("--all", "show_all", is_flag=True, help=f"List every memo, not the newest {DEFAULT_LIMIT}.")
@click.option("--json", "as_json", is_flag=True, help="Print the memos as one JSON array.")
@click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    help="Recordings output root, where a transcribed memo's folder is looked up.",
)
def memos_command(*, show_all: bool, as_json: bool, out: Path | None) -> None:
    """List the iPhone Voice Memos synced to this Mac, newest first: date, duration, title, status and id.

    Reads a copy of the Voice Memos library and hashes nothing. Status: transcribed (a document exists in the output
    root), new, not downloaded (open the memo once in Voice Memos to download it), or unsupported. --json always
    lists every memo.
    """
    try:
        listing = read_library()
    except ConfigError as exc:
        click.echo(f"error: {redact(str(exc))}", err=True)
        if exc.fix:
            click.echo(f"  Fix: {exc.fix}", err=True)
        raise SystemExit(EXIT_FAILED) from None
    for warning in listing.warnings:
        click.echo(f"warning: {warning}", err=True)
    memos = listing.memos
    cache = MemoCache()
    cache.prune({memo.unique_id for memo in memos})
    root = out if out is not None else paths.recordings_root()
    shown = memos if show_all or as_json else memos[:DEFAULT_LIMIT]
    entries = [_entry(memo, cache, root) for memo in shown]
    if as_json:
        click.echo(json.dumps([_json(entry) for entry in entries], ensure_ascii=False))
        return
    _print(entries, hidden=len(memos) - len(shown))


def _entry(memo: Memo, cache: MemoCache, root: Path) -> _Entry:
    if memo.availability != "local":
        return _Entry(memo, _AVAILABILITY_LABEL[memo.availability], None)
    recording_id = cache.lookup(memo)
    transcribed = recording_id is not None and (root / recording_id / CONTEXT_JSON).is_file()
    return _Entry(memo, "transcribed" if transcribed else "new", recording_id)


def _json(entry: _Entry) -> dict[str, object]:
    memo = entry.memo
    return {
        "unique_id": memo.unique_id,
        "title": memo.title,
        "recorded_at": memo.recorded_at.isoformat(timespec="seconds") if memo.recorded_at else None,
        "duration_s": memo.duration_s,
        "status": entry.status,
        "recording_id": entry.recording_id,
    }


def short_id(memo: Memo) -> str:
    return memo.unique_id[:8].lower()


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _width(text: str) -> int:
    return sum(
        0
        if unicodedata.combining(c) or unicodedata.category(c) in {"Mn", "Me", "Cf"}
        else 2
        if unicodedata.east_asian_width(c) in {"W", "F"}
        else 1
        for c in text
    )


def _pad(text: str, width: int) -> str:
    return text + " " * (width - _width(text))


def _title(memo: Memo) -> str:
    title = memo.title or "-"
    if _width(title) <= _TITLE_WIDTH:
        return title
    kept = ""
    for char in title:
        if _width(kept + char) > _TITLE_WIDTH - 1:
            break
        kept += char
    return kept + "…"


def _print(entries: list[_Entry], *, hidden: int) -> None:
    if not entries:
        click.echo("no memos")
        return
    title_width = max(_width(_title(e.memo)) for e in entries)
    status_width = max(len(e.status) for e in entries)
    for entry in entries:
        memo = entry.memo
        when = memo.recorded_at.astimezone().strftime("%Y-%m-%d %H:%M") if memo.recorded_at else "-" * 16
        click.echo(
            f"{when}  {_duration(memo.duration_s):>7}  {_pad(_title(memo), title_width)}  "
            f"{entry.status:<{status_width}}  {short_id(memo)}"
        )
    if hidden:
        click.echo(f"{hidden} older memos not shown; --all lists every memo")
    statuses = {entry.status for entry in entries}
    if "not downloaded" in statuses:
        click.echo("Not downloaded: open the memo once in Voice Memos to download it")
    if "unsupported" in statuses:
        click.echo(
            "Unsupported: ytverbatim cannot read this memo (a .composition folder, or a file outside the library)"
        )
