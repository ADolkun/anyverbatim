"""`index.md` at the recordings root: every transcribed recording, newest first, linked to its `context.md`.

Rebuilt from the `context.json` files present rather than updated in place, so two runs that finish together both
write a complete index and the last one stands.
"""

import logging
import math
import re
from datetime import datetime
from pathlib import Path

from ytverbatim import atomic, ids
from ytverbatim.contract import ContextDocument, RecordingSource, load_document
from ytverbatim.errors import ContractError

log = logging.getLogger(__name__)

INDEX_MD = "index.md"
_CONTEXT_JSON = "context.json"
_CONTEXT_MD = "context.md"
_HOUR = 3600
_CELL_ESCAPE = re.compile(r"([\\`*_\[\]<>|~])")


def write_index(root: Path) -> Path:
    entries = sorted(_entries(root), key=_order)
    lines = ["# Recordings", ""]
    if entries:
        lines += ["| Recorded | Title | Duration | Speakers |", "| --- | --- | --- | --- |"]
        lines += [_row(source, document) for source, document in entries]
    else:
        lines.append("No recordings yet.")
    path = root / INDEX_MD
    root.mkdir(parents=True, exist_ok=True)
    atomic.write_text(path, "\n".join(lines) + "\n")
    return path


def _entries(root: Path) -> list[tuple[RecordingSource, ContextDocument]]:
    found: list[tuple[RecordingSource, ContextDocument]] = []
    for context in sorted(root.glob(f"rec-*/{_CONTEXT_JSON}")):
        if not ids.is_recording_id(context.parent.name):
            continue
        try:
            document = load_document(context.read_text(encoding="utf-8"))
        except (OSError, ContractError) as exc:
            log.warning("%s left out of %s: %s", context.parent.name, INDEX_MD, exc)
            continue
        if isinstance(document.source, RecordingSource):
            found.append((document.source, document))
    return found


def _order(entry: tuple[RecordingSource, ContextDocument]) -> tuple[bool, float, str, str]:
    source = entry[0]
    # Newest first; undated recordings after every dated one.
    newest = -source.recorded_at.timestamp() if source.recorded_at is not None else 0.0
    return source.recorded_at is None, newest, source.title.casefold(), source.id


def _row(source: RecordingSource, document: ContextDocument) -> str:
    title = source.title.strip() or source.file_name
    link = f"[{_cell(title)}]({source.id}/{_CONTEXT_MD})"
    labels = list(dict.fromkeys(speaker.label for speaker in document.speakers))
    speakers = _cell(", ".join(labels)) if labels else "-"
    return f"| {_recorded(source.recorded_at)} | {link} | {_clock(source.duration_s)} | {speakers} |"


def _recorded(moment: datetime | None) -> str:
    return f"{moment:%Y-%m-%d %H:%M}" if moment is not None else "-"


def _clock(seconds: float) -> str:
    total = math.floor(seconds)
    hours, rest = divmod(total, _HOUR)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _cell(text: str) -> str:
    return _CELL_ESCAPE.sub(r"\\\1", " ".join(text.split()))
