"""`index.md` at the recordings root: every transcribed recording, newest first, linked to its `context.md`.

Built from the `context.json` files present rather than kept as a list of its own: a batch scans the root once,
then puts each document it writes into that scan and renders the table again, which costs a string per row instead of
reading every document after every recording. The batch ends with a fresh scan, so documents another run wrote
meanwhile are listed too, and of two runs finishing together the last one's index is complete.
"""

import logging
import math
import re
from datetime import datetime
from pathlib import Path

from anyverbatim import atomic, ids
from anyverbatim.contract import ContextDocument, RecordingSource, load_document
from anyverbatim.errors import ContractError

log = logging.getLogger(__name__)

INDEX_MD = "index.md"
_CONTEXT_JSON = "context.json"
_CONTEXT_MD = "context.md"
_HOUR = 3600
_CELL_ESCAPE = re.compile(r"([\\`*_\[\]<>|~])")


class Index:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._entries: dict[str, RecordingSource] = {}
        self._speakers: dict[str, tuple[str, ...]] = {}

    @classmethod
    def scan(cls, root: Path) -> Index:
        index = cls(root)
        for context in sorted(root.glob(f"rec-*/{_CONTEXT_JSON}")):
            if not ids.is_recording_id(context.parent.name):
                continue
            try:
                document = load_document(context.read_text(encoding="utf-8"))
            except (OSError, ValueError, ContractError) as exc:
                # One bad document must not hide every other recording, nor stop the batch that rebuilds the index.
                log.warning("%s left out of %s: %s", context.parent.name, INDEX_MD, exc)
                continue
            index.add(document)
        return index

    def add(self, document: ContextDocument) -> None:
        source = document.source
        if isinstance(source, RecordingSource):
            self._entries[source.id] = source
            self._speakers[source.id] = tuple(dict.fromkeys(speaker.label for speaker in document.speakers))

    def write(self) -> Path:
        entries = sorted(self._entries.values(), key=_order)
        lines = ["# Recordings", ""]
        if entries:
            lines += ["| Recorded | Title | Duration | Speakers |", "| --- | --- | --- | --- |"]
            lines += [_row(source, self._speakers[source.id]) for source in entries]
        else:
            lines.append("No recordings yet.")
        path = self.root / INDEX_MD
        self.root.mkdir(parents=True, exist_ok=True)
        atomic.write_text(path, "\n".join(lines) + "\n")
        return path


def write_index(root: Path) -> Path:
    return Index.scan(root).write()


def _order(source: RecordingSource) -> tuple[bool, float, str, str]:
    # Newest first; undated recordings after every dated one.
    newest = -source.recorded_at.timestamp() if source.recorded_at is not None else 0.0
    return source.recorded_at is None, newest, source.title.casefold(), source.id


def _row(source: RecordingSource, labels: tuple[str, ...]) -> str:
    title = source.title.strip() or source.file_name
    link = f"[{_cell(title)}]({source.id}/{_CONTEXT_MD})"
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
