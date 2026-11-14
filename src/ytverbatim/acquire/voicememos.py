"""iPhone Voice Memos: list the memos synced to this Mac, and remember which recording id each one has.

The library is Apple's undocumented Sonoma-and-later layout, so the database is read-only and its columns are
resolved at run time. Hashing a memo into its `rec-…` id is the identity stage's job; this module only caches the
result, keyed by size and mtime, through a function the caller passes in.
"""

import json
import os
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ytverbatim import paths
from ytverbatim.atomic import write_json
from ytverbatim.errors import ConfigError

CONTAINER = "~/Library/Group Containers/group.com.apple.VoiceMemos.shared"
DATABASE = "CloudRecordings.db"
TABLE = "ZCLOUDRECORDING"
_APPLE_EPOCH = 978307200  # 2001-01-01 in Unix seconds; Core Data dates count from it.

Availability = Literal["local", "not_downloaded", "unsupported"]
RecordingId = Callable[[Path], str]
"""Hashes a memo's audio into its `rec-<16 hex>` id (§2.3 of the recordings plan)."""

_REQUIRED = ("ZUNIQUEID", "ZPATH")
_OPTIONAL = ("ZENCRYPTEDTITLE", "ZCUSTOMLABEL", "ZDATE", "ZDURATION", "ZEVICTIONDATE")
_TERMINALS = {
    "Apple_Terminal": "Terminal",
    "iTerm.app": "iTerm",
    "vscode": "Visual Studio Code",
    "ghostty": "Ghostty",
    "WezTerm": "WezTerm",
    "WarpTerminal": "Warp",
    "Hyper": "Hyper",
    "zed": "Zed",
}


class MemosAccessError(ConfigError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.fix = _access_fix()


class MemosNotSyncingError(ConfigError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.fix = (
            "Voice Memos is not syncing to this Mac: turn on Voice Memos in iCloud settings on the iPhone and the "
            "Mac, then open Voice Memos once"
        )


def _access_fix() -> str:
    app = _TERMINALS.get(os.environ.get("TERM_PROGRAM", ""))
    target = (
        f"{app} (the app running ytverbatim)"
        if app
        else "the app you run ytverbatim from (Terminal, iTerm, your editor)"
    )
    return (
        f"System Settings → Privacy & Security → Full Disk Access → add {target}, then restart it. "
        "For a cron or launchd job, add the job's program instead: cron itself, or the executable in the launchd "
        "agent's ProgramArguments"
    )


@dataclass(frozen=True)
class Memo:
    unique_id: str
    title: str | None
    recorded_at: datetime | None
    duration_s: float | None
    audio: Path | None
    availability: Availability
    size: int
    mtime_ns: int


def default_recordings_dir() -> Path:
    return Path.home() / "Library" / "Group Containers" / "group.com.apple.VoiceMemos.shared" / "Recordings"


def list_memos(recordings: Path | None = None) -> list[Memo]:
    """Every memo that is not in Recently Deleted, newest first; memos without a date come last."""
    folder = recordings if recordings is not None else default_recordings_dir()
    rows = _read_rows(folder)
    memos = [memo for row in rows if (memo := _memo(row, folder)) is not None]
    memos.sort(
        key=lambda m: (m.recorded_at is not None, m.recorded_at or datetime.min.replace(tzinfo=UTC)), reverse=True
    )
    return memos


def _read_rows(folder: Path) -> list[Mapping[str, object]]:
    database = folder / DATABASE
    try:
        os.stat(folder)
        # SQLite reports a denied open as "unable to open database file", so the permission error is raised here.
        with database.open("rb"):
            pass
    except PermissionError as exc:
        raise MemosAccessError(f"cannot read the Voice Memos library at {CONTAINER}/Recordings") from exc
    except FileNotFoundError as exc:
        raise MemosNotSyncingError(f"no Voice Memos library found at {CONTAINER}/Recordings") from exc
    try:
        with sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            present = {row["name"] for row in connection.execute(f"PRAGMA table_info({TABLE})")}
            missing = [column for column in _REQUIRED if column not in present]
            if missing:
                raise ConfigError(
                    f"the Voice Memos database has no {', '.join(missing)} in {TABLE}; Apple changed its layout"
                )
            columns = [column for column in (*_REQUIRED, *_OPTIONAL) if column in present]
            return [dict(row) for row in connection.execute(f"SELECT {', '.join(columns)} FROM {TABLE}")]
    except sqlite3.Error as exc:
        raise ConfigError(f"cannot read the Voice Memos database: {exc}") from exc


def _memo(row: Mapping[str, object], folder: Path) -> Memo | None:
    unique_id = row["ZUNIQUEID"]
    if row.get("ZEVICTIONDATE") is not None or not isinstance(unique_id, str):
        return None
    zpath = row["ZPATH"]
    audio = folder / zpath if isinstance(zpath, str) and zpath else None
    availability: Availability = "not_downloaded"
    size = mtime_ns = 0
    if audio is not None:
        try:
            stat = audio.stat()
        except FileNotFoundError:
            pass
        except PermissionError as exc:
            raise MemosAccessError(f"cannot read a Voice Memos recording under {CONTAINER}/Recordings") from exc
        else:
            size, mtime_ns = stat.st_size, stat.st_mtime_ns
            if audio.is_dir():
                availability = "unsupported"
            elif size > 0:
                availability = "local"
    return Memo(
        unique_id=unique_id,
        title=_title(row),
        recorded_at=_date(row.get("ZDATE")),
        duration_s=_number(row.get("ZDURATION")),
        audio=audio if availability != "not_downloaded" else None,
        availability=availability,
        size=size,
        mtime_ns=mtime_ns,
    )


def _title(row: Mapping[str, object]) -> str | None:
    for column in ("ZENCRYPTEDTITLE", "ZCUSTOMLABEL"):
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _date(value: object) -> datetime | None:
    seconds = _number(value)
    return None if seconds is None else datetime.fromtimestamp(seconds + _APPLE_EPOCH, UTC)


class MemoCache:
    """`ZUNIQUEID → recording id`, trusted only while the audio file's size and mtime are unchanged."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else paths.voicememos_cache_file()
        self._entries = self._load()

    def _load(self) -> dict[str, dict[str, object]]:
        # A cache only saves hashing, so an unreadable file is the same as an empty one.
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)} if isinstance(raw, dict) else {}

    def lookup(self, memo: Memo) -> str | None:
        entry = self._entries.get(memo.unique_id)
        if (
            memo.availability == "local"
            and entry is not None
            and entry.get("size") == memo.size
            and entry.get("mtime_ns") == memo.mtime_ns
            and isinstance(recording_id := entry.get("id"), str)
        ):
            return recording_id
        return None

    def resolve(self, memo: Memo, recording_id: RecordingId) -> str:
        """Return the memo's id, hashing only on a miss; the entry is stored at once so a batch keeps its progress."""
        cached = self.lookup(memo)
        if cached is not None:
            return cached
        if memo.audio is None or memo.availability != "local":
            raise ValueError(f"memo is not a local audio file ({memo.availability})")
        found = recording_id(memo.audio)
        self._entries[memo.unique_id] = {"id": found, "size": memo.size, "mtime_ns": memo.mtime_ns}
        write_json(self._path, self._entries, mode=0o600)
        return found


@dataclass(frozen=True)
class Probe:
    state: Literal["ok", "no_access", "not_syncing", "unreadable"]
    memos: int | None = None
    detail: str = ""
    fix: str = ""


def probe(recordings: Path | None = None) -> Probe:
    """Report whether the library is readable, as `status` shows it; never raises."""
    try:
        return Probe("ok", memos=len(list_memos(recordings)))
    except MemosAccessError as exc:
        return Probe("no_access", detail=str(exc), fix=exc.fix)
    except MemosNotSyncingError as exc:
        return Probe("not_syncing", detail=str(exc), fix=exc.fix)
    except ConfigError as exc:
        return Probe("unreadable", detail=str(exc))
