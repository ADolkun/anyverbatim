"""iPhone Voice Memos: list the memos synced to this Mac, and remember which recording id each one has.

The library is Apple's undocumented Sonoma-and-later layout, so the database is read-only and its columns are
resolved at run time. Hashing a memo into its `rec-…` id is `acquire.local.recording_id`'s job; this module only
caches the result, keyed by size and mtime, through a function the caller passes in.
"""

import json
import math
import os
import shutil
import sqlite3
import tempfile
import unicodedata
from collections.abc import Callable, Collection, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ytverbatim import paths
from ytverbatim.atomic import write_json
from ytverbatim.errors import ConfigError
from ytverbatim.ytdlp.runner import redact

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
        self.fix = access_fix()


class MemosNotSyncingError(ConfigError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.fix = (
            "Voice Memos is not syncing to this Mac: turn on Voice Memos in iCloud settings on the iPhone and the "
            "Mac, then open Voice Memos once"
        )


def access_fix() -> str:
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


@dataclass(frozen=True)
class Listing:
    memos: list[Memo]
    warnings: tuple[str, ...] = ()


def list_memos(recordings: Path | None = None) -> list[Memo]:
    return read_library(recordings).memos


def read_library(recordings: Path | None = None) -> Listing:
    """Read every memo that is not in Recently Deleted, newest first; memos without a date come last."""
    folder = recordings if recordings is not None else default_recordings_dir()
    rows, columns = _read_rows(folder)
    memos = [memo for row in rows if (memo := _memo(row, folder)) is not None]
    memos.sort(
        key=lambda m: (m.recorded_at is not None, m.recorded_at or datetime.min.replace(tzinfo=UTC)), reverse=True
    )
    warnings = ()
    if "ZEVICTIONDATE" not in columns:
        warnings = (
            f"the Voice Memos database has no ZEVICTIONDATE column in {TABLE}, "
            "so memos in Recently Deleted cannot be told apart and are listed too",
        )
    return Listing(memos, warnings)


def _read_rows(folder: Path) -> tuple[list[dict[str, object]], frozenset[str]]:
    """Query a private copy of the database: even a read-only open creates -wal and -shm files beside the original."""
    with tempfile.TemporaryDirectory(prefix="ytverbatim-memos-") as scratch:
        copy = Path(scratch) / DATABASE
        for attempt in (1, 2):
            _copy_database(folder, copy)
            try:
                return _query(copy)
            except sqlite3.Error as exc:
                # A copy taken mid-checkpoint can be torn; the second copy is almost always whole.
                if attempt == 2:  # noqa: PLR2004
                    raise ConfigError(f"cannot read the Voice Memos database: {exc}") from exc
    raise AssertionError("unreachable")


def _copy_database(folder: Path, copy: Path) -> None:
    database = folder / DATABASE
    # The -shm file is the WAL's index and is rebuilt from the copied WAL on open, so it is not copied.
    wal, wal_copy = Path(f"{database}-wal"), Path(f"{copy}-wal")
    try:
        os.stat(folder)
        shutil.copyfile(database, copy)
        try:
            shutil.copyfile(wal, wal_copy)
        except FileNotFoundError:
            wal_copy.unlink(missing_ok=True)
        Path(f"{copy}-shm").unlink(missing_ok=True)
    except PermissionError as exc:
        raise MemosAccessError(f"cannot read the Voice Memos library at {CONTAINER}/Recordings") from exc
    except FileNotFoundError as exc:
        raise MemosNotSyncingError(f"no Voice Memos library found at {CONTAINER}/Recordings") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read the Voice Memos library at {CONTAINER}/Recordings: {exc.strerror}") from exc


def _query(database: Path) -> tuple[list[dict[str, object]], frozenset[str]]:
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        present = {row["name"] for row in connection.execute(f"PRAGMA table_info({TABLE})")}
        missing = [column for column in _REQUIRED if column not in present]
        if missing:
            raise ConfigError(
                f"the Voice Memos database has no {', '.join(missing)} in {TABLE}; Apple changed its layout"
            )
        columns = [column for column in (*_REQUIRED, *_OPTIONAL) if column in present]
        rows: list[dict[str, object]] = [
            dict(row) for row in connection.execute(f"SELECT {', '.join(columns)} FROM {TABLE}")
        ]
        return rows, frozenset(columns)


def _memo(row: Mapping[str, object], folder: Path) -> Memo | None:
    unique_id = row["ZUNIQUEID"]
    if row.get("ZEVICTIONDATE") is not None or not isinstance(unique_id, str):
        return None
    zpath = row["ZPATH"]
    audio: Path | None = None
    availability: Availability = "not_downloaded"
    size = mtime_ns = 0
    if isinstance(zpath, str) and zpath:
        relative = Path(zpath)
        if relative.is_absolute() or ".." in relative.parts:
            availability = "unsupported"
        else:
            audio = folder / relative
            availability, size, mtime_ns = _stat_audio(audio)
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


def _stat_audio(audio: Path) -> tuple[Availability, int, int]:
    try:
        stat = audio.stat()
    except FileNotFoundError:
        return "not_downloaded", 0, 0
    except PermissionError as exc:
        raise MemosAccessError(f"cannot read a Voice Memos recording under {CONTAINER}/Recordings") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read a Voice Memos recording under {CONTAINER}/Recordings: {exc.strerror}") from exc
    if audio.is_dir():
        return "unsupported", stat.st_size, stat.st_mtime_ns
    return ("local" if stat.st_size > 0 else "not_downloaded"), stat.st_size, stat.st_mtime_ns


def _title(row: Mapping[str, object]) -> str | None:
    for column in ("ZENCRYPTEDTITLE", "ZCUSTOMLABEL"):
        value = row.get(column)
        if isinstance(value, str) and (clean := _printable(value)):
            return clean
    return None


def _printable(text: str) -> str:
    """Titles are user text that reaches a terminal: control characters, escape sequences included, become spaces."""
    return " ".join("".join(" " if unicodedata.category(c) in {"Cc", "Zl", "Zp"} else c for c in text).split())


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) else None


def _date(value: object) -> datetime | None:
    seconds = _number(value)
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(seconds + _APPLE_EPOCH, UTC)
    except (OverflowError, ValueError, OSError):
        return None


class MemoCache:
    """`ZUNIQUEID → recording id`, trusted only while the audio file's size and mtime are unchanged."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._entries = self._load()

    def _file(self, *, create: bool = False) -> Path:
        return self._path if self._path is not None else paths.voicememos_cache_file(create_parents=create)

    def _load(self) -> dict[str, dict[str, object]]:
        # A cache only saves hashing, so an unreadable file is the same as an empty one.
        try:
            raw = json.loads(self._file().read_text(encoding="utf-8"))
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
        # Merge with what is on disk now, so a run that finished meanwhile does not lose its entries.
        self._entries = self._load()
        self._entries[memo.unique_id] = {"id": found, "size": memo.size, "mtime_ns": memo.mtime_ns}
        write_json(self._file(create=True), self._entries, mode=0o600)
        return found

    def prune(self, unique_ids: Collection[str]) -> None:
        """Forget memos that are no longer in the database."""
        self._entries = self._load()
        kept = {key: entry for key, entry in self._entries.items() if key in unique_ids}
        if len(kept) != len(self._entries):
            self._entries = kept
            write_json(self._file(create=True), kept, mode=0o600)


@dataclass(frozen=True)
class Probe:
    state: Literal["ok", "no_access", "not_syncing", "unreadable"]
    memos: int | None = None
    detail: str = ""
    fix: str = ""


def probe(recordings: Path | None = None) -> Probe:
    """Report whether the library is readable, as `status` shows it; never raises."""
    try:
        listing = read_library(recordings)
    except MemosAccessError as exc:
        return Probe("no_access", detail=redact(str(exc)), fix=exc.fix)
    except MemosNotSyncingError as exc:
        return Probe("not_syncing", detail=redact(str(exc)), fix=exc.fix)
    except ConfigError as exc:
        return Probe("unreadable", detail=redact(str(exc)))
    return Probe("ok", memos=len(listing.memos), detail="; ".join(listing.warnings))
