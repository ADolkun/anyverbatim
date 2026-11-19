import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import voicememos_fakes
from voicememos_fakes import Row, make_library

from anyverbatim.acquire import voicememos
from anyverbatim.acquire.voicememos import MemoCache, MemosAccessError, MemosNotSyncingError
from anyverbatim.errors import ConfigError


@pytest.fixture(autouse=True)
def close_wal_connections() -> Iterator[None]:
    yield
    while voicememos_fakes.wal_connections:
        voicememos_fakes.wal_connections.pop().close()


def test_lists_title_date_duration_and_presence_newest_first(tmp_path: Path) -> None:
    folder = make_library(
        tmp_path,
        [
            Row("A", title="Older", unix=1_700_000_000, path="a.m4a"),
            Row("B", title=None, label="Label", unix=1_760_000_000, path="b.m4a", duration=3.0),
            Row("C", title=" ", label=None, unix=None, path="c.m4a"),
        ],
    )
    memos = voicememos.list_memos(folder)
    assert [m.unique_id for m in memos] == ["B", "A", "C"]
    newest = memos[0]
    assert newest.title == "Label"
    assert newest.recorded_at == datetime.fromtimestamp(1_760_000_000, UTC)
    assert newest.duration_s == 3.0
    assert newest.audio == folder / "b.m4a"
    assert newest.availability == "local"
    assert memos[2].title is None
    assert memos[2].recorded_at is None


def test_recently_deleted_rows_are_excluded(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a"), Row("B", path="b.m4a", evicted=7e8)])
    assert [m.unique_id for m in voicememos.list_memos(folder)] == ["A"]


def test_missing_optional_columns_are_tolerated(tmp_path: Path) -> None:
    folder = make_library(
        tmp_path, [Row("A", path="a.m4a")], drop=("ZENCRYPTEDTITLE", "ZDATE", "ZDURATION", "ZEVICTIONDATE")
    )
    (memo,) = voicememos.list_memos(folder)
    assert (memo.title, memo.recorded_at, memo.duration_s) == (None, None, None)
    assert memo.availability == "local"


def test_the_custom_label_is_the_title_when_the_encrypted_title_column_is_gone(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", label="Label", path="a.m4a")], drop=("ZENCRYPTEDTITLE",))
    assert voicememos.list_memos(folder)[0].title == "Label"


@pytest.mark.parametrize("column", ["ZUNIQUEID", "ZPATH"])
def test_a_missing_required_column_fails_loud(tmp_path: Path, column: str) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")], drop=(column,))
    with pytest.raises(ConfigError, match=f"no {column}"):
        voicememos.list_memos(folder)


def test_a_missing_table_fails_loud(tmp_path: Path) -> None:
    folder = tmp_path / "Recordings"
    folder.mkdir()
    sqlite3.connect(folder / voicememos.DATABASE).close()
    with pytest.raises(ConfigError, match="Apple changed its layout"):
        voicememos.list_memos(folder)


def test_audio_that_is_not_local_is_not_downloaded_without_a_hash(tmp_path: Path) -> None:
    folder = make_library(
        tmp_path,
        [
            Row("null", path=None),
            Row("gone", path="gone.m4a", audio=None),
            Row("empty", path="empty.m4a", audio=b""),
            Row("ok", path="ok.m4a"),
        ],
    )
    by_id = {m.unique_id: m for m in voicememos.list_memos(folder)}
    assert {k: (m.availability, m.audio) for k, m in by_id.items() if k != "ok"} == {
        "null": ("not_downloaded", None),
        "gone": ("not_downloaded", None),
        "empty": ("not_downloaded", None),
    }
    assert by_id["ok"].availability == "local"


def test_a_composition_folder_is_unsupported(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.composition", audio=None)])
    (folder / "a.composition").mkdir()
    (memo,) = voicememos.list_memos(folder)
    assert memo.availability == "unsupported"


def test_rows_still_in_the_wal_are_read(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")], wal=True, close=False)
    assert [m.unique_id for m in voicememos.list_memos(folder)] == ["A"]
    assert (folder / f"{voicememos.DATABASE}-wal").stat().st_size > 0


def test_the_database_is_opened_read_only(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")])
    before = (folder / voicememos.DATABASE).read_bytes()
    voicememos.list_memos(folder)
    assert (folder / voicememos.DATABASE).read_bytes() == before
    assert sorted(p.name for p in folder.iterdir()) == ["CloudRecordings.db", "a.m4a"]


def test_a_missing_folder_gives_the_sync_fix(tmp_path: Path) -> None:
    with pytest.raises(MemosNotSyncingError) as caught:
        voicememos.list_memos(tmp_path / "Recordings")
    assert "Voice Memos is not syncing to this Mac" in caught.value.fix
    assert "turn on Voice Memos in iCloud settings on the iPhone and the Mac" in caught.value.fix
    assert str(tmp_path) not in str(caught.value)


def test_a_folder_without_the_database_gives_the_sync_fix(tmp_path: Path) -> None:
    (tmp_path / "Recordings").mkdir()
    with pytest.raises(MemosNotSyncingError):
        voicememos.list_memos(tmp_path / "Recordings")


@pytest.mark.parametrize("denied", ["folder", "database", "wal"])
def test_permission_denied_gives_the_full_disk_access_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, denied: str
) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")], wal=True, close=False)
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    real_stat, real_copy = os.stat, shutil.copyfile

    def stat(
        path: str | os.PathLike[str], *, dir_fd: int | None = None, follow_symlinks: bool = True
    ) -> os.stat_result:
        if denied == "folder" and Path(path) == folder:
            raise PermissionError(1, "Operation not permitted")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    def copyfile(src: str | Path, dst: str | Path, *, follow_symlinks: bool = True) -> str | Path:
        name = Path(src).name
        if (denied, name) in {("database", voicememos.DATABASE), ("wal", f"{voicememos.DATABASE}-wal")}:
            raise PermissionError(13, "Permission denied")
        return real_copy(src, dst, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(shutil, "copyfile", copyfile)
    with pytest.raises(MemosAccessError) as caught:
        voicememos.list_memos(folder)
    fix = caught.value.fix
    assert "Full Disk Access" in fix
    assert "add iTerm (the app running anyverbatim)" in fix
    assert "cron itself" in fix
    assert "ProgramArguments" in fix


def test_the_fix_for_an_unknown_host_app_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert "add the app you run anyverbatim from (Terminal, iTerm, your editor)" in MemosAccessError("x").fix


def test_probe_reports_each_state_without_raising(tmp_path: Path) -> None:
    assert voicememos.probe(tmp_path / "Recordings").state == "not_syncing"
    folder = make_library(tmp_path / "ok", [Row("A", path="a.m4a"), Row("B", path=None)])
    assert voicememos.probe(folder) == voicememos.Probe("ok", memos=2)
    broken = make_library(tmp_path / "broken", [Row("A")], drop=("ZPATH",))
    assert voicememos.probe(broken).state == "unreadable"


def local_memo(tmp_path: Path) -> voicememos.Memo:
    (memo,) = voicememos.list_memos(make_library(tmp_path, [Row("A", path="a.m4a")]))
    return memo


class CountingHasher:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.calls.append(path)
        return f"rec-{len(self.calls):016x}"


def test_the_cache_hashes_once_and_then_hits(tmp_path: Path) -> None:
    memo = local_memo(tmp_path)
    cache_file = tmp_path / "state" / "voicememos.json"
    cache_file.parent.mkdir()
    hasher = CountingHasher()
    cache = MemoCache(cache_file)
    assert cache.lookup(memo) is None
    first = cache.resolve(memo, hasher)
    assert cache.resolve(memo, hasher) == first
    assert len(hasher.calls) == 1
    reloaded = MemoCache(cache_file)
    assert reloaded.lookup(memo) == first
    assert reloaded.resolve(memo, hasher) == first
    assert len(hasher.calls) == 1
    assert (cache_file.stat().st_mode & 0o777) == 0o600
    assert json.loads(cache_file.read_text())["A"]["id"] == first


@pytest.mark.parametrize("change", ["size", "mtime"])
def test_a_size_or_mtime_change_invalidates_the_entry(tmp_path: Path, change: str) -> None:
    memo = local_memo(tmp_path)
    assert memo.audio is not None
    cache = MemoCache(tmp_path / "voicememos.json")
    hasher = CountingHasher()
    first = cache.resolve(memo, hasher)
    if change == "size":
        memo.audio.write_bytes(b"trimmed")
        os.utime(memo.audio, ns=(memo.mtime_ns, memo.mtime_ns))
    else:
        os.utime(memo.audio, ns=(memo.mtime_ns + 1_000_000_000, memo.mtime_ns + 1_000_000_000))
    (changed,) = voicememos.list_memos(tmp_path / "Recordings")
    assert cache.lookup(changed) is None
    second = cache.resolve(changed, hasher)
    assert second != first
    assert len(hasher.calls) == 2
    assert MemoCache(tmp_path / "voicememos.json").lookup(changed) == second


def test_a_memo_that_is_not_local_is_never_hashed_or_cached(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a", audio=b"")])
    (memo,) = voicememos.list_memos(folder)
    cache = MemoCache(tmp_path / "voicememos.json")
    assert cache.lookup(memo) is None
    with pytest.raises(ValueError, match="not a local audio file"):
        cache.resolve(memo, CountingHasher())


def test_a_stale_entry_for_a_memo_that_was_evicted_is_not_trusted(tmp_path: Path) -> None:
    memo = local_memo(tmp_path)
    cache = MemoCache(tmp_path / "voicememos.json")
    cache.resolve(memo, CountingHasher())
    assert memo.audio is not None
    memo.audio.write_bytes(b"")
    (evicted,) = voicememos.list_memos(tmp_path / "Recordings")
    assert cache.lookup(evicted) is None


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"A": "x"}', '{"A": {"id": 3, "size": 5}}'])
def test_an_unreadable_cache_is_an_empty_one(tmp_path: Path, content: str) -> None:
    memo = local_memo(tmp_path)
    path = tmp_path / "voicememos.json"
    path.write_text(content)
    assert MemoCache(path).lookup(memo) is None


def test_reading_a_closed_wal_database_leaves_the_container_untouched(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")], wal=True)
    before = {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}
    assert voicememos.DATABASE in before
    assert f"{voicememos.DATABASE}-wal" not in before
    assert [m.unique_id for m in voicememos.list_memos(folder)] == ["A"]
    assert {p.name: p.stat().st_mtime_ns for p in folder.iterdir()} == before


def test_the_private_copy_is_removed_afterwards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    folder = make_library(tmp_path, [Row("A", path="a.m4a")], wal=True, close=False)
    voicememos.list_memos(folder)
    assert list(scratch.iterdir()) == []


def test_a_torn_copy_is_retried_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")])
    real = sqlite3.connect
    calls: list[int] = []

    def flaky(database: str, *, uri: bool) -> sqlite3.Connection:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return real(database, uri=uri)

    monkeypatch.setattr(sqlite3, "connect", flaky)
    assert [m.unique_id for m in voicememos.list_memos(folder)] == ["A"]
    assert len(calls) == 2


def test_a_second_read_error_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")])
    (folder / voicememos.DATABASE).write_bytes(b"not a database" * 100)
    with pytest.raises(ConfigError, match="cannot read the Voice Memos database"):
        voicememos.list_memos(folder)


def test_out_of_range_values_do_not_crash_the_listing_or_probe(tmp_path: Path) -> None:
    folder = make_library(
        tmp_path,
        [
            Row("A", unix=1e300, path="a.m4a"),
            Row("B", unix=float("inf"), duration=float("inf"), path="b.m4a"),
            Row("C", unix=-1e300, duration=float("nan"), path="c.m4a"),
        ],
    )
    assert voicememos.probe(folder) == voicememos.Probe("ok", memos=3)
    by_id = {m.unique_id: m for m in voicememos.list_memos(folder)}
    assert all(m.recorded_at is None for m in by_id.values())
    assert (by_id["A"].duration_s, by_id["B"].duration_s, by_id["C"].duration_s) == (61.5, None, None)


def test_other_os_errors_become_config_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    with pytest.raises(ConfigError) as caught:
        voicememos.list_memos(blocker / "Recordings")
    assert not isinstance(caught.value, MemosNotSyncingError | MemosAccessError)
    assert voicememos.probe(blocker / "Recordings").state == "unreadable"
    folder = make_library(tmp_path / "lib", [Row("A", path="a.m4a" + "x" * 5000, audio=None)])
    with pytest.raises(ConfigError):
        voicememos.list_memos(folder)


def test_titles_lose_control_characters(tmp_path: Path) -> None:
    folder = make_library(
        tmp_path, [Row("A", title="\x1b[31mred\x1b[0m\nline\ttwo\u2028x", path="a.m4a"), Row("B", title="\x07")]
    )
    titles = {m.unique_id: m.title for m in voicememos.list_memos(folder)}
    assert titles == {"A": "[31mred [0m line two x", "B": None}


@pytest.mark.parametrize("zpath", ["/etc/hosts", "../outside.m4a", "a/../../outside.m4a"])
def test_a_path_outside_the_library_is_unsupported(tmp_path: Path, zpath: str) -> None:
    (tmp_path / "outside.m4a").write_bytes(b"audio")
    folder = make_library(tmp_path / "lib", [Row("A", path=zpath, audio=None)])
    (memo,) = voicememos.list_memos(folder)
    assert (memo.availability, memo.audio) == ("unsupported", None)


def test_a_missing_eviction_column_is_a_warning_not_silence(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a")], drop=("ZEVICTIONDATE",))
    listing = voicememos.read_library(folder)
    assert "ZEVICTIONDATE" in listing.warnings[0]
    assert "Recently Deleted" in listing.warnings[0]
    assert voicememos.probe(folder).detail == listing.warnings[0]
    assert voicememos.read_library(make_library(tmp_path / "ok", [Row("A")])).warnings == ()


def test_concurrent_runs_keep_each_others_cache_entries(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a"), Row("B", path="b.m4a")])
    a, b = sorted(voicememos.list_memos(folder), key=lambda m: m.unique_id)
    path = tmp_path / "voicememos.json"
    first, second = MemoCache(path), MemoCache(path)
    first.resolve(a, lambda _p: "rec-aaaaaaaaaaaaaaaa")
    second.resolve(b, lambda _p: "rec-bbbbbbbbbbbbbbbb")
    fresh = MemoCache(path)
    assert (fresh.lookup(a), fresh.lookup(b)) == ("rec-aaaaaaaaaaaaaaaa", "rec-bbbbbbbbbbbbbbbb")


def test_prune_forgets_memos_that_left_the_database(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [Row("A", path="a.m4a"), Row("B", path="b.m4a")])
    a, b = sorted(voicememos.list_memos(folder), key=lambda m: m.unique_id)
    path = tmp_path / "voicememos.json"
    cache = MemoCache(path)
    cache.resolve(a, lambda _p: "rec-aaaaaaaaaaaaaaaa")
    cache.resolve(b, lambda _p: "rec-bbbbbbbbbbbbbbbb")
    cache.prune({"A"})
    assert list(json.loads(path.read_text())) == ["A"]
    mtime = path.stat().st_mtime_ns
    cache.prune({"A"})
    assert path.stat().st_mtime_ns == mtime


def test_the_default_cache_is_created_only_when_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    folder = make_library(tmp_path / "lib", [Row("A", path="a.m4a")])
    (memo,) = voicememos.list_memos(folder)
    cache = MemoCache()
    assert cache.lookup(memo) is None
    cache.prune({"A"})
    assert not (tmp_path / "state").exists()
    cache.resolve(memo, lambda _p: "rec-aaaaaaaaaaaaaaaa")
    assert MemoCache().lookup(memo) == "rec-aaaaaaaaaaaaaaaa"
