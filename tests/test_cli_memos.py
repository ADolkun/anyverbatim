import json
import unicodedata
from pathlib import Path

import pytest
from click.testing import CliRunner
from voicememos_fakes import Row, make_library

from anyverbatim import paths
from anyverbatim.acquire import voicememos
from anyverbatim.cli import main


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("ANYVERBATIM_HOME", str(tmp_path / "ytv"))
    return tmp_path


def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[Row]) -> Path:
    folder = make_library(tmp_path / "container", rows)
    monkeypatch.setattr(voicememos, "default_recordings_dir", lambda: folder)
    return folder


def no_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voicememos.MemoCache, "resolve", lambda *_a, **_k: pytest.fail("the listing hashed a memo"))


def test_the_listing_shows_date_duration_title_status_and_a_short_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [
            Row("11112222-AAAA", title="Standup", unix=1_760_000_000, duration=3725, path="a.m4a"),
            Row("33334444-BBBB", title="Cloud only", unix=1_750_000_000, duration=59, path=None),
        ],
    )
    no_hashing(monkeypatch)
    result = CliRunner().invoke(main, ["memos"])
    assert result.exit_code == 0, result.output
    first, second, hint = result.output.splitlines()
    assert first.endswith("1:02:05  Standup     new             11112222")
    assert second.endswith("0:59  Cloud only  not downloaded  33334444")
    assert hint == "Not downloaded: open the memo once in Voice Memos to download it"


def test_a_cached_memo_with_a_document_is_transcribed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = library(
        tmp_path, monkeypatch, [Row("A", path="a.m4a"), Row("B", path="b.m4a"), Row("C", path="c.m4a", unix=1e9)]
    )
    cache = voicememos.MemoCache()
    by_id = {m.unique_id: m for m in voicememos.list_memos(folder)}
    cache.resolve(by_id["A"], lambda _p: "rec-aaaaaaaaaaaaaaaa")
    cache.resolve(by_id["B"], lambda _p: "rec-bbbbbbbbbbbbbbbb")
    document = paths.recordings_root() / "rec-aaaaaaaaaaaaaaaa" / "context.json"
    document.parent.mkdir(parents=True)
    document.write_text("{}")
    no_hashing(monkeypatch)
    payload = json.loads(CliRunner().invoke(main, ["memos", "--json"]).output)
    assert {row["unique_id"]: (row["status"], row["recording_id"]) for row in payload} == {
        "A": ("transcribed", "rec-aaaaaaaaaaaaaaaa"),
        "B": ("new", "rec-bbbbbbbbbbbbbbbb"),
        "C": ("new", None),
    }


def test_out_picks_the_root_where_documents_are_looked_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = library(tmp_path, monkeypatch, [Row("A", path="a.m4a")])
    (memo,) = voicememos.list_memos(folder)
    voicememos.MemoCache().resolve(memo, lambda _p: "rec-aaaaaaaaaaaaaaaa")
    root = tmp_path / "elsewhere"
    (root / "rec-aaaaaaaaaaaaaaaa").mkdir(parents=True)
    (root / "rec-aaaaaaaaaaaaaaaa" / "context.json").write_text("{}")
    result = CliRunner().invoke(main, ["memos", "--out", str(root)])
    assert " transcribed " in result.output


def test_the_newest_twenty_are_shown_unless_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library(tmp_path, monkeypatch, [Row(f"id{n:02d}", unix=1_700_000_000 + n, path=f"{n}.m4a") for n in range(25)])
    short = CliRunner().invoke(main, ["memos"]).output.splitlines()
    assert len(short) == 21 and short[-1] == "5 older memos not shown; --all lists every memo"
    assert short[0].endswith("id24")
    assert len(CliRunner().invoke(main, ["memos", "--all"]).output.splitlines()) == 25
    assert len(json.loads(CliRunner().invoke(main, ["memos", "--json"]).output)) == 25


def display_column(line: str, needle: str) -> int:
    before = line[: line.index(needle)]
    return sum(2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1 for c in before)


def test_wide_characters_keep_the_columns_aligned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library(
        tmp_path,
        monkeypatch,
        [Row("A", title="周会纪要", unix=1_760_000_001, path="a.m4a"), Row("B", title="abcdefgh", path="b.m4a")],
    )
    first, second = CliRunner().invoke(main, ["memos"]).output.splitlines()
    assert display_column(first, "new") == display_column(second, "new")


def test_a_long_wide_title_is_cut_to_the_column_width(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library(tmp_path, monkeypatch, [Row("A", title="会" * 30, path="a.m4a")])
    line = CliRunner().invoke(main, ["memos"]).output.splitlines()[0]
    assert "会" * 19 + "…" in line and "会" * 20 not in line


def test_the_eviction_warning_goes_to_stderr_and_the_listing_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = make_library(tmp_path / "container", [Row("A", path="a.m4a")], drop=("ZEVICTIONDATE",))
    monkeypatch.setattr(voicememos, "default_recordings_dir", lambda: folder)
    result = CliRunner().invoke(main, ["memos"])
    assert result.exit_code == 0
    assert "warning: the Voice Memos database has no ZEVICTIONDATE" in result.stderr


def test_listing_forgets_cached_memos_that_are_gone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = library(tmp_path, monkeypatch, [Row("A", path="a.m4a")])
    (memo,) = voicememos.list_memos(folder)
    entry = {"id": "rec-aaaaaaaaaaaaaaaa", "size": memo.size, "mtime_ns": memo.mtime_ns}
    cache_file = paths.voicememos_cache_file(create_parents=True)
    cache_file.write_text(json.dumps({"A": entry, "GONE": entry}))
    CliRunner().invoke(main, ["memos"])
    assert list(json.loads(cache_file.read_text())) == ["A"]
