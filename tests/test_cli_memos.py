import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from voicememos_fakes import Row, make_library

from ytverbatim import paths
from ytverbatim.acquire import voicememos
from ytverbatim.cli import main


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("YTVERBATIM_HOME", str(tmp_path / "ytv"))
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


def test_a_composition_memo_says_it_is_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = library(tmp_path, monkeypatch, [Row("A", path="a.composition", audio=None)])
    (folder / "a.composition").mkdir()
    output = CliRunner().invoke(main, ["memos"]).output
    assert " unsupported " in output and "cannot read yet" in output


def test_no_memos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    library(tmp_path, monkeypatch, [])
    assert CliRunner().invoke(main, ["memos"]).output == "no memos\n"


def test_a_missing_library_prints_the_sync_fix_and_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voicememos, "default_recordings_dir", lambda: tmp_path / "missing")
    result = CliRunner().invoke(main, ["memos"])
    assert result.exit_code == 1
    assert result.output.startswith("error: no Voice Memos library found at ~/Library/Group Containers/")
    assert "Fix: Voice Memos is not syncing to this Mac" in result.output
    assert str(tmp_path) not in result.output
