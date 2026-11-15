"""`run memos:…` over a synthetic Core Data-shaped library, with fake ffprobe/ffmpeg, worker and no LLM."""

# The fixtures are imported from test_pipeline_files, so every test argument that names one looks like a redefinition.
# ruff: noqa: F811

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from test_pipeline_files import Media, Worker, engine_installed, media, rec_id, worker  # noqa: F401
from voicememos_fakes import Row, make_library

from ytverbatim.acquire import local, voicememos
from ytverbatim.acquire.voicememos import MemoCache, select
from ytverbatim.cli import main
from ytverbatim.contract import RecordingSource, load_document
from ytverbatim.errors import DependencyError, MediaValidationError

SECRET = "Quarterly layoffs"
NEWEST, MIDDLE, OLDEST = 1_760_000_000, 1_750_000_000, 1_740_000_000


def audio(packets: str) -> bytes:
    return f"{packets}\n.m4a\n".encode()


def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[Row], media: Media | None = None) -> Path:
    folder = make_library(tmp_path / "container", rows)
    monkeypatch.setattr(voicememos, "default_recordings_dir", lambda: folder)
    if media is not None:
        for row in rows:
            if row.path:
                media.tags[Path(row.path).name] = {}
    return folder


def memo_row(uid: str, packets: str, *, unix: float = NEWEST, title: str | None = SECRET) -> Row:
    return Row(uid, title=title, unix=unix, path=f"{uid}.m4a", audio=audio(packets))


def cli(kit, *args: str):
    return CliRunner().invoke(main, ["run", "--out", str(kit.out), "--llm", "none", *args])


def names(output: str) -> list[str]:
    return [line.split()[1] for line in output.splitlines() if line.startswith("✓")]


def context(kit, packets: str) -> Path:
    return kit.out / rec_id(packets) / "context.json"


def source_of(kit, packets: str) -> RecordingSource:
    source = load_document(context(kit, packets)).source
    assert isinstance(source, RecordingSource)
    return source


def test_latest_runs_the_newest_local_memo_with_its_title_and_date(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [
            Row("CLOUD-0001", unix=NEWEST + 10, path=None),
            memo_row("AAAA1111", "alpha", unix=NEWEST),
            memo_row("BBBB2222", "beta", unix=MIDDLE),
        ],
        media,
    )
    media.tags["AAAA1111.m4a"] = {"title": "Container title", "creation_time": "2020-01-01T00:00:00Z"}
    result = cli(session_kit, "memos:latest")
    assert result.exit_code == 0, result.output
    assert names(result.output) == ["AAAA1111.m4a"]
    assert "1 memo(s) not downloaded, skipped: open Voice Memos to download them" in result.output
    assert not context(session_kit, "beta").exists()
    source = source_of(session_kit, "alpha")
    assert (source.origin, source.title) == ("voice_memos", SECRET)
    assert source.recorded_at == datetime.fromtimestamp(NEWEST, UTC)
    assert SECRET not in result.output


def test_the_container_title_and_then_the_file_name_stand_in_for_a_missing_memo_title(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [memo_row("TAGGED01", "alpha", title=None, unix=NEWEST), memo_row("PLAIN001", "beta", title=None, unix=MIDDLE)],
        media,
    )
    media.tags["TAGGED01.m4a"] = {"title": "Container title"}
    assert cli(session_kit, "memos:all").exit_code == 0
    assert source_of(session_kit, "alpha").title == "Container title"
    assert source_of(session_kit, "beta").title == "PLAIN001"


def test_all_runs_every_local_memo_and_skips_the_rest_without_hashing(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [
            memo_row("AAAA1111", "alpha", unix=NEWEST),
            Row("CLOUD-0001", unix=MIDDLE, path="gone.m4a", audio=None),
            Row("EMPTY-0001", unix=MIDDLE, path="empty.m4a", audio=b""),
            memo_row("BBBB2222", "beta", unix=OLDEST),
        ],
        media,
    )
    result = cli(session_kit, "memos:all")
    assert result.exit_code == 0, result.output
    assert names(result.output) == ["AAAA1111.m4a", "BBBB2222.m4a"]
    assert "2 memo(s) not downloaded, skipped: open Voice Memos to download them" in result.output
    assert "2 ok, 0 failed" in result.output


def test_new_runs_only_untranscribed_memos_and_hashes_only_uncached_local_ones(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = library(
        tmp_path,
        monkeypatch,
        [
            memo_row("DONE0001", "alpha", unix=NEWEST),
            memo_row("CACHED01", "beta", unix=MIDDLE),
            memo_row("FRESH001", "gamma", unix=OLDEST),
            Row("CLOUD-0001", unix=OLDEST - 1, path=None),
        ],
        media,
    )
    by_id = {memo.unique_id: memo for memo in voicememos.list_memos(folder)}
    cache = MemoCache()
    cache.resolve(by_id["DONE0001"], lambda _p: rec_id("alpha"))
    cache.resolve(by_id["CACHED01"], lambda _p: rec_id("beta"))
    context(session_kit, "alpha").parent.mkdir(parents=True)
    context(session_kit, "alpha").write_text("{}")

    hashed: list[str] = []
    real = local.recording_id

    def spy(path: Path) -> str:
        hashed.append(path.name)
        return real(path)

    monkeypatch.setattr(local, "recording_id", spy)
    result = cli(session_kit, "memos:new")
    assert result.exit_code == 0, result.output
    assert hashed == ["FRESH001.m4a"]
    assert names(result.output) == ["CACHED01.m4a", "FRESH001.m4a"]
    assert "1 memo(s) not downloaded, skipped" in result.output

    again = cli(session_kit, "memos:new")
    assert again.exit_code == 0 and "no new memos to transcribe" in again.output
    assert hashed == ["FRESH001.m4a"], "the second run hashed nothing"


def test_a_run_remembers_the_memo_so_the_listing_calls_it_transcribed(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha")], media)
    assert cli(session_kit, "memos:latest").exit_code == 0
    listed = CliRunner().invoke(main, ["memos", "--json", "--out", str(session_kit.out)])
    assert [(row["status"], row["recording_id"]) for row in json.loads(listed.output)] == [
        ("transcribed", rec_id("alpha"))
    ]


def test_an_id_prefix_picks_one_memo_whatever_its_case(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [memo_row("ABCD1111-0000", "alpha", unix=NEWEST), memo_row("EFGH2222-0000", "beta", unix=MIDDLE)],
        media,
    )
    result = cli(session_kit, "memos:efgh2222")
    assert result.exit_code == 0, result.output
    assert names(result.output) == ["EFGH2222-0000.m4a"]
    assert not context(session_kit, "alpha").exists()


def test_an_ambiguous_prefix_is_refused_with_the_ids_to_choose_from(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [memo_row("ABCD1111-0000", "alpha"), memo_row("ABCD1111-9999", "beta", unix=MIDDLE)],
        media,
    )
    result = cli(session_kit, "memos:abcd")
    assert result.exit_code == 1
    assert "'abcd' matches 2 memos: abcd1111-0, abcd1111-9" in result.output
    assert "Fix: give more characters of the id" in result.output
    assert SECRET not in result.output and not worker.calls


def test_an_unknown_or_empty_prefix_is_refused_with_a_fix(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("ABCD1111", "alpha")], media)
    unknown = cli(session_kit, "memos:zzzz")
    assert unknown.exit_code == 1 and "no Voice Memo has an id starting with 'zzzz'" in unknown.output
    assert "Fix: `ytverbatim memos --all` lists every memo with its id" in unknown.output
    empty = cli(session_kit, "memos:")
    assert empty.exit_code == 1 and "memos: needs new, latest, all" in empty.output and "Fix:" in empty.output
    assert not worker.calls


def test_a_prefix_naming_a_memo_that_is_not_downloaded_says_how_to_get_it(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [Row("CLOUD-0001", path=None), memo_row("ABCD1111", "alpha")], media)
    result = cli(session_kit, "memos:cloud")
    assert result.exit_code == 1
    assert "memo cloud-00 is not downloaded to this Mac" in result.output
    assert "Fix: open the memo once in Voice Memos to download it" in result.output


def test_latest_and_all_without_a_local_memo_are_errors_and_new_is_not(
    session_kit, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [Row("CLOUD-0001", path=None)])
    for selector in ("memos:latest", "memos:all"):
        result = cli(session_kit, selector)
        assert result.exit_code == 1 and "no Voice Memos are downloaded to this Mac" in result.output
        assert "Fix: open a memo once in Voice Memos to download it" in result.output
    nothing = cli(session_kit, "memos:new")
    assert nothing.exit_code == 0 and "1 memo(s) not downloaded" in nothing.output
    assert "no new memos to transcribe" in nothing.output
    assert not (session_kit.out / "batch_report.json").exists()


def test_the_sync_fix_comes_through_run(session_kit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voicememos, "default_recordings_dir", lambda: tmp_path / "no-container" / "Recordings")
    result = cli(session_kit, "memos:new")
    assert result.exit_code == 1
    assert "no Voice Memos library found" in result.output
    assert "Fix: Voice Memos is not syncing to this Mac: turn on Voice Memos in iCloud settings" in result.output


def test_the_access_fix_comes_through_run(session_kit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha")])
    folder.chmod(0)
    try:
        result = cli(session_kit, "memos:latest")
    finally:
        folder.chmod(0o700)
    assert result.exit_code == 1
    assert "cannot read the Voice Memos library" in result.output
    assert "Fix: System Settings → Privacy & Security → Full Disk Access" in result.output
    assert "launchd" in result.output


def test_memos_mix_with_files_but_not_with_youtube(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha")], media)
    both = cli(session_kit, "memos:latest", str(media.add(tmp_path / "talk.m4a", "beta")))
    assert both.exit_code == 0 and names(both.output) == ["talk.m4a", "AAAA1111.m4a"]
    youtube = cli(session_kit, "memos:latest", "pubVideo001")
    assert youtube.exit_code == 1 and "run YouTube and local targets separately" in youtube.output
    flags = cli(session_kit, "--no-captions", "memos:latest")
    assert flags.exit_code == 1 and "only apply to YouTube targets" in flags.output


def test_a_memo_that_cannot_be_hashed_stays_selected_for_new(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [memo_row("AAAA1111", "alpha"), memo_row("BBBB2222", "beta", unix=MIDDLE)])
    memos = voicememos.list_memos(folder)

    def broken(path: Path) -> str:
        if path.name.startswith("AAAA"):
            raise MediaValidationError("unreadable")
        return rec_id("beta")

    chosen = select(
        "new", memos, cache=MemoCache(tmp_path / "cache.json"), recording_id=broken, transcribed=lambda _id: False
    )
    assert [m.unique_id for m in chosen.memos] == ["AAAA1111", "BBBB2222"]


def test_a_missing_ffmpeg_is_not_swallowed_by_selection(tmp_path: Path) -> None:
    folder = make_library(tmp_path, [memo_row("AAAA1111", "alpha")])

    def missing(_path: Path) -> str:
        raise DependencyError("ffmpeg is not installed")

    with pytest.raises(DependencyError):
        select(
            "new",
            voicememos.list_memos(folder),
            cache=MemoCache(tmp_path / "cache.json"),
            recording_id=missing,
            transcribed=lambda _id: False,
        )
