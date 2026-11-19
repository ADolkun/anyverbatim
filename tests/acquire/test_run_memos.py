"""`run memos:…` over a synthetic Core Data-shaped library, with fake ffprobe/ffmpeg, worker and no LLM."""

# The fixtures are imported from test_pipeline_files, so every test argument that names one looks like a redefinition.
# ruff: noqa: F811

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from test_pipeline_files import (  # noqa: F401
    SPOKEN,
    FakeLLM,
    Media,
    Worker,
    diarizer,
    engine_installed,
    media,
    rec_id,
    run_files,
    worker,
)
from voicememos_fakes import Row, make_library

from anyverbatim import pipeline
from anyverbatim.acquire import local, voicememos
from anyverbatim.acquire.voicememos import MemoCache, select
from anyverbatim.asr import runner as asr_runner
from anyverbatim.asr.models import WorkerRequest
from anyverbatim.asr.runner import WorkerResult
from anyverbatim.cli import main
from anyverbatim.contract import RecordingSource, load_document
from anyverbatim.errors import DependencyError, MediaValidationError
from anyverbatim.store import lock as video_lock

SECRET = "Quarterly layoffs"
NEWEST, MIDDLE, OLDEST = 1_760_000_000, 1_750_000_000, 1_740_000_000


def audio(packets: str) -> bytes:
    return f"{packets}\n.m4a\n".encode()


def library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[Row],
    media: Media | None = None,
    name: str = "container",
) -> Path:
    folder = make_library(tmp_path / name, rows)
    monkeypatch.setattr(voicememos, "default_recordings_dir", lambda: folder)
    if media is not None:
        for row in rows:
            if row.path:
                media.tags[Path(row.path).name] = {}
    return folder


def memo_row(uid: str, packets: str, *, unix: float | None = NEWEST, title: str | None = SECRET) -> Row:
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
    real = local.packet_sha256

    def spy(path: Path, stream: int) -> str:
        hashed.append(path.name)
        return real(path, stream)

    monkeypatch.setattr(local, "packet_sha256", spy)
    result = cli(session_kit, "memos:new")
    assert result.exit_code == 0, result.output
    assert sorted(hashed) == ["CACHED01.m4a", "FRESH001.m4a"], "the uncached memo is hashed once, the cached one once"
    assert names(result.output) == ["CACHED01.m4a", "FRESH001.m4a"]
    assert "1 memo(s) not downloaded, skipped" in result.output

    again = cli(session_kit, "memos:new")
    assert again.exit_code == 0 and "no new memos to transcribe" in again.output
    assert len(hashed) == 2, "the second run hashed nothing"


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
    assert "Fix: `anyverbatim memos --all` lists every memo with its id" in unknown.output
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


def test_a_memo_another_run_is_transcribing_is_skipped_not_failed(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha"), memo_row("BBBB2222", "beta", unix=MIDDLE)], media)
    with video_lock(session_kit.out / rec_id("alpha")):
        result = cli(session_kit, "memos:all")
    assert result.exit_code == 0, result.output
    assert "= AAAA1111.m4a  being transcribed by another run" in result.output
    assert names(result.output) == ["BBBB2222.m4a"]
    assert "1 ok, 0 failed, 1 busy in another run" in result.output
    batch = json.loads((session_kit.out / "batch_report.json").read_text(encoding="utf-8"))
    assert [v["status"] for v in batch["videos"]] == ["ok", "ok"]


def test_quiet_prints_nothing_when_every_memo_is_done_busy_or_not_downloaded(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha"), Row("CLOUD-0001", path=None)], media)
    first = cli(session_kit, "--quiet", "memos:new")
    assert first.exit_code == 0 and names(first.output) == ["AAAA1111.m4a"]
    assert "not downloaded" not in first.output

    idle = cli(session_kit, "--quiet", "memos:new")
    assert (idle.exit_code, idle.output) == (0, "")

    shown = cli(session_kit, "memos:new")
    assert "no new memos to transcribe" in shown.output and "not downloaded" in shown.output

    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha")], media, name="second")
    with video_lock(session_kit.out / rec_id("alpha")):
        busy = cli(session_kit, "--quiet", "memos:all")
    assert (busy.exit_code, busy.output) == (0, "")


def test_quiet_is_for_local_targets(session_kit) -> None:
    result = cli(session_kit, "--quiet", "pubVideo001")
    assert result.exit_code == 1 and "--quiet only applies to local and Voice Memos targets" in result.output


def test_new_with_a_folder_that_has_no_media_is_still_an_error(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha")], media)
    empty = tmp_path / "empty"
    empty.mkdir()
    result = cli(session_kit, "memos:new", str(empty))
    assert result.exit_code == 1 and "no audio or video files found in the folders given" in result.output
    assert not worker.calls


def test_selectors_are_case_insensitive(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha"), memo_row("BBBB2222", "beta", unix=MIDDLE)], media)
    assert names(cli(session_kit, "memos:LATEST").output) == ["AAAA1111.m4a"]
    result = cli(session_kit, "memos:New")
    assert result.exit_code == 0 and names(result.output) == ["BBBB2222.m4a"]


def test_overlapping_selectors_run_each_memo_once_newest_first(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [
            memo_row("ABCD1111", "alpha", unix=NEWEST),
            memo_row("EFGH2222", "beta", unix=MIDDLE),
            memo_row("IJKL3333", "gamma", unix=OLDEST),
        ],
        media,
    )
    result = cli(session_kit, "memos:ijkl", "memos:efgh", "memos:all", "memos:ijkl")
    assert result.exit_code == 0, result.output
    assert names(result.output) == ["ABCD1111.m4a", "EFGH2222.m4a", "IJKL3333.m4a"]
    assert "3 ok, 0 failed" in result.output


def test_the_container_date_stands_in_for_a_missing_memo_date(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(
        tmp_path,
        monkeypatch,
        [memo_row("DATED001", "alpha", unix=NEWEST), memo_row("NODATE01", "beta", unix=None)],
        media,
    )
    media.tags["DATED001.m4a"] = {"creation_time": "2020-01-01T00:00:00Z"}
    media.tags["NODATE01.m4a"] = {"creation_time": "2021-02-03T04:05:06Z"}
    library_rows = voicememos.list_memos()
    assert [m.recorded_at is None for m in library_rows] == [False, True]
    assert cli(session_kit, "memos:all").exit_code == 0
    assert source_of(session_kit, "alpha").recorded_at == datetime.fromtimestamp(NEWEST, UTC)
    assert source_of(session_kit, "beta").recorded_at == datetime(2021, 2, 3, 4, 5, 6, tzinfo=UTC)


def test_an_unreadable_composition_memo_gets_its_own_notice(
    session_kit, media: Media, worker: Worker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = library(
        tmp_path,
        monkeypatch,
        [memo_row("AAAA1111", "alpha"), Row("COMPOSE1", path="edit.composition", audio=None, unix=MIDDLE)],
        media,
    )
    (folder / "edit.composition").mkdir()
    result = cli(session_kit, "memos:all")
    assert result.exit_code == 0 and names(result.output) == ["AAAA1111.m4a"]
    assert "1 memo(s) cannot be read (a .composition folder, or a file outside the library), skipped" in result.output
    assert "not downloaded" not in result.output
    named = cli(session_kit, "memos:compose1")
    assert named.exit_code == 1 and "memo compose1 cannot be read" in named.output
    assert "Fix: share the memo out of Voice Memos and pass the file" in named.output


def test_a_rerun_with_new_provenance_rebuilds_only_the_context_record(
    session_kit, media: Media, worker: Worker, diarizer, tmp_path: Path
) -> None:
    path = media.add(tmp_path / "memo.m4a", "alpha", tags={"title": "Container title"})
    llm = FakeLLM()
    run_files(session_kit, [path], llm=llm)
    calls, requests, converted = len(worker.calls), len(llm.requests), list(media.converted)
    assert calls == 1 and requests > 0

    known = {path: local.Provenance("voice_memos", "Renamed memo", datetime.fromtimestamp(NEWEST, UTC))}
    _, results, events = run_files(session_kit, [path], llm=llm, provenance=known)
    source = source_of(session_kit, "alpha")
    assert (source.title, source.origin) == ("Renamed memo", "voice_memos")
    assert source.recorded_at == datetime.fromtimestamp(NEWEST, UTC)
    assert (len(worker.calls), len(llm.requests), media.converted) == (calls, requests, converted)
    assert len(diarizer.calls) == 1 and [s.id for s in load_document(context(session_kit, "alpha")).speakers] == ["S1"]
    assert "skip" not in [e.stage for e in events]
    assert results[0].document is not None and results[0].document.source == source
    assert (session_kit.out / rec_id() / "context.md").read_text(encoding="utf-8").startswith("# Renamed memo\n")

    _, again, events = run_files(session_kit, [path], llm=llm, provenance=known)
    assert again[0].outcome.reused == ("context",) and [e.stage for e in events] == ["skip"]

    _, plain, _ = run_files(session_kit, [path], llm=llm)
    assert plain[0].outcome.reused == ("context",), "a plain file never overwrites the memo's record"


class SecondOpinionWorker(Worker):
    """Hears "50" where the first decode heard "5" whenever it re-decodes chosen spans."""

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.spoken = tuple("50" if w == "5" else w for w in SPOKEN) if request.files[0].chunks else SPOKEN
        return super().__call__(request, python=python)


def test_no_title_file_name_memo_title_or_speaker_name_reaches_the_llm_but_the_glossary_does(
    session_kit, media: Media, worker: Worker, diarizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library(tmp_path, monkeypatch, [memo_row("AAAA1111", "alpha")], media)
    media.tags["AAAA1111.m4a"] = {"title": "Container secret title"}
    # "we met 5 people today" is one voice, "and then we left" another.
    diarizer.turns = [(0, 4, 0), (5, 8, 3)]
    # A second opinion that hears "50" disputes the "5", so a numeric verdict is asked for too.
    config_file = session_kit.home / ".config" / "anyverbatim" / "config.toml"
    config_file.write_text(config_file.read_text() + '[text]\nnumeric_second_opinion = "w-vad"\n')
    monkeypatch.setattr(asr_runner, "run_worker", SecondOpinionWorker())
    llm = FakeLLM()
    monkeypatch.setattr(pipeline, "check_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "client_from_options", lambda *_args, **_kwargs: llm)

    def run(*args: str):
        argv = ["run", "--out", str(session_kit.out), "--llm", "claude-cli:fake-model", *args, "memos:latest"]
        result = CliRunner().invoke(main, argv)
        assert result.exit_code == 0, result.output

    run("--glossary", "Zorblax, Quuxcorp,")
    named = CliRunner().invoke(main, ["speakers", "--out", str(session_kit.out), rec_id(), "S1=Zelda", "S2=Quentin"])
    assert named.exit_code == 0, named.output
    run("--glossary", "Zorblax, Quuxcorp", "--force")

    sent = [request.system + request.user for request in llm.requests]
    assert any('"speaker": "S2"' in text for text in sent), "the text stages ran on the diarized memo twice"
    assert any("ITEMS:" in text and '"50"' in text for text in sent), "a numeric verdict was asked for"
    for secret in (SECRET, "Container secret title", "AAAA1111", ".m4a", "Zelda", "Quentin"):
        assert not any(secret in text for text in sent), secret
    corrections = [text for text in sent if '"fixed"' in text and "SEGMENTS:" in text]
    assert corrections and all("Prefer these domain terms when relevant: Zorblax, Quuxcorp." in t for t in corrections)
    assert [(s.id, s.name) for s in load_document(context(session_kit, "alpha")).speakers] == [
        ("S1", "Zelda"),
        ("S2", "Quentin"),
    ]
