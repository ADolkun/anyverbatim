"""Moving from the old name, ytverbatim: directories, environment variables and the keys of stored work."""

import errno
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest
from chatgpt_fakes import CLIENT_ID, NOW, FakeOpener, form, json_response, store_account, token_payload
from click.testing import CliRunner
from pipeline_fixture import CAPTIONED, FIXTURE_ROOT, MIXED, FakeLLM, isolated, run
from test_pipeline_fixture import STAGES, RefusingYtDlp, refuse, stage_keys

from anyverbatim import legacy, paths, proc
from anyverbatim.asr import registry as asr_registry
from anyverbatim.asr import runner as asr_runner
from anyverbatim.asr.models import FINGERPRINT_FILE, FileJob, WorkerRequest
from anyverbatim.cli import main
from anyverbatim.diarize import registry as diarize_registry
from anyverbatim.diarize.protocol import DiarizeRequest
from anyverbatim.diarize.protocol import FileJob as DiarizeJob
from anyverbatim.errors import ConfigError
from anyverbatim.llm import chatgpt_auth
from anyverbatim.llm.chatgpt_auth import TOKEN_URL
from anyverbatim.store import STAGES_DIR


@pytest.fixture
def home() -> Path:
    return Path.home()


def old_roots(home: Path) -> dict[str, Path]:
    return {
        "config": home / ".config" / "ytverbatim",
        "state": home / ".local" / "state" / "ytverbatim",
        "cache": home / "Library" / "Caches" / "ytverbatim",
        "data": home / ".local" / "share" / "ytverbatim",
    }


def new_roots(home: Path) -> dict[str, Path]:
    return {name: path.with_name("anyverbatim") for name, path in old_roots(home).items()}


def make_old_layout(home: Path) -> None:
    roots = old_roots(home)
    for root in roots.values():
        root.mkdir(parents=True, mode=0o700)
    (roots["config"] / "config.toml").write_text("[llm]\nprovider = 'none'\n", encoding="utf-8")
    (roots["state"] / "voicememos.json").write_text("{}", encoding="utf-8")
    (roots["cache"] / "pins").mkdir()
    (roots["data"] / "index.md").write_text("# index\n", encoding="utf-8")


def test_an_old_variable_without_its_new_counterpart_fails_naming_the_new_one() -> None:
    with pytest.raises(ConfigError, match="ANYVERBATIM_HOME") as raised:
        legacy.check_environment({"YTVERBATIM_HOME": "/data", "YTVERBATIM_AUTH_X": "", "PATH": "/bin"})
    assert "YTVERBATIM_AUTH_X, YTVERBATIM_HOME are set" in str(raised.value)
    assert "ANYVERBATIM_AUTH_X, ANYVERBATIM_HOME" in raised.value.fix


def test_an_old_variable_beside_its_new_counterpart_is_left_alone() -> None:
    legacy.check_environment({"YTVERBATIM_HOME": "/old", "ANYVERBATIM_HOME": "/new"})
    legacy.check_environment({"ANYVERBATIM_AUTH": "chrome"})


@pytest.mark.parametrize(
    "name", ["YTVERBATIM_HOME", "YTVERBATIM_AUTH", "YTVERBATIM_JS_RUNTIMES", "YTVERBATIM_EVAL_DIR"]
)
def test_every_path_lookup_refuses_an_old_variable(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, "set")
    with pytest.raises(ConfigError, match="ANYVERBATIM_" + name.removeprefix("YTVERBATIM_")):
        paths.state_dir()


def test_the_move_works_in_a_process_without_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, new = tmp_path / "ytverbatim", tmp_path / "anyverbatim"
    old.mkdir(mode=0o700)
    monkeypatch.setattr(sys, "stderr", None)
    assert legacy.move_into_place(old, new) == new
    assert new.is_dir()


def test_only_the_old_directory_is_moved_with_one_notice(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old, new = tmp_path / "ytverbatim", tmp_path / "anyverbatim"
    (old / "auth").mkdir(parents=True, mode=0o700)
    (old / "auth" / "source.json").write_text("{}", encoding="utf-8")
    assert legacy.move_into_place(old, new) == new
    assert not old.exists()
    assert (new / "auth" / "source.json").read_text(encoding="utf-8") == "{}"
    assert capsys.readouterr().err == f"anyverbatim: moved {old} to {new} (ytverbatim was renamed to anyverbatim)\n"
    assert legacy.move_into_place(old, new) == new
    assert capsys.readouterr().err == ""


def test_neither_directory_creates_nothing(tmp_path: Path) -> None:
    assert legacy.move_into_place(tmp_path / "ytverbatim", tmp_path / "anyverbatim") == tmp_path / "anyverbatim"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("new_is_empty", [True, False])
def test_both_directories_fail_loud_and_move_nothing(tmp_path: Path, new_is_empty: bool) -> None:
    old, new = tmp_path / "ytverbatim", tmp_path / "anyverbatim"
    old.mkdir()
    (old / "kept").write_text("old", encoding="utf-8")
    new.mkdir()
    if not new_is_empty:
        (new / "kept").write_text("new", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"both .* exist") as raised:
        legacy.move_into_place(old, new)
    assert "merge" in raised.value.fix
    assert (old / "kept").read_text(encoding="utf-8") == "old"
    assert sorted(p.name for p in new.iterdir()) == ([] if new_is_empty else ["kept"])


def test_a_failed_move_fails_loud_and_copies_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old, new = tmp_path / "ytverbatim", tmp_path / "anyverbatim"
    old.mkdir()
    (old / "kept").write_text("old", encoding="utf-8")

    def cross_device(_src: object, _dst: object) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))

    monkeypatch.setattr(os, "rename", cross_device)
    with pytest.raises(ConfigError, match="cannot move") as raised:
        legacy.move_into_place(old, new)
    assert "yourself" in raised.value.fix
    assert (old / "kept").exists()
    assert not new.exists()


def test_losing_the_race_to_another_process_carries_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old, new = tmp_path / "ytverbatim", tmp_path / "anyverbatim"
    old.mkdir()
    real_rename = os.rename

    def other_process_wins(src: str, dst: str) -> None:
        real_rename(src, dst)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), src)

    monkeypatch.setattr(os, "rename", other_process_wins)
    assert legacy.move_into_place(old, new) == new
    assert capsys.readouterr().err == ""


def test_concurrent_moves_leave_one_winner_and_no_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old, new = tmp_path / "ytverbatim", tmp_path / "anyverbatim"
    old.mkdir()
    (old / "kept").write_text("old", encoding="utf-8")
    start = threading.Barrier(8)
    outcomes: list[object] = []

    def migrate() -> None:
        start.wait()
        try:
            outcomes.append(legacy.move_into_place(old, new))
        except Exception as exc:  # noqa: BLE001 - any failure of a racing thread is an outcome to assert on
            outcomes.append(exc)

    threads = [threading.Thread(target=migrate) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == [new] * 8
    assert (new / "kept").read_text(encoding="utf-8") == "old"
    assert capsys.readouterr().err.count("moved") == 1


def test_migrate_moves_every_root(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_old_layout(home)
    paths.migrate()
    new = new_roots(home)
    assert all(not path.exists() for path in old_roots(home).values())
    assert paths.config_file() == new["config"] / "config.toml"
    assert paths.config_file().read_text(encoding="utf-8").startswith("[llm]")
    assert paths.voicememos_cache_file() == new["state"] / "voicememos.json"
    assert paths.voicememos_cache_file().exists()
    assert paths.pins_dir() == new["cache"] / "pins"
    assert (paths.home_dir() / "index.md").exists()
    assert capsys.readouterr().err.count("moved") == len(new)


@pytest.mark.parametrize("root", ["config", "state", "cache", "data"])
def test_the_first_lookup_of_a_root_moves_it(home: Path, root: str) -> None:
    make_old_layout(home)
    lookup = {
        "config": paths.config_file,
        "state": paths.chatgpt_auth_dir,
        "cache": paths.cache_dir,
        "data": paths.default_output_root,
    }[root]
    assert str(lookup()).startswith(str(new_roots(home)[root]))
    assert not old_roots(home)[root].exists()


def test_the_xdg_roots_are_moved_where_they_are(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "xdg-state"))
    (home / "xdg-state" / "ytverbatim").mkdir(parents=True)
    assert paths.state_dir() == home / "xdg-state" / "anyverbatim"
    assert not (home / "xdg-state" / "ytverbatim").exists()


def test_a_home_the_user_chose_leaves_the_old_default_home_alone(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_old_layout(home)
    monkeypatch.setenv("ANYVERBATIM_HOME", str(home / "chosen"))
    paths.migrate()
    assert paths.home_dir() == home / "chosen"
    assert old_roots(home)["data"].exists()
    assert not new_roots(home)["data"].exists()


def test_the_cli_reports_a_conflict_with_its_fix_before_any_command(home: Path) -> None:
    make_old_layout(home)
    new_roots(home)["state"].mkdir(parents=True)
    result = CliRunner().invoke(main, ["memos", "--help"])
    assert result.exit_code == 1
    assert "both" in result.output
    assert "Fix: merge" in result.output
    assert old_roots(home)["state"].exists()


def test_the_cli_refuses_an_old_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YTVERBATIM_AUTH", "chrome")
    result = CliRunner().invoke(main, ["memos", "--help"])
    assert result.exit_code == 1
    assert "ANYVERBATIM_AUTH" in result.output


def test_fingerprints_keep_the_adapter_path_of_the_old_package() -> None:
    assert legacy.fingerprint_factory("anyverbatim.asr.engines.qwen3.create") == "ytverbatim.asr.engines.qwen3.create"
    assert legacy.fingerprint_factory("tests.asr.fake_engine.create") == "tests.asr.fake_engine.create"
    assert legacy.fingerprint_factory("anyverbatim") == "anyverbatim"
    engine = asr_registry.get("w-prod")
    assert engine.factory.startswith("anyverbatim.")
    request = WorkerRequest(engine=engine, out_dir="/out", files=(FileJob(id="a", wav="/a.wav"),))
    keyed = json.loads(json.dumps(request.fingerprint()))
    assert keyed["engine"]["factory"] == "ytverbatim.asr.engines.whisper_mlx.create"
    assert json.loads(request.model_dump_json())["engine"]["factory"] == engine.factory
    diarizer = next(iter(diarize_registry.DIARIZERS.values()))
    diarize = DiarizeRequest(diarizer=diarizer, out_dir="/out", files=(DiarizeJob(id="a", wav="/a.wav"),))
    assert json.loads(json.dumps(diarize.fingerprint()))["diarizer"]["factory"].startswith("ytverbatim.diarize.")


def test_the_committed_worker_folders_were_made_under_the_old_name() -> None:
    folders = sorted((FIXTURE_ROOT / MIXED / STAGES_DIR).glob(f"*/*/*/{FINGERPRINT_FILE}"))
    assert folders
    for fingerprint in folders:
        assert json.loads(fingerprint.read_text(encoding="utf-8"))["engine"]["factory"].startswith("ytverbatim.")


@pytest.mark.parametrize("video_id", [MIXED, CAPTIONED])
def test_a_store_in_the_old_data_home_is_moved_and_every_stage_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, video_id: str
) -> None:
    home = tmp_path / "home"
    old_videos = home / ".local" / "share" / "ytverbatim" / "videos"
    shutil.copytree(FIXTURE_ROOT, old_videos)
    vdir = old_videos / video_id
    # Built by the old name: the document is made again, cheaply, from the stored stages.
    for name in ("context.json", "context.md", f"{STAGES_DIR}/context.json"):
        (vdir / name).unlink()
    monkeypatch.setattr(proc, "run", refuse)
    monkeypatch.setattr(asr_runner, "run_worker", refuse)
    llm = FakeLLM()
    with isolated(home):
        result = run(video_id, None, llm, RefusingYtDlp())
        moved = paths.default_output_root() / video_id
    assert moved == home / ".local" / "share" / "anyverbatim" / "videos" / video_id
    assert not old_videos.exists()
    assert result.document is not None, result.outcome
    assert result.document.generator.name == "anyverbatim"
    assert {t.stage: t.reused for t in result.timings if t.stage not in ("fetch", "write")} == dict.fromkeys(
        STAGES[video_id], True
    )
    assert llm.requests == []
    assert stage_keys(moved) == stage_keys(FIXTURE_ROOT / video_id)


def test_a_chatgpt_sign_in_keeps_working_after_its_store_moves(home: Path) -> None:
    stored = store_account(expires_at=NOW)
    state_root = new_roots(home)["state"]
    state_root.rename(old_roots(home)["state"])
    opener = FakeOpener({f"POST {TOKEN_URL}": json_response(token_payload(access="access-2", refresh="refresh-2"))})
    assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW) == "access-2"
    (request,) = opener.requests
    assert form(request)["client_id"] == CLIENT_ID
    assert not old_roots(home)["state"].exists()
    with chatgpt_auth._locked() as state:
        assert (state.host_id, state.client_id) == (stored.host_id, CLIENT_ID)
