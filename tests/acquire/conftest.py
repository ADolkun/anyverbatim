from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from acquire_fixture import INFO_DURATION, FakeCookieSnapshot, FakeProc, Fakes, SessionKit

from anyverbatim import proc
from anyverbatim.auth import snapshot
from anyverbatim.ytdlp import jsruntime, version


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def fake_proc(monkeypatch: pytest.MonkeyPatch) -> FakeProc:
    fake = FakeProc()
    monkeypatch.setattr(proc, "run", fake)
    return fake


@pytest.fixture
def session_kit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_proc: FakeProc) -> SessionKit:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "ANYVERBATIM_HOME", "ANYVERBATIM_AUTH"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The pipeline under test runs without an LLM unless a test picks one; the default LLM needs a ChatGPT sign-in.
    config_file = home / ".config" / "anyverbatim" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('[llm]\nprovider = "none"\n', encoding="utf-8")
    released = datetime.now(UTC).date() - timedelta(days=10)
    monkeypatch.setattr(version, "installed", lambda: version.YtDlpVersion("2099.01.01", released))
    monkeypatch.setattr(jsruntime, "detect", lambda: jsruntime.JsRuntime("deno", (), None))
    snaps = FakeCookieSnapshot()
    monkeypatch.setattr(snapshot, "cookie_snapshot", snaps)
    fake_proc.probe_returns(f"{INFO_DURATION}\n")

    def ffmpeg(argv: list[str]) -> proc.ProcResult:
        Path(argv[-1]).write_bytes(b"RIFF fake wav")
        return proc.ProcResult(0, "", "", 0.0)

    fake_proc.responses["ffmpeg"] = ffmpeg
    return SessionKit(home, tmp_path / "out", snaps, fake_proc)
