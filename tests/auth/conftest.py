import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from auth_fakes import PROFILES, FakeSnapshot, Kit

from anyverbatim.auth import chromium, snapshot, source
from anyverbatim.ytdlp import jsruntime, version


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "ANYVERBATIM_HOME", source.ENV_VAR, jsruntime.ENV_OVERRIDE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = chromium.browser_root("chrome")
    for directory in PROFILES:
        (root / directory).mkdir(parents=True)
        (root / directory / "Cookies").write_bytes(b"sqlite")
    (root / "Local State").write_text(
        json.dumps({"profile": {"last_used": "Default", "info_cache": PROFILES}}), encoding="utf-8"
    )
    released = datetime.now().astimezone().date() - timedelta(days=10)
    monkeypatch.setattr(version, "installed", lambda: version.YtDlpVersion("2099.01.01", released))
    monkeypatch.setattr(jsruntime, "detect", lambda: jsruntime.JsRuntime("deno", (), None))
    return tmp_path


@pytest.fixture
def kit(home: Path, monkeypatch: pytest.MonkeyPatch) -> Kit:
    snap = FakeSnapshot()
    monkeypatch.setattr(snapshot, "cookie_snapshot", snap)
    return Kit(home, snap)


@pytest.fixture
def fakes() -> type[Kit]:
    """The runner doubles without the fake snapshot, for tests that take real cookie-file snapshots."""
    return Kit
