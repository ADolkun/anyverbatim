import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from chatgpt_fakes import EMAIL, store_account
from click.testing import CliRunner

from ytverbatim import paths
from ytverbatim.acquire import voicememos
from ytverbatim.auth import cookiefile, source
from ytverbatim.auth.cookiefile import CookieEntry
from ytverbatim.auth.cooldown import read_cooldown, start_cooldown
from ytverbatim.auth.source import BrowserSource, CookieFileSource
from ytverbatim.cli import auth as cli_auth
from ytverbatim.cli import main
from ytverbatim.cli import session as cli_session
from ytverbatim.ytdlp.types import Signal

JSON_KEYS = {
    "state",
    "ok",
    "usable",
    "fix",
    "detail",
    "source",
    "profile",
    "logged_in",
    "premium",
    "counts",
    "checked_at",
    "cached",
    "ytdlp_version",
    "ytdlp_age_days",
    "js_runtime",
    "warnings",
    "cooldown_until",
    "probe_skipped",
}


@pytest.fixture(autouse=True)
def voice_memos_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voicememos, "probe", lambda: voicememos.Probe("ok", memos=3))


def _use_runner(monkeypatch: pytest.MonkeyPatch, runner: object) -> None:
    monkeypatch.setattr(cli_auth, "SubprocessRunner", lambda: runner)
    monkeypatch.setattr(cli_session, "SubprocessRunner", lambda: runner)


def _invoke(*args: str):
    return CliRunner().invoke(main, ["auth", *args])


def _status(*args: str):
    return CliRunner().invoke(main, ["status", *args])


# The ChatGPT sign-in is absent in these homes, so with the default chatgpt LLM a usable YouTube login exits 1.
EXIT_CHATGPT_SIGNED_OUT = 1


def test_auth_is_registered() -> None:
    assert "auth" in CliRunner().invoke(main, ["--help"]).output
    commands = set(_invoke("--help").output.split())
    assert {"doctor", "profiles", "use"} <= commands
    assert "status" not in commands, "the YouTube check moved to `ytverbatim status`"


def test_status_json_shape_and_cache(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    first = _status("--json", "--refresh")
    assert first.exit_code == EXIT_CHATGPT_SIGNED_OUT, first.output
    payload = json.loads(first.output)["youtube"]
    assert set(payload) == JSON_KEYS
    assert payload["state"] == "ok" and payload["ok"] is True and payload["usable"] is True
    assert payload["cached"] is False
    assert payload["source"] == "chrome:Default" and payload["profile"] == "Default"
    assert payload["premium"] is None

    second = json.loads(_status("--json").output)["youtube"]
    assert second["cached"] is True
    assert len(runner.calls) == 1


def test_status_human_output(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    _use_runner(monkeypatch, kit.ok())
    result = _status()
    assert result.exit_code == EXIT_CHATGPT_SIGNED_OUT
    first = result.output.splitlines()[0]
    assert first.startswith("YouTube      chrome:Default  ok, signed in · Premium: unknown · verified ")
    assert "fakeLikedId" not in result.output


def test_status_auth_problem_needs_attention(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    _use_runner(monkeypatch, kit.fail(Signal.LOGIN_REQUIRED))
    result = _status()
    assert result.exit_code == 1
    assert result.output.startswith("YouTube      chrome:Default  logged_out: ")
    assert "Fix: chrome profile 'Default' is not signed in" in result.output


def test_status_without_a_browser_login_is_fine_for_public_videos(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_runner(monkeypatch, kit.ok())
    store_account(expires_at=time.time() + 3600)
    result = _status()
    assert result.exit_code == 0, result.output
    youtube = result.output.splitlines()[0]
    assert youtube == f"YouTube      {'none':<{len(EMAIL)}}  public videos only (no browser login)"
    assert "Fix:" not in result.output
    payload = json.loads(_status("--json").output)
    assert payload["youtube"]["state"] == "no_source" and payload["youtube"]["usable"] is True


def test_status_with_broken_saved_source_needs_attention(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    paths.auth_source_file().write_text("{", encoding="utf-8")
    _use_runner(monkeypatch, kit.ok())
    store_account(expires_at=time.time() + 3600)
    result = _status()
    assert result.exit_code == 1
    assert "no_source" in result.output and str(kit.home) not in result.output


def test_profiles_masks_emails_by_default() -> None:
    result = _invoke("profiles")
    assert result.exit_code == 0
    assert "a***@example.com" in result.output and "b***@example.com" in result.output
    assert "alice@" not in result.output and "bob@" not in result.output
    assert "Default" in result.output and "Work" in result.output
    assert "ytverbatim auth use chrome:<DIR>" in result.output


def test_profiles_show_emails() -> None:
    assert "alice@example.com" in _invoke("profiles", "--show-emails").output


def test_profiles_missing_browser_exits_2() -> None:
    result = _invoke("profiles", "--browser", "brave")
    assert result.exit_code == 2
    assert "browser_not_found" in result.output


def test_use_saves_a_valid_profile() -> None:
    result = _invoke("use", "chrome:Default")
    assert result.exit_code == 0, result.output
    assert source.resolve_source() == BrowserSource("chrome", "Default")


def test_use_rejects_an_unknown_profile_with_the_list() -> None:
    result = _invoke("use", "chrome:Nope")
    assert result.exit_code == 2
    assert "✗ profile_not_found" in result.output
    assert "'Default' (Person 1)" in result.output and "'Work'" in result.output
    assert not paths.auth_source_file().exists()


def test_use_bare_browser_refuses_to_guess() -> None:
    result = _invoke("use", "chrome")
    assert result.exit_code == 2
    assert "profile_not_found" in result.output


def test_use_bare_browser_saves_the_only_profile(kit) -> None:
    (kit.chrome_root / "Work" / "Cookies").unlink()
    assert _invoke("use", "chrome").exit_code == 0
    assert source.resolve_source() == BrowserSource("chrome", "Default")


def test_use_unknown_browser() -> None:
    result = _invoke("use", "firefox")
    assert result.exit_code == 2
    assert "browser_not_found" in result.output


def _cookie_file(path: Path, *, signed_in: bool) -> Path:
    rows = [CookieEntry(".youtube.com", "/", True, 4_000_000_000, "PREF", "x")]
    if signed_in:
        rows += [
            CookieEntry(".youtube.com", "/", True, 4_000_000_000, "LOGIN" + "_INFO", "secret-login"),
            CookieEntry(".youtube.com", "/", True, 4_000_000_000, "SAPISID", "secret-sapisid"),
        ]
    path.write_text(cookiefile.render(rows), encoding="utf-8")
    return path


def test_use_cookie_file_validation(home: Path) -> None:
    missing = _invoke("use", f"cookie-file:{home / 'missing.txt'}")
    assert missing.exit_code == 2 and "cookie_file_invalid" in missing.output
    assert str(home) not in missing.output

    signed_out = _cookie_file(home / "out.txt", signed_in=False)
    result = _invoke("use", f"cookie-file:{signed_out}")
    assert result.exit_code == 2 and "cookie_file_logged_out" in result.output

    good = _cookie_file(home / "good.txt", signed_in=True)
    result = _invoke("use", f"cookie-file:{good}")
    assert result.exit_code == 0, result.output
    assert "cookie-file:~/good.txt" in result.output
    assert "secret" not in result.output
    assert source.resolve_source() == CookieFileSource(good)


def test_use_none_clears() -> None:
    _invoke("use", "chrome:Default")
    result = _invoke("use", "none")
    assert result.exit_code == 0
    assert source.resolve_source() is None


def test_doctor_prints_one_line_per_check(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/bin/{name}")
    _use_runner(monkeypatch, kit.ok())
    result = _invoke("doctor")
    assert result.exit_code == 2
    lines = result.output.splitlines()
    assert lines[0] == "✓ ffmpeg: ffmpeg found"
    assert any(line.startswith("✗ source: no_source") for line in lines)
    assert any(line.startswith("    Fix: No auth source set") for line in lines)


def test_status_during_a_cooldown_skips_the_probe_unless_ignored(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    start_cooldown(Signal.BOT_CHECK, datetime.now(UTC))
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    result = _status("--refresh")
    assert result.exit_code == 1 and runner.calls == []
    assert result.output.startswith("YouTube      chrome:Default  bot_check: cooldown active until ")
    assert "Wait until " in result.output and "an hour has passed" not in result.output
    assert "--ignore-cooldown" in result.output
    payload = json.loads(_status("--json", "--refresh").output)["youtube"]
    assert payload["probe_skipped"] is True and payload["cooldown_until"] is not None

    allowed = _status("--refresh", "--ignore-cooldown")
    assert allowed.exit_code == EXIT_CHATGPT_SIGNED_OUT and len(runner.calls) == 1


def test_status_probe_stop_signal_starts_a_cooldown(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    _use_runner(monkeypatch, kit.fail(Signal.BOT_CHECK))
    result = _status("--refresh")
    assert result.exit_code == 1
    assert result.output.startswith("YouTube      chrome:Default  bot_check: the :ytfav probe hit bot_check")
    cooldown = read_cooldown()
    assert cooldown is not None and cooldown.until > datetime.now(UTC) + timedelta(minutes=59)
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    assert _status("--refresh").exit_code == 1 and runner.calls == []


def test_doctor_during_a_cooldown_skips_the_probe_unless_ignored(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/bin/{name}")
    source.save_source(BrowserSource("chrome", "Default"))
    start_cooldown(Signal.RATE_LIMITED, datetime.now(UTC))
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    result = _invoke("doctor")
    assert result.exit_code == 2 and runner.calls == []
    assert any(line.startswith("! login: skipped: cooldown active") for line in result.output.splitlines())
    assert any(line.startswith("✗ cooldown: rate_limited") for line in result.output.splitlines())
    _invoke("doctor", "--ignore-cooldown")
    assert len(runner.calls) == 1


def _signed_in_youtube(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    _use_runner(monkeypatch, kit.ok())


def test_status_signed_out_of_chatgpt(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    _signed_in_youtube(kit, monkeypatch)
    result = _status()
    assert result.exit_code == 1
    lines = result.output.splitlines()
    assert lines[0].startswith("YouTube      chrome:Default  ok, signed in")
    assert lines[1] == "ChatGPT      -               not signed in"
    assert lines[2] == "                             Fix: run `ytverbatim login`"
    assert lines[3] == "LLM          chatgpt:gpt-6.1-sol (medium effort)"


def test_status_signed_in_to_chatgpt(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    _signed_in_youtube(kit, monkeypatch)
    store_account(expires_at=time.time() + 3600)
    result = _status()
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith(f"YouTube      {'chrome:Default':<{len(EMAIL)}}  ok, signed in")
    assert lines[1] == f"ChatGPT      {EMAIL}  ok, plan permission, session ready"
    assert lines[2] == "LLM          chatgpt:gpt-6.1-sol (medium effort)"
    assert "access-1" not in result.output and "refresh-1" not in result.output


@pytest.mark.parametrize(
    ("stored", "said", "code"),
    [
        ({"expires_at": time.time() + 30}, "ok, plan permission, refreshes automatically on the next call", 0),
        ({"expires_at": time.time() - 1, "refresh": ""}, "needs re-sign-in", 1),
        ({"scopes": ("openid",)}, "signed in without ChatGPT plan permission", 1),
        (
            {"refresh_refused": True},
            "needs re-sign-in: OpenAI refused to renew it, so it works only until its token expires",
            1,
        ),
    ],
)
def test_status_shows_when_chatgpt_renews_on_its_own_or_needs_a_sign_in(
    kit, monkeypatch: pytest.MonkeyPatch, stored: dict[str, object], said: str, code: int
) -> None:
    _signed_in_youtube(kit, monkeypatch)
    store_account(**stored)  # type: ignore[arg-type]
    result = _status()
    assert result.exit_code == code, result.output
    assert result.output.splitlines()[1] == f"ChatGPT      {EMAIL}  {said}"
    assert ("Fix: run `ytverbatim login`" in result.output) is (code == 1)


def test_status_with_no_llm_does_not_fail_on_chatgpt(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    _signed_in_youtube(kit, monkeypatch)
    config = paths.config_file()
    config.parent.mkdir(parents=True)
    config.write_text('[llm]\nprovider = "none"\n', encoding="utf-8")
    result = _status()
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-2] == "LLM          none (the LLM stages are skipped)"
    payload = json.loads(_status("--json").output)
    assert payload["llm"] == {"provider": "none", "label": "none (the LLM stages are skipped)"}
    assert payload["chatgpt"] == {"email": "", "status": "signed_out", "usable": False}


def test_status_shows_voice_memos_after_the_llm(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    _signed_in_youtube(kit, monkeypatch)
    store_account(expires_at=time.time() + 3600)
    line = _status().output.splitlines()[3]
    assert line.startswith("Voice Memos  -") and line.endswith("  ok · 3 memos")
    assert json.loads(_status("--json").output)["voice_memos"] == {
        "state": "ok",
        "memos": 3,
        "detail": "",
        "fix": "",
    }


@pytest.mark.parametrize(
    ("probe", "line"),
    [
        (voicememos.Probe("no_access", fix="grant it"), "no access (Full Disk Access)"),
        (voicememos.Probe("not_syncing", fix="turn on sync"), "not syncing"),
        (voicememos.Probe("unreadable", detail="layout changed"), "unreadable: layout changed"),
    ],
)
def test_the_voice_memos_line_never_changes_the_exit_code(
    kit, monkeypatch: pytest.MonkeyPatch, probe: voicememos.Probe, line: str
) -> None:
    _signed_in_youtube(kit, monkeypatch)
    store_account(expires_at=time.time() + 3600)
    monkeypatch.setattr(voicememos, "probe", lambda: probe)
    result = _status()
    assert result.exit_code == 0, result.output
    assert line in result.output.splitlines()[3]
    assert (f"Fix: {probe.fix}" in result.output) is bool(probe.fix)
    assert json.loads(_status("--json").output)["voice_memos"]["state"] == probe.state
