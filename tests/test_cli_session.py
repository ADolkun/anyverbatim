import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from chatgpt_fakes import EMAIL, store_account
from click.testing import CliRunner

from ytverbatim.cli import main
from ytverbatim.cli import session as cli_session
from ytverbatim.config import LLMOptions
from ytverbatim.llm import chatgpt_auth


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path


def test_the_command_tree() -> None:
    top = CliRunner().invoke(main, ["--help"]).output
    commands = {line.split()[0] for line in top.split("Commands:", 1)[1].splitlines() if line.strip()}
    assert commands == {"auth", "fetch", "llm", "login", "logout", "memos", "render", "run", "status"}
    llm = CliRunner().invoke(main, ["llm", "--help"]).output
    assert {line.split()[0] for line in llm.split("Commands:", 1)[1].splitlines() if line.strip()} == {"test"}
    for gone in (["llm", "login"], ["llm", "logout"], ["llm", "check"], ["auth", "status"]):
        assert CliRunner().invoke(main, gone).exit_code == 2, gone


@dataclass
class FakeLogin:
    plan: bool = True
    calls: list[bool] = field(default_factory=list)

    def __call__(self, *, open_url: Callable[[str], object], switch: bool = False) -> chatgpt_auth.LoginResult:
        self.calls.append(switch)
        assert open_url("https://auth.openai.com/api/accounts/authorize?client_id=dynamic_agent_client") is not False
        return chatgpt_auth.LoginResult(EMAIL, plan=self.plan)


def test_login_when_already_signed_in_opens_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    store_account(expires_at=time.time() + 3600)
    monkeypatch.setattr(chatgpt_auth, "login", lambda **_kwargs: pytest.fail("opened a sign-in"))
    result = CliRunner().invoke(main, ["login"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"Already signed in as {EMAIL} (")
    assert "`ytverbatim login --switch` to use another account" in result.output
    assert "access-1" not in result.output


def test_login_switch_signs_in_even_when_signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    store_account(expires_at=time.time() + 3600)
    login = FakeLogin()
    monkeypatch.setattr(chatgpt_auth, "login", login)
    monkeypatch.setattr(cli_session.webbrowser, "open", lambda _url: True)
    result = CliRunner().invoke(main, ["login", "--switch"])
    assert result.exit_code == 0, result.output
    assert login.calls == [True]


@pytest.mark.parametrize(
    "stored",
    [None, {"expires_at": 1.0, "refresh": ""}, {"scopes": ("openid",)}, {"refresh_refused": True}],
)
def test_login_signs_in_when_the_session_needs_it(
    monkeypatch: pytest.MonkeyPatch, stored: dict[str, object] | None
) -> None:
    if stored is not None:
        store_account(**stored)  # type: ignore[arg-type]
    login = FakeLogin()
    opened: list[str] = []
    monkeypatch.setattr(chatgpt_auth, "login", login)
    monkeypatch.setattr(cli_session.webbrowser, "open", lambda url: opened.append(url) or True)
    result = CliRunner().invoke(main, ["login"])
    assert result.exit_code == 0, result.output
    assert login.calls == [False]
    assert "https://auth.openai.com/api/accounts/authorize" in result.output and len(opened) == 1
    assert f"chatgpt: signed in as {EMAIL}" in result.output


def test_login_no_browser_only_prints_the_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "login", FakeLogin())
    monkeypatch.setattr(cli_session.webbrowser, "open", lambda _url: pytest.fail("opened a browser"))
    result = CliRunner().invoke(main, ["login", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert "https://auth.openai.com/api/accounts/authorize" in result.output


def test_login_without_plan_permission_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "login", FakeLogin(plan=False))
    result = CliRunner().invoke(main, ["login", "--no-browser"])
    assert result.exit_code == 1
    assert "without ChatGPT plan permission" in result.output


def test_login_failures_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**_kwargs: object) -> object:
        raise chatgpt_auth.ChatGPTAuthError("sign-in timed out after 300s; the stored sign-in was kept", code="timeout")

    monkeypatch.setattr(chatgpt_auth, "login", fail)
    result = CliRunner().invoke(main, ["login", "--no-browser"])
    assert result.exit_code == 1 and "sign-in timed out" in result.output


@pytest.mark.parametrize(
    ("outcome", "said"),
    [
        ("confirmed", "chatgpt: signed out; OpenAI confirmed the revocation"),
        ("unconfirmed", "chatgpt: signed out; OpenAI did not confirm the revocation"),
        ("no_refresh_token", "chatgpt: signed out; there was no refresh token to revoke"),
        ("not_signed_in", "chatgpt: not signed in; nothing to revoke"),
    ],
)
def test_logout_reports_what_happened_to_the_revocation(
    monkeypatch: pytest.MonkeyPatch, outcome: str, said: str
) -> None:
    monkeypatch.setattr(chatgpt_auth, "logout", lambda: outcome)
    result = CliRunner().invoke(main, ["logout"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith(said)


@pytest.mark.parametrize(
    ("options", "label"),
    [
        (LLMOptions(), "chatgpt:gpt-6.1-sol (medium effort)"),
        (LLMOptions(effort="high"), "chatgpt:gpt-6.1-sol (high effort)"),
        (LLMOptions(provider="claude-cli", model="claude-sonnet-5-5"), "claude-cli:claude-sonnet-5-5"),
        (LLMOptions(provider="none"), "none (the LLM stages are skipped)"),
    ],
)
def test_the_llm_label(options: LLMOptions, label: str) -> None:
    assert cli_session.llm_label(options) == label
