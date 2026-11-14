import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from chatgpt_fakes import EMAIL, PLAN_SCOPES, FakeOpener, Route, http_error, json_response, store_account, token_payload
from click.testing import CliRunner

import ytverbatim.cli as cli_module
from ytverbatim.cli import llm as llm_cli
from ytverbatim.cli import main
from ytverbatim.cli import run as run_cli
from ytverbatim.config import LLMOptions, load_options
from ytverbatim.errors import ConfigError, DependencyError, LLMUnavailableError
from ytverbatim.llm import (
    ChatGPTClient,
    ClaudeCLIClient,
    OpenAICompatClient,
    apply_spec,
    chatgpt_auth,
    check_ready,
    claude_cli,
    client_from_options,
    openai_compat,
)
from ytverbatim.llm.chatgpt_auth import TOKEN_URL
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Readiness, Usage


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    path = tmp_path / "ytverbatim" / "config.toml"
    path.parent.mkdir()
    return path


@dataclass
class ProbeClient:
    requests: list[LLMRequest] = field(default_factory=list)

    @property
    def id(self) -> str:
        return "fake:probe"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text='{"ok": true}', usage=Usage(30, 4), elapsed_s=0.5)


def test_check_without_provider_fails(config: Path) -> None:
    config.write_text('[llm]\nprovider = "none"\n', encoding="utf-8")
    result = CliRunner().invoke(main, ["llm", "check"])
    assert result.exit_code == 1
    assert "no LLM provider configured" in result.output


def test_check_claude_ready_sends_no_generation(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(True, "logged in through claude.ai (firstParty)"))
    monkeypatch.setattr(llm_cli, "client_from_options", _forbidden)
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "claude-cli: ready - logged in through claude.ai (firstParty)"


def test_check_not_ready_exits_nonzero(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(False, "not logged in"))
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli"])
    assert result.exit_code == 1
    assert "claude-cli: not ready - not logged in" in result.output


def test_check_openai_uses_configured_base_url(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config.write_text('[llm]\nprovider = "openai-compat"\nbase_url = "http://127.0.0.1:9/v1"\n', encoding="utf-8")
    seen: list[tuple[str | None, str | None]] = []

    def fake(base_url: str | None, api_key_env: str | None = None) -> Readiness:
        seen.append((base_url, api_key_env))
        return Readiness(True, "answered")

    monkeypatch.setattr(openai_compat, "readiness", fake)
    result = CliRunner().invoke(main, ["llm", "check"])
    assert result.exit_code == 0, result.output
    assert seen == [("http://127.0.0.1:9/v1", None)]


def test_probe_sends_one_json_call_and_reports_usage(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = ProbeClient()
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(True, "ok"))
    monkeypatch.setattr(llm_cli, "client_from_options", lambda options, **_kwargs: probe)
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli", "--model", "haiku", "--probe"])
    assert result.exit_code == 0, result.output
    assert len(probe.requests) == 1
    assert 'probe fake:probe: {"ok":true} (1 call(s), input_tokens=30, output_tokens=4)' in result.output


def test_probe_errors_are_reported(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(True, "ok"))
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli", "--probe"])
    assert result.exit_code == 1
    assert "needs a model" in result.output


def _forbidden(options: LLMOptions, **_kwargs: object) -> object:
    raise AssertionError("readiness must not build a client")


def test_client_from_options() -> None:
    claude = client_from_options(LLMOptions(provider="claude-cli", model="sonnet", effort="low", timeout_s=60.0))
    assert isinstance(claude, ClaudeCLIClient)
    assert claude.id == "claude-cli:sonnet@low", "effort changes replies, so it is part of the identity"
    assert ClaudeCLIClient(model="sonnet").id == "claude-cli:sonnet"
    compat = client_from_options(LLMOptions(provider="openai-compat", model="m", base_url="http://127.0.0.1:9/v1"))
    assert isinstance(compat, OpenAICompatClient)
    assert compat.id == "openai-compat:m@127.0.0.1:9"


def test_llm_spec_keeps_its_own_providers_settings_and_drops_the_others() -> None:
    compat = LLMOptions(provider="openai-compat", model="m", base_url="http://127.0.0.1:9/v1", timeout_s=60.0)
    switched = apply_spec(compat, "claude-cli:claude-sonnet-5-5")
    assert switched == LLMOptions(provider="claude-cli", model="claude-sonnet-5-5", timeout_s=60.0)
    assert apply_spec(compat, "openai-compat:other") == replace(compat, model="other")
    claude = LLMOptions(provider="claude-cli", model="sonnet", effort="high")
    assert apply_spec(claude, "openai-compat:m").effort is None
    assert apply_spec(claude, "none").provider == "none"
    with pytest.raises(ConfigError, match="PROVIDER:MODEL"):
        apply_spec(claude, "sonnet")


def test_check_ready_is_local_and_fails_on_a_missing_cli_or_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _name: None)
    with pytest.raises(DependencyError, match="claude CLI not found"):
        check_ready(LLMOptions(provider="claude-cli", model="sonnet"))
    with pytest.raises(ConfigError, match="base_url"):
        check_ready(LLMOptions(provider="openai-compat", model="m"))
    monkeypatch.delenv("SYNTHETIC_KEY", raising=False)
    with pytest.raises(ConfigError, match="SYNTHETIC_KEY"):
        check_ready(
            LLMOptions(
                provider="openai-compat", model="m", base_url="https://x.example/v1", api_key_env="SYNTHETIC_KEY"
            )
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (LLMOptions(provider="none"), "no LLM provider"),
        (LLMOptions(provider="claude-cli"), "needs a model"),
        (LLMOptions(provider="claude-cli", model="s", base_url="http://x"), "base_url"),
        (LLMOptions(provider="claude-cli", model="s", api_key_env="K"), "api_key_env"),
        (LLMOptions(provider="openai-compat", model="m", base_url="http://x", effort="low"), "effort"),
        (LLMOptions(provider="openai-compat", model="m"), "base_url"),
    ],
)
def test_client_from_options_rejects(options: LLMOptions, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        client_from_options(options)


def test_the_default_llm_is_chatgpt_with_its_model_and_effort_left_to_the_provider() -> None:
    options = LLMOptions()
    assert (options.provider, options.model, options.effort) == ("chatgpt", None, None)
    default = client_from_options(options)
    assert isinstance(default, ChatGPTClient)
    assert default.id == "chatgpt:gpt-6.1-sol@medium"
    assert client_from_options(LLMOptions(effort="high")).id == "chatgpt:gpt-6.1-sol@high"
    assert client_from_options(LLMOptions(model="gpt-other", effort="low")).id == "chatgpt:gpt-other@low"


def test_switching_provider_never_inherits_chatgpts_model(config: Path) -> None:
    config.write_text('[llm]\nprovider = "claude-cli"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="needs a model"):
        client_from_options(load_options().llm)
    switched = apply_spec(LLMOptions(effort="high", timeout_s=60.0), "claude-cli:claude-sonnet-5-5")
    assert switched == LLMOptions(provider="claude-cli", model="claude-sonnet-5-5", timeout_s=60.0)


def test_llm_spec_selects_chatgpt() -> None:
    claude = LLMOptions(provider="claude-cli", model="sonnet", effort="low", timeout_s=60.0)
    assert apply_spec(claude, "chatgpt:gpt-6.1-sol") == LLMOptions(
        provider="chatgpt", model="gpt-6.1-sol", timeout_s=60.0
    )
    assert apply_spec(LLMOptions(effort="high"), "chatgpt:gpt-6.1-sol") == LLMOptions(
        model="gpt-6.1-sol", effort="high"
    )
    with pytest.raises(ConfigError, match="chatgpt, claude-cli, openai-compat"):
        apply_spec(claude, "gpt:gpt-6.1-sol")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (LLMOptions(base_url="https://api.example.com/v1"), "base_url do not apply to chatgpt"),
        (LLMOptions(api_key_env="K"), "api_key_env do not apply to chatgpt"),
        (LLMOptions(effort="minimal"), "chatgpt effort must be one of: low, medium, high, xhigh, max"),
        (LLMOptions(provider="claude-cli", model="s", effort="minimal"), "claude-cli effort must be one of"),
    ],
)
def test_chatgpt_options_it_would_ignore_are_rejected(options: LLMOptions, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        client_from_options(options)


@pytest.mark.parametrize(
    ("account", "detail"),
    [
        ({}, "not signed in"),
        ({"scopes": ("openid", "offline_access")}, "without ChatGPT plan permission"),
        ({"expires_at": time.time() - 1, "refresh": ""}, "expired"),
    ],
)
def test_check_ready_for_chatgpt_names_login_and_llm_none(
    config: Path, account: dict[str, object], detail: str
) -> None:
    if account:
        store_account(**account)
    with pytest.raises(ConfigError, match=detail) as raised:
        check_ready(LLMOptions())
    message = str(raised.value)
    assert "`ytverbatim llm login`" in message and "`--llm none`" in message
    assert '`LLMOptions(provider="none")`' in message
    assert "ytverbatim llm login" in raised.value.fix and "--llm none" in raised.value.fix


def test_check_ready_accepts_a_ready_account_without_a_request(config: Path) -> None:
    store_account(expires_at=time.time() + 3600)
    # tests/conftest.py makes any real request to OpenAI fail this test.
    check_ready(LLMOptions())
    check_ready(LLMOptions(), interactive=True)


@dataclass
class SignIns:
    """Stands in for the browser sign-in: records each reason and stores what the sign-in would."""

    plan: bool = True
    reasons: list[str] = field(default_factory=list)

    def __call__(self, reason: str) -> chatgpt_auth.LoginResult:
        self.reasons.append(reason)
        store_account(expires_at=time.time() + 3600, scopes=PLAN_SCOPES if self.plan else ("openid",))
        return chatgpt_auth.LoginResult(EMAIL, plan=self.plan)


def token_endpoint(monkeypatch: pytest.MonkeyPatch, route: Route) -> FakeOpener:
    opener = FakeOpener({f"POST {TOKEN_URL}": route})
    real = chatgpt_auth.access_token
    monkeypatch.setattr(chatgpt_auth, "access_token", lambda **_kwargs: real(opener=opener))
    return opener


def test_check_ready_renews_a_due_token_now_without_llm_quota(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_account(expires_at=time.time() - 1)
    opener = token_endpoint(monkeypatch, json_response(token_payload(access="access-2", refresh="refresh-2")))
    check_ready(LLMOptions())
    assert [r.full_url for r in opener.requests] == [TOKEN_URL]
    assert chatgpt_auth.status() == "ready"


@pytest.mark.parametrize("account", ["signed_out", "dead_refresh_chain"])
def test_check_ready_raises_when_not_interactive_and_signs_in_when_interactive(
    config: Path, monkeypatch: pytest.MonkeyPatch, account: str
) -> None:
    if account == "dead_refresh_chain":
        store_account(expires_at=time.time() - 1)
    token_endpoint(monkeypatch, http_error(400, {"error": "invalid_grant"}))
    sign_ins = SignIns()
    monkeypatch.setattr(chatgpt_auth, "interactive_login", sign_ins)
    with pytest.raises(ConfigError, match="ytverbatim llm login"):
        check_ready(LLMOptions())
    assert sign_ins.reasons == []

    if account == "dead_refresh_chain":
        store_account(expires_at=time.time() - 1)
    check_ready(LLMOptions(), interactive=True)
    assert len(sign_ins.reasons) == 1
    assert chatgpt_auth.status() == "ready"


def test_check_ready_signs_in_at_most_once(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sign_ins = SignIns(plan=False)
    monkeypatch.setattr(chatgpt_auth, "interactive_login", sign_ins)
    with pytest.raises(ConfigError, match="plan"):
        check_ready(LLMOptions(), interactive=True)
    assert sign_ins.reasons == ["ChatGPT is not signed in"]


def test_a_transient_refresh_failure_in_pre_flight_never_opens_a_browser(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_account(expires_at=time.time() - 1)
    token_endpoint(monkeypatch, http_error(503))
    monkeypatch.setattr(chatgpt_auth, "interactive_login", lambda _reason: pytest.fail("opened a browser"))
    with pytest.raises(chatgpt_auth.ChatGPTAuthError) as raised:
        check_ready(LLMOptions(), interactive=True)
    assert raised.value.code == "request_failed"


def test_only_an_interactive_chatgpt_client_signs_in_again(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sign_ins = SignIns()
    monkeypatch.setattr(chatgpt_auth, "interactive_login", sign_ins)
    request = LLMRequest(system="s", user="u")
    with pytest.raises(LLMUnavailableError, match="not signed in"):
        client_from_options(LLMOptions()).complete(request)
    assert sign_ins.reasons == []
    # Signed in again, the client goes on to send the request, which tests/conftest.py stops at the network.
    with pytest.raises(AssertionError, match=r"tried to contact https://api\.openai\.com/v1/responses"):
        client_from_options(LLMOptions(), interactive=True).complete(request)
    assert sign_ins.reasons == ["ChatGPT sign-in lapsed"]


def test_the_terminal_decides_interactivity(monkeypatch: pytest.MonkeyPatch) -> None:
    for stdin, stderr, expected in [(True, True, True), (True, False, False), (False, True, False)]:
        monkeypatch.setattr(cli_module.sys, "stdin", Terminal(tty=stdin))
        monkeypatch.setattr(cli_module.sys, "stderr", Terminal(tty=stderr))
        assert cli_module.interactive() is expected


@dataclass
class Terminal:
    tty: bool

    def isatty(self) -> bool:
        return self.tty


@pytest.mark.parametrize("tty", [True, False])
def test_run_and_probe_pass_the_terminals_interactivity(
    config: Path, monkeypatch: pytest.MonkeyPatch, tty: bool
) -> None:
    seen: dict[str, object] = {}

    def process_many(*_args: object, **kwargs: object) -> object:
        seen["run"] = kwargs["interactive"]
        raise ConfigError("stopped by the test")

    monkeypatch.setattr(run_cli, "interactive", lambda: tty)
    monkeypatch.setattr(run_cli.pipeline, "process_many", process_many)
    result = CliRunner().invoke(main, ["run", "--llm", "none", "https://www.youtube.com/watch?v=BaW_jenozKc"])
    assert result.exit_code == 1 and "stopped by the test" in result.output

    store_account()
    probe = ProbeClient()

    def build(options: LLMOptions, *, interactive: bool = False) -> ProbeClient:
        seen["probe"] = interactive
        return probe

    monkeypatch.setattr(llm_cli, "interactive", lambda: tty)
    monkeypatch.setattr(llm_cli, "client_from_options", build)
    assert CliRunner().invoke(main, ["llm", "check", "--probe"]).exit_code == 0
    assert seen == {"run": tty, "probe": tty}


def test_library_calls_default_to_non_interactive() -> None:
    import inspect

    from ytverbatim import pipeline

    for function in (pipeline.process, pipeline.process_many, check_ready, client_from_options):
        assert inspect.signature(function).parameters["interactive"].default is False


def test_check_reports_chatgpt_status_without_contacting_openai(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_cli, "client_from_options", _forbidden)
    runner = CliRunner()
    signed_out = runner.invoke(main, ["llm", "check"])
    assert signed_out.exit_code == 1
    assert signed_out.output.strip() == "chatgpt: not ready - not signed in; run `ytverbatim llm login`"

    store_account(scopes=("openid",))
    no_plan = runner.invoke(main, ["llm", "check"])
    assert no_plan.exit_code == 1 and "without ChatGPT plan permission" in no_plan.output

    store_account(expires_at=time.time() - 1, refresh="")
    assert "the ChatGPT sign-in expired" in runner.invoke(main, ["llm", "check"]).output

    store_account()
    ready = runner.invoke(main, ["llm", "check", "--effort", "xhigh"])
    assert ready.exit_code == 0, ready.output
    assert ready.output.strip() == "chatgpt: ready - signed in with ChatGPT plan permission"
    assert "access-1" not in ready.output


def test_check_probe_goes_through_the_chatgpt_client(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_account()
    probe = ProbeClient()
    built: list[LLMOptions] = []

    def build(options: LLMOptions, **_kwargs: object) -> ProbeClient:
        built.append(options)
        return probe

    monkeypatch.setattr(llm_cli, "client_from_options", build)
    result = CliRunner().invoke(main, ["llm", "check", "--probe", "--effort", "high"])
    assert result.exit_code == 0, result.output
    assert (built[0].provider, built[0].effort) == ("chatgpt", "high")
    assert len(probe.requests) == 1


def test_check_validates_effort_per_provider(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openai_compat, "readiness", lambda *_a, **_k: Readiness(True, "answered"))
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "openai-compat", "--effort", "low"])
    assert result.exit_code == 1
    assert "effort does not apply to openai-compat" in result.output
    unknown = CliRunner().invoke(main, ["llm", "check", "--effort", "minimal"])
    assert unknown.exit_code == 2


@dataclass
class FakeLogin:
    plan: bool = True
    urls: list[str] = field(default_factory=list)

    def __call__(self, *, open_url: Callable[[str], object]) -> chatgpt_auth.LoginResult:
        url = "https://auth.openai.com/api/accounts/authorize?client_id=dynamic_agent_client"
        self.urls.append(url)
        assert open_url(url) is not False
        return chatgpt_auth.LoginResult(EMAIL, plan=self.plan)


def test_login_prints_the_url_and_opens_the_browser(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(chatgpt_auth, "login", FakeLogin())
    monkeypatch.setattr(llm_cli.webbrowser, "open", lambda url: opened.append(url) or True)
    result = CliRunner().invoke(main, ["llm", "login"])
    assert result.exit_code == 0, result.output
    assert "https://auth.openai.com/api/accounts/authorize" in result.output
    assert opened == ["https://auth.openai.com/api/accounts/authorize?client_id=dynamic_agent_client"]
    assert f"chatgpt: signed in as {EMAIL}" in result.output


def test_login_no_browser_only_prints_the_url(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "login", FakeLogin())
    monkeypatch.setattr(llm_cli.webbrowser, "open", lambda _url: pytest.fail("opened a browser"))
    result = CliRunner().invoke(main, ["llm", "login", "--no-browser"])
    assert result.exit_code == 0, result.output
    assert "https://auth.openai.com/api/accounts/authorize" in result.output


def test_login_without_plan_permission_exits_nonzero(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "login", FakeLogin(plan=False))
    result = CliRunner().invoke(main, ["llm", "login", "--no-browser"])
    assert result.exit_code == 1
    assert "without ChatGPT plan permission" in result.output


def test_login_failures_are_reported(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(**_kwargs: object) -> object:
        raise chatgpt_auth.ChatGPTAuthError("sign-in timed out after 300s; the stored sign-in was kept", code="timeout")

    monkeypatch.setattr(chatgpt_auth, "login", fail)
    result = CliRunner().invoke(main, ["llm", "login", "--no-browser"])
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
    config: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, said: str
) -> None:
    monkeypatch.setattr(chatgpt_auth, "logout", lambda: outcome)
    result = CliRunner().invoke(main, ["llm", "logout"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith(said)


def test_check_creates_no_directory(config: Path, tmp_path: Path) -> None:
    assert CliRunner().invoke(main, ["llm", "check"]).exit_code == 1
    with pytest.raises(ConfigError):
        check_ready(LLMOptions())
    assert not (tmp_path / "state").exists()
