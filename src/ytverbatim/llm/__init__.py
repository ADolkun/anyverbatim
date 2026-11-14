from dataclasses import replace
from typing import Literal, cast

from ytverbatim.config import LLMOptions
from ytverbatim.errors import ConfigError
from ytverbatim.llm import chatgpt, chatgpt_auth, claude_cli, openai_compat
from ytverbatim.llm.chatgpt import ChatGPTClient
from ytverbatim.llm.claude_cli import ClaudeCLIClient
from ytverbatim.llm.jsonio import complete_json, extract_json_object
from ytverbatim.llm.openai_compat import OpenAICompatClient
from ytverbatim.llm.protocol import LLMClient, LLMRequest, LLMResponse, Readiness, Usage
from ytverbatim.llm.usage import UsageMeter

__all__ = [
    "EFFORTS",
    "ChatGPTClient",
    "ClaudeCLIClient",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "OpenAICompatClient",
    "Readiness",
    "Usage",
    "UsageMeter",
    "apply_spec",
    "check_effort",
    "check_ready",
    "client_from_options",
    "complete_json",
    "extract_json_object",
]

Provider = Literal["chatgpt", "claude-cli", "openai-compat"]
_PROVIDERS = ("chatgpt", "claude-cli", "openai-compat")
EFFORTS: dict[str, tuple[str, ...]] = {"chatgpt": chatgpt.EFFORTS, "claude-cli": claude_cli.EFFORTS}


def client_from_options(
    options: LLMOptions, *, interactive: bool = False
) -> ChatGPTClient | ClaudeCLIClient | OpenAICompatClient:
    """Build the configured transport; options the provider would ignore are rejected rather than dropped.

    An unset chatgpt model or effort means the provider's defaults, gpt-6.1-sol at medium effort. With `interactive`,
    a chatgpt sign-in that lapses mid-run reopens the browser to sign in again; pass it only when a person is at the
    terminal.
    """
    if options.provider == "none":
        raise ConfigError("no LLM provider configured; set [llm] provider to chatgpt, claude-cli or openai-compat")
    check_effort(options)
    if options.provider == "openai-compat":
        return OpenAICompatClient(
            base_url=options.base_url or "",
            model=options.model or "",
            api_key_env=options.api_key_env,
            timeout_s=options.timeout_s,
        )
    stray = [name for name in ("base_url", "api_key_env") if getattr(options, name) is not None]
    if stray:
        raise ConfigError(f"[llm] {', '.join(stray)} do not apply to {options.provider}")
    if options.provider == "chatgpt":
        return ChatGPTClient(
            model=options.model or chatgpt.DEFAULT_MODEL,
            effort=options.effort or chatgpt.DEFAULT_EFFORT,
            timeout_s=options.timeout_s,
            reauth=_sign_in_again if interactive else None,
        )
    return ClaudeCLIClient(model=options.model or "", effort=options.effort, timeout_s=options.timeout_s)


def check_effort(options: LLMOptions) -> None:
    """Raise `ConfigError` unless the configured effort is one the provider takes; openai-compat takes none."""
    if options.effort is None:
        return
    efforts = EFFORTS.get(options.provider)
    if efforts is None:
        raise ConfigError(f"[llm] effort does not apply to {options.provider}")
    if options.effort not in efforts:
        raise ConfigError(f"{options.provider} effort must be one of: {', '.join(efforts)}")


def apply_spec(options: LLMOptions, spec: str) -> LLMOptions:
    """Apply a `--llm` spec: `none`, or `PROVIDER:MODEL` with a full model id (`chatgpt:gpt-6.1-sol`).

    The same provider keeps its configured settings (effort, base URL, key). Another provider starts from its
    defaults: a configured `base_url` must not block `claude-cli`, nor a configured `effort` block `openai-compat`.
    """
    if spec == "none":
        return LLMOptions(provider="none", timeout_s=options.timeout_s)
    provider, _, model = spec.partition(":")
    if provider not in _PROVIDERS or not model:
        known = ", ".join(_PROVIDERS)
        raise ConfigError(f"--llm takes none or PROVIDER:MODEL with PROVIDER one of {known}; got {spec!r}")
    if provider == options.provider:
        return replace(options, model=model)
    return LLMOptions(provider=cast("Provider", provider), model=model, timeout_s=options.timeout_s)


def check_ready(options: LLMOptions, *, interactive: bool = False) -> None:
    """Fail before any work when the configured transport cannot run.

    That is: no ChatGPT sign-in with plan permission, no `claude` CLI, or no base URL or key. It spends no LLM quota;
    `ytverbatim llm test` makes a real call. For chatgpt it renews an access token that is due, so a dead
    refresh chain shows now rather than after hours of download and ASR. With `interactive`, an unusable chatgpt
    sign-in opens the browser to sign in again instead of raising; pass it only when a person is at the terminal.
    """
    if options.provider == "chatgpt":
        _ready_chatgpt(interactive=interactive)
    elif options.provider == "claude-cli":
        claude_cli.executable()
    elif options.provider == "openai-compat":
        if not options.base_url:
            raise ConfigError("openai-compat needs [llm] base_url, e.g. http://127.0.0.1:8080/v1")
        openai_compat.check_key(options.api_key_env)


def _ready_chatgpt(*, interactive: bool) -> None:
    signed_in = False
    current = chatgpt_auth.status()
    if current not in ("ready", "refreshable"):
        if not interactive:
            raise _chatgpt_not_ready(chatgpt_auth.STATUS_DETAIL[current])
        chatgpt_auth.interactive_login(_NOT_READY_REASON.get(current, "ChatGPT sign-in lapsed"))
        signed_in = True
    try:
        chatgpt_auth.access_token()
    except chatgpt_auth.ChatGPTAuthError as exc:
        if not exc.needs_sign_in:
            raise
        if not interactive or signed_in:
            raise _chatgpt_not_ready(str(exc)) from None
        _sign_in_again()
        chatgpt_auth.access_token()


_NOT_READY_REASON = {
    "signed_out": "ChatGPT is not signed in",
    "no_plan": "the ChatGPT sign-in lacks plan permission",
}


def _sign_in_again() -> object:
    return chatgpt_auth.interactive_login("ChatGPT sign-in lapsed")


def _chatgpt_not_ready(detail: str) -> ConfigError:
    error = ConfigError(
        f"the chatgpt LLM is not ready: {detail}. To run without an LLM, pass `--llm none` on the command line or "
        '`LLMOptions(provider="none")` from Python'
    )
    error.fix = f'run `{chatgpt_auth.LOGIN_COMMAND}`, or opt out: --llm none, LLMOptions(provider="none")'
    return error
