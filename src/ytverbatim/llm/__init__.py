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
_EFFORTS: dict[str, tuple[str, ...]] = {"chatgpt": chatgpt.EFFORTS, "claude-cli": claude_cli.EFFORTS}


def client_from_options(options: LLMOptions) -> ChatGPTClient | ClaudeCLIClient | OpenAICompatClient:
    """Build the configured transport; options the provider would ignore are rejected rather than dropped.

    An unset chatgpt model or effort means the provider's defaults, gpt-6.1-sol at medium effort.
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
        )
    return ClaudeCLIClient(model=options.model or "", effort=options.effort, timeout_s=options.timeout_s)


def check_effort(options: LLMOptions) -> None:
    """Raise `ConfigError` unless the configured effort is one the provider takes; openai-compat takes none."""
    if options.effort is None:
        return
    efforts = _EFFORTS.get(options.provider)
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


def check_ready(options: LLMOptions) -> None:
    """Fail before any work when the configured transport cannot run.

    That is: no ChatGPT sign-in with plan permission, no `claude` CLI, or no base URL or key. Cheap and local: it
    sends nothing and spends nothing; `ytverbatim llm check --probe` makes a real call.
    """
    if options.provider == "chatgpt":
        status = chatgpt_auth.readiness()
        if not status.ready:
            raise ConfigError(
                f"the chatgpt LLM is not ready: {status.detail}, or pass `--llm none` to skip the LLM stages"
            )
    elif options.provider == "claude-cli":
        claude_cli.executable()
    elif options.provider == "openai-compat":
        if not options.base_url:
            raise ConfigError("openai-compat needs [llm] base_url, e.g. http://127.0.0.1:8080/v1")
        openai_compat.check_key(options.api_key_env)
