import webbrowser
from typing import Literal

import click
from pydantic import BaseModel

from ytverbatim.config import LLMOptions, load_options
from ytverbatim.errors import YtverbatimError
from ytverbatim.llm import (
    EFFORTS,
    LLMRequest,
    Readiness,
    UsageMeter,
    chatgpt_auth,
    check_effort,
    claude_cli,
    client_from_options,
    complete_json,
    openai_compat,
)

_EFFORTS = list(dict.fromkeys(effort for efforts in EFFORTS.values() for effort in efforts))


_LOGOUT_MESSAGES: dict[chatgpt_auth.Revocation, str] = {
    "confirmed": "signed out; OpenAI confirmed the revocation",
    "unconfirmed": "signed out; OpenAI did not confirm the revocation, so the sign-in may stay valid until it expires",
    "no_refresh_token": "signed out; there was no refresh token to revoke",
    "not_signed_in": "not signed in; nothing to revoke",
}


class _Probe(BaseModel):
    ok: Literal[True]


@click.group()
def llm() -> None:
    """Sign in to ChatGPT and check the configured LLM transport."""


@llm.command()
@click.option("--no-browser", is_flag=True, help="Only print the sign-in URL; open it in a browser yourself.")
def login(*, no_browser: bool) -> None:
    """Sign in with your ChatGPT plan for the default chatgpt LLM.

    Opens OpenAI's sign-in page and waits up to five minutes for it to return to this machine. Signing in again
    replaces the stored account.
    """

    def open_url(url: str) -> object:
        click.echo(f"Sign in to ChatGPT at:\n  {url}", err=True)
        return None if no_browser else webbrowser.open(url)

    try:
        result = chatgpt_auth.login(open_url=open_url)
    except YtverbatimError as exc:
        raise click.ClickException(str(exc)) from exc
    who = f" as {result.email}" if result.email else ""
    if result.plan:
        click.echo(f"chatgpt: signed in{who}")
        return
    click.echo(
        f"chatgpt: signed in{who}, but without ChatGPT plan permission; run `ytverbatim llm login` again and grant it",
        err=True,
    )
    raise SystemExit(1)


@llm.command()
def logout() -> None:
    """Revoke the ChatGPT sign-in and remove its tokens from this machine."""
    try:
        outcome = chatgpt_auth.logout()
    except YtverbatimError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"chatgpt: {_LOGOUT_MESSAGES[outcome]}")


@llm.command()
@click.option(
    "--provider", type=click.Choice(["chatgpt", "claude-cli", "openai-compat"]), help="Override [llm] provider."
)
@click.option("--model", help="Override [llm] model.")
@click.option("--effort", type=click.Choice(_EFFORTS), help="Override [llm] effort (chatgpt and claude-cli).")
@click.option("--probe", is_flag=True, help='Also send one tiny generation asking for {"ok": true}; this spends quota.')
def check(provider: str | None, model: str | None, effort: str | None, *, probe: bool) -> None:
    """Report whether the LLM transport is ready. Contacts no provider unless --probe is given."""
    try:
        options = load_options(overrides={"llm": {"provider": provider, "model": model, "effort": effort}}).llm
        if options.provider == "none":
            raise click.ClickException("no LLM provider configured; set [llm] provider or pass --provider")
        check_effort(options)
        status = _readiness(options)
        click.echo(f"{options.provider}: {'ready' if status.ready else 'not ready'} - {status.detail}")
        if not status.ready:
            raise SystemExit(1)
        if probe:
            _probe(options)
    except YtverbatimError as exc:
        raise click.ClickException(str(exc)) from exc


def _readiness(options: LLMOptions) -> Readiness:
    if options.provider == "chatgpt":
        return chatgpt_auth.readiness()
    if options.provider == "claude-cli":
        return claude_cli.readiness()
    return openai_compat.readiness(options.base_url, options.api_key_env)


def _probe(options: LLMOptions) -> None:
    meter = UsageMeter()
    client = meter.metered("probe", client_from_options(options))
    request = LLMRequest(system="You are a connectivity probe.", user='Return {"ok": true}.', max_output_tokens=256)
    result = complete_json(client, request, _Probe)
    (usage,) = meter.summary()
    click.echo(
        f"probe {client.id}: {result.model_dump_json()} "
        f"({usage.calls} call(s), input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens})"
    )
