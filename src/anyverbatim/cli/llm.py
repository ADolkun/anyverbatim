from typing import Literal

import click
from pydantic import BaseModel

from anyverbatim.cli import interactive
from anyverbatim.config import load_options
from anyverbatim.errors import AnyverbatimError
from anyverbatim.llm import (
    EFFORTS,
    LLMRequest,
    UsageMeter,
    check_effort,
    check_ready,
    client_from_options,
    complete_json,
)

_EFFORTS = list(dict.fromkeys(effort for efforts in EFFORTS.values() for effort in efforts))


class _Probe(BaseModel):
    ok: Literal[True]


@click.group()
def llm() -> None:
    """Test the configured LLM; `anyverbatim status` shows it without a call."""


@llm.command()
@click.option(
    "--provider", type=click.Choice(["chatgpt", "claude-cli", "openai-compat"]), help="Override [llm] provider."
)
@click.option("--model", help="Override [llm] model.")
@click.option("--effort", type=click.Choice(_EFFORTS), help="Override [llm] effort (chatgpt and claude-cli).")
def test(provider: str | None, model: str | None, effort: str | None) -> None:
    """Send one tiny generation asking for {"ok": true}; this spends a little quota."""
    try:
        options = load_options(overrides={"llm": {"provider": provider, "model": model, "effort": effort}}).llm
        if options.provider == "none":
            raise click.ClickException("no LLM provider configured; set [llm] provider or pass --provider")
        check_effort(options)
        at_terminal = interactive()
        check_ready(options, interactive=at_terminal)
        meter = UsageMeter()
        client = meter.metered("test", client_from_options(options, interactive=at_terminal))
        request = LLMRequest(system="You are a connectivity probe.", user='Return {"ok": true}.', max_output_tokens=256)
        result = complete_json(client, request, _Probe)
    except AnyverbatimError as exc:
        raise click.ClickException(str(exc)) from exc
    (usage,) = meter.summary()
    click.echo(
        f"{client.id}: {result.model_dump_json()} "
        f"({usage.calls} call(s), input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens})"
    )
