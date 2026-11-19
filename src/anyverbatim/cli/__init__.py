import sys
from collections.abc import Mapping
from importlib import import_module
from typing import Any, override

import click

from anyverbatim import __version__, paths
from anyverbatim.errors import ConfigError
from anyverbatim.ytdlp.runner import redact

# Subcommand name -> "module:attribute". Modules are imported on first use so a heavy
# subcommand never slows down (or breaks) the others.
LAZY_COMMANDS: Mapping[str, str] = {
    "auth": "anyverbatim.cli.auth:auth",
    "fetch": "anyverbatim.cli.fetch:fetch_command",
    "llm": "anyverbatim.cli.llm:llm",
    "login": "anyverbatim.cli.session:login_command",
    "logout": "anyverbatim.cli.session:logout_command",
    "memos": "anyverbatim.cli.memos:memos_command",
    "render": "anyverbatim.cli.render:render_command",
    "run": "anyverbatim.cli.run:run_command",
    "speakers": "anyverbatim.cli.speakers:speakers_command",
    "status": "anyverbatim.cli.session:status_command",
}


def interactive() -> bool:
    """Whether a person is at the terminal to finish a browser sign-in; cron, pipes and CI are not.

    Only the CLI decides this. Library calls default to non-interactive and never open a browser on their own.
    """
    return sys.stdin.isatty() and sys.stderr.isatty()


class LazyGroup(click.Group):
    def __init__(self, *, lazy_commands: Mapping[str, str], **attrs: Any) -> None:  # noqa: ANN401 - click.Group takes **Any
        super().__init__(**attrs)
        self._lazy_commands = lazy_commands

    @override
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self._lazy_commands})

    @override
    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = self._lazy_commands.get(cmd_name)
        if target is None:
            return super().get_command(ctx, cmd_name)
        module_name, _, attribute = target.partition(":")
        command = getattr(import_module(module_name), attribute)
        if not isinstance(command, click.Command):
            raise TypeError(f"{target} is not a click command")
        return command


@click.group(cls=LazyGroup, lazy_commands=LAZY_COMMANDS)
@click.version_option(__version__, prog_name="anyverbatim")
def main() -> None:
    """Produce verified, timestamped transcripts for LLM context from YouTube videos and your own recordings."""
    try:
        paths.migrate()
    except ConfigError as exc:
        click.echo(f"error: {redact(str(exc))}", err=True)
        click.echo(f"  Fix: {redact(exc.fix)}", err=True)
        raise SystemExit(1) from None
