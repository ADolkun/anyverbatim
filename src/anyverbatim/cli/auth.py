import time
from pathlib import Path
from typing import NoReturn

import click

from anyverbatim.auth import chromium, cookiefile, snapshot
from anyverbatim.auth.doctor import Status, exit_code, run_doctor
from anyverbatim.auth.fixes import fix_for
from anyverbatim.auth.preflight import fix_context
from anyverbatim.auth.source import BrowserSource, CookieFileSource, format_spec, parse_spec, save_source
from anyverbatim.auth.state import AuthState
from anyverbatim.errors import AnyverbatimError, AuthError
from anyverbatim.ytdlp.runner import SubprocessRunner, redact

EXIT_AUTH = 2
EXIT_OTHER = 1
_MARKS = {Status.OK: "✓", Status.WARN: "!", Status.FAIL: "✗"}


def fail_auth(exc: AuthError, **context: str | None) -> NoReturn:
    click.echo(f"✗ {exc.state.value}: {exc.detail}" if exc.detail else f"✗ {exc.state.value}", err=True)
    click.echo(f"  Fix: {fix_for(exc.state, **context)}", err=True)
    raise SystemExit(EXIT_AUTH)


def fail_other(exc: AnyverbatimError) -> NoReturn:
    click.echo(f"error: {redact(str(exc))}", err=True)
    raise SystemExit(EXIT_OTHER)


@click.group()
def auth() -> None:
    """Set up the browser login used for members-only and Premium videos; `anyverbatim status` checks it."""


IGNORE_COOLDOWN = click.option(
    "--ignore-cooldown", is_flag=True, help="Probe YouTube even though a bot or rate-limit cooldown is active."
)


@auth.command()
@IGNORE_COOLDOWN
def doctor(*, ignore_cooldown: bool) -> None:
    """Check every dependency of authenticated fetching, one line each with its fix."""
    try:
        checks = run_doctor(SubprocessRunner(), ignore_cooldown=ignore_cooldown)
    except AnyverbatimError as exc:
        fail_other(exc)
    for check in checks:
        click.echo(f"{_MARKS[check.status]} {check.name}: {check.message}")
        if check.fix and check.status is not Status.OK:
            click.echo(f"    Fix: {check.fix}")
    raise SystemExit(exit_code(checks))


@auth.command()
@click.option(
    "--browser", type=click.Choice(chromium.SUPPORTED_BROWSERS), default="chrome", show_default=True, help="Browser."
)
@click.option("--show-emails", is_flag=True, help="Show full account emails instead of masked ones.")
def profiles(browser: str, *, show_emails: bool) -> None:
    """List browser profiles with their directory names, for `anyverbatim auth use BROWSER:DIR`."""
    try:
        found = chromium.list_profiles(browser, emails="full" if show_emails else "masked")
    except AuthError as exc:
        fail_auth(exc, browser=browser)
    if not found:
        click.echo(f"no {browser} profiles found")
        raise SystemExit(EXIT_AUTH)
    width = max(len(p.dir) for p in found)
    for profile in found:
        notes = [
            "cookies" if profile.cookie_db is not None else "no cookies",
            *(["last used"] if profile.last_used else []),
            *([profile.email] if profile.email else []),
        ]
        click.echo(f"{profile.dir:<{width}}  {profile.name!r}  ({', '.join(notes)})")
    click.echo(f"Use one with: anyverbatim auth use {browser}:<DIR>")


@auth.command()
@click.argument("spec")
def use(spec: str) -> None:
    """Validate and save the auth source: BROWSER[:DIR], cookie-file:PATH, or none to clear it.

    Validation is offline: the profile must have a cookie database, or the file must be a readable,
    signed-in cookies.txt. Run `anyverbatim status` afterwards to check the login with YouTube.
    """
    try:
        source = parse_spec(spec)
        match source:
            case None:
                save_source(None)
                click.echo("✓ auth source cleared; videos are fetched anonymously")
                return
            case BrowserSource(browser=browser, profile=profile):
                # Saved with the directory resolved, so adding a second profile later cannot change accounts.
                source = BrowserSource(browser, chromium.cookie_db(browser, profile).profile)
            case CookieFileSource(path=path):
                _validate_cookie_file(path)
        save_source(source)
    except AuthError as exc:
        fail_auth(exc, **_spec_context(spec))
    except AnyverbatimError as exc:
        fail_other(exc)
    click.echo(f"✓ auth source set to {redact(format_spec(source))}; check it with: anyverbatim status --refresh")


def _validate_cookie_file(path: Path) -> None:
    if not snapshot.is_logged_in(cookiefile.read_cookie_file(path), time.time()):
        raise AuthError(
            AuthState.COOKIE_FILE_LOGGED_OUT, "the cookie file has no signed-in YouTube session (or it expired)"
        )


def _spec_context(spec: str) -> dict[str, str | None]:
    browser, _, profile = spec.partition(":")
    if browser in chromium.SUPPORTED_BROWSERS:
        return fix_context(BrowserSource(browser, profile or None))
    return {}
