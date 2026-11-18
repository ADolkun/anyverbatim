"""Top-level sign-in commands: `login` and `logout` for ChatGPT, and `status` for every sign-in at once."""

import json
import webbrowser
from dataclasses import dataclass, field

import click

from ytverbatim.acquire import voicememos
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.preflight import AuthReport, preflight
from ytverbatim.auth.source import resolve_source
from ytverbatim.auth.state import AuthState
from ytverbatim.cli.auth import IGNORE_COOLDOWN, fail_other
from ytverbatim.config import LLMOptions, load_options
from ytverbatim.errors import AuthError, ConfigError, YtverbatimError
from ytverbatim.llm import chatgpt, chatgpt_auth, chatgpt_models
from ytverbatim.ytdlp.runner import SubprocessRunner, redact

EXIT_ATTENTION = 1
_CHATGPT_STATE: dict[chatgpt_auth.Status, str] = {
    "ready": "ok, plan permission, session ready",
    "refreshable": "ok, plan permission, refreshes automatically on the next call",
    "refused": "needs re-sign-in: OpenAI refused to renew it, so it works only until its token expires",
    "expired": "needs re-sign-in",
    "no_plan": "signed in without ChatGPT plan permission",
    "signed_out": "not signed in",
}


@click.command(name="login")
@click.option(
    "--switch", is_flag=True, help="Pick another ChatGPT account; the current one stays until the new one is verified."
)
@click.option("--no-browser", is_flag=True, help="Only print the sign-in URL; open it in a browser yourself.")
def login_command(*, switch: bool, no_browser: bool) -> None:
    """Sign in with your ChatGPT plan for the default chatgpt LLM.

    Opens OpenAI's sign-in page and waits up to five minutes for it to return to this machine. A usable sign-in is
    kept as it is, and renewed automatically; --switch signs in with another account instead.
    """
    try:
        current = chatgpt_auth.session()
        if current.usable and not switch:
            who = current.email or "your ChatGPT account"
            click.echo(
                f"Already signed in as {who} ({_CHATGPT_STATE[current.status]}); "
                "`ytverbatim login --switch` to use another account"
            )
            return

        def open_url(url: str) -> object:
            click.echo(f"Sign in to ChatGPT at:\n  {url}", err=True)
            return None if no_browser else webbrowser.open(url)

        result = chatgpt_auth.login(open_url=open_url, switch=switch)
    except YtverbatimError as exc:
        raise click.ClickException(str(exc)) from exc
    who = f" as {result.email}" if result.email else ""
    if result.plan:
        click.echo(f"chatgpt: signed in{who}")
        return
    click.echo(
        f"chatgpt: signed in{who}, but without ChatGPT plan permission; run `ytverbatim login` again and grant it",
        err=True,
    )
    raise SystemExit(EXIT_ATTENTION)


_LOGOUT_MESSAGES: dict[chatgpt_auth.Revocation, str] = {
    "confirmed": "signed out; OpenAI confirmed the revocation",
    "unconfirmed": "signed out; OpenAI did not confirm the revocation, so the sign-in may stay valid until it expires",
    "no_refresh_token": "signed out; there was no refresh token to revoke",
    "unregistered": (
        "signed out; the tokens were removed from this machine, but without a registration the refresh token could "
        "not be revoked"
    ),
    "not_signed_in": "not signed in; nothing to revoke",
}


@click.command(name="logout")
def logout_command() -> None:
    """Sign out of ChatGPT: revoke the sign-in and remove its tokens from this machine."""
    try:
        outcome = chatgpt_auth.logout()
    except YtverbatimError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"chatgpt: {_LOGOUT_MESSAGES[outcome]}")


@dataclass
class _Row:
    label: str
    who: str
    what: str = ""
    notes: list[str] = field(default_factory=list[str])


@click.command(name="status")
@click.option("--json", "as_json", is_flag=True, help="Print everything as one JSON object.")
@click.option("--refresh", is_flag=True, help="Ignore the one-hour cache and probe YouTube again.")
@IGNORE_COOLDOWN
def status_command(*, as_json: bool, refresh: bool, ignore_cooldown: bool) -> None:
    """Show the YouTube login, the ChatGPT sign-in, the LLM in use and Voice Memos access. Makes no LLM call.

    The YouTube check takes a cookie snapshot and makes one :ytfav request, cached for an hour. Nothing printed is a
    cookie or a token. Exit codes: 0 when everything configured is usable (no browser login, which limits YouTube to
    public videos, counts as usable), 1 when a line says what needs attention. The Voice Memos line is informational
    and never changes the exit code.
    """
    try:
        options = load_options().llm
    except YtverbatimError as exc:
        fail_other(exc)
    report, youtube_error = _youtube(refresh=refresh, ignore_cooldown=ignore_cooldown)
    try:
        session: chatgpt_auth.Session | None = chatgpt_auth.session()
        store_error = None
    except chatgpt_auth.ChatGPTAuthError as exc:
        session, store_error = None, exc
    memos = voicememos.probe()
    resolution = _resolve_model(options, session)
    if as_json:
        click.echo(
            json.dumps(
                _json(report, youtube_error, session, store_error, options, resolution, memos), ensure_ascii=False
            )
        )
    else:
        _print_rows(
            [
                _youtube_row(report, youtube_error),
                _chatgpt_row(session, store_error),
                _llm_row(options, resolution),
                _memos_row(memos),
            ]
        )
    youtube_usable = report is not None and (report.ok or _anonymous(report))
    chatgpt_usable = options.provider != "chatgpt" or (session is not None and session.usable)
    if not (youtube_usable and chatgpt_usable):
        raise SystemExit(EXIT_ATTENTION)


def _anonymous(report: AuthReport) -> bool:
    return report.state is AuthState.NO_SOURCE


def _resolve_model(options: LLMOptions, session: chatgpt_auth.Session | None) -> str | ConfigError | None:
    """Resolve a chatgpt model family to its slug: the slug, the error when it cannot be read, or None if not tried.

    Only a usable sign-in is tried; the ChatGPT row and the exit code already say what a missing one needs.
    """
    if options.provider != "chatgpt" or session is None or not session.usable:
        return None
    try:
        return chatgpt_models.resolve(options.model or chatgpt.DEFAULT_MODEL)
    except ConfigError as exc:
        return exc


def llm_label(options: LLMOptions, resolution: str | ConfigError | None = None) -> str:
    """One line for the LLM; `resolution` is what `_resolve_model` returned for a chatgpt model without a version."""
    if options.provider == "none":
        return "none (the LLM stages are skipped)"
    is_chatgpt = options.provider == "chatgpt"
    model = options.model or (chatgpt.DEFAULT_MODEL if is_chatgpt else "?")
    effort = options.effort or (chatgpt.DEFAULT_EFFORT if is_chatgpt else None)
    details: list[str] = []
    if is_chatgpt and chatgpt_models.is_alias(model):
        if isinstance(resolution, str):
            details.append(f"newest {model}")
            model = resolution
        else:
            details.append(f"newest {model}, not resolved")
    if effort:
        details.append(f"{effort} effort")
    return f"{options.provider}:{model}" + (f" ({', '.join(details)})" if details else "")


def _llm_row(options: LLMOptions, resolution: str | ConfigError | None) -> _Row:
    row = _Row("LLM", llm_label(options, resolution))
    if isinstance(resolution, ConfigError):
        row.notes.append(f"! {resolution}")
    return row


def _youtube(*, refresh: bool, ignore_cooldown: bool) -> tuple[AuthReport | None, AuthError | None]:
    try:
        source = resolve_source()
        return preflight(source, SubprocessRunner(), refresh=refresh, ignore_cooldown=ignore_cooldown), None
    except AuthError as exc:
        return None, exc
    except YtverbatimError as exc:
        fail_other(exc)


def _youtube_row(report: AuthReport | None, error: AuthError | None) -> _Row:
    if report is None:
        assert error is not None
        state = f"{error.state.value}: {redact(error.detail)}" if error.detail else error.state.value
        return _Row("YouTube", "?", state, [f"Fix: {fix_for(error.state)}"])
    if _anonymous(report):
        return _Row("YouTube", "none", "public videos only (no browser login)")
    if report.ok:
        verdict = report.fix[:1].lower() + report.fix[1:]
        row = _Row("YouTube", report.source, f"ok, {verdict}{' (cached)' if report.cached else ''}")
    else:
        state = f"{report.state.value}: {report.detail}" if report.detail else report.state.value
        row = _Row("YouTube", report.source, state, [f"Fix: {report.fix}"])
    if report.probe_skipped:
        row.notes.append("(--ignore-cooldown probes anyway, at the account's risk.)")
    context = {"version": report.ytdlp_version, "days": str(report.ytdlp_age_days)}
    row.notes += [f"! {fix_for(AuthState(warning), **context)}" for warning in report.warnings]
    return row


def _chatgpt_row(session: chatgpt_auth.Session | None, error: chatgpt_auth.ChatGPTAuthError | None) -> _Row:
    if session is None:
        return _Row("ChatGPT", "?", str(error))
    row = _Row("ChatGPT", session.email or "-", _CHATGPT_STATE[session.status])
    if not session.usable:
        row.notes.append(f"Fix: run `{chatgpt_auth.LOGIN_COMMAND}`")
    return row


_MEMOS_STATE = {
    "no_access": "no access (Full Disk Access)",
    "not_syncing": "not syncing",
}


def _memos_row(probe: voicememos.Probe) -> _Row:
    match probe.state:
        case "ok":
            ok = f"ok · {probe.memos} memo{'' if probe.memos == 1 else 's'}"
            return _Row("Voice Memos", "-", ok, [f"! {probe.detail}"] if probe.detail else [])
        case "unreadable":
            return _Row("Voice Memos", "-", f"unreadable: {probe.detail}")
        case state:
            return _Row("Voice Memos", "-", _MEMOS_STATE[state], notes=[f"Fix: {probe.fix}"])


def _print_rows(rows: list[_Row]) -> None:
    label_width = max(len(row.label) for row in rows)
    who_width = max(len(row.who) for row in rows if row.what)
    indent = " " * (label_width + 2 + who_width + 2)
    for row in rows:
        line = (
            f"{row.label:<{label_width}}  {row.who:<{who_width}}  {row.what}"
            if row.what
            else (f"{row.label:<{label_width}}  {row.who}")
        )
        click.echo(line.rstrip())
        for note in row.notes:
            click.echo(f"{indent}{note}")


def _json(
    report: AuthReport | None,
    youtube_error: AuthError | None,
    session: chatgpt_auth.Session | None,
    store_error: chatgpt_auth.ChatGPTAuthError | None,
    options: LLMOptions,
    resolution: str | ConfigError | None,
    memos: voicememos.Probe,
) -> dict[str, object]:
    if report is not None:
        youtube: dict[str, object] = {**report.to_json(), "usable": report.ok or _anonymous(report)}
    else:
        assert youtube_error is not None
        state = youtube_error.state
        youtube = {
            "state": state.value,
            "ok": False,
            "usable": False,
            "fix": fix_for(state),
            "detail": redact(youtube_error.detail),
        }
    chatgpt_status: dict[str, object] = (
        {"email": session.email, "status": session.status, "usable": session.usable}
        if session is not None
        else {"error": str(store_error)}
    )
    llm: dict[str, object] = {"provider": options.provider, "label": llm_label(options, resolution)}
    if isinstance(resolution, str):
        llm["model"] = resolution
    elif resolution is not None:
        llm["error"] = str(resolution)
    voice_memos: dict[str, object] = {
        "state": memos.state,
        "memos": memos.memos,
        "detail": memos.detail,
        "fix": memos.fix,
    }
    return {"youtube": youtube, "chatgpt": chatgpt_status, "llm": llm, "voice_memos": voice_memos}
