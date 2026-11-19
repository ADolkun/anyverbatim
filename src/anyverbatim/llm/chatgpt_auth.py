"""App-owned ChatGPT sign-in: OAuth with PKCE on a loopback callback, an owner-only store, and rotating refresh.

One account at a time. The store keeps this machine's agent host id and the client id OpenAI registered for it, so
signing in again reuses the registration. Tokens are never printed or logged, and errors carry safe codes only.
"""

import base64
import binascii
import errno
import fcntl
import hashlib
import hmac
import json
import math
import os
import secrets
import socket
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Literal, cast, override

from anyverbatim import atomic, paths
from anyverbatim.errors import AnyverbatimError
from anyverbatim.llm.chatgpt_http import AUTH_HOST, Opener, error_code, https_open, require_host
from anyverbatim.llm.jsonio import as_object
from anyverbatim.llm.protocol import Readiness

ISSUER = "https://" + AUTH_HOST
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
DYNAMIC_CLIENT = "dynamic_agent_client"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
AGENT_NAME = "anyverbatim"
CALLBACK_PATH = "/auth/callback"
LOGIN_TIMEOUT_S = 300.0
LOGIN_COMMAND = "anyverbatim login"

_HTTP_TIMEOUT_S = 15.0
_REFRESH_MARGIN_S = 60.0
# A token with less left than this is not handed out: the request it would authorize could outlive it.
_USABLE_MARGIN_S = 30.0
_CLOCK_SKEW_S = 30.0
_SCHEMA_VERSION = 1
_FILE_MODE = 0o600
_DIR_MODE = 0o700
_MAX_STORE_BYTES = 1_048_576
_MAX_CALLBACK = 16_384
_TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)
_KNOWN_ERRORS = _TERMINAL_REFRESH_ERRORS | {"invalid_client", "access_denied"}
# Failures only a new browser sign-in can mend; a network error, a refused request or an early refresh never count.
_NEEDS_SIGN_IN = _TERMINAL_REFRESH_ERRORS | {"not_signed_in", "expired", "plan_disabled", "invalid_client"}

Status = Literal["signed_out", "no_plan", "ready", "refreshable", "refused", "expired"]
STATUS_DETAIL: dict[Status, str] = {
    "signed_out": f"not signed in; run `{LOGIN_COMMAND}`",
    "no_plan": f"signed in without ChatGPT plan permission; run `{LOGIN_COMMAND}` and grant it",
    "ready": "signed in with ChatGPT plan permission",
    "refreshable": "signed in with ChatGPT plan permission; the access token renews on the next call",
    "refused": (
        f"OpenAI refused to renew the ChatGPT sign-in, which works only until its token expires; run `{LOGIN_COMMAND}`"
    ),
    "expired": f"the ChatGPT sign-in expired; run `{LOGIN_COMMAND}`",
}


class ChatGPTAuthError(AnyverbatimError):
    """A sign-in, refresh or store failure; the message and `code` never carry tokens, URLs or response bodies."""

    def __init__(self, message: str, *, code: str = "auth_error") -> None:
        super().__init__(message)
        self.code = code

    @property
    def needs_sign_in(self) -> bool:
        return self.code in _NEEDS_SIGN_IN


@dataclass(frozen=True)
class Account:
    subject: str
    email: str = ""
    access_token: str = field(default="", repr=False)
    refresh_token: str = field(default="", repr=False)
    scopes: tuple[str, ...] = ()
    expires_at: float = 0.0
    earliest_refresh_at: float = 0.0
    # Set when OpenAI terminally refused the refresh token. The tokens stay, since only a sign-in or a logout changes
    # the stored session, but the refused token is not sent again and the account needs a new sign-in.
    refresh_refused: bool = False

    def renewable(self, *, registered: bool) -> bool:
        return bool(self.refresh_token) and registered and not self.refresh_refused

    def status(self, now: float, *, registered: bool = True) -> Status:
        """Classify the account; without `registered` (OpenAI dropped the client) the token works but cannot renew."""
        if not self.access_token:
            return "signed_out"
        if PLAN_SCOPE not in self.scopes:
            return "no_plan"
        if not self.renewable(registered=registered):
            if not self.usable(now):
                return "expired"
            # A dead registration is mended by the next sign-in on its own; a refused refresh token is not.
            return "refused" if self.refresh_refused else "ready"
        return "ready" if self.expires_at > now + _REFRESH_MARGIN_S else "refreshable"

    def usable(self, now: float) -> bool:
        return self.expires_at > now + _USABLE_MARGIN_S

    def refresh_due(self, now: float) -> bool:
        """Refresh at the first chance the server allows, or a minute before expiry, whichever comes first.

        Refresh tokens rotate, so renewing as early as allowed keeps the chain as fresh as each use permits.
        """
        if self.refresh_refused:
            return False
        opened = self.earliest_refresh_at > 0 and now >= self.earliest_refresh_at
        return opened or self.expires_at <= now + _REFRESH_MARGIN_S

    def signed_out(self) -> "Account":
        return Account(self.subject, self.email)


@dataclass(frozen=True)
class State:
    host_id: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    client_id: str | None = None
    account: Account | None = None


@dataclass(frozen=True)
class Session:
    email: str
    status: Status

    @property
    def usable(self) -> bool:
        return self.status in ("ready", "refreshable")


def session(*, clock: Callable[[], float] = time.time) -> Session:
    """Read the stored account's email and status without a lock, a refresh or any request."""
    state = _read(_require_dir(create=False))
    if state.account is None:
        return Session("", "signed_out")
    return Session(state.account.email, state.account.status(clock(), registered=state.client_id is not None))


def status(*, clock: Callable[[], float] = time.time) -> Status:
    """Read the stored account's status without a lock, a refresh or any request."""
    return session(clock=clock).status


def readiness(*, clock: Callable[[], float] = time.time) -> Readiness:
    current = status(clock=clock)
    return Readiness(current in ("ready", "refreshable"), STATUS_DETAIL[current])


@dataclass(frozen=True)
class LoginResult:
    email: str
    plan: bool


def login(
    *,
    open_url: Callable[[str], object] = webbrowser.open,
    opener: Opener = https_open,
    timeout_s: float = LOGIN_TIMEOUT_S,
    clock: Callable[[], float] = time.time,
    switch: bool = False,
) -> LoginResult:
    """Sign in through the browser; the stored account is replaced only once the new one is verified.

    `open_url` gets the authorization URL; returning False means the browser could not be opened. `switch` shows
    OpenAI's account chooser instead of preselecting the stored account.
    """
    with _locked() as state:
        # The host id must be on disk before OpenAI first sees it, so the registration it makes stays ours.
        _save(state)
    previous = state.account
    # Asking for consent again is the only way to add the plan scope to an account that signed in without it.
    consent = previous is not None and bool(previous.access_token) and PLAN_SCOPE not in previous.scopes
    prompt = "login" if switch else "consent" if consent else None
    # The identity page preselects the account signed in before, so a re-sign-in while the browser session lives
    # can finish without a click.
    hint = previous.email if previous is not None and previous.email and not switch else None
    client_id = state.client_id or DYNAMIC_CLIENT
    # Each fallback below runs at most once per sign-in, so a server that keeps refusing ends the attempt.
    reauthorized = reregistered = False
    while True:
        pending, callback = _authorize_in_browser(
            state.host_id, client_id, open_url=open_url, timeout_s=timeout_s, prompt=prompt, login_hint=hint
        )
        client_id = _issued_client(pending, callback)
        # Kept before the exchange: a first sign-in that fails after this point must not register the host again.
        _remember_client(client_id)
        try:
            payload = _exchange(pending, callback, client_id, opener)
        except ChatGPTAuthError as exc:
            if exc.code == "invalid_client" and not reregistered:
                # OpenAI dropped the registration, or a forged callback named one. A dead client id at authorize
                # shows an error page that never redirects, so forget it and register the host afresh, once.
                with _locked() as current:
                    _save(_unregistered(current))
                client_id = DYNAMIC_CLIENT
                reregistered = True
                continue
            # OpenAI can refuse the code once and expect a fresh authorization under the client it issued; that is
            # the protocol's step, not a retry, and a second refusal is final.
            if exc.code != "invalid_grant" or reauthorized:
                raise
            reauthorized = True
            continue
        break
    account = _credentials(payload, client_id, clock(), nonce=pending.nonce)
    with _locked() as current:
        _save(replace(current, client_id=client_id, account=account))
    return LoginResult(account.email, PLAN_SCOPE in account.scopes)


def interactive_login(reason: str, *, opener: Opener = https_open) -> LoginResult:
    """Sign in again through the browser for a caller at a terminal: say why on stderr, print the URL, open it.

    Only callers that know a person is watching may use this; a library never opens a browser on its own.
    """
    print(
        f"{reason}; opening the browser to sign in again (waiting up to {LOGIN_TIMEOUT_S / 60:g} minutes)",
        file=sys.stderr,
    )

    def open_url(url: str) -> None:
        # The URL is printed, so a browser that will not open still leaves a way through: keep waiting.
        print(f"If no browser opens, visit:\n  {url}", file=sys.stderr)
        webbrowser.open(url)

    return login(open_url=open_url, opener=opener)


def access_token(*, opener: Opener = https_open, clock: Callable[[], float] = time.time) -> str:
    """Return a current access token with plan permission, refreshed first when `Account.refresh_due`.

    The refresh runs under the store's flock, which every caller takes on its own open file, so concurrent processes
    and threads alike never present the same rotating refresh token twice.
    """
    with _locked() as state:
        account = state.account
        if account is None or not account.access_token:
            raise ChatGPTAuthError(f"not signed in to ChatGPT; run `{LOGIN_COMMAND}`", code="not_signed_in")
        if PLAN_SCOPE not in account.scopes:
            raise ChatGPTAuthError(STATUS_DETAIL["no_plan"], code="plan_disabled")
        now = clock()
        client_id = state.client_id
        if client_id is None or not account.renewable(registered=True):
            if account.usable(now):
                return account.access_token
            raise ChatGPTAuthError(STATUS_DETAIL["expired"], code="expired")
        if not account.refresh_due(now):
            return account.access_token
        if now < account.earliest_refresh_at:
            if account.usable(now):
                return account.access_token
            raise ChatGPTAuthError("the ChatGPT token cannot be refreshed yet; retry later", code="refresh_not_yet")
        try:
            payload = _post_form(
                TOKEN_URL,
                {
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": account.refresh_token,
                    "resource": RESOURCE,
                },
                opener,
            )
            renewed = _credentials(payload, client_id, clock(), previous=account)
        except ChatGPTAuthError as exc:
            # The stored sign-in is never cleared here: only a new sign-in or an explicit logout changes it. A dead
            # registration is forgotten so the next sign-in registers afresh, and the account and tokens stay.
            if exc.code == "invalid_client":
                _save(_unregistered(state))
            elif exc.code in _TERMINAL_REFRESH_ERRORS:
                _save(replace(state, account=replace(account, refresh_refused=True)))
            # Refresh starts minutes before expiry, so a failed one leaves a token that still works for now.
            if account.usable(clock()):
                return account.access_token
            raise _refresh_failure(exc) from None
        _save(replace(state, account=renewed))
    if PLAN_SCOPE not in renewed.scopes:
        raise ChatGPTAuthError(STATUS_DETAIL["no_plan"], code="plan_disabled")
    return renewed.access_token


Revocation = Literal["confirmed", "unconfirmed", "no_refresh_token", "unregistered", "not_signed_in"]


def logout(*, opener: Opener = https_open) -> Revocation:
    """Revoke the refresh token (best effort), then clear the tokens, and say how the revocation went.

    The registration and host id stay, so the next sign-in reuses them.
    """
    with _locked() as state:
        account = state.account
        if account is None or not account.access_token:
            return "not_signed_in"
        outcome: Revocation = "no_refresh_token" if not account.refresh_token else "unregistered"
        try:
            if account.refresh_token and state.client_id is not None:
                try:
                    _post_form(
                        _endpoint(_discovery(opener), "revocation_endpoint"),
                        {
                            "token": account.refresh_token,
                            "token_type_hint": "refresh_token",
                            "client_id": state.client_id,
                        },
                        opener,
                    )
                    outcome = "confirmed"
                except ChatGPTAuthError:
                    outcome = "unconfirmed"
        finally:
            _save(replace(state, account=account.signed_out()))
    return outcome


def _require_private(info: os.stat_result, *, directory: bool) -> None:
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ChatGPTAuthError(
            f"the ChatGPT credential store in {paths.chatgpt_auth_dir()} must be owned by you, owner-only, and not a "
            "symlink",
            code="unsafe_store",
        )


def _require_dir(*, create: bool) -> Path | None:
    root = paths.chatgpt_auth_dir(create_parents=create)
    if create:
        with suppress(FileExistsError):
            root.mkdir(mode=_DIR_MODE)
    try:
        info = root.lstat()
    except FileNotFoundError:
        return None
    _require_private(info, directory=True)
    return root


@contextmanager
def _locked() -> Iterator[State]:
    """Yield the stored state under an exclusive lock, for a read-modify-write that `_save` finishes."""
    _require_dir(create=True)
    try:
        fd = os.open(paths.chatgpt_lock_file(), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, _FILE_MODE)
    except OSError as exc:
        raise _unsafe(exc) from None
    try:
        _require_private(os.fstat(fd), directory=False)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield _read(paths.chatgpt_auth_dir())
    finally:
        os.close(fd)


def _unsafe(exc: OSError) -> ChatGPTAuthError:
    if exc.errno == errno.ELOOP:
        return ChatGPTAuthError("the ChatGPT credential store must not be a symlink", code="unsafe_store")
    return ChatGPTAuthError(f"cannot open the ChatGPT credential store ({exc.strerror})", code="unreadable_store")


def _read(root: Path | None) -> State:
    if root is None:
        return State()
    try:
        fd = os.open(paths.chatgpt_credentials_file(), os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return State()
    except OSError as exc:
        raise _unsafe(exc) from None
    with os.fdopen(fd, "rb") as handle:
        _require_private(os.fstat(handle.fileno()), directory=False)
        raw = handle.read(_MAX_STORE_BYTES + 1)
    try:
        if len(raw) > _MAX_STORE_BYTES:
            raise ValueError("oversized")
        return _decode(json.loads(raw))
    except (ValueError, TypeError):
        raise ChatGPTAuthError(
            f"the ChatGPT credential store is invalid; remove {paths.chatgpt_credentials_file()} and run "
            f"`{LOGIN_COMMAND}`",
            code="invalid_store",
        ) from None


def _save(state: State) -> None:
    account = state.account
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "host_id": state.host_id,
        "client_id": state.client_id,
        "account": None
        if account is None
        else {
            "subject": account.subject,
            "email": account.email,
            "access_token": account.access_token,
            "refresh_token": account.refresh_token,
            "scopes": list(account.scopes),
            "expires_at": account.expires_at,
            "earliest_refresh_at": account.earliest_refresh_at,
            "refresh_refused": account.refresh_refused,
        },
    }
    root = paths.chatgpt_auth_dir()
    atomic.write_bytes(paths.chatgpt_credentials_file(), json.dumps(payload).encode(), mode=_FILE_MODE)
    # The replace is durable only once the directory entry is.
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _decode(value: object) -> State:
    raw = as_object(value)
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("schema_version")
    host_id = _string(raw, "host_id")
    if not host_id.startswith("urn:uuid:"):
        raise ValueError("host_id")
    uuid.UUID(host_id.removeprefix("urn:uuid:"))
    client_id = raw.get("client_id")
    if client_id is not None and (not isinstance(client_id, str) or client_id in ("", DYNAMIC_CLIENT)):
        raise ValueError("client_id")
    stored = raw.get("account")
    if stored is None:
        return State(host_id, client_id)
    fields = as_object(stored)
    scopes = fields.get("scopes")
    if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
        raise ValueError("scopes")
    account = Account(
        subject=_string(fields, "subject"),
        email=_string(fields, "email", allow_empty=True),
        access_token=_string(fields, "access_token", allow_empty=True),
        refresh_token=_string(fields, "refresh_token", allow_empty=True),
        scopes=tuple(cast("list[str]", scopes)),
        expires_at=_number(fields, "expires_at"),
        earliest_refresh_at=_number(fields, "earliest_refresh_at"),
        refresh_refused=_flag(fields, "refresh_refused"),
    )
    return State(host_id, client_id, account)


def _string(value: dict[str, object], key: str, *, allow_empty: bool = False) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        raise ValueError(key)
    return result


def _flag(value: dict[str, object], key: str) -> bool:
    """Read an optional boolean; a store written before the field existed lacks it, which reads as false."""
    flag = value.get(key, False)
    if not isinstance(flag, bool):
        raise ValueError(key)
    return flag


def _number(value: dict[str, object], key: str) -> float:
    number = value.get(key)
    if isinstance(number, bool) or not isinstance(number, int | float) or not math.isfinite(number) or number < 0:
        raise ValueError(key)
    return float(number)


def _request_json(request: urllib.request.Request, opener: Opener) -> dict[str, object]:
    try:
        response = opener(request, _HTTP_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        found = error_code(exc)
        code = found if found in _KNOWN_ERRORS else "request_failed"
        raise ChatGPTAuthError(f"OpenAI refused the sign-in request (HTTP {exc.code}, {code})", code=code) from None
    except (urllib.error.URLError, OSError):
        raise ChatGPTAuthError("could not reach OpenAI; the stored sign-in was kept", code="network_error") from None
    try:
        body = response.read()
    except OSError:
        raise ChatGPTAuthError("the OpenAI sign-in response broke off", code="network_error") from None
    finally:
        response.close()
    if not body:
        return {}
    try:
        decoded: object = json.loads(body)
    except ValueError:
        decoded = None
    if not isinstance(decoded, dict):
        raise ChatGPTAuthError("OpenAI returned an unreadable sign-in response", code="invalid_response")
    return as_object(decoded)


def _post_form(url: str, data: dict[str, str], opener: Opener) -> dict[str, object]:
    require_host(url)
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    return _request_json(request, opener)


def _discovery(opener: Opener) -> dict[str, object]:
    metadata = _request_json(
        urllib.request.Request(DISCOVERY_URL, headers={"Accept": "application/json"}, method="GET"), opener
    )
    if metadata.get("issuer") != ISSUER:
        raise ChatGPTAuthError("OpenAI identity discovery named an unexpected issuer", code="invalid_discovery")
    return metadata


def _endpoint(metadata: dict[str, object], key: str) -> str:
    url = metadata.get(key)
    if not isinstance(url, str):
        raise ChatGPTAuthError(f"OpenAI identity discovery has no {key}", code="invalid_discovery")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.netloc != AUTH_HOST or parts.query or parts.fragment:
        raise ChatGPTAuthError(f"OpenAI identity discovery named an unsafe {key}", code="invalid_discovery")
    return url


def id_token_claims(token: str, client_id: str, now: float, *, nonce: str | None = None) -> dict[str, object]:
    """Return the ID token's claims once `iss`, `aud`, `azp`, `exp`, `sub` and, at sign-in, `nonce` check out.

    The signature is not verified. OIDC Core 1.0 §3.1.3.7 step 6 allows TLS server validation in its place when the
    ID token comes straight from the token endpoint, as it does here: over HTTPS to the fixed issuer host, never
    through a redirect.
    """
    try:
        _header, body, _signature = token.split(".")
        claims = as_object(json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))))
        if claims.get("iss") != ISSUER:
            raise ValueError("iss")
        audience = claims.get("aud")
        audiences = [audience] if isinstance(audience, str) else audience
        if not isinstance(audiences, list) or client_id not in audiences:
            raise ValueError("aud")
        if (len(audiences) > 1 and claims.get("azp") != client_id) or claims.get("azp", client_id) != client_id:
            raise ValueError("azp")
        expires = claims.get("exp")
        if isinstance(expires, bool) or not isinstance(expires, int | float) or expires + _CLOCK_SKEW_S <= now:
            raise ValueError("exp")
        _string(claims, "sub")
        if nonce is not None and not hmac.compare_digest(_string(claims, "nonce"), nonce):
            raise ValueError("nonce")
    except (ValueError, binascii.Error) as exc:
        claim = exc.args[0] if exc.args and exc.args[0] in ("iss", "aud", "azp", "exp", "sub", "nonce") else "format"
        raise ChatGPTAuthError(
            f"OpenAI returned an ID token that failed its {claim} check; the stored sign-in was kept",
            code="invalid_identity",
        ) from None
    return claims


def _credentials(
    payload: dict[str, object],
    client_id: str,
    now: float,
    *,
    nonce: str | None = None,
    previous: Account | None = None,
) -> Account:
    """Validate a token response: a sign-in (`nonce`) or a refresh of `previous`, which must keep its identity."""
    try:
        access = _string(payload, "access_token")
        if _string(payload, "token_type").lower() != "bearer":
            raise ValueError("token_type")
        expires_in = _number(payload, "expires_in")
        if expires_in <= 0:
            raise ValueError("expires_in")
        refresh = _string(payload, "refresh_token") if "refresh_token" in payload else ""
        returned = payload.get("id_token")
        if returned is None and previous is not None:
            identity: dict[str, object] = {"sub": previous.subject, "email": previous.email}
        elif isinstance(returned, str) and returned:
            identity = id_token_claims(returned, client_id, now, nonce=nonce)
        else:
            raise ValueError("id_token")
        subject = _string(identity, "sub")
        if previous is not None and subject != previous.subject:
            raise ValueError("subject")
        granted = payload.get("scope")
        if granted is None and previous is not None:
            scopes = previous.scopes
        elif isinstance(granted, str):
            scopes = tuple(sorted(set(granted.split())))
        else:
            raise ValueError("scope")
        # Refresh tokens rotate: a refresh that returns none leaves nothing valid to refresh with next time.
        if not refresh and (previous is not None or "offline_access" in scopes):
            raise ValueError("refresh_token")
        email = identity.get("email", "")
        if not isinstance(email, str) or not email.isprintable():
            raise ValueError("email")
        earliest = _number(payload, "earliest_refresh_at") if "earliest_refresh_at" in payload else 0.0
        # A window that does not open between issue and expiry is meaningless; refresh then follows expiry alone.
        if not now < earliest < now + expires_in:
            earliest = 0.0
    except (ValueError, TypeError):
        raise ChatGPTAuthError(
            "OpenAI returned unusable credentials; the stored sign-in was kept", code="invalid_credentials"
        ) from None
    return Account(
        subject=subject,
        email=email,
        access_token=access,
        refresh_token=refresh,
        scopes=scopes,
        expires_at=now + expires_in,
        earliest_refresh_at=earliest,
    )


@dataclass(repr=False)
class _Pending:
    client_id: str
    redirect_uri: str
    state: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    verifier: str = field(default_factory=lambda: secrets.token_urlsafe(64))


def _authorization_url(pending: _Pending, host_id: str, *, prompt: str | None, login_hint: str | None) -> str:
    challenge = base64.urlsafe_b64encode(hashlib.sha256(pending.verifier.encode()).digest()).rstrip(b"=").decode()
    query = {
        "client_id": pending.client_id,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": pending.redirect_uri,
        "resource": RESOURCE,
        "scope": SCOPES,
        "state": pending.state,
        "nonce": pending.nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    if pending.client_id == DYNAMIC_CLIENT:
        query["agent_name_hint"] = AGENT_NAME
    if prompt is not None:
        query["prompt"] = prompt
    if login_hint is not None:
        query["login_hint"] = login_hint
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(query)


def _callback_params(target: str, pending: _Pending) -> dict[str, str] | None:
    parts = urllib.parse.urlsplit(target)
    if len(target) > _MAX_CALLBACK or parts.path != CALLBACK_PATH:
        return None
    query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    if any(len(values) != 1 for values in query.values()):
        return None
    if not hmac.compare_digest(query.get("state", [""])[0], pending.state):
        return None
    return {key: values[0] for key, values in query.items()}


class _LoopbackServer(HTTPServer):
    deadline: float = 0.0
    _closer: threading.Timer | None = None

    @override
    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        left = self.deadline - time.monotonic()
        # A browser preconnect may send nothing; bounding each read keeps it from holding the loop. A peer that
        # trickles bytes beats a per-read bound, so the connection is also closed outright at the login deadline.
        connection.settimeout(max(0.01, min(1.0, left)))
        self._closer = threading.Timer(max(0.0, left), _close_connection, args=(connection,))
        self._closer.daemon = True
        self._closer.start()
        return connection, address

    @override
    def shutdown_request(self, request: object) -> None:
        if self._closer is not None:
            self._closer.cancel()
            self._closer = None
        super().shutdown_request(cast("socket.socket", request))

    @override
    def handle_error(self, request: object, client_address: object) -> None:
        pass


def _close_connection(connection: socket.socket) -> None:
    with suppress(OSError):
        connection.shutdown(socket.SHUT_RDWR)
    connection.close()


def _authorize_in_browser(
    host_id: str,
    client_id: str,
    *,
    open_url: Callable[[str], object],
    timeout_s: float,
    prompt: str | None,
    login_hint: str | None,
) -> tuple[_Pending, dict[str, str]]:
    received: list[dict[str, str]] = []
    pending: _Pending | None = None

    class Callback(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert pending is not None
            params = _callback_params(self.path, pending)
            host_matches = self.headers.get("Host") == urllib.parse.urlsplit(pending.redirect_uri).netloc
            valid = params is not None and host_matches and not received
            self.send_response(200 if valid else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"Return to anyverbatim to finish signing in." if valid else b"Invalid sign-in callback.")
            if valid and params is not None:
                received.append(params)

        @override
        def log_message(self, format: str, *args: object) -> None:
            # Request lines carry the authorization code.
            pass

    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ChatGPTAuthError("the sign-in timeout must be finite and positive", code="invalid_timeout")
    with _LoopbackServer(("127.0.0.1", 0), Callback) as server:
        server.timeout = 0.25
        server.deadline = time.monotonic() + timeout_s
        pending = _Pending(client_id, f"http://127.0.0.1:{server.server_port}{CALLBACK_PATH}")
        if open_url(_authorization_url(pending, host_id, prompt=prompt, login_hint=login_hint)) is False:
            raise ChatGPTAuthError("could not open a browser; retry with --no-browser", code="no_browser")
        while not received and time.monotonic() < server.deadline:
            server.handle_request()
    if not received:
        raise ChatGPTAuthError(f"sign-in timed out after {timeout_s:g}s; the stored sign-in was kept", code="timeout")
    return pending, received[0]


def _issued_client(pending: _Pending, callback: dict[str, str]) -> str:
    """Return the client id a callback with the right state names; the first sign-in registers this host."""
    if not hmac.compare_digest(callback.get("state", ""), pending.state):
        raise ChatGPTAuthError("the sign-in callback had the wrong state", code="invalid_state")
    if callback.get("error"):
        raise ChatGPTAuthError("sign-in was not authorized; the stored sign-in was kept", code="access_denied")
    client_id = callback.get("client_id", pending.client_id)
    if client_id in ("", DYNAMIC_CLIENT) or (pending.client_id != DYNAMIC_CLIENT and client_id != pending.client_id):
        raise ChatGPTAuthError("the sign-in callback named an unexpected registration", code="invalid_client")
    return client_id


def _unregistered(state: State) -> State:
    return replace(state, client_id=None)


def _refresh_failure(exc: ChatGPTAuthError) -> ChatGPTAuthError:
    if exc.code == "invalid_client":
        return ChatGPTAuthError(
            f"OpenAI no longer accepts this sign-in's registration; run `{LOGIN_COMMAND}`", code=exc.code
        )
    if exc.code in _TERMINAL_REFRESH_ERRORS:
        return ChatGPTAuthError(
            f"the ChatGPT sign-in was revoked or expired ({exc.code}); run `{LOGIN_COMMAND}`", code=exc.code
        )
    return exc


def _remember_client(client_id: str) -> None:
    with _locked() as state:
        if state.client_id != client_id:
            _save(replace(state, client_id=client_id))


def _exchange(pending: _Pending, callback: dict[str, str], client_id: str, opener: Opener) -> dict[str, object]:
    code = callback.get("code")
    if not code:
        raise ChatGPTAuthError("the sign-in callback had no authorization code", code="no_code")
    return _post_form(
        TOKEN_URL,
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": pending.verifier,
            "redirect_uri": pending.redirect_uri,
            "resource": RESOURCE,
        },
        opener,
    )
