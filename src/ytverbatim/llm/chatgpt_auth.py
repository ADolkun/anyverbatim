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

from ytverbatim import atomic, paths
from ytverbatim.errors import YtverbatimError
from ytverbatim.llm.chatgpt_http import AUTH_HOST, Opener, error_code, https_open, require_host
from ytverbatim.llm.jsonio import as_object
from ytverbatim.llm.protocol import Readiness

ISSUER = "https://" + AUTH_HOST
RESOURCE = "https://api.openai.com/v1"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
DYNAMIC_CLIENT = "dynamic_agent_client"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
AGENT_NAME = "ytverbatim"
CALLBACK_PATH = "/auth/callback"
LOGIN_TIMEOUT_S = 300.0
LOGIN_COMMAND = "ytverbatim llm login"

_HTTP_TIMEOUT_S = 15.0
_REFRESH_MARGIN_S = 60.0
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

Status = Literal["signed_out", "no_plan", "ready", "refreshable", "expired"]
STATUS_DETAIL: dict[Status, str] = {
    "signed_out": f"not signed in; run `{LOGIN_COMMAND}`",
    "no_plan": f"signed in without ChatGPT plan permission; run `{LOGIN_COMMAND}` and grant it",
    "ready": "signed in with ChatGPT plan permission",
    "refreshable": "signed in with ChatGPT plan permission; the access token renews on the next call",
    "expired": f"the ChatGPT sign-in expired; run `{LOGIN_COMMAND}`",
}


class ChatGPTAuthError(YtverbatimError):
    """A sign-in, refresh or store failure; the message and `code` never carry tokens, URLs or response bodies."""

    def __init__(self, message: str, *, code: str = "auth_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Account:
    subject: str
    email: str = ""
    access_token: str = field(default="", repr=False)
    refresh_token: str = field(default="", repr=False)
    id_token: str = field(default="", repr=False)
    scopes: tuple[str, ...] = ()
    expires_at: float = 0.0
    earliest_refresh_at: float = 0.0

    def status(self, now: float) -> Status:
        if not self.access_token:
            return "signed_out"
        if PLAN_SCOPE not in self.scopes:
            return "no_plan"
        if self.expires_at > now + _REFRESH_MARGIN_S:
            return "ready"
        return "refreshable" if self.refresh_token else "expired"

    def signed_out(self) -> "Account":
        return Account(self.subject, self.email)


@dataclass(frozen=True)
class State:
    host_id: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    client_id: str | None = None
    account: Account | None = None


def status(*, clock: Callable[[], float] = time.time) -> Status:
    """Read the stored account's status without a lock, a refresh or any request."""
    account = _read(_require_dir(create=False)).account
    return "signed_out" if account is None else account.status(clock())


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
) -> LoginResult:
    """Sign in through the browser; the stored account is replaced only once the new one is verified.

    `open_url` gets the authorization URL; returning False means the browser could not be opened.
    """
    with _locked() as state:
        # The host id must be on disk before OpenAI first sees it, so the registration it makes stays ours.
        _save(state)
    previous = state.account
    # Asking for consent again is the only way to add the plan scope to an account that signed in without it.
    consent = previous is not None and bool(previous.access_token) and PLAN_SCOPE not in previous.scopes
    pending, callback = _authorize_in_browser(
        state.host_id, state.client_id or DYNAMIC_CLIENT, open_url=open_url, timeout_s=timeout_s, consent=consent
    )
    client_id, payload = _exchange(pending, callback, opener)
    account = _credentials(payload, client_id, clock(), nonce=pending.nonce)
    with _locked() as current:
        _save(replace(current, client_id=client_id, account=account))
    return LoginResult(account.email, PLAN_SCOPE in account.scopes)


def access_token(*, opener: Opener = https_open, clock: Callable[[], float] = time.time) -> str:
    """Return a current access token with plan permission, refreshed first when it expires within a minute."""
    with _locked() as state:
        account = state.account
        if account is None or not account.access_token:
            raise ChatGPTAuthError(f"not signed in to ChatGPT; run `{LOGIN_COMMAND}`", code="not_signed_in")
        if PLAN_SCOPE not in account.scopes:
            raise ChatGPTAuthError(STATUS_DETAIL["no_plan"], code="plan_disabled")
        now = clock()
        if account.expires_at > now + _REFRESH_MARGIN_S:
            return account.access_token
        if now < account.earliest_refresh_at:
            if account.expires_at > now:
                return account.access_token
            raise ChatGPTAuthError("the ChatGPT token cannot be refreshed yet; retry later", code="refresh_not_yet")
        if not account.refresh_token or state.client_id is None:
            raise ChatGPTAuthError(STATUS_DETAIL["expired"], code="expired")
        try:
            payload = _post_form(
                TOKEN_URL,
                {
                    "grant_type": "refresh_token",
                    "client_id": state.client_id,
                    "refresh_token": account.refresh_token,
                    "resource": RESOURCE,
                },
                opener,
            )
            renewed = _credentials(payload, state.client_id, clock(), previous=account)
        except ChatGPTAuthError as exc:
            if exc.code not in _TERMINAL_REFRESH_ERRORS:
                raise
            # A refused refresh token never works again; keeping it would only repeat the refusal.
            _save(replace(state, account=account.signed_out()))
            raise ChatGPTAuthError(
                f"the ChatGPT sign-in was revoked or expired ({exc.code}); run `{LOGIN_COMMAND}`", code=exc.code
            ) from None
        _save(replace(state, account=renewed))
    if PLAN_SCOPE not in renewed.scopes:
        raise ChatGPTAuthError(STATUS_DETAIL["no_plan"], code="plan_disabled")
    return renewed.access_token


def logout(*, opener: Opener = https_open) -> bool:
    """Revoke the refresh token (best effort), then clear the tokens; True when OpenAI confirmed the revocation.

    The registration and host id stay, so the next sign-in reuses them. Nothing to revoke counts as confirmed.
    """
    with _locked() as state:
        account = state.account
        if account is None:
            return True
        confirmed = not account.refresh_token
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
                confirmed = True
            except ChatGPTAuthError:
                confirmed = False
        _save(replace(state, account=account.signed_out()))
    return confirmed


def _require_private(info: os.stat_result, *, directory: bool) -> None:
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise ChatGPTAuthError(
            f"the ChatGPT credential store in {paths.chatgpt_auth_dir()} must be owned by you, owner-only, and not a "
            "symlink",
            code="unsafe_store",
        )


def _require_dir(*, create: bool) -> Path | None:
    root = paths.chatgpt_auth_dir()
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
            "id_token": account.id_token,
            "scopes": list(account.scopes),
            "expires_at": account.expires_at,
            "earliest_refresh_at": account.earliest_refresh_at,
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
        id_token=_string(fields, "id_token", allow_empty=True),
        scopes=tuple(cast("list[str]", scopes)),
        expires_at=_number(fields, "expires_at"),
        earliest_refresh_at=_number(fields, "earliest_refresh_at"),
    )
    if account.access_token and client_id is None:
        raise ValueError("tokens without a registration")
    return State(host_id, client_id, account)


def _string(value: dict[str, object], key: str, *, allow_empty: bool = False) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        raise ValueError(key)
    return result


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
            id_token = previous.id_token
        elif isinstance(returned, str) and returned:
            identity = id_token_claims(returned, client_id, now, nonce=nonce)
            id_token = returned
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
    except (ValueError, TypeError):
        raise ChatGPTAuthError(
            "OpenAI returned unusable credentials; the stored sign-in was kept", code="invalid_credentials"
        ) from None
    return Account(
        subject=subject,
        email=email,
        access_token=access,
        refresh_token=refresh,
        id_token=id_token,
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


def authorization_url(pending: _Pending, host_id: str, *, consent: bool) -> str:
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
    if consent:
        query["prompt"] = "consent"
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

    @override
    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        # A browser preconnect may send nothing; bounding each accepted socket keeps it from holding the loop.
        connection.settimeout(max(0.01, min(1.0, self.deadline - time.monotonic())))
        return connection, address

    @override
    def handle_error(self, request: object, client_address: object) -> None:
        pass


def _authorize_in_browser(
    host_id: str,
    client_id: str,
    *,
    open_url: Callable[[str], object],
    timeout_s: float,
    consent: bool,
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
            self.wfile.write(b"Return to ytverbatim to finish signing in." if valid else b"Invalid sign-in callback.")
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
        if open_url(authorization_url(pending, host_id, consent=consent)) is False:
            raise ChatGPTAuthError("could not open a browser; retry with --no-browser", code="no_browser")
        while not received and time.monotonic() < server.deadline:
            server.handle_request()
    if not received:
        raise ChatGPTAuthError(f"sign-in timed out after {timeout_s:g}s; the stored sign-in was kept", code="timeout")
    return pending, received[0]


def _exchange(pending: _Pending, callback: dict[str, str], opener: Opener) -> tuple[str, dict[str, object]]:
    if not hmac.compare_digest(callback.get("state", ""), pending.state):
        raise ChatGPTAuthError("the sign-in callback had the wrong state", code="invalid_state")
    if callback.get("error"):
        raise ChatGPTAuthError("sign-in was not authorized; the stored sign-in was kept", code="access_denied")
    # The first sign-in registers this host; the callback names the client id OpenAI assigned it.
    client_id = callback.get("client_id", pending.client_id)
    if client_id in ("", DYNAMIC_CLIENT) or (pending.client_id != DYNAMIC_CLIENT and client_id != pending.client_id):
        raise ChatGPTAuthError("the sign-in callback named an unexpected registration", code="invalid_client")
    code = callback.get("code")
    if not code:
        raise ChatGPTAuthError("the sign-in callback had no authorization code", code="no_code")
    payload = _post_form(
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
    return client_id, payload
