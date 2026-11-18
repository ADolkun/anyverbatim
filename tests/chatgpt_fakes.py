"""Offline doubles for the chatgpt provider: a routing opener, canned responses and ID tokens. No network."""

import base64
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import Message

from anyverbatim.llm import chatgpt_auth
from anyverbatim.llm.chatgpt_auth import ISSUER, PLAN_SCOPE, Account, State

NOW = 2_000_000_000.0
CLIENT_ID = "app_registered_client"
SUBJECT = "user-subject-1"
EMAIL = "someone@example.com"
PLAN_SCOPES = ("email", "offline_access", "openid", "profile", "resource.invoke", PLAN_SCOPE)


@dataclass
class FakeResponse:
    body: bytes = b""
    lines: list[bytes] = field(default_factory=list[bytes])
    closed: bool = False
    on_readline: Callable[[], None] | None = None
    read_error: Exception | None = None

    def read(self) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        return self.body

    def readline(self) -> bytes:
        if self.on_readline is not None:
            self.on_readline()
        return self.lines.pop(0) if self.lines else b""

    def close(self) -> None:
        self.closed = True


Route = Callable[[urllib.request.Request], FakeResponse]


@dataclass
class FakeOpener:
    """Answers by `METHOD URL` (query dropped); an unrouted request fails the test."""

    routes: dict[str, Route] = field(default_factory=dict[str, Route])
    requests: list[urllib.request.Request] = field(default_factory=list[urllib.request.Request])
    timeouts: list[float] = field(default_factory=list[float])

    def __call__(self, request: urllib.request.Request, timeout_s: float) -> FakeResponse:
        self.requests.append(request)
        self.timeouts.append(timeout_s)
        key = f"{request.get_method()} {request.full_url.split('?', 1)[0]}"
        if key not in self.routes:
            raise AssertionError(f"unexpected request {key}")
        return self.routes[key](request)

    def to(self, url: str) -> list[urllib.request.Request]:
        return [r for r in self.requests if r.full_url.split("?", 1)[0] == url]


def form(request: urllib.request.Request) -> dict[str, str]:
    data = request.data
    assert isinstance(data, bytes)
    return {key: values[0] for key, values in urllib.parse.parse_qs(data.decode()).items()}


def json_response(payload: object) -> Route:
    return lambda _request: FakeResponse(body=json.dumps(payload).encode())


def http_error(code: int, payload: object = None) -> Route:
    def route(request: urllib.request.Request) -> FakeResponse:
        body = b"" if payload is None else json.dumps(payload).encode()
        raise urllib.error.HTTPError(request.full_url, code, "error", Message(), io.BytesIO(body))

    return route


def _b64(obj: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def id_token(**claims: object) -> str:
    """An unsigned ID token; a claim passed as None is left out."""
    base: dict[str, object] = {"iss": ISSUER, "aud": CLIENT_ID, "sub": SUBJECT, "exp": NOW + 3600, "email": EMAIL}
    base.update(claims)
    return f"{_b64({'alg': 'RS256', 'kid': 'k1'})}.{_b64({k: v for k, v in base.items() if v is not None})}.sig"


def token_payload(*, access: str = "access-1", refresh: str | None = "refresh-1", **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "access_token": access,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": " ".join(PLAN_SCOPES),
    }
    if refresh is not None:
        payload["refresh_token"] = refresh
    payload.update(extra)
    return {k: v for k, v in payload.items() if v is not None}


def store_account(
    *,
    expires_at: float = NOW + 3600,
    scopes: tuple[str, ...] = PLAN_SCOPES,
    access: str = "access-1",
    refresh: str = "refresh-1",
    earliest_refresh_at: float = 0.0,
    refresh_refused: bool = False,
) -> State:
    with chatgpt_auth._locked() as state:
        stored = State(
            state.host_id,
            CLIENT_ID,
            Account(
                subject=SUBJECT,
                email=EMAIL,
                access_token=access,
                refresh_token=refresh,
                scopes=scopes,
                expires_at=expires_at,
                earliest_refresh_at=earliest_refresh_at,
                refresh_refused=refresh_refused,
            ),
        )
        chatgpt_auth._save(stored)
    return stored


def stored_state() -> State:
    with chatgpt_auth._locked() as state:
        return state
