import base64
import contextlib
import dataclasses
import hashlib
import http.client
import json
import os
import socket
import stat
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
from chatgpt_fakes import (
    CLIENT_ID,
    EMAIL,
    NOW,
    PLAN_SCOPES,
    FakeOpener,
    FakeResponse,
    Route,
    form,
    http_error,
    id_token,
    json_response,
    store_account,
    stored_state,
    token_payload,
)

from anyverbatim import paths
from anyverbatim.llm import chatgpt_auth
from anyverbatim.llm.chatgpt_auth import (
    DISCOVERY_URL,
    TOKEN_URL,
    ChatGPTAuthError,
    id_token_claims,
)

REVOKE_URL = "https://auth.openai.com/oauth/revoke"
DISCOVERY = {"issuer": "https://auth.openai.com", "revocation_endpoint": REVOKE_URL}


@pytest.fixture(autouse=True)
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "state" / "anyverbatim" / "llm" / "chatgpt"


Callbacks = Callable[[dict[str, str]], list[dict[str, str]]]


class Browser:
    """Stands in for the user's browser: records the authorization URL and calls back to the loopback server."""

    def __init__(self, callbacks: Callbacks) -> None:
        self.callbacks = callbacks
        self.query: dict[str, str] = {}
        self.statuses: list[int] = []

    def __call__(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == chatgpt_auth.AUTHORIZE_URL
        self.query = {key: values[0] for key, values in urllib.parse.parse_qs(parts.query).items()}
        redirect = urllib.parse.urlsplit(self.query["redirect_uri"])

        def visit() -> None:
            for params in self.callbacks(self.query):
                connection = http.client.HTTPConnection(redirect.hostname or "", redirect.port, timeout=5)
                connection.request("GET", f"{redirect.path}?{urllib.parse.urlencode(params)}")
                self.statuses.append(connection.getresponse().status)
                connection.close()

        threading.Thread(target=visit, daemon=True).start()
        return True


def approve(query: dict[str, str]) -> list[dict[str, str]]:
    return [{"code": "auth-code", "state": query["state"], "client_id": CLIENT_ID}]


def token_endpoint(browser: Browser, **claims: object) -> FakeOpener:
    def route(_request: object) -> FakeResponse:
        token = id_token(**{"nonce": browser.query["nonce"], **claims})
        return FakeResponse(body=json.dumps(token_payload(id_token=token)).encode())

    return FakeOpener({f"POST {TOKEN_URL}": route})


def login(browser: Browser, opener: FakeOpener, *, timeout_s: float = 10.0) -> chatgpt_auth.LoginResult:
    return chatgpt_auth.login(open_url=browser, opener=opener, timeout_s=timeout_s, clock=lambda: NOW)


def test_login_registers_this_host_exchanges_the_code_and_stores_the_account(state_dir: Path) -> None:
    browser = Browser(approve)
    opener = token_endpoint(browser)
    result = login(browser, opener)

    assert result == chatgpt_auth.LoginResult(EMAIL, plan=True)
    query = browser.query
    redirect = urllib.parse.urlsplit(query["redirect_uri"])
    assert (redirect.scheme, redirect.hostname, redirect.path) == ("http", "127.0.0.1", "/auth/callback")
    assert redirect.port is not None
    assert redirect.port > 0
    assert query["client_id"] == "dynamic_agent_client"
    assert query["agent_name_hint"] == "anyverbatim"
    assert query["ext_agent_host_id"].startswith("urn:uuid:")
    assert query["response_type"] == "code"
    assert query["resource"] == "https://api.openai.com/v1"
    assert query["scope"] == "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    assert query["code_challenge_method"] == "S256"
    assert query["state"]
    assert query["nonce"]
    assert query["state"] != query["nonce"]
    assert "prompt" not in query
    assert "login_hint" not in query, "nothing is stored before the first sign-in"
    assert browser.statuses == [200]

    (exchange,) = opener.requests
    sent = form(exchange)
    verifier = sent.pop("code_verifier")
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == challenge
    assert sent == {
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "code": "auth-code",
        "redirect_uri": query["redirect_uri"],
        "resource": "https://api.openai.com/v1",
    }

    state = stored_state()
    assert state.host_id == query["ext_agent_host_id"]
    assert state.client_id == CLIENT_ID
    assert state.account is not None
    assert state.account.access_token == "access-1"
    assert state.account.expires_at == NOW + 3600
    assert chatgpt_auth.status(clock=lambda: NOW) == "ready"
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths.chatgpt_credentials_file().stat().st_mode) == 0o600
    assert json.loads(paths.chatgpt_credentials_file().read_text())["schema_version"] == 1


def test_signing_in_again_reuses_the_registration_and_host_id() -> None:
    first = Browser(approve)
    login(first, token_endpoint(first))
    again = Browser(lambda query: [{"code": "auth-code-2", "state": query["state"]}])
    opener = token_endpoint(again)
    login(again, opener)

    assert again.query["client_id"] == CLIENT_ID
    assert again.query["login_hint"] == EMAIL
    assert "agent_name_hint" not in again.query
    assert again.query["ext_agent_host_id"] == first.query["ext_agent_host_id"]
    assert form(opener.requests[0])["client_id"] == CLIENT_ID


def test_an_account_without_plan_permission_is_asked_for_consent_again() -> None:
    store_account(scopes=("openid", "offline_access"))
    browser = Browser(approve)
    login(browser, token_endpoint(browser))
    assert browser.query["prompt"] == "consent"
    assert chatgpt_auth.status(clock=lambda: NOW) == "ready"


def test_a_callback_with_the_wrong_state_is_refused_and_the_right_one_still_completes() -> None:
    browser = Browser(
        lambda query: [
            {"code": "forged", "state": "not-the-state", "client_id": CLIENT_ID},
            {"code": "auth-code", "state": query["state"], "client_id": CLIENT_ID},
        ]
    )
    opener = token_endpoint(browser)
    login(browser, opener)
    assert browser.statuses == [400, 200]
    assert form(opener.requests[0])["code"] == "auth-code"


def test_only_wrong_state_callbacks_time_out_without_an_exchange() -> None:
    browser = Browser(lambda _query: [{"code": "forged", "state": "not-the-state"}])
    opener = FakeOpener()
    with pytest.raises(ChatGPTAuthError) as raised:
        login(browser, opener, timeout_s=1.0)
    assert raised.value.code == "timeout"
    assert opener.requests == []
    assert stored_state().account is None


def test_a_denied_sign_in_keeps_the_stored_account() -> None:
    store_account()
    browser = Browser(lambda query: [{"error": "access_denied", "state": query["state"]}])
    with pytest.raises(ChatGPTAuthError) as raised:
        login(browser, FakeOpener())
    assert raised.value.code == "access_denied"
    account = stored_state().account
    assert account is not None
    assert account.access_token == "access-1"


def test_a_first_sign_in_without_an_assigned_client_id_is_refused() -> None:
    browser = Browser(lambda query: [{"code": "auth-code", "state": query["state"]}])
    with pytest.raises(ChatGPTAuthError) as raised:
        login(browser, FakeOpener())
    assert raised.value.code == "invalid_client"


@pytest.mark.parametrize(
    ("claims", "failed"),
    [
        ({"iss": "https://issuer.example.com"}, "iss"),
        ({"aud": "someone-else"}, "aud"),
        ({"aud": [CLIENT_ID, "other"]}, "azp"),
        ({"azp": "someone-else"}, "azp"),
        ({"exp": NOW - 31}, "exp"),
        ({"sub": ""}, "sub"),
        ({"nonce": "replayed-nonce"}, "nonce"),
    ],
)
def test_an_id_token_failing_a_claim_check_stores_nothing(claims: dict[str, object], failed: str) -> None:
    browser = Browser(approve)
    with pytest.raises(ChatGPTAuthError, match=f"failed its {failed} check") as raised:
        login(browser, token_endpoint(browser, **claims))
    assert raised.value.code == "invalid_identity"
    assert stored_state().account is None


def test_id_token_claims_allow_thirty_seconds_of_clock_skew_and_several_audiences_with_azp() -> None:
    assert id_token_claims(id_token(exp=NOW - 29), CLIENT_ID, NOW)["sub"]
    assert id_token_claims(id_token(aud=[CLIENT_ID, "other"], azp=CLIENT_ID), CLIENT_ID, NOW)
    with pytest.raises(ChatGPTAuthError, match="format"):
        id_token_claims("not-a-jwt", CLIENT_ID, NOW)


def test_a_browser_that_cannot_open_points_at_no_browser() -> None:
    with pytest.raises(ChatGPTAuthError, match="--no-browser"):
        chatgpt_auth.login(open_url=lambda _url: False, opener=FakeOpener(), timeout_s=1.0)


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
def test_a_store_readable_by_others_is_refused(mode: int) -> None:
    store_account()
    paths.chatgpt_credentials_file().chmod(mode)
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.status()
    assert raised.value.code == "unsafe_store"


def test_a_symlinked_credentials_file_is_refused(tmp_path: Path) -> None:
    store_account()
    elsewhere = tmp_path / "elsewhere.json"
    paths.chatgpt_credentials_file().rename(elsewhere)
    paths.chatgpt_credentials_file().symlink_to(elsewhere)
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.status()
    assert raised.value.code == "unsafe_store"
    with pytest.raises(ChatGPTAuthError, match="symlink"):
        chatgpt_auth.access_token(opener=FakeOpener())


def test_a_symlinked_or_loose_store_directory_is_refused(tmp_path: Path, state_dir: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    state_dir.parent.mkdir(parents=True)
    state_dir.symlink_to(target)
    with pytest.raises(ChatGPTAuthError, match="not a symlink"):
        chatgpt_auth.status()
    state_dir.unlink()
    state_dir.mkdir(mode=0o700)
    state_dir.chmod(0o755)
    with pytest.raises(ChatGPTAuthError, match="owner-only"):
        chatgpt_auth.status()


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"schema_version": 2, "host_id": "urn:uuid:00000000-0000-4000-8000-000000000000"}),
        json.dumps({"schema_version": 1, "host_id": "not-a-urn"}),
    ],
)
def test_an_invalid_store_fails_loud(content: str) -> None:
    store_account()
    path = paths.chatgpt_credentials_file()
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.status()
    assert raised.value.code == "invalid_store"
    assert "anyverbatim login" in str(raised.value)


def test_status_reports_every_state_without_a_request() -> None:
    assert chatgpt_auth.status() == "signed_out"
    store_account(scopes=("openid",))
    assert chatgpt_auth.status(clock=lambda: NOW) == "no_plan"
    store_account(expires_at=NOW + 30)
    assert chatgpt_auth.status(clock=lambda: NOW) == "refreshable"
    store_account(expires_at=NOW - 1, refresh="")
    assert chatgpt_auth.status(clock=lambda: NOW) == "expired"
    assert chatgpt_auth.readiness(clock=lambda: NOW).ready is False


def test_a_current_token_is_returned_without_a_request() -> None:
    store_account(expires_at=NOW + 61)
    opener = FakeOpener()
    assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW) == "access-1"
    assert opener.requests == []


def test_a_token_expiring_within_a_minute_is_refreshed_and_the_refresh_token_rotates() -> None:
    store_account(expires_at=NOW + 59)
    opener = FakeOpener({f"POST {TOKEN_URL}": json_response(token_payload(access="access-2", refresh="refresh-2"))})
    assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW) == "access-2"
    (request,) = opener.requests
    assert form(request) == {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": "refresh-1",
        "resource": "https://api.openai.com/v1",
    }
    account = stored_state().account
    assert account is not None
    assert (account.access_token, account.refresh_token) == ("access-2", "refresh-2")
    assert account.expires_at == NOW + 3600


def test_a_refresh_without_a_new_refresh_token_is_refused_and_keeps_the_old_one() -> None:
    store_account(expires_at=NOW)
    opener = FakeOpener({f"POST {TOKEN_URL}": json_response(token_payload(access="access-2", refresh=None))})
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.access_token(opener=opener, clock=lambda: NOW)
    assert raised.value.code == "invalid_credentials"
    account = stored_state().account
    assert account is not None
    assert account.refresh_token == "refresh-1"


@pytest.mark.parametrize("code", ["invalid_grant", "refresh_token_reused", "refresh_token_expired"])
def test_a_terminal_refresh_error_keeps_the_stored_sign_in(code: str) -> None:
    stored = store_account(expires_at=NOW)
    opener = FakeOpener({f"POST {TOKEN_URL}": http_error(400, {"error": code, "error_description": "secret detail"})})
    with pytest.raises(ChatGPTAuthError, match="anyverbatim login") as raised:
        chatgpt_auth.access_token(opener=opener, clock=lambda: NOW)
    assert raised.value.code == code
    assert raised.value.needs_sign_in
    assert "secret detail" not in str(raised.value)
    assert stored.account is not None
    assert stored_state() == dataclasses.replace(
        stored, account=dataclasses.replace(stored.account, refresh_refused=True)
    ), "only a new sign-in or an explicit logout changes the session; the refusal is only marked"


def test_a_transient_refresh_failure_keeps_the_tokens() -> None:
    store_account(expires_at=NOW)
    opener = FakeOpener({f"POST {TOKEN_URL}": http_error(503)})
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.access_token(opener=opener, clock=lambda: NOW)
    assert raised.value.code == "request_failed"
    account = stored_state().account
    assert account is not None
    assert account.refresh_token == "refresh-1"


def test_earliest_refresh_at_holds_off_a_refresh_while_the_token_still_works() -> None:
    store_account(expires_at=NOW + 45, earliest_refresh_at=NOW + 100)
    opener = FakeOpener()
    assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW) == "access-1"
    assert opener.requests == []
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.access_token(opener=opener, clock=lambda: NOW + 20)
    assert raised.value.code == "refresh_not_yet"
    assert opener.requests == []


def test_no_account_or_no_plan_refuses_a_token() -> None:
    with pytest.raises(ChatGPTAuthError, match="anyverbatim login") as raised:
        chatgpt_auth.access_token(opener=FakeOpener())
    assert raised.value.code == "not_signed_in"
    store_account(scopes=("openid",))
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.access_token(opener=FakeOpener(), clock=lambda: NOW)
    assert raised.value.code == "plan_disabled"


def test_logout_revokes_at_the_discovered_endpoint_and_clears_the_tokens() -> None:
    store_account()
    opener = FakeOpener(
        {f"GET {DISCOVERY_URL}": json_response(DISCOVERY), f"POST {REVOKE_URL}": lambda _r: FakeResponse()}
    )
    assert chatgpt_auth.logout(opener=opener) == "confirmed"
    (revoke,) = opener.to(REVOKE_URL)
    assert form(revoke) == {"token": "refresh-1", "token_type_hint": "refresh_token", "client_id": CLIENT_ID}
    state = stored_state()
    assert state.account is not None
    assert state.account.access_token == ""
    assert state.client_id == CLIENT_ID
    assert chatgpt_auth.status() == "signed_out"


@pytest.mark.parametrize(
    "routes",
    [
        {f"GET {DISCOVERY_URL}": json_response(DISCOVERY), f"POST {REVOKE_URL}": http_error(500)},
        {f"GET {DISCOVERY_URL}": json_response({**DISCOVERY, "revocation_endpoint": "https://revoke.example.com/"})},
        {f"GET {DISCOVERY_URL}": json_response({**DISCOVERY, "issuer": "https://issuer.example.com"})},
    ],
)
def test_an_unconfirmed_revocation_still_clears_the_tokens(routes: dict[str, Route]) -> None:
    store_account()
    opener = FakeOpener(routes)
    assert chatgpt_auth.logout(opener=opener) == "unconfirmed"
    account = stored_state().account
    assert account is not None
    assert account.access_token == ""
    assert account.refresh_token == ""


def test_logout_says_when_there_was_nothing_to_revoke() -> None:
    opener = FakeOpener()
    assert chatgpt_auth.logout(opener=opener) == "not_signed_in"
    store_account(refresh="")
    assert chatgpt_auth.logout(opener=opener) == "no_refresh_token"
    assert chatgpt_auth.status() == "signed_out"
    assert chatgpt_auth.logout(opener=opener) == "not_signed_in"
    assert opener.requests == []


def test_a_revocation_response_cut_off_mid_body_still_clears_the_tokens() -> None:
    store_account()

    def truncated(_request: object) -> FakeResponse:
        return FakeResponse(read_error=ConnectionError("the HTTP response broke off (IncompleteRead)"))

    opener = FakeOpener({f"GET {DISCOVERY_URL}": json_response(DISCOVERY), f"POST {REVOKE_URL}": truncated})
    assert chatgpt_auth.logout(opener=opener) == "unconfirmed"
    account = stored_state().account
    assert account is not None
    assert account.access_token == account.refresh_token == ""


def test_the_store_never_holds_more_than_one_account() -> None:
    store_account()
    raw = json.loads(paths.chatgpt_credentials_file().read_text())
    assert set(raw) == {"schema_version", "host_id", "client_id", "account"}
    assert raw["account"]["scopes"] == list(PLAN_SCOPES)
    assert "id_token" not in raw["account"]
    assert os.stat(paths.chatgpt_lock_file()).st_mode & 0o077 == 0


def test_reading_the_status_creates_no_directory(tmp_path: Path) -> None:
    assert chatgpt_auth.status() == "signed_out"
    assert chatgpt_auth.readiness().ready is False
    assert not (tmp_path / "state").exists()


class Browsers:
    """Several sign-in rounds: one `Browser` per authorization URL opened."""

    def __init__(self, callbacks: Callbacks) -> None:
        self.callbacks = callbacks
        self.rounds: list[Browser] = []

    def __call__(self, url: str) -> bool:
        browser = Browser(self.callbacks)
        self.rounds.append(browser)
        return browser(url)


def test_invalid_grant_on_a_first_registration_reauthorizes_once_under_the_issued_client() -> None:
    browsers = Browsers(
        lambda query: [{"code": f"code-{query['state'][:4]}", "state": query["state"], "client_id": CLIENT_ID}]
    )
    calls: list[str] = []

    def token(request: urllib.request.Request) -> FakeResponse:
        calls.append(form(request)["code_verifier"])
        if len(calls) == 1:
            return http_error(400, {"error": "invalid_grant"})(request)
        nonce = browsers.rounds[-1].query["nonce"]
        return FakeResponse(body=json.dumps(token_payload(id_token=id_token(nonce=nonce))).encode())

    login(browsers, FakeOpener({f"POST {TOKEN_URL}": token}))  # type: ignore[arg-type]
    first, second = (b.query for b in browsers.rounds)
    assert first["client_id"] == "dynamic_agent_client"
    assert "agent_name_hint" in first
    assert second["client_id"] == CLIENT_ID
    assert "agent_name_hint" not in second
    assert first["state"] != second["state"]
    assert first["nonce"] != second["nonce"]
    assert first["code_challenge"] != second["code_challenge"]
    assert calls[0] != calls[1]
    assert chatgpt_auth.status(clock=lambda: NOW) == "ready"


def test_a_second_invalid_grant_is_not_retried_again() -> None:
    browsers = Browsers(approve)
    opener = FakeOpener({f"POST {TOKEN_URL}": http_error(400, {"error": "invalid_grant"})})
    with pytest.raises(ChatGPTAuthError) as raised:
        login(browsers, opener)  # type: ignore[arg-type]
    assert raised.value.code == "invalid_grant"
    assert len(browsers.rounds) == 2
    assert len(opener.requests) == 2


def test_the_issued_client_id_is_kept_when_the_first_exchange_fails() -> None:
    browser = Browser(approve)
    with pytest.raises(ChatGPTAuthError) as raised:
        login(browser, FakeOpener({f"POST {TOKEN_URL}": http_error(503)}))
    assert raised.value.code == "request_failed"
    state = stored_state()
    assert state.client_id == CLIENT_ID
    assert state.account is None

    again = Browser(lambda query: [{"code": "auth-code", "state": query["state"]}])
    login(again, token_endpoint(again))
    assert again.query["client_id"] == CLIENT_ID


def test_a_peer_that_trickles_bytes_cannot_hold_login_past_its_deadline() -> None:
    stop = threading.Event()

    def trickle(url: str) -> bool:
        redirect = urllib.parse.urlsplit(dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))["redirect_uri"])

        def send() -> None:
            with socket.create_connection((redirect.hostname or "", redirect.port or 0), timeout=5) as connection:
                for byte in b"GET /auth/callback?state=x HTTP/1.1\r\n":
                    if stop.is_set():
                        return
                    with contextlib.suppress(OSError):
                        connection.sendall(bytes([byte]))
                    time.sleep(0.2)

        threading.Thread(target=send, daemon=True).start()
        return True

    started = time.monotonic()
    try:
        with pytest.raises(ChatGPTAuthError) as raised:
            chatgpt_auth.login(open_url=trickle, opener=FakeOpener(), timeout_s=1.0)
    finally:
        stop.set()
    assert raised.value.code == "timeout"
    assert time.monotonic() - started < 3.0


def test_a_refresh_refused_for_its_client_forgets_only_the_client() -> None:
    stored = store_account(expires_at=NOW + 200, earliest_refresh_at=NOW - 1)
    opener = FakeOpener({f"POST {TOKEN_URL}": http_error(401, {"error": "invalid_client"})})
    assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW) == "access-1"
    state = stored_state()
    assert state.client_id is None
    assert state.account == stored.account
    assert chatgpt_auth.status(clock=lambda: NOW + 150) == "ready", "the token still works for 50 s"
    assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW + 150) == "access-1"
    assert len(opener.requests) == 1, "no refresh is attempted without a registration"
    assert chatgpt_auth.status(clock=lambda: NOW + 180) == "expired"
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.access_token(opener=opener, clock=lambda: NOW + 180)
    assert raised.value.code == "expired"
    assert raised.value.needs_sign_in
    assert stored_state().account == stored.account


@pytest.mark.parametrize(
    "route",
    [
        http_error(400, {"error": "invalid_grant"}),
        http_error(401, {"error": "invalid_client"}),
        http_error(503),
        json_response({"access_token": "half"}),
    ],
)
def test_a_failed_refresh_with_time_left_returns_the_current_token(route: Route) -> None:
    store_account(expires_at=NOW + 31, earliest_refresh_at=NOW - 1)
    opener = FakeOpener({f"POST {TOKEN_URL}": route})
    assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW) == "access-1"
    assert len(opener.requests) == 1
    account = stored_state().account
    assert account is not None
    assert account.refresh_token == "refresh-1"


def test_a_failed_refresh_raises_once_the_token_is_nearly_gone() -> None:
    store_account(expires_at=NOW + 29, earliest_refresh_at=NOW - 1)
    opener = FakeOpener({f"POST {TOKEN_URL}": http_error(503)})
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.access_token(opener=opener, clock=lambda: NOW)
    assert raised.value.code == "request_failed"
    assert not raised.value.needs_sign_in


NEW_CLIENT = "app_new_client"


def scripted_token_endpoint(browsers: Browsers, *outcomes: object) -> FakeOpener:
    """Token responses in turn: an error code string refuses, None issues tokens for the client that asked."""
    queue = list(outcomes)

    def route(request: urllib.request.Request) -> FakeResponse:
        outcome = queue.pop(0)
        if isinstance(outcome, str):
            return http_error(401 if outcome == "invalid_client" else 400, {"error": outcome})(request)
        client = form(request)["client_id"]
        token = id_token(aud=client, nonce=browsers.rounds[-1].query["nonce"])
        return FakeResponse(body=json.dumps(token_payload(id_token=token)).encode())

    return FakeOpener({f"POST {TOKEN_URL}": route})


def issue(query: dict[str, str]) -> list[dict[str, str]]:
    client = NEW_CLIENT if query["client_id"] == chatgpt_auth.DYNAMIC_CLIENT else query["client_id"]
    return [{"code": "auth-code", "state": query["state"], "client_id": client}]


def test_an_exchange_refused_for_its_client_registers_afresh_within_one_login() -> None:
    store_account()
    browsers = Browsers(issue)
    login(browsers, scripted_token_endpoint(browsers, "invalid_client", None))  # type: ignore[arg-type]
    first, second = (b.query for b in browsers.rounds)
    assert first["client_id"] == CLIENT_ID
    assert second["client_id"] == chatgpt_auth.DYNAMIC_CLIENT
    assert second["agent_name_hint"] == "anyverbatim"
    assert stored_state().client_id == NEW_CLIENT
    assert chatgpt_auth.status(clock=lambda: NOW) == "ready"


@pytest.mark.parametrize(
    ("outcomes", "rounds", "code"),
    [
        (("invalid_client", "invalid_client"), 2, "invalid_client"),
        (("invalid_grant", "invalid_grant"), 2, "invalid_grant"),
        (("invalid_grant", "invalid_client", "invalid_grant"), 3, "invalid_grant"),
        (("invalid_client", "invalid_grant", "invalid_client"), 3, "invalid_client"),
    ],
)
def test_each_sign_in_fallback_runs_at_most_once(outcomes: tuple[str, ...], rounds: int, code: str) -> None:
    store_account()
    browsers = Browsers(issue)
    with pytest.raises(ChatGPTAuthError) as raised:
        login(browsers, scripted_token_endpoint(browsers, *outcomes))  # type: ignore[arg-type]
    assert raised.value.code == code
    assert len(browsers.rounds) == rounds
    account = stored_state().account
    assert account is not None and account.access_token == "access-1", "a failed sign-in keeps the old session"


def test_logout_clears_the_tokens_even_when_revocation_fails_unexpectedly() -> None:
    store_account()

    def broken(_request: object) -> FakeResponse:
        raise RuntimeError("unexpected")

    opener = FakeOpener({f"GET {DISCOVERY_URL}": json_response(DISCOVERY), f"POST {REVOKE_URL}": broken})
    with pytest.raises(RuntimeError):
        chatgpt_auth.logout(opener=opener)
    account = stored_state().account
    assert account is not None
    assert account.access_token == ""
    assert account.refresh_token == ""


def test_a_signed_out_account_still_hints_its_email_and_no_email_means_no_hint() -> None:
    store_account()
    chatgpt_auth.logout(opener=FakeOpener({f"GET {DISCOVERY_URL}": http_error(503)}))
    browser = Browser(approve)
    login(browser, token_endpoint(browser))
    assert browser.query["login_hint"] == EMAIL

    with chatgpt_auth._locked() as state:
        assert state.account is not None
        chatgpt_auth._save(dataclasses.replace(state, account=dataclasses.replace(state.account, email="")))
    unnamed = Browser(lambda query: [{"code": "auth-code", "state": query["state"]}])
    login(unnamed, token_endpoint(unnamed, email=None))
    assert "login_hint" not in unnamed.query


@pytest.mark.parametrize(
    ("expires_at", "earliest", "refreshed"),
    [
        (NOW + 1800, NOW - 1, True),
        (NOW + 1800, NOW, True),
        (NOW + 1800, NOW + 1, False),
        (NOW + 1800, 0.0, False),
        (NOW + 59, 0.0, True),
        (NOW + 61, 0.0, False),
    ],
)
def test_the_token_is_refreshed_at_the_first_chance_the_server_allows(
    expires_at: float, earliest: float, refreshed: bool
) -> None:
    store_account(expires_at=expires_at, earliest_refresh_at=earliest)
    opener = FakeOpener({f"POST {TOKEN_URL}": json_response(token_payload(access="access-2", refresh="refresh-2"))})
    token = chatgpt_auth.access_token(opener=opener, clock=lambda: NOW)
    assert (token == "access-2") is refreshed
    assert len(opener.requests) == int(refreshed)


@pytest.mark.parametrize(
    ("code", "needs"),
    [
        ("not_signed_in", True),
        ("expired", True),
        ("plan_disabled", True),
        ("invalid_client", True),
        ("invalid_grant", True),
        ("invalid_refresh_token", True),
        ("token_expired", True),
        ("refresh_token_expired", True),
        ("refresh_token_invalidated", True),
        ("refresh_token_reused", True),
        ("network_error", False),
        ("request_failed", False),
        ("refresh_not_yet", False),
        ("timeout", False),
        ("unsafe_store", False),
    ],
)
def test_needs_sign_in_marks_only_what_a_new_sign_in_mends(code: str, needs: bool) -> None:
    assert ChatGPTAuthError("x", code=code).needs_sign_in is needs


def test_concurrent_refreshes_send_one_token_request() -> None:
    store_account(expires_at=NOW)
    calls: list[str] = []

    def refresh(request: urllib.request.Request) -> FakeResponse:
        calls.append(form(request)["refresh_token"])
        time.sleep(0.3)
        return json_response(token_payload(access="access-2", refresh="refresh-2"))(request)

    opener = FakeOpener({f"POST {TOKEN_URL}": refresh})
    tokens: list[str] = []
    errors: list[BaseException] = []

    def use() -> None:
        try:
            tokens.append(chatgpt_auth.access_token(opener=opener, clock=lambda: NOW))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=use) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert errors == []
    assert tokens == ["access-2", "access-2"]
    assert calls == ["refresh-1"], "the rotated refresh token is presented exactly once"


def test_interactive_login_says_why_prints_the_url_and_opens_the_browser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    url = "https://auth.openai.com/api/accounts/authorize?client_id=dynamic_agent_client"

    def fake_login(*, open_url: Callable[[str], object], opener: object) -> chatgpt_auth.LoginResult:
        assert open_url(url) is None, "a browser that will not open must not abort the wait"
        return chatgpt_auth.LoginResult(EMAIL, plan=True)

    monkeypatch.setattr(chatgpt_auth, "login", fake_login)
    monkeypatch.setattr(chatgpt_auth.webbrowser, "open", lambda target: opened.append(target) or False)
    assert chatgpt_auth.interactive_login("ChatGPT sign-in lapsed").plan
    err = capsys.readouterr().err
    assert err.startswith("ChatGPT sign-in lapsed; opening the browser to sign in again (waiting up to 5 minutes)\n")
    assert url in err
    assert opened == [url]
    assert "access-" not in err
    assert "refresh-" not in err


def test_switch_shows_the_account_chooser_and_replaces_the_account_only_once_verified() -> None:
    store_account()
    browser = Browser(approve)
    opener = token_endpoint(browser, sub="another-subject", email="other@example.com")
    result = chatgpt_auth.login(open_url=browser, opener=opener, timeout_s=10.0, clock=lambda: NOW, switch=True)
    assert browser.query["prompt"] == "login"
    assert "login_hint" not in browser.query
    assert result.email == "other@example.com"
    account = stored_state().account
    assert account is not None
    assert account.subject == "another-subject"


def test_a_failed_switch_keeps_the_current_account() -> None:
    stored = store_account()
    browser = Browser(lambda query: [{"error": "access_denied", "state": query["state"]}])
    with pytest.raises(ChatGPTAuthError):
        chatgpt_auth.login(open_url=browser, opener=FakeOpener(), timeout_s=10.0, clock=lambda: NOW, switch=True)
    assert stored_state() == stored


def test_a_refused_refresh_token_is_marked_and_never_sent_again() -> None:
    store_account(expires_at=NOW + 200, earliest_refresh_at=NOW - 1)
    opener = FakeOpener({f"POST {TOKEN_URL}": http_error(400, {"error": "invalid_grant"})})
    for _ in range(5):
        assert chatgpt_auth.access_token(opener=opener, clock=lambda: NOW) == "access-1"
    assert len(opener.requests) == 1, "the refused token is sent once, not on every call"
    account = stored_state().account
    assert account is not None
    assert account.refresh_refused
    assert (account.access_token, account.refresh_token) == ("access-1", "refresh-1")
    assert chatgpt_auth.session(clock=lambda: NOW).status == "refused"
    assert not chatgpt_auth.session(clock=lambda: NOW).usable
    assert chatgpt_auth.status(clock=lambda: NOW + 180) == "expired"
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.access_token(opener=opener, clock=lambda: NOW + 180)
    assert raised.value.code == "expired"
    assert raised.value.needs_sign_in
    assert len(opener.requests) == 1


def test_a_new_sign_in_clears_the_refusal_and_hints_the_account() -> None:
    store_account(refresh_refused=True)
    browser = Browser(lambda query: [{"code": "auth-code", "state": query["state"]}])
    login(browser, token_endpoint(browser))
    assert browser.query["login_hint"] == EMAIL
    account = stored_state().account
    assert account is not None
    assert not account.refresh_refused
    assert chatgpt_auth.status(clock=lambda: NOW) == "ready"


def test_a_store_written_before_the_refusal_marker_reads_it_as_false() -> None:
    store_account()
    path = paths.chatgpt_credentials_file()
    raw = json.loads(path.read_text())
    del raw["account"]["refresh_refused"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    account = stored_state().account
    assert account is not None
    assert not account.refresh_refused
    raw["account"]["refresh_refused"] = "yes"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ChatGPTAuthError) as raised:
        chatgpt_auth.status()
    assert raised.value.code == "invalid_store"


@pytest.mark.parametrize(
    ("earliest", "kept"),
    [
        (NOW + 3300, True),
        (NOW, False),
        (NOW - 10, False),
        (NOW + 3600, False),
        (NOW + 7200, False),
    ],
)
def test_earliest_refresh_at_is_kept_only_between_issue_and_expiry(earliest: float, kept: bool) -> None:
    store_account(expires_at=NOW)
    payload = token_payload(access="access-2", refresh="refresh-2", earliest_refresh_at=earliest)
    chatgpt_auth.access_token(opener=FakeOpener({f"POST {TOKEN_URL}": json_response(payload)}), clock=lambda: NOW)
    account = stored_state().account
    assert account is not None
    assert account.earliest_refresh_at == (earliest if kept else 0.0)


def test_logout_without_a_registration_says_the_refresh_token_was_not_revoked() -> None:
    store_account()
    with chatgpt_auth._locked() as state:
        chatgpt_auth._save(dataclasses.replace(state, client_id=None))
    opener = FakeOpener()
    assert chatgpt_auth.logout(opener=opener) == "unregistered"
    assert opener.requests == []
    account = stored_state().account
    assert account is not None
    assert account.access_token == account.refresh_token == ""
