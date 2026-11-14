import contextlib
import io
import json
import socket
import threading
import time
import urllib.request
from collections.abc import Callable
from email.message import Message
from pathlib import Path

import pytest
from chatgpt_fakes import (
    NOW,
    FakeOpener,
    FakeResponse,
    Route,
    form,
    http_error,
    json_response,
    store_account,
    token_payload,
)

from ytverbatim.errors import ConfigError, LLMOutputError, LLMUnavailableError
from ytverbatim.llm import chatgpt, chatgpt_auth, chatgpt_http
from ytverbatim.llm.chatgpt import RESPONSES_URL, ChatGPTClient
from ytverbatim.llm.chatgpt_auth import TOKEN_URL, ChatGPTAuthError
from ytverbatim.llm.protocol import LLMRequest, Usage

MODEL = "gpt-6.1-sol"
REQUEST = LLMRequest(system="Be terse.", user="Hi")


@pytest.fixture(autouse=True)
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    handed_out: list[str] = []

    def access_token(*, opener: object) -> str:
        handed_out.append("token")
        return "test-access-token"

    monkeypatch.setattr(chatgpt_auth, "access_token", access_token)
    return handed_out


def sse(*events: object, done: bool = False) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        kind = event.get("type") if isinstance(event, dict) else None
        lines += [f"event: {kind}\n".encode(), f"data: {json.dumps(event)}\n".encode(), b"\n"]
    if done:
        lines += [b"data: [DONE]\n", b"\n"]
    return lines


def message(*texts: str, item_id: str = "msg_1") -> dict[str, object]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []} for text in texts],
    }


REASONING = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"}
USAGE = {
    "input_tokens": 120,
    "output_tokens": 45,
    "total_tokens": 165,
    "output_tokens_details": {"reasoning_tokens": 30},
}


def completed(output: list[object], *, usage: object = USAGE, model: str = MODEL) -> dict[str, object]:
    response: dict[str, object] = {"id": "resp_1", "model": model, "status": "completed", "error": None}
    response["output"] = output
    if usage is not None:
        response["usage"] = usage
    return {"type": "response.completed", "response": response}


def stream(lines: list[bytes], *, on_readline: Callable[[], None] | None = None) -> FakeOpener:
    return FakeOpener({f"POST {RESPONSES_URL}": lambda _r: FakeResponse(lines=list(lines), on_readline=on_readline)})


def client(opener: FakeOpener, *, timeout_s: float = 900.0, clock: Callable[[], float] | None = None) -> ChatGPTClient:
    if clock is None:
        return ChatGPTClient(model=MODEL, effort="medium", timeout_s=timeout_s, opener=opener)
    return ChatGPTClient(model=MODEL, effort="medium", timeout_s=timeout_s, opener=opener, clock=clock)


def test_id_and_the_exact_payload() -> None:
    chat = client(FakeOpener())
    assert chat.id == "chatgpt:gpt-6.1-sol@medium"
    assert ChatGPTClient(model=MODEL, effort="high").id == "chatgpt:gpt-6.1-sol@high"
    request = LLMRequest(
        system="Fix words.", user="text", json_schema={"type": "object"}, max_output_tokens=64, temperature=0.0
    )
    assert chat.payload(request) == {
        "model": "gpt-6.1-sol",
        "input": [{"role": "developer", "content": "Fix words."}, {"role": "user", "content": "text"}],
        "store": False,
        "stream": True,
        "reasoning": {"effort": "medium"},
        "include": ["reasoning.encrypted_content"],
        "parallel_tool_calls": False,
    }


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_every_effort_is_accepted(effort: str) -> None:
    assert ChatGPTClient(model=MODEL, effort=effort).id.endswith(f"@{effort}")


def test_unknown_effort_or_empty_model_is_refused() -> None:
    with pytest.raises(ConfigError, match="effort must be one of"):
        ChatGPTClient(model=MODEL, effort="minimal")
    with pytest.raises(ConfigError, match="needs a model"):
        ChatGPTClient(model="", effort="medium")


def test_a_completed_stream_returns_the_message_text_and_usage(signed_in: list[str]) -> None:
    opener = stream(
        sse(
            {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
            {"type": "response.output_text.delta", "output_index": 1, "delta": "Hel"},
            completed([REASONING, message("Hel", "lo")]),
            done=True,
        )
    )
    response = client(opener, timeout_s=30.0).complete(REQUEST)
    assert response.text == "Hello"
    assert response.usage == Usage(input_tokens=120, output_tokens=45)

    (http_request,) = opener.requests
    assert http_request.full_url == "https://api.openai.com/v1/responses"
    assert http_request.get_method() == "POST"
    assert http_request.get_header("Authorization") == "Bearer test-access-token"
    assert http_request.get_header("Content-type") == "application/json"
    assert isinstance(http_request.data, bytes)
    assert json.loads(http_request.data) == client(opener).payload(REQUEST)
    assert opener.timeouts[0] <= 30.0
    assert signed_in == ["token"]


def test_an_empty_completed_output_is_assembled_from_the_finished_items(signed_in: list[str]) -> None:
    opener = stream(
        sse(
            {"type": "response.output_item.done", "output_index": 0, "item": REASONING},
            {"type": "response.output_item.done", "output_index": 1, "item": message('{"ok": true}')},
            completed([]),
        )
    )
    assert client(opener).complete(REQUEST).text == '{"ok": true}'


def test_sse_framing_tolerates_crlf_comments_and_multiline_data(signed_in: list[str]) -> None:
    payload = json.dumps(completed([message("ok")]), indent=1).splitlines()
    lines = [b": keep-alive\r\n", b"event: response.completed\r\n"]
    lines += [f"data: {part}\r\n".encode() for part in payload]
    lines += [b"\r\n"]
    assert client(stream(lines)).complete(REQUEST).text == "ok"


def test_a_final_event_without_a_trailing_blank_line_still_counts(signed_in: list[str]) -> None:
    lines = [f"data: {json.dumps(completed([message('ok')]))}\n".encode()]
    assert client(stream(lines)).complete(REQUEST).text == "ok"


def test_a_completed_reply_without_text_is_an_output_error(signed_in: list[str]) -> None:
    with pytest.raises(LLMOutputError, match="empty reply") as raised:
        client(stream(sse(completed([REASONING])))).complete(REQUEST)
    assert raised.value.responded is True
    assert raised.value.usage == Usage(120, 45)


def test_another_model_answering_is_refused(signed_in: list[str]) -> None:
    with pytest.raises(LLMOutputError, match="not the requested one") as raised:
        client(stream(sse(completed([message("ok")], model="gpt-other")))).complete(REQUEST)
    assert raised.value.usage == Usage(120, 45)


def test_a_completed_response_without_usage_is_refused(signed_in: list[str]) -> None:
    with pytest.raises(LLMUnavailableError, match="no token usage") as raised:
        client(stream(sse(completed([message("ok")], usage=None)))).complete(REQUEST)
    assert raised.value.responded is True and raised.value.usage is None


def test_a_failed_response_carries_its_code_and_is_not_retried(signed_in: list[str]) -> None:
    failed = {
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": {"code": "server_is_overloaded", "message": "secret server detail"},
            "usage": USAGE,
        },
    }
    opener = stream(sse(failed))
    with pytest.raises(LLMUnavailableError, match="server_is_overloaded") as raised:
        client(opener).complete(REQUEST)
    assert raised.value.code == "server_is_overloaded"
    assert raised.value.responded is True and raised.value.usage == Usage(120, 45)
    assert "secret server detail" not in str(raised.value)
    assert len(opener.requests) == 1


def test_an_incomplete_response_is_an_output_error_with_its_reason(signed_in: list[str]) -> None:
    incomplete = {
        "type": "response.incomplete",
        "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "usage": USAGE},
    }
    with pytest.raises(LLMOutputError, match="max_output_tokens") as raised:
        client(stream(sse(incomplete))).complete(REQUEST)
    assert raised.value.code == "max_output_tokens"
    assert raised.value.responded is True and raised.value.usage == Usage(120, 45)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "error", "code": "rate_limit_exceeded", "message": "slow down", "param": None},
        {"type": "error", "error": {"type": "invalid_request_error", "code": "rate_limit_exceeded", "message": "x"}},
    ],
)
def test_an_error_event_raises_with_its_code(signed_in: list[str], event: dict[str, object]) -> None:
    with pytest.raises(LLMUnavailableError, match="rate_limit_exceeded") as raised:
        client(stream(sse(event))).complete(REQUEST)
    assert raised.value.code == "rate_limit_exceeded"
    assert raised.value.responded is False and raised.value.usage is None


def test_a_stream_that_ends_without_completing_raises(signed_in: list[str]) -> None:
    lines = sse({"type": "response.output_text.delta", "output_index": 0, "delta": "partial"})
    with pytest.raises(LLMUnavailableError, match=r"ended without response\.completed") as raised:
        client(stream(lines)).complete(REQUEST)
    assert raised.value.responded is False


def test_a_malformed_event_raises(signed_in: list[str]) -> None:
    with pytest.raises(LLMUnavailableError, match="malformed event"):
        client(stream([b"data: {not json\n", b"\n"])).complete(REQUEST)


def test_the_deadline_covers_the_whole_stream(signed_in: list[str]) -> None:
    now = [0.0]

    def tick() -> None:
        now[0] += 4.0

    lines = sse({"type": "response.created", "response": {}}) * 10
    opener = FakeOpener({f"POST {RESPONSES_URL}": lambda _r: response})
    response = FakeResponse(lines=lines, on_readline=tick)
    chat = client(opener, timeout_s=10.0, clock=lambda: now[0])
    with pytest.raises(LLMUnavailableError, match="within 10s"):
        chat.complete(REQUEST)
    assert opener.timeouts == [10.0], "the opener is handed the whole time left"
    assert now[0] == 12.0, "three reads, then the clock is past the deadline"
    assert response.closed


def test_a_read_that_times_out_is_the_deadline(signed_in: list[str]) -> None:
    def stall() -> None:
        raise TimeoutError

    with pytest.raises(LLMUnavailableError, match="no complete response within"):
        client(stream([], on_readline=stall)).complete(REQUEST)


@pytest.mark.parametrize(
    ("status", "body", "expected", "code"),
    [
        (401, {"error": {"code": "invalid_token", "message": "secret"}}, "ytverbatim login", "invalid_token"),
        (429, {"error": {"code": "usage_limit_reached"}}, "rate limit", "usage_limit_reached"),
        (503, None, "OpenAI is unavailable", None),
        (400, {"error": {"type": "invalid_request_error", "message": "secret"}}, "refused", "invalid_request_error"),
    ],
)
def test_http_errors_raise_without_a_retry(
    signed_in: list[str], status: int, body: object, expected: str, code: str | None
) -> None:
    opener = FakeOpener({f"POST {RESPONSES_URL}": http_error(status, body)})
    with pytest.raises(LLMUnavailableError, match=expected) as raised:
        client(opener).complete(REQUEST)
    assert raised.value.code == code
    assert f"HTTP {status}" in str(raised.value)
    assert "secret" not in str(raised.value)
    assert len(opener.requests) == 1


def test_a_missing_sign_in_fails_before_any_request() -> None:
    opener = FakeOpener()
    with pytest.raises(LLMUnavailableError, match="ytverbatim login") as raised:
        client(opener).complete(REQUEST)
    assert raised.value.code == "not_signed_in"
    assert opener.requests == []


def test_the_token_is_refreshed_before_the_request() -> None:
    store_account(expires_at=NOW - 5)
    order: list[str] = []

    def refresh(request: urllib.request.Request) -> FakeResponse:
        order.append(form(request)["grant_type"])
        return json_response(token_payload(access="fresh-token", refresh="refresh-2"))(request)

    def responses(request: urllib.request.Request) -> FakeResponse:
        order.append(str(request.get_header("Authorization")))
        return FakeResponse(lines=sse(completed([message("ok")])))

    opener = FakeOpener({f"POST {TOKEN_URL}": refresh, f"POST {RESPONSES_URL}": responses})
    real_access_token: Callable[..., str] = chatgpt_auth.access_token
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            chatgpt_auth, "access_token", lambda *, opener: real_access_token(opener=opener, clock=lambda: NOW)
        )
        assert client(opener).complete(REQUEST).text == "ok"
    assert order == ["refresh_token", "Bearer fresh-token"]


@pytest.mark.parametrize(
    "url",
    [
        "http://api.openai.com/v1/responses",
        "https://api.openai.com.example.com/v1/responses",
        "https://example.com/v1/responses",
        "https://api.openai.com:8443/v1/responses",
        "https://user@api.openai.com/v1/responses",
    ],
)
def test_https_open_refuses_anything_but_the_fixed_hosts(url: str) -> None:
    with pytest.raises(ConfigError, match="refusing to contact"):
        chatgpt_http.https_open(urllib.request.Request(url), 1.0)


def test_redirects_are_refused_rather_than_followed() -> None:
    handler = chatgpt_http._RefuseRedirects()
    request = urllib.request.Request(RESPONSES_URL, data=b"{}", headers={"Authorization": "Bearer t"})
    assert handler.redirect_request(request, io.BytesIO(), 307, "redirect", Message(), "https://example.com/") is None


def test_the_provider_defaults() -> None:
    assert (chatgpt.DEFAULT_MODEL, chatgpt.DEFAULT_EFFORT) == ("gpt-6.1-sol", "medium")
    assert chatgpt.EFFORTS == ("low", "medium", "high", "xhigh", "max")


def test_a_reply_item_that_did_not_complete_is_refused(signed_in: list[str]) -> None:
    unfinished = {**message("half an answer"), "status": "in_progress"}
    with pytest.raises(LLMOutputError, match="did not complete") as raised:
        client(stream(sse(completed([unfinished])))).complete(REQUEST)
    assert raised.value.usage == Usage(120, 45)


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "response.output_item.done", "output_index": 1, "item": message("ok")}],
        [
            {"type": "response.output_text.delta", "output_index": 1, "delta": "lost"},
            {"type": "response.output_item.done", "output_index": 0, "item": message("ok")},
        ],
    ],
)
def test_rebuilt_output_must_cover_every_index_without_a_gap(
    signed_in: list[str], events: list[dict[str, object]]
) -> None:
    with pytest.raises(LLMOutputError, match="output items are incomplete"):
        client(stream(sse(*events, completed([])))).complete(REQUEST)


def test_a_stream_that_breaks_off_is_unavailable(signed_in: list[str]) -> None:
    def broken() -> None:
        raise ConnectionError("the HTTP response broke off (IncompleteRead)")

    with pytest.raises(LLMUnavailableError, match="broke off") as raised:
        client(stream([], on_readline=broken)).complete(REQUEST)
    assert raised.value.responded is False


def test_the_suite_cannot_reach_openai() -> None:
    with pytest.raises(AssertionError, match="tried to contact"):
        chatgpt_http.https_open(urllib.request.Request(RESPONSES_URL), 1.0)


Script = Callable[[socket.socket], None]

HEADERS = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"


def chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def read_request(connection: socket.socket) -> None:
    # The whole request, body included: bytes left unread would turn the server's close into a reset.
    data = b""
    while b"\r\n\r\n" not in data:
        data += connection.recv(65536)
    head, _, body = data.partition(b"\r\n\r\n")
    length = next(
        (int(line.split(b":", 1)[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")),
        0,
    )
    while len(body) < length:
        body += connection.recv(65536)


def real_client(script: Script, timeout_s: float) -> tuple[ChatGPTClient, threading.Event]:
    """A client whose responses are real http.client responses from a server on 127.0.0.1 running `script`."""
    done = threading.Event()
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]

    def serve() -> None:
        with server:
            connection, _ = server.accept()
            with connection:
                read_request(connection)
                with contextlib.suppress(OSError):
                    script(connection)
                done.wait(5)

    threading.Thread(target=serve, daemon=True).start()

    def open_local(request: urllib.request.Request, timeout_s: float) -> chatgpt_http.Response:
        # urllib, as in production: it hands the socket to the response, so http.client's close at the end is final.
        local = urllib.request.Request(f"http://127.0.0.1:{port}/v1/responses", data=request.data, method="POST")
        return chatgpt_http.BoundedResponse(urllib.request.build_opener().open(local, timeout=timeout_s), timeout_s)

    return ChatGPTClient(model=MODEL, effort="medium", timeout_s=timeout_s, opener=open_local), done


def test_the_deadline_fires_on_a_real_stream_that_stalls(signed_in: list[str]) -> None:
    def stall(connection: socket.socket) -> None:
        connection.sendall(HEADERS + chunk(b'data: {"type": "response.created", "response": {}}\n\n'))

    chat, done = real_client(stall, 0.5)
    started = time.monotonic()
    try:
        with pytest.raises(LLMUnavailableError, match=r"within 0\.5s"):
            chat.complete(REQUEST)
    finally:
        done.set()
    assert time.monotonic() - started < 2.0


def test_the_deadline_fires_on_a_real_stream_that_trickles_inside_one_chunk(signed_in: list[str]) -> None:
    def trickle(connection: socket.socket) -> None:
        connection.sendall(HEADERS + b"400\r\n")
        for _ in range(40):
            connection.sendall(b"x")
            time.sleep(0.1)

    chat, done = real_client(trickle, 0.5)
    started = time.monotonic()
    try:
        with pytest.raises(LLMUnavailableError, match=r"within 0\.5s"):
            chat.complete(REQUEST)
    finally:
        done.set()
    assert time.monotonic() - started < 2.0


def test_a_real_stream_cut_off_mid_chunk_is_unavailable(signed_in: list[str]) -> None:
    def truncate(connection: socket.socket) -> None:
        connection.sendall(HEADERS + b"100\r\n" + b"data: {")
        connection.shutdown(socket.SHUT_RDWR)

    chat, done = real_client(truncate, 5.0)
    try:
        with pytest.raises(LLMUnavailableError, match=r"broke off \(the HTTP response broke off \(IncompleteRead\)\)"):
            chat.complete(REQUEST)
    finally:
        done.set()


def test_a_real_stream_completes_through_the_bounded_reader(signed_in: list[str]) -> None:
    body = b"".join(sse(completed([message("Hel", "lo")])))

    def reply(connection: socket.socket) -> None:
        connection.sendall(HEADERS + chunk(body[:7]) + chunk(body[7:]) + b"0\r\n\r\n")

    chat, done = real_client(reply, 5.0)
    try:
        assert chat.complete(REQUEST).text == "Hello"
    finally:
        done.set()


def test_a_response_without_a_reachable_socket_fails_loud_and_is_closed() -> None:
    class Stripped:
        fp = io.BytesIO(b"body")
        closed = False

        def close(self) -> None:
            self.closed = True

    stripped = Stripped()
    with pytest.raises(RuntimeError, match="deadline cannot be enforced"):
        chatgpt_http.BoundedResponse(stripped, 1.0)  # type: ignore[arg-type]
    assert stripped.closed


@pytest.mark.parametrize(
    "outcome",
    [ValueError("Read on closed or unwrapped SSL socket."), AttributeError("read"), b"data: {}\n"],
)
def test_whatever_a_read_the_deadline_interrupts_returns_is_the_timeout(outcome: Exception | bytes) -> None:
    class Interrupted:
        fp = None

        def read1(self, _size: int) -> bytes:
            bounded._expired.set()  # the deadline timer firing in the middle of this read
            if isinstance(outcome, bytes):
                return outcome
            raise outcome

        def isclosed(self) -> bool:
            return False

        def close(self) -> None:
            pass

    bounded = chatgpt_http.BoundedResponse(Interrupted(), 5.0)  # type: ignore[arg-type]
    with pytest.raises(TimeoutError):
        bounded.readline()


def test_reply_items_with_a_null_status_count_as_complete(signed_in: list[str]) -> None:
    reasoning = {"type": "reasoning", "id": "rs_1", "status": None, "summary": []}
    reply = {**message("ok"), "status": None}
    response = client(stream(sse(completed([reasoning, reply])))).complete(REQUEST)
    assert response.text == "ok"


def test_a_whole_body_read_through_urllib_ends_cleanly() -> None:
    body = b'{"issuer": "https://auth.example.com"}'
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]

    def serve() -> None:
        with server:
            connection, _ = server.accept()
            with connection:
                connection.recv(65536)
                connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))

    threading.Thread(target=serve, daemon=True).start()
    response = urllib.request.build_opener().open(f"http://127.0.0.1:{port}/", timeout=5)
    bounded = chatgpt_http.BoundedResponse(response, 5.0)
    try:
        assert bounded.read() == body
    finally:
        bounded.close()


class Tokens:
    """Stands in for `chatgpt_auth.access_token`: raises the queued errors in turn, then hands out a token."""

    def __init__(self, *errors: ChatGPTAuthError) -> None:
        self.errors = list(errors)
        self.calls = 0

    def __call__(self, *, opener: object) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "test-access-token"


def lapsed() -> ChatGPTAuthError:
    return ChatGPTAuthError("the ChatGPT sign-in was revoked or expired (invalid_grant)", code="invalid_grant")


def reauth_client(opener: FakeOpener, reauth: Callable[[], object] | None) -> ChatGPTClient:
    return ChatGPTClient(model=MODEL, effort="medium", opener=opener, reauth=reauth)


def test_a_lapsed_sign_in_signs_in_again_once_then_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = Tokens(lapsed())
    monkeypatch.setattr(chatgpt_auth, "access_token", tokens)
    signed_in: list[str] = []
    opener = stream(sse(completed([message("ok")])))
    assert reauth_client(opener, lambda: signed_in.append("browser")).complete(REQUEST).text == "ok"
    assert signed_in == ["browser"] and tokens.calls == 2
    assert len(opener.requests) == 1


def test_a_second_failure_after_signing_in_again_raises_without_a_third_try(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = Tokens(lapsed(), lapsed(), lapsed())
    monkeypatch.setattr(chatgpt_auth, "access_token", tokens)
    signed_in: list[str] = []
    opener = FakeOpener()
    chat = reauth_client(opener, lambda: signed_in.append("browser"))
    with pytest.raises(LLMUnavailableError, match="invalid_grant") as raised:
        chat.complete(REQUEST)
    assert raised.value.code == "invalid_grant"
    assert signed_in == ["browser"] and tokens.calls == 2 and opener.requests == []
    with pytest.raises(LLMUnavailableError):
        chat.complete(REQUEST)
    assert signed_in == ["browser"], "a failed sign-in is not reopened on every later call"


def test_a_sign_in_that_fails_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "access_token", Tokens(lapsed()))

    def time_out() -> object:
        raise ChatGPTAuthError("sign-in timed out after 300s; the stored sign-in was kept", code="timeout")

    with pytest.raises(LLMUnavailableError, match="timed out") as raised:
        reauth_client(FakeOpener(), time_out).complete(REQUEST)
    assert raised.value.code == "timeout"


@pytest.mark.parametrize("code", ["network_error", "request_failed", "refresh_not_yet"])
def test_a_transient_failure_never_opens_a_browser(monkeypatch: pytest.MonkeyPatch, code: str) -> None:
    monkeypatch.setattr(chatgpt_auth, "access_token", Tokens(ChatGPTAuthError("transient", code=code)))
    with pytest.raises(LLMUnavailableError) as raised:
        reauth_client(FakeOpener(), lambda: pytest.fail("opened a browser")).complete(REQUEST)
    assert raised.value.code == code


def test_without_reauth_a_lapsed_sign_in_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "access_token", Tokens(lapsed()))
    with pytest.raises(LLMUnavailableError, match="invalid_grant"):
        reauth_client(FakeOpener(), None).complete(REQUEST)


def responses(*routes: Route) -> FakeOpener:
    queue = list(routes)
    return FakeOpener({f"POST {RESPONSES_URL}": lambda request: queue.pop(0)(request)})


def ok_stream(_request: urllib.request.Request) -> FakeResponse:
    return FakeResponse(lines=sse(completed([message("ok")])))


def test_a_401_signs_in_again_once_and_resends(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens = Tokens()
    monkeypatch.setattr(chatgpt_auth, "access_token", tokens)
    signed_in: list[str] = []
    opener = responses(http_error(401, {"error": {"code": "invalid_token"}}), ok_stream)
    assert reauth_client(opener, lambda: signed_in.append("browser")).complete(REQUEST).text == "ok"
    assert signed_in == ["browser"] and len(opener.requests) == 2 and tokens.calls == 2


def test_a_second_401_is_not_answered_again(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "access_token", Tokens())
    signed_in: list[str] = []
    opener = responses(http_error(401), http_error(401))
    with pytest.raises(LLMUnavailableError, match="ytverbatim login"):
        reauth_client(opener, lambda: signed_in.append("browser")).complete(REQUEST)
    assert signed_in == ["browser"] and len(opener.requests) == 2


def test_a_401_without_reauth_keeps_the_login_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chatgpt_auth, "access_token", Tokens())
    opener = responses(http_error(401))
    with pytest.raises(LLMUnavailableError, match=r"HTTP 401.*ytverbatim login"):
        reauth_client(opener, None).complete(REQUEST)
    assert len(opener.requests) == 1


def test_threads_sharing_a_client_sign_in_once(monkeypatch: pytest.MonkeyPatch) -> None:
    signed_in = threading.Event()
    sign_ins: list[str] = []

    def access_token(*, opener: object) -> str:
        if not signed_in.is_set():
            raise lapsed()
        return "test-access-token"

    def reauth() -> None:
        sign_ins.append("browser")
        time.sleep(0.2)
        signed_in.set()

    monkeypatch.setattr(chatgpt_auth, "access_token", access_token)
    chat = reauth_client(FakeOpener({f"POST {RESPONSES_URL}": ok_stream}), reauth)
    texts: list[str] = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            texts.append(chat.complete(REQUEST).text)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert errors == [] and texts == ["ok", "ok"]
    assert sign_ins == ["browser"]


def test_reauth_stays_wired_when_only_the_token_fetch_after_it_failed_transiently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transient = ChatGPTAuthError("could not reach OpenAI", code="network_error")
    monkeypatch.setattr(chatgpt_auth, "access_token", Tokens(lapsed(), transient, lapsed()))
    sign_ins: list[str] = []
    chat = reauth_client(FakeOpener({f"POST {RESPONSES_URL}": ok_stream}), lambda: sign_ins.append("browser"))
    with pytest.raises(LLMUnavailableError) as raised:
        chat.complete(REQUEST)
    assert raised.value.code == "network_error"
    assert chat.complete(REQUEST).text == "ok"
    assert sign_ins == ["browser", "browser"]


def test_a_real_body_cut_short_of_its_content_length_is_unavailable(signed_in: list[str]) -> None:
    def truncate(connection: socket.socket) -> None:
        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 500\r\n\r\ndata: {")
        connection.shutdown(socket.SHUT_RDWR)

    chat, done = real_client(truncate, 5.0)
    try:
        with pytest.raises(LLMUnavailableError, match="broke off before its Content-Length"):
            chat.complete(REQUEST)
    finally:
        done.set()


def test_a_real_body_of_exactly_its_content_length_completes(signed_in: list[str]) -> None:
    body = b"".join(sse(completed([message("ok")])))

    def reply(connection: socket.socket) -> None:
        head = f"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {len(body)}\r\n\r\n"
        connection.sendall(head.encode() + body)

    chat, done = real_client(reply, 5.0)
    try:
        assert chat.complete(REQUEST).text == "ok"
    finally:
        done.set()
