import contextlib
import http.client
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
from chatgpt_fakes import NOW, FakeOpener, FakeResponse, form, http_error, json_response, store_account, token_payload

from ytverbatim.errors import ConfigError, LLMOutputError, LLMUnavailableError
from ytverbatim.llm import chatgpt, chatgpt_auth, chatgpt_http
from ytverbatim.llm.chatgpt import RESPONSES_URL, ChatGPTClient
from ytverbatim.llm.chatgpt_auth import TOKEN_URL
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
        (401, {"error": {"code": "invalid_token", "message": "secret"}}, "ytverbatim llm login", "invalid_token"),
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
    with pytest.raises(LLMUnavailableError, match="ytverbatim llm login") as raised:
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


def real_client(script: Script, timeout_s: float) -> tuple[ChatGPTClient, threading.Event]:
    """A client whose responses are real http.client responses from a server on 127.0.0.1 running `script`."""
    done = threading.Event()
    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]

    def serve() -> None:
        with server:
            connection, _ = server.accept()
            with connection:
                connection.recv(65536)
                with contextlib.suppress(OSError):
                    script(connection)
                done.wait(5)

    threading.Thread(target=serve, daemon=True).start()

    def open_local(request: urllib.request.Request, timeout_s: float) -> chatgpt_http.Response:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
        connection.request("POST", "/v1/responses", body=request.data)
        return chatgpt_http.BoundedResponse(connection.getresponse(), timeout_s)

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


def test_a_response_without_a_reachable_socket_fails_loud() -> None:
    class Stripped:
        fp = io.BytesIO(b"body")

    with pytest.raises(RuntimeError, match="deadline cannot be enforced"):
        chatgpt_http.BoundedResponse(Stripped(), 1.0)  # type: ignore[arg-type]
