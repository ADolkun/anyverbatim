"""HTTPS to OpenAI's two fixed hosts for the chatgpt provider: stdlib urllib, redirects refused, never followed."""

import http.client
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from email.message import Message
from typing import IO, Protocol, override

from anyverbatim.errors import ConfigError
from anyverbatim.llm.jsonio import as_object

AUTH_HOST = "auth.openai.com"
API_HOST = "api.openai.com"
HOSTS = frozenset({AUTH_HOST, API_HOST})
_ERROR_BODY_LIMIT = 65_536
_READ_SIZE = 65_536
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_.]{0,95}")


class Response(Protocol):
    def read(self) -> bytes: ...

    def readline(self) -> bytes: ...

    def close(self) -> None: ...


# (request, timeout_s) -> the open response, which must end within timeout_s of the call: a read past it raises
# TimeoutError. A status other than 2xx, a refused redirect included, raises urllib.error.HTTPError; network errors
# are OSErrors. Tests inject a stub.
Opener = Callable[[urllib.request.Request, float], Response]


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    # A redirect would carry the bearer token or the authorization code to wherever it points.
    @override
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


_OPENER = urllib.request.build_opener(_RefuseRedirects)


class BoundedResponse:
    """An open response that ends at its deadline, whatever the server does.

    Each socket read waits at most the time left, and a timer shuts the socket down at the deadline, so a read that
    http.client splits into several receives cannot outlast it either. http.client's own exceptions, such as
    `IncompleteRead` from a stream cut off mid-chunk, become `ConnectionError`, so callers handle one family.
    """

    def __init__(self, response: http.client.HTTPResponse, timeout_s: float) -> None:
        self._response = response
        self._deadline = time.monotonic() + timeout_s
        self._buffer = bytearray()
        self._expired = threading.Event()
        try:
            self._socket = _socket(response)
        except RuntimeError:
            response.close()
            raise
        self._timer: threading.Timer | None = None
        if self._socket is not None:
            self._timer = threading.Timer(max(timeout_s, 0.0), self._expire)
            self._timer.daemon = True
            self._timer.start()

    def readline(self) -> bytes:
        searched = 0
        while (end := self._buffer.find(b"\n", searched)) == -1:
            searched = len(self._buffer)
            chunk = self._read_some()
            if not chunk:
                line = bytes(self._buffer)
                self._buffer.clear()
                return line
            self._buffer.extend(chunk)
        line = bytes(self._buffer[: end + 1])
        del self._buffer[: end + 1]
        return line

    def read(self) -> bytes:
        while chunk := self._read_some():
            self._buffer.extend(chunk)
        body = bytes(self._buffer)
        self._buffer.clear()
        return body

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._response.close()

    def _read_some(self) -> bytes:
        # http.client closes the connection once the body is complete; urllib has already released the socket, so
        # that close is final and the socket must not be touched again.
        if self._response.isclosed():
            return self._end()
        remaining = self._deadline - time.monotonic()
        if remaining <= 0 or self._expired.is_set():
            raise TimeoutError
        if self._socket is not None:
            self._socket.settimeout(remaining)
        try:
            chunk = self._response.read1(_READ_SIZE)
        except Exception as exc:
            # A read the deadline's shutdown interrupts can fail in ways no live stream does; it is the timeout.
            if self._expired.is_set() or isinstance(exc, TimeoutError):
                raise TimeoutError from None
            if isinstance(exc, http.client.HTTPException):
                raise ConnectionError(f"the HTTP response broke off ({type(exc).__name__})") from None
            raise
        # The deadline's shutdown reads as a clean end of stream; it must not pass for one.
        if self._expired.is_set():
            raise TimeoutError
        return chunk or self._end()

    def _end(self) -> bytes:
        # http.client's read1 meets an early end of a Content-Length body with b"" and no error, leaving the bytes
        # still owed in `length`; a chunked body raises IncompleteRead instead, and has no length.
        if self._response.length:
            raise ConnectionError("the response broke off before its Content-Length")
        return b""

    def _expire(self) -> None:
        self._expired.set()
        if self._socket is not None:
            with suppress(OSError):
                # The plain socket's shutdown, not SSLSocket's: that one unwraps the SSL object under a concurrent
                # read, which then fails with ValueError or AttributeError instead of seeing the end of the stream.
                socket.socket.shutdown(self._socket, socket.SHUT_RDWR)


def _socket(response: http.client.HTTPResponse) -> socket.socket | None:
    if response.fp is None:
        return None
    # http.client keeps the connection's socket only inside its buffered reader. Without it no read could be bounded
    # by the deadline, so its absence is an error, never a silently weaker deadline.
    found = getattr(getattr(response.fp, "raw", None), "_sock", None)
    if not isinstance(found, socket.socket):
        # http.client's internals changed, which is no caller's type error.
        raise RuntimeError(  # noqa: TRY004
            "http.client keeps no reachable socket; the chatgpt stream deadline cannot be enforced"
        )
    return found


def require_host(url: str) -> None:
    """Refuse any URL but HTTPS to one of the fixed OpenAI hosts on the default port."""
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.hostname not in HOSTS
        or parts.port not in (None, 443)
        or parts.username is not None
        or parts.fragment
    ):
        raise ConfigError(f"refusing to contact {parts.scheme}://{parts.hostname}: chatgpt only talks HTTPS to OpenAI")


def https_open(request: urllib.request.Request, timeout_s: float) -> Response:
    """Open `request`; connecting and each header read wait at most `timeout_s`, the body ends by then."""
    require_host(request.full_url)
    deadline = time.monotonic() + timeout_s
    try:
        response: http.client.HTTPResponse = _OPENER.open(request, timeout=timeout_s)
    except http.client.HTTPException as exc:
        raise ConnectionError(f"OpenAI sent a malformed HTTP response ({type(exc).__name__})") from None
    return BoundedResponse(response, deadline - time.monotonic())


def safe_code(value: object) -> str | None:
    """`value` when it looks like a provider error code; anything else could carry a payload into a message."""
    return value if isinstance(value, str) and _SAFE_CODE.fullmatch(value) else None


def error_code(exc: urllib.error.HTTPError) -> str | None:
    """Read the provider's error code from an error body: `{"error": "code"}` or `{"error": {"code"|"type": ...}}`."""
    try:
        body: object = json.loads(exc.read(_ERROR_BODY_LIMIT))
    except OSError, ValueError, http.client.HTTPException:
        return None
    finally:
        exc.close()
    error = as_object(body).get("error")
    if isinstance(error, str):
        return safe_code(error)
    details = as_object(error)
    return safe_code(details.get("code")) or safe_code(details.get("type"))
