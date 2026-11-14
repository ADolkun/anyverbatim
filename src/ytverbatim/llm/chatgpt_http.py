"""HTTPS to OpenAI's two fixed hosts for the chatgpt provider: stdlib urllib, redirects refused, never followed."""

import http.client
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from email.message import Message
from typing import IO, Protocol, override

from ytverbatim.errors import ConfigError
from ytverbatim.llm.jsonio import as_object

AUTH_HOST = "auth.openai.com"
API_HOST = "api.openai.com"
HOSTS = frozenset({AUTH_HOST, API_HOST})
_ERROR_BODY_LIMIT = 65_536
_SAFE_CODE = re.compile(r"[a-z][a-z0-9_.]{0,95}")


class Response(Protocol):
    def read(self) -> bytes: ...

    def readline(self) -> bytes: ...

    def settimeout(self, seconds: float) -> None:
        """Bound the next read; a read that waits longer raises `TimeoutError`."""
        ...

    def close(self) -> None: ...


# (request, timeout_s) -> the open response. A status other than 2xx, a refused redirect included, raises
# urllib.error.HTTPError; network errors propagate as urllib raises them. Tests inject a stub.
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


class _UrllibResponse:
    def __init__(self, response: http.client.HTTPResponse) -> None:
        self._response = response

    def read(self) -> bytes:
        return self._response.read()

    def readline(self) -> bytes:
        return self._response.readline()

    def settimeout(self, seconds: float) -> None:
        # http.client keeps the connection's socket only inside its buffered reader. Bounding each read by the time
        # left is what turns urllib's per-operation timeout into one deadline over the whole stream.
        raw = getattr(getattr(self._response, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        if isinstance(sock, socket.socket):
            sock.settimeout(seconds)

    def close(self) -> None:
        self._response.close()


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
    require_host(request.full_url)
    return _UrllibResponse(_OPENER.open(request, timeout=timeout_s))


def safe_code(value: object) -> str | None:
    """`value` when it looks like a provider error code; anything else could carry a payload into a message."""
    return value if isinstance(value, str) and _SAFE_CODE.fullmatch(value) else None


def error_code(exc: urllib.error.HTTPError) -> str | None:
    """Read the provider's error code from an error body: `{"error": "code"}` or `{"error": {"code"|"type": ...}}`."""
    try:
        body: object = json.loads(exc.read(_ERROR_BODY_LIMIT))
    except (OSError, ValueError):
        return None
    finally:
        exc.close()
    error = as_object(body).get("error")
    if isinstance(error, str):
        return safe_code(error)
    details = as_object(error)
    return safe_code(details.get("code")) or safe_code(details.get("type"))
