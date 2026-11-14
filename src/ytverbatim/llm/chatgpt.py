"""OpenAI models on the user's ChatGPT plan: one streamed `/v1/responses` call per completion, under one deadline.

The access token comes from `chatgpt_auth` (`ytverbatim llm login`) and is refreshed before each request. There are
no automatic retries: a failure raises, and retry policy belongs to the caller.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping

from ytverbatim.errors import ConfigError, LLMOutputError, LLMUnavailableError
from ytverbatim.llm import chatgpt_auth
from ytverbatim.llm.chatgpt_auth import LOGIN_COMMAND, ChatGPTAuthError
from ytverbatim.llm.chatgpt_http import Opener, Response, error_code, https_open, safe_code
from ytverbatim.llm.jsonio import as_object
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage

PROVIDER = "chatgpt"
DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_EFFORT = "medium"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
RESPONSES_URL = "https://api.openai.com/v1/responses"
_UNAUTHORIZED = 401
_RATE_LIMITED = 429
_SERVER_ERROR = 500


class ChatGPTClient:
    def __init__(
        self,
        *,
        model: str,
        effort: str,
        timeout_s: float = 900.0,
        opener: Opener = https_open,
        clock: Callable[[], float] = time.monotonic,
        reauth: Callable[[], object] | None = None,
    ) -> None:
        """Build the client; `reauth`, if given, signs in again when the stored sign-in can no longer be used.

        Wire `reauth` only when a person is at hand to finish a browser sign-in. Without it, a lapsed sign-in raises.
        """
        if not model:
            raise ConfigError("chatgpt needs a model")
        if effort not in EFFORTS:
            raise ConfigError(f"chatgpt effort must be one of: {', '.join(EFFORTS)}")
        self._model = model
        self._effort = effort
        self._timeout_s = timeout_s
        self._opener = opener
        self._clock = clock
        self._reauth = reauth

    @property
    def id(self) -> str:
        # Effort changes the replies, so it is part of the identity that checkpoints are keyed by.
        return f"{PROVIDER}:{self._model}@{self._effort}"

    def payload(self, request: LLMRequest) -> dict[str, object]:
        # No max_output_tokens: the Responses cap counts reasoning tokens too, so a cap sized for the answer cuts
        # replies off mid-thought. No temperature: reasoning models reject it. `json_schema` is not sent either,
        # since complete_json already states the schema in the prompt. This is the payload the correction eval ran.
        return {
            "model": self._model,
            "input": [
                {"role": "developer", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "store": False,
            "stream": True,
            "reasoning": {"effort": self._effort},
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": False,
        }

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = self._clock()
        deadline = started + self._timeout_s
        token = self._access_token()
        http_request = urllib.request.Request(
            RESPONSES_URL,
            data=json.dumps(self.payload(request), ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        response = self._open(http_request, deadline)
        try:
            text, usage = self._read(response, deadline)
        finally:
            response.close()
        return LLMResponse(text=text, usage=usage, elapsed_s=self._clock() - started)

    def _access_token(self) -> str:
        """Return a usable token, signing in again once through `reauth` when only that can help.

        All of this happens before the request is sent, so nothing is billed twice.
        """
        try:
            return chatgpt_auth.access_token(opener=self._opener)
        except ChatGPTAuthError as exc:
            if self._reauth is None or not exc.needs_sign_in:
                raise self._auth_error(exc) from None
        reauth, self._reauth = self._reauth, None
        try:
            reauth()
            token = chatgpt_auth.access_token(opener=self._opener)
        except ChatGPTAuthError as exc:
            # Left unwired after a failed sign-in, so later calls in the batch do not reopen the browser each time.
            raise self._auth_error(exc) from None
        self._reauth = reauth
        return token

    def _auth_error(self, exc: ChatGPTAuthError) -> LLMUnavailableError:
        return LLMUnavailableError(f"{self.id}: {exc}; usage unknown", code=exc.code)

    def _open(self, request: urllib.request.Request, deadline: float) -> Response:
        try:
            return self._opener(request, self._remaining(deadline))
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from None
        except TimeoutError:
            raise self._timeout() from None
        except (urllib.error.URLError, OSError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            raise LLMUnavailableError(f"{self.id}: cannot reach OpenAI ({reason}); usage unknown") from None

    def _http_error(self, exc: urllib.error.HTTPError) -> LLMUnavailableError:
        code = error_code(exc)
        status = f"HTTP {exc.code}" + (f", {code}" if code else "")
        if exc.code == _UNAUTHORIZED:
            message = f"OpenAI rejected the ChatGPT sign-in ({status}); run `{LOGIN_COMMAND}`"
        elif exc.code == _RATE_LIMITED:
            message = f"plan usage or rate limit reached ({status}); wait, then retry"
        elif exc.code >= _SERVER_ERROR:
            message = f"OpenAI is unavailable ({status})"
        else:
            message = f"the request was refused ({status})"
        return LLMUnavailableError(f"{self.id}: {message}. No retry was attempted.", code=code)

    def _read(self, response: Response, deadline: float) -> tuple[str, Usage]:
        done_items: dict[int, dict[str, object]] = {}
        seen: set[object] = set()
        for event in self._events(response, deadline):
            kind = event.get("type")
            if "output_index" in event:
                seen.add(event.get("output_index"))
            if kind == "response.output_item.done":
                index = event.get("output_index")
                if isinstance(index, int) and not isinstance(index, bool):
                    done_items[index] = as_object(event.get("item"))
            elif kind == "response.completed":
                return self._completed(as_object(event.get("response")), done_items, seen)
            elif kind == "response.failed":
                raise self._failed(as_object(event.get("response")))
            elif kind == "response.incomplete":
                raise self._incomplete(as_object(event.get("response")))
            elif kind == "error":
                details = as_object(event.get("error"))
                code = safe_code(event.get("code")) or safe_code(details.get("code")) or safe_code(details.get("type"))
                raise LLMUnavailableError(
                    f"{self.id}: the stream reported an error ({code or 'no code'}); usage unknown", code=code
                )
        raise LLMUnavailableError(f"{self.id}: the stream ended without response.completed; usage unknown")

    def _events(self, response: Response, deadline: float) -> Iterator[dict[str, object]]:
        """Server-sent events' `data` fields, decoded; the opener ends any read at the deadline."""
        data: list[str] = []
        while True:
            self._remaining(deadline)
            try:
                raw = response.readline()
            except TimeoutError:
                raise self._timeout() from None
            except OSError as exc:
                raise LLMUnavailableError(f"{self.id}: the stream broke off ({exc}); usage unknown") from None
            if not raw:
                if data:
                    yield from self._decode(data)
                return
            try:
                line = raw.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                raise LLMUnavailableError(f"{self.id}: the stream is not UTF-8; usage unknown") from None
            if not line:
                if data:
                    yield from self._decode(data)
                    data = []
                continue
            name, _, value = line.partition(":")
            if name == "data":
                data.append(value.removeprefix(" "))

    def _decode(self, data: list[str]) -> Iterator[dict[str, object]]:
        text = "\n".join(data)
        if text == "[DONE]":
            return
        try:
            event: object = json.loads(text)
        except ValueError:
            event = None
        if not isinstance(event, dict):
            raise LLMUnavailableError(f"{self.id}: the stream sent a malformed event; usage unknown")
        yield as_object(event)

    def _completed(
        self, body: Mapping[str, object], done_items: Mapping[int, dict[str, object]], seen: set[object]
    ) -> tuple[str, Usage]:
        usage = _usage(body)
        if body.get("status") != "completed" or body.get("error") is not None:
            raise LLMUnavailableError(
                f"{self.id}: response.completed reports status {body.get('status')!r}",
                responded=usage is not None,
                usage=usage,
            )
        if usage is None:
            raise LLMUnavailableError(f"{self.id}: the response carries no token usage", responded=True)
        if body.get("model") != self._model:
            raise LLMOutputError(
                f"{self.id}: OpenAI answered with model {body.get('model')!r}, not the requested one",
                responded=True,
                usage=usage,
            )
        # The completed event can arrive with an empty output list; the items it finished arrived one by one, and
        # then they must account for every output index the stream mentioned, with no gap.
        output = body.get("output")
        if isinstance(output, list) and output:
            items = [as_object(item) for item in output]
        elif set(done_items) | seen == set(range(len(done_items))):
            items = [done_items[index] for index in sorted(done_items)]
        else:
            raise LLMOutputError(f"{self.id}: the streamed output items are incomplete", responded=True, usage=usage)
        if any(item.get("status") not in (None, "completed") for item in items):
            raise LLMOutputError(f"{self.id}: an output item did not complete", responded=True, usage=usage)
        text = "".join(
            str(part.get("text", ""))
            for item in items
            if item.get("type") == "message"
            for part in map(as_object, _list(item.get("content")))
            if part.get("type") == "output_text"
        )
        if not text.strip():
            raise LLMOutputError(f"{self.id}: empty reply", responded=True, usage=usage)
        return text, usage

    def _failed(self, body: Mapping[str, object]) -> LLMUnavailableError:
        code = safe_code(as_object(body.get("error")).get("code"))
        usage = _usage(body)
        return LLMUnavailableError(
            f"{self.id}: the response failed ({code or 'no code'}). No retry was attempted.",
            responded=usage is not None,
            usage=usage,
            code=code,
        )

    def _incomplete(self, body: Mapping[str, object]) -> LLMOutputError:
        reason = safe_code(as_object(body.get("incomplete_details")).get("reason"))
        return LLMOutputError(
            f"{self.id}: the response stopped incomplete ({reason or 'no reason given'})",
            responded=True,
            usage=_usage(body),
            code=reason,
        )

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise self._timeout()
        return remaining

    def _timeout(self) -> LLMUnavailableError:
        return LLMUnavailableError(f"{self.id}: no complete response within {self._timeout_s:g}s; usage unknown")


def _usage(body: Mapping[str, object]) -> Usage | None:
    counts = as_object(body.get("usage"))
    input_tokens, output_tokens = counts.get("input_tokens"), counts.get("output_tokens")
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    return None


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []
