"""Caller-side retry of an LLM call that failed with a transient server condition, and nothing else.

The provider decides what is transient (`LLMUnavailableError.transient`). Every other failure, a plan limit, a lapsed
sign-in, a timeout, an unusable reply, raises on the first attempt: a retry would only repeat it, or multiply a
deadline that is already minutes long. Calls are independent, so repeating one is safe.
"""

import random
import time
from collections.abc import Callable

from anyverbatim.errors import LLMUnavailableError
from anyverbatim.llm.protocol import LLMClient, LLMRequest, LLMResponse

TRANSIENT_HTTP_STATUS = frozenset({500, 502, 503, 504})
# About 225 s of waiting in all (15, 30, 60, 120), before jitter: long enough for a brief overload to pass, short
# enough that a real outage still fails the run within minutes.
RETRIES = 4
FIRST_DELAY_S = 15.0
MAX_DELAY_S = 120.0
JITTER = 0.2


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class RetryingClient:
    """`inner`, with each transient failure retried under the module's backoff; `id` is the inner client's own."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        notify: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = _sleep,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._inner = inner
        self._notify = notify
        self._sleep = sleep
        self._uniform = uniform

    @property
    def id(self) -> str:
        return self._inner.id

    @property
    def inner(self) -> LLMClient:
        return self._inner

    def around(self, wrap: Callable[[LLMClient], LLMClient]) -> RetryingClient:
        """Return the same retrying over `wrap(inner)`, so `wrap` sees each attempt rather than the final outcome."""
        return RetryingClient(wrap(self._inner), notify=self._notify, sleep=self._sleep, uniform=self._uniform)

    def complete(self, request: LLMRequest) -> LLMResponse:
        retries_used = 0
        while True:
            try:
                return self._inner.complete(request)
            except LLMUnavailableError as exc:
                if not exc.transient:
                    raise
                if retries_used == RETRIES:
                    exc.args = (f"{exc} (gave up after {retries_used + 1} attempts)",)
                    raise
                delay = min(FIRST_DELAY_S * 2**retries_used, MAX_DELAY_S)
                delay *= self._uniform(1 - JITTER, 1 + JITTER)
                retries_used += 1
                if self._notify is not None:
                    self._notify(
                        f"LLM unavailable ({exc.code or 'server error'}); retry {retries_used}/{RETRIES} "
                        f"in {delay:.0f}s"
                    )
                self._sleep(delay)
