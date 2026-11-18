from datetime import datetime
from typing import TYPE_CHECKING, Literal

from anyverbatim.auth.state import AuthState
from anyverbatim.ytdlp.types import Signal

if TYPE_CHECKING:
    from anyverbatim.llm.protocol import Usage


__all__ = [
    "AnyverbatimError",
    "AuthError",
    "AuthState",
    "BatchStopError",
    "BusyError",
    "ConfigError",
    "ContractError",
    "DependencyError",
    "EngineError",
    "EngineMemoryError",
    "LLMError",
    "LLMOutputError",
    "LLMUnavailableError",
    "MediaValidationError",
    "ProcMemoryExceeded",
    "ProcTimeout",
    "Signal",
    "UnavailableError",
    "YtDlpError",
]


class AnyverbatimError(Exception):
    """Base class for every error raised on purpose by this package.

    `fix` is a one-line, actionable remedy for the caller to show; empty when there is nothing specific to say.
    `stage` names where a failure happened when it is known: `info`, `captions`, `audio` or `video` while fetching a
    video, otherwise the progress stage that was running (`asr`, `correct`, ...); None when unknown.
    """

    fix: str = ""
    stage: str | None = None


class ConfigError(AnyverbatimError):
    pass


class DependencyError(AnyverbatimError):
    pass


class ProcTimeout(AnyverbatimError):  # noqa: N818
    def __init__(self, command: str, timeout_s: float) -> None:
        super().__init__(f"{command} did not finish within {timeout_s:g}s and was killed")
        self.command = command
        self.timeout_s = timeout_s


MemoryTrigger = Literal["footprint", "available"]
_GIB = 1 << 30


class ProcMemoryExceeded(AnyverbatimError):  # noqa: N818
    """A child killed by the memory watch: its own footprint passed the cap, or the machine ran short of memory."""

    def __init__(
        self,
        command: str,
        trigger: MemoryTrigger,
        *,
        footprint_bytes: int | None,
        cap_bytes: int,
        available_percent: int,
        floor_percent: int,
    ) -> None:
        used = "unknown" if footprint_bytes is None else f"{footprint_bytes / _GIB:.2f} GiB"
        if trigger == "footprint":
            detail = (
                f"its physical footprint reached {used}, over the {cap_bytes / _GIB:.2f} GiB cap "
                f"(system memory available: {available_percent}%, floor {floor_percent}%)"
            )
        else:
            detail = (
                f"system memory available fell to {available_percent}%, under the {floor_percent}% floor "
                f"(its physical footprint: {used}, cap {cap_bytes / _GIB:.2f} GiB)"
            )
        super().__init__(f"{command} was killed: {detail}")
        self.command = command
        self.trigger: MemoryTrigger = trigger
        self.footprint_bytes = footprint_bytes
        self.cap_bytes = cap_bytes
        self.available_percent = available_percent
        self.floor_percent = floor_percent


class YtDlpError(AnyverbatimError):
    def __init__(
        self,
        primary: Signal | None,
        secondary: tuple[Signal, ...],
        preview: str,
        returncode: int,
    ) -> None:
        cause = primary.value if primary is not None else "unclassified"
        extra = f" (also: {', '.join(s.value for s in secondary)})" if secondary else ""
        super().__init__(f"yt-dlp failed with exit {returncode}: {cause}{extra}\n{preview}")
        self.primary = primary
        self.secondary = secondary
        self.preview = preview
        self.returncode = returncode


class AuthError(AnyverbatimError):
    def __init__(self, state: AuthState, detail: str = "") -> None:
        super().__init__(f"{state.value}: {detail}" if detail else state.value)
        self.state = state
        self.detail = detail


class BatchStopError(AnyverbatimError):
    def __init__(self, signal: Signal, until: datetime | None) -> None:
        suffix = f"; wait until {until.isoformat(timespec='seconds')}" if until is not None else ""
        super().__init__(f"batch stopped by {signal.value}{suffix}")
        self.signal = signal
        self.until = until


class UnavailableError(AnyverbatimError):
    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"video unavailable ({kind}): {detail}" if detail else f"video unavailable ({kind})")
        self.kind = kind


class MediaValidationError(AnyverbatimError):
    pass


class EngineError(AnyverbatimError):
    pass


class EngineMemoryError(EngineError):
    """The memory watch killed the ASR worker; the batch stops, since the next video would hit the same limit."""

    def __init__(self, detail: str, fix: str, cause: ProcMemoryExceeded) -> None:
        super().__init__(f"{detail}; {fix}")
        self.detail = detail
        self.fix = fix
        self.trigger: MemoryTrigger = cause.trigger


class LLMError(AnyverbatimError):
    """A failed LLM call.

    `responded` is True when a reply arrived, and was billed, but could not be used, for example when it was cut off
    at the token limit; `usage` then holds its tokens, or None when the transport did not report them. `code` is the
    provider's own error code (`server_is_overloaded`, say) when it reported one.
    """

    def __init__(
        self, message: str, *, responded: bool = False, usage: "Usage | None" = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.responded = responded
        self.usage = usage
        self.code = code


class LLMUnavailableError(LLMError):
    """The LLM could not be used. `transient` marks a server-side overload or error that passes by itself.

    Only the provider knows its own codes, so it sets `transient`; the pipeline retries such a call a few times with
    backoff before giving up. A library `LLMClient` opts in by raising this with `transient=True`.
    """

    def __init__(
        self,
        message: str,
        *,
        responded: bool = False,
        usage: "Usage | None" = None,
        code: str | None = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message, responded=responded, usage=usage, code=code)
        self.transient = transient


class LLMOutputError(LLMError):
    pass


class BusyError(AnyverbatimError):
    pass


class ContractError(AnyverbatimError):
    """A `context.json` that cannot be read: bad JSON, an unsupported version, or a contract violation."""


def stderr_preview(text: str, *, head: int = 800, tail: int = 800) -> str:
    """Head and tail of long output; yt-dlp prints the cause at the top and the exit detail at the bottom."""
    text = text.strip()
    if len(text) <= head + tail:
        return text
    dropped = len(text) - head - tail
    return f"{text[:head]}\n... [truncated {dropped} chars] ...\n{text[-tail:]}"
