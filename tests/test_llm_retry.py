import shutil
import threading
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pipeline_fixture import CAPTIONED, FIXTURE_ROOT, FakeLLM, isolated, run

from ytverbatim import proc
from ytverbatim.asr import runner as asr_runner
from ytverbatim.cli import run as cli_run
from ytverbatim.contract.models import LLMUsage
from ytverbatim.errors import LLMError, LLMOutputError, LLMUnavailableError
from ytverbatim.llm import retry
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.llm.retry import FIRST_DELAY_S, JITTER, MAX_DELAY_S, RETRIES, RetryingClient
from ytverbatim.llm.usage import UsageMeter
from ytverbatim.pipeline import ProgressEvent
from ytverbatim.store import STAGES_DIR
from ytverbatim.ytdlp.types import YtDlpResult

REQUEST = LLMRequest(system="", user="Hi")
OVERLOADED = "server_is_overloaded"


def overloaded(*, usage: Usage | None = None) -> LLMUnavailableError:
    return LLMUnavailableError(
        f"fake:model: the stream reported an error ({OVERLOADED})",
        responded=usage is not None,
        usage=usage,
        code=OVERLOADED,
        transient=True,
    )


@dataclass
class Scripted:
    """Answers each call with the next outcome: an exception is raised, a response is returned."""

    outcomes: list[Exception | LLMResponse]
    name: str = "fake:model"
    calls: int = 0

    @property
    def id(self) -> str:
        return self.name

    def complete(self, request: LLMRequest) -> LLMResponse:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class Waits:
    """The injected sleep and notice sinks; nothing ever really waits."""

    slept: list[float] = field(default_factory=list[float])
    notices: list[str] = field(default_factory=list[str])

    def client(self, inner: Scripted) -> RetryingClient:
        return RetryingClient(inner, notify=self.notices.append, sleep=self.slept.append, uniform=lambda _lo, _hi: 1.0)


def ok(usage: Usage | None = None) -> LLMResponse:
    return LLMResponse(text="{}", usage=usage, elapsed_s=0.0)


def test_a_transient_failure_is_retried_until_the_call_succeeds() -> None:
    waits = Waits()
    inner = Scripted([overloaded(), overloaded(), ok()])
    assert waits.client(inner).complete(REQUEST).text == "{}"
    assert inner.calls == 3
    assert waits.slept == [15.0, 30.0]
    assert waits.notices == [
        f"LLM unavailable ({OVERLOADED}); retry 1/4 in 15s",
        f"LLM unavailable ({OVERLOADED}); retry 2/4 in 30s",
    ]


def test_the_backoff_doubles_to_its_cap_with_jitter_around_each_wait() -> None:
    bounds: list[tuple[float, float]] = []

    def uniform(low: float, high: float) -> float:
        bounds.append((low, high))
        return high

    waits = Waits()
    client = RetryingClient(Scripted([overloaded()]), sleep=waits.slept.append, uniform=uniform)
    with pytest.raises(LLMUnavailableError):
        client.complete(REQUEST)
    assert bounds == [(1 - JITTER, 1 + JITTER)] * RETRIES
    base = [min(FIRST_DELAY_S * 2**n, MAX_DELAY_S) for n in range(RETRIES)]
    assert waits.slept == pytest.approx([delay * (1 + JITTER) for delay in base])
    assert base == [15.0, 30.0, 60.0, 120.0]


def test_exhausted_retries_raise_the_original_error_with_the_attempt_count() -> None:
    waits = Waits()
    error = overloaded()
    inner = Scripted([error])
    with pytest.raises(LLMUnavailableError, match=r"gave up after 5 attempts") as raised:
        waits.client(inner).complete(REQUEST)
    assert raised.value is error
    assert OVERLOADED in str(error) and error.code == OVERLOADED and error.transient is True
    assert inner.calls == RETRIES + 1
    assert waits.slept == [15.0, 30.0, 60.0, 120.0]


@pytest.mark.parametrize(
    "error",
    [
        LLMUnavailableError("plan usage or rate limit reached (HTTP 429)", code="usage_limit_reached"),
        LLMUnavailableError("OpenAI rejected the ChatGPT sign-in (HTTP 401)", code="invalid_token"),
        LLMUnavailableError("no complete response within 900s; usage unknown"),
        LLMUnavailableError("cannot reach OpenAI (connection refused)"),
        LLMOutputError("empty reply", responded=True, usage=Usage(1, 1)),
        LLMError("something else"),
        ValueError("a bug is not an LLM failure"),
    ],
)
def test_other_failures_are_raised_on_the_first_attempt(error: Exception) -> None:
    waits = Waits()
    inner = Scripted([error])
    with pytest.raises(type(error)) as raised:
        waits.client(inner).complete(REQUEST)
    assert raised.value is error
    assert inner.calls == 1
    assert waits.slept == [] and waits.notices == []


def test_a_transient_failure_followed_by_another_kind_stops_there() -> None:
    waits = Waits()
    inner = Scripted([overloaded(), LLMUnavailableError("plan usage or rate limit reached (HTTP 429)")])
    with pytest.raises(LLMUnavailableError, match="rate limit"):
        waits.client(inner).complete(REQUEST)
    assert inner.calls == 2
    assert waits.slept == [15.0]


def test_the_client_keeps_its_identity() -> None:
    assert RetryingClient(Scripted([ok()], name="chatgpt:gpt-sol@medium")).id == "chatgpt:gpt-sol@medium"


def test_no_notify_is_fine() -> None:
    waits = Waits()
    client = RetryingClient(Scripted([overloaded(), ok()]), sleep=waits.slept.append, uniform=lambda _lo, _hi: 1.0)
    assert client.complete(REQUEST).text == "{}"


def test_a_failed_attempt_that_reports_usage_is_counted_once_with_the_success() -> None:
    meter = UsageMeter()
    waits = Waits()
    inner = Scripted([overloaded(usage=Usage(100, 7)), overloaded(), ok(Usage(30, 5))])
    metered = meter.metered("correct", waits.client(inner))
    assert metered.id == "fake:model"
    metered.complete(REQUEST)
    assert inner.calls == 3
    assert meter.summary() == (
        LLMUsage(stage="correct", client="fake:model", calls=2, input_tokens=130, output_tokens=12),
    )
    assert waits.slept == [15.0, 30.0]


def test_usage_of_attempts_is_counted_when_the_retries_run_out() -> None:
    meter = UsageMeter()
    inner = Scripted([overloaded(usage=Usage(10, 1))])
    with pytest.raises(LLMUnavailableError):
        meter.metered("numeric", Waits().client(inner)).complete(REQUEST)
    assert meter.summary() == (
        LLMUsage(stage="numeric", client="fake:model", calls=RETRIES + 1, input_tokens=50, output_tokens=5),
    )


def test_a_plain_client_is_metered_as_before() -> None:
    meter = UsageMeter()
    meter.metered("correct", Scripted([ok(Usage(3, 4))])).complete(REQUEST)
    assert meter.summary() == (
        LLMUsage(stage="correct", client="fake:model", calls=1, input_tokens=3, output_tokens=4),
    )


def test_concurrent_calls_retry_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Barrier(2, timeout=5)
    waits = Waits()

    @dataclass
    class Flaky:
        failed: set[str] = field(default_factory=set[str])
        id: str = "fake:model"

        def complete(self, request: LLMRequest) -> LLMResponse:
            if request.user not in self.failed:
                self.failed.add(request.user)
                raise overloaded()
            return ok()

    def sleep(seconds: float) -> None:
        waits.slept.append(seconds)
        started.wait()

    client = RetryingClient(Flaky(), sleep=sleep, uniform=lambda _lo, _hi: 1.0)
    results: list[str] = []
    threads = [
        threading.Thread(target=lambda user=user: results.append(client.complete(LLMRequest("", user)).text))
        for user in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert results == ["{}", "{}"]
    assert waits.slept == [15.0, 15.0]


# The pipeline: a transient failure inside a stage is retried, and tells the progress callback.


@dataclass
class FlakyOnce(FakeLLM):
    failures: int = 2

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.failures:
            self.failures -= 1
            raise overloaded()
        return super().complete(request)


class RefusingYtDlp:
    def run(self, args: object, *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        raise AssertionError(f"the rerun asked yt-dlp for {label}")


def refuse(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("the rerun ran a subprocess or the ASR worker")


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "home"
    home.mkdir()
    copy = tmp_path / "out"
    shutil.copytree(FIXTURE_ROOT, copy)
    monkeypatch.setattr(proc, "run", refuse)
    monkeypatch.setattr(asr_runner, "run_worker", refuse)
    with isolated(home):
        yield copy


def forget_correction(vdir: Path) -> None:
    """Leave the correct stage to run again, with no stored batch to answer its call."""
    for name in ("context.json", "context.md", f"{STAGES_DIR}/context.json", f"{STAGES_DIR}/correct.json"):
        (vdir / name).unlink()
    shutil.rmtree(vdir / STAGES_DIR / "correct")


def test_a_stage_retries_a_transient_failure_and_reports_each_retry(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(retry, "time", types.SimpleNamespace(sleep=slept.append))
    vdir = out / CAPTIONED
    forget_correction(vdir)
    llm = FlakyOnce()
    events: list[ProgressEvent] = []
    result = run(CAPTIONED, out, llm, RefusingYtDlp(), events)
    assert result.document is not None, result.outcome
    assert len(slept) == 2 and 12 <= slept[0] <= 18
    retries = [event for event in events if event.retry]
    assert [(event.id, event.stage) for event in retries] == [(CAPTIONED, "correct")] * 2
    assert retries[0].detail.startswith(f"LLM unavailable ({OVERLOADED}); retry 1/4 in ")
    assert [event.stage for event in events if not event.retry].count("correct") == 1
    assert llm.requests, "the call went through after the retries"


def test_the_run_fails_loud_once_the_retries_are_spent(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(retry, "time", types.SimpleNamespace(sleep=slept.append))
    vdir = out / CAPTIONED
    forget_correction(vdir)
    result = run(CAPTIONED, out, FlakyOnce(failures=99), RefusingYtDlp())
    assert result.document is None
    assert isinstance(result.error, LLMUnavailableError)
    assert "gave up after 5 attempts" in str(result.error)
    assert len(slept) == RETRIES


# The command line: --quiet keeps retry notices out of a scheduled job's output.


def test_a_retry_notice_prints_like_any_progress_line(capsys: pytest.CaptureFixture[str]) -> None:
    event = ProgressEvent("rec-1", "correct", "LLM unavailable (server_is_overloaded); retry 1/4 in 17s", retry=True)
    cli_run._print_progress(event)
    assert capsys.readouterr().err == "… rec-1  correct LLM unavailable (server_is_overloaded); retry 1/4 in 17s\n"


def test_quiet_drops_retry_notices_but_keeps_stage_lines(capsys: pytest.CaptureFixture[str]) -> None:
    notice = ProgressEvent("rec-1", "correct", "LLM unavailable (server_is_overloaded); retry 1/4 in 17s", retry=True)
    cli_run._print_progress(notice, quiet=True)
    assert capsys.readouterr().err == ""
    cli_run._print_progress(ProgressEvent("rec-1", "correct", "chatgpt:gpt-sol@medium"), quiet=True)
    assert capsys.readouterr().err == "… rec-1  correct chatgpt:gpt-sol@medium\n"
