from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from anyverbatim.auth import chromium, source
from anyverbatim.auth.snapshot import Snapshot, SnapshotSummary
from anyverbatim.auth.source import CookieSource
from anyverbatim.errors import AuthError, YtDlpError
from anyverbatim.ytdlp.types import Signal, YtDlpResult

PROFILES = {
    "Default": {"name": "Person 1", "user_name": "alice@example.com"},
    "Work": {"name": "Work", "user_name": "bob@example.com"},
}


@dataclass(frozen=True)
class RunnerCall:
    args: list[str]
    cookies: Path | None


@dataclass
class FakeRunner:
    outcome: YtDlpResult | YtDlpError
    calls: list[RunnerCall] = field(default_factory=list)

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        self.calls.append(RunnerCall(list(args), cookies))
        if isinstance(self.outcome, YtDlpError):
            raise self.outcome
        return self.outcome


@dataclass
class FakeSnapshot:
    """Stands in for the browser snapshot (which would need Chrome and the Keychain)."""

    error: AuthError | None = None
    counts: dict[str, int] = field(default_factory=lambda: {"youtube.com": 12, "google.com": 30})
    sources: list[CookieSource] = field(default_factory=list)

    @contextmanager
    def __call__(self, src: CookieSource, **_: object) -> Generator[Snapshot]:
        self.sources.append(src)
        if self.error is not None:
            raise self.error
        profile = src.profile if isinstance(src, source.BrowserSource) else None
        yield Snapshot(Path("/nonexistent/snapshot.txt"), SnapshotSummary(self.counts, 0, True, (), profile))


class Kit:
    def __init__(self, home: Path, snap: FakeSnapshot) -> None:
        self.home = home
        self.snapshot = snap

    @staticmethod
    def ok(*secondary: Signal, stdout: str = "fakeLikedId\n") -> FakeRunner:
        return FakeRunner(YtDlpResult(0, stdout, "", None, secondary, ()))

    @staticmethod
    def fail(primary: Signal | None, *secondary: Signal) -> FakeRunner:
        return FakeRunner(YtDlpError(primary, secondary, "ERROR: synthetic fakeLikedId", 1))

    @property
    def chrome_root(self) -> Path:
        return chromium.browser_root("chrome")
