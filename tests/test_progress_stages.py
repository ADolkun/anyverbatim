"""`STAGES` and `LLM_STAGES` describe what a real run does: the stages it reaches and the ones that call the LLM."""

from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import pytest
from pipeline_fixture import CAPTIONED, MIXED, FakeLLM, FakeWorker, FakeYtDlp, _fake_proc, isolated, run

from anyverbatim import LLM_STAGES, STAGES, proc
from anyverbatim.asr import runner as asr_runner
from anyverbatim.llm.protocol import LLMRequest, LLMResponse
from anyverbatim.pipeline import ProgressEvent


@dataclass
class StageRecordingLLM:
    """Answers as the fixture's LLM does, and notes the progress stage each request was made in."""

    events: list[ProgressEvent]
    inner: FakeLLM = field(default_factory=FakeLLM)
    stages: list[str] = field(default_factory=list[str])

    @property
    def id(self) -> str:
        return self.inner.id

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.stages.append(self.events[-1].stage)
        return self.inner.complete(request)


def run_video(video_id: str, out: Path) -> tuple[list[ProgressEvent], StageRecordingLLM]:
    events: list[ProgressEvent] = []
    llm = StageRecordingLLM(events)
    result = run(video_id, out, llm, FakeYtDlp(), events)  # type: ignore[arg-type]
    assert result.document is not None, result.outcome
    return events, llm


@pytest.fixture
def offline(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    with (
        isolated(home),
        mock.patch.object(proc, "run", _fake_proc),
        mock.patch.object(asr_runner, "run_worker", FakeWorker()),
    ):
        yield tmp_path / "out"


def test_the_stages_that_call_the_llm_are_the_declared_ones(offline: Path) -> None:
    # MIXED is unpunctuated, corrected, and has a number the captions and the second opinion dispute.
    _, llm = run_video(MIXED, offline)
    assert set(llm.stages) == LLM_STAGES


@pytest.mark.parametrize("video_id", [MIXED, CAPTIONED])
def test_a_run_reaches_declared_stages_in_the_declared_order(offline: Path, video_id: str) -> None:
    events, llm = run_video(video_id, offline)
    stages = [event.stage for event in events]
    assert set(stages) <= set(STAGES)
    assert [STAGES.index(stage) for stage in stages] == sorted(STAGES.index(stage) for stage in stages)
    assert set(llm.stages) <= LLM_STAGES
