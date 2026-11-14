"""Rerunning the committed pipeline fixture computes nothing and reproduces every stored file byte for byte."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from pipeline_fixture import (
    CAPTIONED,
    FIXTURE_ROOT,
    MIXED,
    FakeLLM,
    comparable,
    fixture_files,
    isolated,
    run,
)

from ytverbatim import proc
from ytverbatim.asr import runner as asr_runner
from ytverbatim.store import STAGES_DIR
from ytverbatim.ytdlp.types import YtDlpResult

STAGES = {
    MIXED: {"asr", "second_opinion", "normalize", "punctuate", "correct", "numeric", "numerals"},
    CAPTIONED: {"captions", "normalize", "punctuate", "correct", "numeric"},
}


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


def files(vdir: Path) -> dict[str, bytes]:
    return {path.relative_to(vdir).as_posix(): comparable(path) for path in fixture_files(vdir)}


def stage_keys(vdir: Path) -> dict[str, str]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))["input_hash"]
        for path in sorted((vdir / STAGES_DIR).glob("*.json"))
    }


@pytest.mark.parametrize("video_id", [MIXED, CAPTIONED])
def test_a_rerun_reuses_the_finished_document(out: Path, video_id: str) -> None:
    llm = FakeLLM()
    result = run(video_id, out, llm, RefusingYtDlp())
    assert result.document is not None, result.outcome
    assert result.outcome.reused == ("context",)
    assert llm.requests == []
    assert files(out / video_id) == files(FIXTURE_ROOT / video_id)


@pytest.mark.parametrize("video_id", [MIXED, CAPTIONED])
def test_rebuilding_the_document_reuses_every_stage_and_reproduces_every_file(out: Path, video_id: str) -> None:
    vdir = out / video_id
    for name in ("context.json", "context.md", f"{STAGES_DIR}/context.json"):
        (vdir / name).unlink()
    llm = FakeLLM()
    result = run(video_id, out, llm, RefusingYtDlp())
    assert result.document is not None, result.outcome
    timings = {t.stage: t.reused for t in result.timings if t.stage not in ("fetch", "write")}
    assert timings == dict.fromkeys(STAGES[video_id], True)
    assert llm.requests == []
    assert stage_keys(vdir) == stage_keys(FIXTURE_ROOT / video_id)
    assert files(vdir) == files(FIXTURE_ROOT / video_id)
