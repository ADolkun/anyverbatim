"""Rerunning the committed pipeline fixture computes nothing and reproduces every stored file byte for byte."""

import hashlib
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
    generate,
    isolated,
    run,
)

from ytverbatim import proc
from ytverbatim.asr import runner as asr_runner
from ytverbatim.store import STAGES_DIR
from ytverbatim.ytdlp.types import YtDlpResult

EVENTS = {
    MIXED: ["fetch", "asr", "second_opinion", "normalize", "punctuate", "correct", "numeric", "numerals", "write"],
    CAPTIONED: ["fetch", "captions", "normalize", "punctuate", "correct", "numeric", "write"],
}
# The 1.1 document and context record a run made of MIXED before contract 2.0 (context record version 3).
V1_1_DOCUMENT = Path(__file__).parent / "fixtures" / "contract" / "context_v1_1.json"
V1_1_RECORD = {
    "stage_version": "3+stages-v1",
    "input_hash": "76045c8589d2b6bdb50e20d59a628218db155d059ac2d99f797c686ad723746d",
    "output": {"sha256": "458004b895f1cca1b9f4eb14a010a8e2155a227ffbc1fa6c44d408900710421a"},
}
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


def test_regenerating_computes_every_stage_into_the_committed_keys_and_bytes(tmp_path: Path) -> None:
    regen = tmp_path / "regen"
    events = generate(regen)
    for video_id in (MIXED, CAPTIONED):
        assert [e.stage for e in events[video_id]] == EVENTS[video_id]
        assert stage_keys(regen / video_id) == stage_keys(FIXTURE_ROOT / video_id)
        assert files(regen / video_id) == files(FIXTURE_ROOT / video_id)


def test_a_1_1_run_is_upgraded_by_rebuilding_only_the_context_record(out: Path) -> None:
    vdir = out / MIXED
    old = V1_1_DOCUMENT.read_bytes()
    assert hashlib.sha256(old).hexdigest() == V1_1_RECORD["output"]["sha256"]
    (vdir / "context.json").write_bytes(old)
    (vdir / STAGES_DIR / "context.json").write_text(json.dumps(V1_1_RECORD), encoding="utf-8")
    before = {name: key for name, key in stage_keys(vdir).items() if name != "context"}
    llm = FakeLLM()
    result = run(MIXED, out, llm, RefusingYtDlp())
    assert result.document is not None and result.document.schema_version == "2.0"
    assert result.outcome.reused != ("context",)
    timings = {t.stage: t.reused for t in result.timings if t.stage not in ("fetch", "write")}
    assert timings == dict.fromkeys(STAGES[MIXED], True)
    assert llm.requests == []
    assert {name: key for name, key in stage_keys(vdir).items() if name != "context"} == before
    assert files(vdir) == files(FIXTURE_ROOT / MIXED)
