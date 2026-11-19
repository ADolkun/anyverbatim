"""The full `run` pipeline with a fake yt-dlp, a fake ASR worker and a fake LLM: stages, store, LLM and captions."""

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from chatgpt_fakes import store_account
from click.testing import CliRunner

from anyverbatim import config, pipeline, stages, store
from anyverbatim.acquire import session
from anyverbatim.asr import registry
from anyverbatim.asr import runner as asr_runner
from anyverbatim.asr.models import HYPOTHESES_FILE, METRICS_FILE, ChunkHypothesis, RunMetrics, WorkerRequest
from anyverbatim.asr.runner import WorkerResult
from anyverbatim.asr.worker import _claim_output as claim_output
from anyverbatim.cli import main
from anyverbatim.contract import ContextDocument, Word, YouTubeSource, load_document
from anyverbatim.errors import ConfigError, DependencyError, EngineError, LLMUnavailableError
from anyverbatim.llm import chatgpt_models, claude_cli
from anyverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from anyverbatim.pipeline import ProgressEvent, VideoResult, process_many
from anyverbatim.text import clean, guards, numbers, numerals, numeric
from anyverbatim.text import punctuate as punctuate_module
from anyverbatim.ytdlp.types import Signal

ZH = "zhVideo0001"
EN = "enVideo0001"
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
# Whisper-style Mandarin: no punctuation, a space at each pause. The third phrase holds the number in dispute.
ZH_PHRASES = ("大家好欢迎回来", "今天聊聊这只股票", "上周涨了十五个百分点", "非常强劲")
ZH_AUTO = {
    "events": [
        {
            "tStartMs": 0,
            "dDurationMs": 12000,
            "segs": [
                {"utf8": "大家好欢迎回来"},
                {"utf8": "今天聊聊这只股票", "tOffsetMs": 3000},
                {"utf8": "上周涨了五十个百分点", "tOffsetMs": 6000},
                {"utf8": "非常强劲", "tOffsetMs": 9000},
            ],
        }
    ]
}
EN_MANUAL = {
    "events": [
        {"tStartMs": 5000, "dDurationMs": 3000, "segs": [{"utf8": "Markets rose today."}]},
        {"tStartMs": 60000, "dDurationMs": 3000, "segs": [{"utf8": "Then they fell sharply."}]},
    ]
}


def zh_rows(request: WorkerRequest, *, pause: float = 0.5) -> list[ChunkHypothesis]:
    words = tuple(
        Word(text=phrase, start=3.0 * index, end=3.0 * index + 3.0 - pause, conf=0.9)
        for index, phrase in enumerate(ZH_PHRASES)
    )
    return [
        ChunkHypothesis(
            hypothesis=request.engine.spec.id,
            file=request.files[0].id,
            chunk=0,
            start=0.0,
            end=12.0,
            text=" ".join(ZH_PHRASES),
            language="zh",
            words=words,
        )
    ]


# Whisper's usual English failure: the opening stretch lowercase and unpunctuated, the rest punctuated.
EN_OPENING = (
    "so today we are going to talk about how the market reacts when the fed changes its mind about rates "
    "and why that matters for anyone who holds stocks or bonds right now because the next few months could be "
    "very different from the last few"
)
EN_REST = "Markets rose today. Then they fell sharply. Nobody was surprised by that outcome at all."


def en_rows(request: WorkerRequest) -> list[ChunkHypothesis]:
    texts = (EN_OPENING, EN_REST)
    return [
        ChunkHypothesis(
            hypothesis=request.engine.spec.id,
            file=request.files[0].id,
            chunk=index,
            start=10.0 * index,
            end=10.0 * index + 8.0,
            text=text,
            language="en",
        )
        for index, text in enumerate(texts)
    ]


@dataclass
class FakeWorker:
    """Claims its output folder exactly as the real worker does, then writes the rows a real decode would."""

    calls: list[WorkerRequest] = field(default_factory=list[WorkerRequest])
    # Raise this after claiming the folder, as a worker killed mid-decode would.
    error: EngineError | None = None
    # Seconds between the Mandarin phrases; from 0.6 s each phrase becomes a stretch of its own.
    pause: float = 0.5

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        claim_output(out, request)
        (out / METRICS_FILE).unlink(missing_ok=True)
        if self.error is not None:
            raise self.error
        rows = en_rows(request) if request.files[0].id == EN else zh_rows(request, pause=self.pause)
        (out / HYPOTHESES_FILE).write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
        metrics = RunMetrics(
            hypothesis=request.engine.spec.id,
            files=1,
            chunks_decoded=1,
            chunks_skipped=0,
            load_s=0.0,
            decode_s=0.0,
            audio_s=12.0,
            peak_memory_bytes=None,
            max_rss_bytes=0,
        )
        (out / METRICS_FILE).write_text(metrics.model_dump_json(), encoding="utf-8")
        return WorkerResult(out / HYPOTHESES_FILE, metrics, 0, 0.0)


@dataclass
class FakeLLM:
    """Punctuates by turning pause spaces into commas and ending with 。; picks the caption reading in disputes."""

    model: str = "fake-model-1"
    fail: bool = False
    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])

    @property
    def id(self) -> str:
        return f"fake:{self.model}"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.fail:
            raise LLMUnavailableError(f"{self.id}: synthetic outage")
        if "SEGMENTS:" in request.user and '"items"' in request.user:
            items = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
            reply: object = {"items": [{"i": it["i"], "text": _punctuated(it["text"])} for it in items]}
        elif "SEGMENTS:" in request.user:
            reply = {"fixed": []}
        else:
            items = json.loads(request.user.split("ITEMS:\n", 1)[1])
            reply = {
                "verdicts": [
                    {"id": it["id"], "value": it["readings"][-1], "confidence": "high", "reason": "context"}
                    for it in items
                ]
            }
        return LLMResponse(text=json.dumps(reply, ensure_ascii=False), usage=Usage(100, 20), elapsed_s=0.0)

    def stages(self) -> list[str]:
        return [
            "numeric" if "ITEMS:" in r.user else "punctuate" if '"items"' in r.user else "correct"
            for r in self.requests
        ]


def _punctuated(text: str) -> str:
    if text.isascii():
        return text[:1].upper() + text[1:] + "."
    return text.replace(" ", "，") + "。"


@pytest.fixture(autouse=True)
def engine_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "find_spec", lambda _name: object())


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> FakeWorker:
    fake = FakeWorker()
    monkeypatch.setattr(asr_runner, "run_worker", fake)
    return fake


def zh_yt(kit):
    yt = kit.yt()
    yt.language[ZH] = "zh"
    yt.tracks[ZH] = {"zh-orig": json.dumps(ZH_AUTO, ensure_ascii=False)}
    return yt


def run(kit, yt, targets: Sequence[str] = (ZH,), *, options: config.Options | None = None, **kwargs: object):
    results: list[VideoResult] = []
    events: list[ProgressEvent] = []
    report = process_many(
        targets,
        options or config.load_options(),
        out=kit.out,
        runner_factory=lambda _sleep: yt,
        sleep=kit.slept.append,
        clock=lambda: NOW,
        progress=events.append,
        on_video=results.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return report, results, events


def only_document(results: list[VideoResult]) -> ContextDocument:
    (result,) = results
    assert result.document is not None, result.outcome
    return result.document


def only_youtube(results: list[VideoResult]) -> YouTubeSource:
    source = only_document(results).source
    assert isinstance(source, YouTubeSource)
    return source


def stored(result: VideoResult) -> dict[str, bool]:
    return {t.stage: t.reused for t in result.timings if t.stage not in ("fetch", "write")}


def ffmpeg_calls(kit) -> int:
    return sum(1 for call in kit.proc.calls if Path(call[0]).name == "ffmpeg")


def test_end_to_end_with_a_fake_llm_punctuates_verifies_numbers_and_meters_tokens(session_kit, worker) -> None:
    llm = FakeLLM()
    report, results, events = run(session_kit, zh_yt(session_kit), llm=llm)
    assert report.exit_code == 0
    doc = only_document(results)
    vdir = session_kit.out / ZH
    assert load_document(vdir / "context.json") == doc

    assert doc.transcript.punctuation == "llm"
    assert doc.transcript.language == "zh"
    text = "".join(s.text for s in doc.segments)
    assert text == "大家好欢迎回来，今天聊聊这只股票，上周涨了50个百分点，非常强劲。"
    fixed = [(s, f) for s in doc.segments for f in s.flags if f.kind == "numeric_fixed"]
    ((segment, flag),) = fixed
    assert segment.text[flag.char_start : flag.char_end] == "50个百分点"
    assert [r.source for r in flag.readings] == ["asr:w-prod", "caption_auto"]
    assert ("numeric", "15", "50") in [(e.stage, e.before, e.after) for e in segment.edits]
    # The fixed word no longer matches its engine word, so its time is interpolated (JSON only).
    assert doc.quality.flags_by_kind == {"numeric_fixed": 1, "timing_interpolated": 1}
    assert {(u.stage, u.client, u.calls, u.input_tokens) for u in doc.llm_usage} == {
        ("punctuation", llm.id, 1, 100),
        ("correction", llm.id, 1, 100),
        ("numeric", llm.id, 1, 100),
    }

    markdown = (vdir / "context.md").read_text(encoding="utf-8")
    assert "50个百分点[^1]" in markdown
    assert "五十个百分点: context; was 15" in markdown
    assert "punctuation: llm" in markdown
    assert "timing_interpolated" not in markdown
    assert [e.stage for e in events] == [
        "fetch",
        "asr",
        "normalize",
        "punctuate",
        "correct",
        "numeric",
        "numerals",
        "write",
    ]
    assert results[0].notices == ()
    assert not (vdir / "audio.wav").exists()
    assert {".lock", "audio.webm", "context.json", "context.md", ".stages"} <= {p.name for p in vdir.iterdir()}


def test_without_an_llm_the_stages_degrade_and_say_so(session_kit, worker) -> None:
    _, results, _ = run(session_kit, zh_yt(session_kit))
    doc = only_document(results)
    assert doc.transcript.punctuation == "none"
    assert doc.llm_usage == ()
    assert len(doc.segments) >= 1
    assert "，" not in "".join(s.text for s in doc.segments)
    (disputed,) = [f for s in doc.segments for f in s.flags]
    assert disputed.kind == "numeric_disputed"
    assert "no LLM" in disputed.note
    notices = results[0].notices
    assert len(notices) == 2
    assert notices[0].startswith("punctuation not restored")
    assert notices[1].startswith("1 disputed number(s)")
    assert set(notices) <= set(doc.quality.warnings)


def test_an_option_change_recomputes_only_the_stages_downstream_of_it(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    _, first, _ = run(session_kit, yt)
    assert stored(first[0]) == {
        "asr": False,
        "normalize": False,
        "punctuate": False,
        "numeric": False,
        "numerals": False,
    }
    converted = ffmpeg_calls(session_kit)

    llm = FakeLLM()
    _, second, events = run(session_kit, yt, llm=llm)
    assert stored(second[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": False,
        "correct": False,
        "numeric": False,
        "numerals": False,
    }
    assert len(worker.calls) == 1 and ffmpeg_calls(session_kit) == converted, "no audio once the ASR is stored"
    assert len(yt.labels("audio")) == 1
    assert "asr" not in [e.stage for e in events]
    assert only_document(second).transcript.punctuation == "llm"

    asked = len(llm.requests)
    no_chapters = config.load_options(overrides={"text": {"chapters": "none"}})
    _, third, _ = run(session_kit, yt, options=no_chapters, llm=llm)
    assert stored(third[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": True,
        "correct": True,
        "numeric": True,
        "numerals": True,
    }
    assert len(llm.requests) == asked
    assert only_document(third).chapters == ()

    keep_script = config.load_options(overrides={"text": {"script": "keep"}})
    _, fourth, _ = run(session_kit, yt, options=keep_script, llm=llm)
    # The text was Simplified already, so normalize's output is unchanged and nothing after it reruns.
    assert stored(fourth[0]) == {
        "asr": True,
        "normalize": False,
        "punctuate": True,
        "correct": True,
        "numeric": True,
        "numerals": True,
    }
    assert len(llm.requests) == asked


def test_unchanged_options_reuse_the_document_without_any_stage(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    calls, asked = len(yt.calls), len(llm.requests)
    _, again, events = run(session_kit, yt, llm=FakeLLM())
    assert [e.stage for e in events] == ["skip"]
    assert again[0].outcome.reused == ("context",)
    assert len(yt.calls) == calls
    assert len(llm.requests) == asked

    (session_kit.out / ZH / "context.json").write_text("{}", encoding="utf-8")
    _, edited, events = run(session_kit, yt, llm=FakeLLM())
    assert "skip" not in [e.stage for e in events], "a context.json that is not the one written is rebuilt"
    assert all(stored(edited[0]).values())
    assert len(worker.calls) == 1


def test_an_llm_failure_fails_the_video_and_keeps_every_finished_stage(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    report, results, _ = run(session_kit, yt, llm=FakeLLM(fail=True))
    assert report.exit_code == 1
    assert results[0].outcome.error == "llm"
    assert "synthetic outage" in results[0].outcome.detail
    stages_dir = session_kit.out / ZH / ".stages"
    assert {p.name for p in stages_dir.iterdir() if p.is_file()} == {"asr.json", "normalize.json"}
    assert not (session_kit.out / ZH / "context.json").exists()

    _, retry, _ = run(session_kit, yt, llm=FakeLLM())
    assert stored(retry[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": False,
        "correct": False,
        "numeric": False,
        "numerals": False,
    }
    assert len(worker.calls) == 1
    assert len(yt.labels("audio")) == 1


def test_force_discards_every_stored_stage(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt, llm=FakeLLM())
    llm = FakeLLM()
    _, forced, _ = run(session_kit, yt, llm=llm, force=True)
    assert not any(stored(forced[0]).values())
    assert len(worker.calls) == 2
    assert llm.stages() == ["punctuate", "correct", "numeric"]


def test_captions_mode_trusts_the_manual_track_and_skips_the_audio(session_kit, worker) -> None:
    yt = session_kit.yt()
    yt.tracks[EN] = {"en": json.dumps(EN_MANUAL)}
    options = config.load_options(overrides={"asr": {"source": "captions"}})
    _, results, events = run(session_kit, yt, [EN], options=options)
    doc = only_document(results)
    assert worker.calls == []
    assert yt.labels("audio") == []
    assert ffmpeg_calls(session_kit) == 0
    assert [e.stage for e in events] == ["fetch", "captions", "normalize", "punctuate", "numeric", "write"]
    assert doc.transcript.runs[0].kind == "caption_manual"
    assert doc.transcript.punctuation == "caption"
    assert [s.text for s in doc.segments] == ["Markets rose today.", "Then they fell sharply."]
    assert [s.chapter for s in doc.segments] == [0, 1]
    assert "Transcript: manual captions" in (session_kit.out / EN / "context.md").read_text(encoding="utf-8")


def test_correction_needs_an_llm_and_runs_with_one(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    correct = config.load_options(overrides={"text": {"correct": True}})
    with pytest.raises(ConfigError, match="correction needs an LLM"):
        run(session_kit, yt, options=correct)
    assert yt.calls == []

    llm = FakeLLM()
    _, results, _ = run(session_kit, yt, options=correct, llm=llm)
    assert only_document(results).transcript.correction_llm == llm.id
    assert llm.stages() == ["punctuate", "correct", "numeric"]


def with_correct(value: bool | None) -> config.Options:
    return config.load_options(overrides={"text": {"correct": value}})


def test_correction_runs_by_default_when_there_is_an_llm(session_kit, worker) -> None:
    llm = FakeLLM()
    _, results, events = run(session_kit, zh_yt(session_kit), llm=llm)
    assert only_document(results).transcript.correction_llm == llm.id
    assert llm.stages() == ["punctuate", "correct", "numeric"]
    assert "correct" in [e.stage for e in events]


def test_correction_is_skipped_without_an_llm_and_the_notice_says_so(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    _, results, events = run(session_kit, yt)
    doc = only_document(results)
    assert doc.transcript.correction_llm is None
    assert "correct" not in [e.stage for e in events]
    notice = results[0].notices[0]
    assert notice.startswith("punctuation not restored")
    assert "correction is skipped" in notice
    assert notice in doc.quality.warnings
    assert len(doc.quality.warnings) == len(results[0].notices)


def test_an_explicit_off_skips_correction_even_with_an_llm(session_kit, worker) -> None:
    llm = FakeLLM()
    _, results, events = run(session_kit, zh_yt(session_kit), options=with_correct(False), llm=llm)
    assert only_document(results).transcript.correction_llm is None
    assert llm.stages() == ["punctuate", "numeric"]
    assert "correct" not in [e.stage for e in events]


@pytest.mark.parametrize(
    ("setting", "with_llm", "resolved"),
    [(None, True, True), (None, False, False), (True, True, True), (False, True, False), (False, False, False)],
)
def test_the_tri_state_is_resolved_once_into_the_settings(session_kit, setting, with_llm, resolved) -> None:
    settings = pipeline._settings(
        with_correct(setting),
        FakeLLM() if with_llm else None,
        session_kit.out,
        kind="youtube",
        force=False,
        captions=True,
        emit=lambda _event: None,
        clock=lambda: NOW,
    )
    assert settings.text.correct is resolved


def test_the_resolved_value_is_what_the_stored_document_depends_on(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    _, _, events = run(session_kit, yt, options=with_correct(True), llm=llm)
    assert [e.stage for e in events] == ["skip"], "unset with an LLM and True are the same identity"

    _, off, events = run(session_kit, yt, options=with_correct(False), llm=llm)
    assert "skip" not in [e.stage for e in events]
    assert only_document(off).transcript.correction_llm is None
    assert stored(off[0]) == {"asr": True, "normalize": True, "punctuate": True, "numeric": True, "numerals": True}

    _, on, _ = run(session_kit, yt, llm=llm)
    assert only_document(on).transcript.correction_llm == llm.id


def test_cli_correct_flag_is_tri_state(session_kit, worker, monkeypatch) -> None:
    llm = FakeLLM()
    monkeypatch.setattr(pipeline, "client_from_options", lambda _options, **_kwargs: llm)
    monkeypatch.setattr(pipeline, "check_ready", lambda _options, **_kwargs: None)
    spec = ("--llm", "claude-cli:claude-sonnet-5-5")

    def correction_requests(*flags: str) -> int:
        llm.requests.clear()
        result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--force", *spec, *flags)
        assert result.exit_code == 0, result.output
        return llm.stages().count("correct")

    assert correction_requests() == 1
    assert correction_requests("--correct") == 1
    assert correction_requests("--no-correct") == 0

    without = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--force")
    assert without.exit_code == 0
    assert "correction is skipped" in without.output
    refused = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--force", "--correct")
    assert refused.exit_code == 1
    assert "correction needs an LLM" in refused.output


def test_a_whole_file_engine_is_refused_as_the_numeric_second_opinion(session_kit, worker) -> None:
    options = config.load_options(overrides={"text": {"numeric_second_opinion": "w-prod"}})
    with pytest.raises(ConfigError, match="decodes whole files"):
        run(session_kit, zh_yt(session_kit), options=options)


def test_a_runs_engine_is_refused_as_the_numeric_second_opinion(session_kit, worker) -> None:
    options = config.load_options(overrides={"text": {"numeric_second_opinion": "w-runs-lid"}})
    with pytest.raises(ConfigError, match="merges same-language spans into runs"):
        run(session_kit, zh_yt(session_kit), options=options)


def invoke(kit, yt, monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setattr(session, "SubprocessRunner", lambda *, sleep_args=(): yt)
    return CliRunner().invoke(main, ["run", "--out", str(kit.out), *args])


def test_cli_without_an_llm_prints_a_notice_per_degraded_stage(session_kit, worker, monkeypatch) -> None:
    result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH)
    assert result.exit_code == 0, result.output
    assert "! punctuation not restored" in result.output
    assert "punctuation: none" in result.output
    assert "stages: fetch" in result.output
    assert "asr 0s" in result.output


def test_cli_llm_spec_reaches_the_pipeline(session_kit, worker, monkeypatch) -> None:
    built: list[config.LLMOptions] = []
    llm = FakeLLM(model="claude-sonnet-5-5")

    def client_from_options(options: config.LLMOptions, **_kwargs: object) -> FakeLLM:
        built.append(options)
        return llm

    monkeypatch.setattr(pipeline, "client_from_options", client_from_options)
    monkeypatch.setattr(pipeline, "check_ready", lambda _options, **_kwargs: None)
    result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "claude-cli:claude-sonnet-5-5")
    assert result.exit_code == 0, result.output
    assert (built[0].provider, built[0].model) == ("claude-cli", "claude-sonnet-5-5")
    assert "punctuation: llm" in result.output
    assert "document LLM tokens: punctuation 1 call(s), 100 in / 20 out" in result.output

    bad = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "sonnet")
    assert bad.exit_code == 1
    assert "PROVIDER:MODEL" in bad.output
    refused = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "none", "--correct")
    assert refused.exit_code == 1
    assert "correction needs an LLM" in refused.output


def test_render_rewrites_the_markdown_from_context_json(session_kit, worker, monkeypatch) -> None:
    invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH)
    vdir = session_kit.out / ZH
    expected = (vdir / "context.md").read_text(encoding="utf-8")
    (vdir / "context.md").unlink()
    result = CliRunner().invoke(main, ["render", str(vdir / "context.json")])
    assert result.exit_code == 0, result.output
    assert (vdir / "context.md").read_text(encoding="utf-8") == expected
    printed = CliRunner().invoke(main, ["render", "--stdout", str(vdir / "context.json")])
    assert printed.output == expected
    (vdir / "broken.json").write_text("{}", encoding="utf-8")
    assert CliRunner().invoke(main, ["render", str(vdir / "broken.json")]).exit_code == 1


def test_only_the_unpunctuated_stretch_of_a_punctuated_video_goes_to_the_llm(session_kit, worker) -> None:
    llm = FakeLLM()
    _, results, _ = run(session_kit, session_kit.yt(), [EN], llm=llm)
    doc = only_document(results)
    (asked,) = [r for r in llm.requests if '"items"' in r.user]
    sent = [item["text"] for item in json.loads(asked.user.split("SEGMENTS:\n", 1)[1])]
    assert len(sent) >= 2 and " ".join(sent) in EN_OPENING, "only the unpunctuated opening is sent"
    assert doc.segments[0].text.startswith("So today")
    assert doc.transcript.punctuation == "llm"
    assert [s.text for s in doc.segments[-2:]] == ["Then they fell sharply.", EN_REST[44:]]


def set_info(kit, video: str, **fields: object) -> None:
    info = kit.out / video / "info.json"
    data = json.loads(info.read_text(encoding="utf-8"))
    info.write_text(json.dumps({**data, **fields}, ensure_ascii=False), encoding="utf-8")


def worker_dirs(kit, video: str = ZH, stage: str = "asr") -> list[str]:
    root = kit.out / video / ".stages" / stage
    return sorted(str(path.relative_to(root)) for path in root.glob("*/*"))


def test_a_changed_asr_request_decodes_into_a_fresh_folder_and_prunes_the_old_one(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    (first,) = worker_dirs(session_kit)
    # The folder V0 used for every request of an engine, holding another request's output: the worker refuses it.
    legacy = session_kit.out / ZH / ".stages" / "asr" / "w-prod"
    (legacy / "fingerprint.json").write_text('{"engine": "something else"}', encoding="utf-8")

    set_info(session_kit, ZH, language="zh-Hant")
    store.StageStore(session_kit.out / ZH).discard("context")
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert stored(results[0])["asr"] is False
    assert len(worker.calls) == 2
    (second,) = worker_dirs(session_kit)
    assert second != first
    assert worker.calls[1].files[0].language == "zh-Hant"
    assert not (legacy / "fingerprint.json").exists(), "stale folders go once the new stage is stored"


def test_switching_engines_keeps_the_other_engines_output(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    store.StageStore(session_kit.out / ZH).discard("context")
    run(session_kit, yt, options=config.load_options(overrides={"asr": {"engine": "w-vad"}}))
    assert [path.split("/")[0] for path in worker_dirs(session_kit)] == ["w-prod", "w-vad"]
    store.StageStore(session_kit.out / ZH).discard("context")
    _, results, _ = run(session_kit, yt)
    assert stored(results[0])["asr"] is False and len(worker.calls) == 2, "switching back decodes nothing"
    assert [path.split("/")[0] for path in worker_dirs(session_kit)] == ["w-prod", "w-vad"]


def bump(monkeypatch: pytest.MonkeyPatch, *stage_names: str) -> None:
    for stage in stage_names:
        monkeypatch.setitem(pipeline._PIPELINE_VERSIONS, stage, pipeline._PIPELINE_VERSIONS[stage] + 1)


def test_an_asr_version_bump_rereads_the_kept_rows_without_audio_or_decoding(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    (folder,) = worker_dirs(session_kit)
    assert not (session_kit.out / ZH / "audio.wav").exists()
    converted, fetched = ffmpeg_calls(session_kit), len(yt.labels("audio"))

    bump(monkeypatch, "asr")
    real = stages.from_asr_rows
    monkeypatch.setattr(
        stages,
        "from_asr_rows",
        lambda *args, **kwargs: real(*args, **kwargs).model_copy(update={"warnings": ("re-derived",)}),
    )
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert stored(results[0])["asr"] is False
    assert len(worker.calls) == 1
    assert ffmpeg_calls(session_kit) == converted
    assert len(yt.labels("audio")) == fetched
    assert "re-derived" in only_document(results).quality.warnings
    assert worker_dirs(session_kit) == [folder]


def test_a_folder_of_the_same_request_under_an_older_name_is_adopted(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    (folder,) = worker_dirs(session_kit)
    engine_dir = session_kit.out / ZH / ".stages" / "asr" / "w-prod"
    (engine_dir / folder.split("/")[1]).rename(engine_dir / "0123456789abcdef")

    bump(monkeypatch, "asr")
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert len(worker.calls) == 1
    assert worker_dirs(session_kit) == [folder]


def test_a_moved_library_rereads_its_kept_rows_after_an_asr_bump(session_kit, worker, monkeypatch, tmp_path) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    (folder,) = worker_dirs(session_kit)
    session_kit.out = session_kit.out.rename(tmp_path / "moved")

    bump(monkeypatch, "asr")
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert stored(results[0])["asr"] is False
    assert len(worker.calls) == 1
    assert worker_dirs(session_kit) == [folder]


def test_a_relative_out_reuses_the_folder_an_absolute_out_made(session_kit, worker, monkeypatch, tmp_path) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    (folder,) = worker_dirs(session_kit)
    monkeypatch.chdir(tmp_path)
    session_kit.out = Path("out")

    bump(monkeypatch, "asr")
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert len(worker.calls) == 1
    assert worker_dirs(session_kit) == [folder]


def test_an_interrupted_run_resumes_its_folder_after_the_library_moves(session_kit, worker, tmp_path) -> None:
    yt = zh_yt(session_kit)
    worker.error = EngineError("killed mid-decode")
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 1
    (folder,) = worker_dirs(session_kit)
    session_kit.out = session_kit.out.rename(tmp_path / "moved")

    worker.error = None
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert worker.calls[1].out_dir == str(session_kit.out / ZH / ".stages" / "asr" / folder)
    assert worker_dirs(session_kit) == [folder]


def test_adoption_prefers_the_complete_folder_of_two_with_the_same_request(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    (folder,) = worker_dirs(session_kit)
    engine_dir = session_kit.out / ZH / ".stages" / "asr" / "w-prod"
    complete = (engine_dir / folder.split("/")[1]).rename(engine_dir / "ffffffffffffffff")
    partial = shutil.copytree(complete, engine_dir / "0000000000000000")
    (partial / METRICS_FILE).unlink()

    bump(monkeypatch, "asr")
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert len(worker.calls) == 1
    assert worker_dirs(session_kit) == [folder]


def test_a_second_opinion_with_nothing_to_check_drops_that_engines_folders(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    options = config.load_options(overrides={"text": {"numeric_second_opinion": "w-vad"}})
    run(session_kit, yt, options=options)
    (folder,) = worker_dirs(session_kit, stage="second_opinion")
    other = session_kit.out / ZH / ".stages" / "second_opinion" / "q-base" / "0123456789abcdef"
    other.mkdir(parents=True)

    bump(monkeypatch, "second_opinion")
    monkeypatch.setattr(numeric, "second_opinion_request", lambda *_args, **_kwargs: None)
    report, results, _ = run(session_kit, yt, options=options)
    assert report.exit_code == 0, results[0].outcome
    assert worker_dirs(session_kit, stage="second_opinion") == ["q-base/0123456789abcdef"]
    assert folder.startswith("w-vad/")


def test_the_second_opinion_folder_follows_its_request_not_the_stage_versions(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    options = config.load_options(overrides={"text": {"numeric_second_opinion": "w-vad"}})
    run(session_kit, yt, options=options)
    assert [call.engine.spec.id for call in worker.calls] == ["w-prod", "w-vad"]
    (folder,) = worker_dirs(session_kit, stage="second_opinion")
    converted = ffmpeg_calls(session_kit)

    bump(monkeypatch, "asr", "second_opinion")
    report, results, _ = run(session_kit, yt, options=options)
    assert report.exit_code == 0, results[0].outcome
    assert stored(results[0])["second_opinion"] is False
    assert len(worker.calls) == 2
    assert ffmpeg_calls(session_kit) == converted
    assert worker_dirs(session_kit, stage="second_opinion") == [folder]

    set_info(session_kit, ZH, language="zh-Hant")
    store.StageStore(session_kit.out / ZH).discard("context")
    report, results, _ = run(session_kit, yt, options=options)
    assert report.exit_code == 0, results[0].outcome
    assert [call.files[0].language for call in worker.calls[2:]] == ["zh-Hant", "zh-Hant"]
    (moved,) = worker_dirs(session_kit, stage="second_opinion")
    assert moved != folder
    assert moved.startswith("w-vad/")


def test_an_interrupted_asr_run_resumes_in_its_own_folder(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    worker.error = EngineError("killed mid-decode")
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 1
    worker.error = None
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 0
    assert worker.calls[0].out_dir == worker.calls[1].out_dir
    assert len(worker_dirs(session_kit)) == 1


def test_the_memory_cap_reaches_the_worker_but_does_not_miss_the_stored_asr(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    capped = config.load_options(overrides={"asr": {"memory_cap_gib": 1.0}})
    run(session_kit, yt, options=capped)
    assert worker.calls[0].memory_cap_bytes == 1 << 30
    store.StageStore(session_kit.out / ZH).discard("context")
    _, results, _ = run(session_kit, yt)
    assert stored(results[0])["asr"] is True
    assert len(worker.calls) == 1


def test_a_module_version_bump_redoes_its_stage_and_the_document(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    monkeypatch.setattr(punctuate_module, "PROMPT_VERSION", punctuate_module.PROMPT_VERSION + "-bumped")
    _, results, events = run(session_kit, yt, llm=llm)
    assert "skip" not in [e.stage for e in events], "the context record names the module versions"
    # Same reply, so the text after punctuation is unchanged and numeric is reused.
    assert stored(results[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": False,
        "correct": True,
        "numeric": True,
        "numerals": True,
    }


def test_a_normalize_rule_bump_redoes_only_normalize_when_the_text_comes_out_the_same(
    session_kit, worker, monkeypatch
) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    asked = len(llm.requests)
    monkeypatch.setattr(clean, "VERSION", clean.VERSION + "-bumped")
    _, results, _ = run(session_kit, yt, llm=llm)
    assert stored(results[0]) == {
        "asr": True,
        "normalize": False,
        "punctuate": True,
        "correct": True,
        "numeric": True,
        "numerals": True,
    }, "the later stages are keyed by the normalized text, which no rule changed"
    assert len(llm.requests) == asked


@pytest.mark.parametrize("module", [numbers, guards])
def test_a_tokenizer_or_guard_bump_rechecks_the_stored_replies(session_kit, worker, monkeypatch, module) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    asked = len(llm.requests)
    assert set(llm.stages()) == {"punctuate", "correct", "numeric"}
    monkeypatch.setattr(module, "VERSION", module.VERSION + "-bumped")
    _, results, events = run(session_kit, yt, llm=llm)
    assert "skip" not in [e.stage for e in events]
    assert stored(results[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": True,
        "correct": False,
        "numeric": False,
        # The numerals stage reads quantities too; the guards are not part of it.
        "numerals": module is guards,
    }
    assert len(llm.requests) == asked, (
        "no request changed, so correct and numeric re-apply their rules to the checkpointed replies"
    )
    assert only_document(results).quality.flags_by_kind == {"numeric_fixed": 1, "timing_interpolated": 1}


def test_the_numerals_stage_is_stored_and_reused(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    _, first, _ = run(session_kit, yt, llm=llm)
    assert stored(first[0])["numerals"] is False
    assert (session_kit.out / ZH / ".stages" / "numerals.json").is_file()
    store.StageStore(session_kit.out / ZH).discard("context")
    _, again, events = run(session_kit, yt, llm=llm)
    assert stored(again[0])["numerals"] is True
    assert "numerals" not in [e.stage for e in events]
    assert "50个百分点" in "".join(s.text for s in only_document(again).segments)


def test_keep_skips_the_numerals_stage_and_reruns_nothing_before_it(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    asked = len(llm.requests)
    keep = config.load_options(overrides={"text": {"numerals": "keep"}})
    _, results, events = run(session_kit, yt, options=keep, llm=llm)
    assert "skip" not in [e.stage for e in events], "the document depends on the option"
    assert stored(results[0]) == {"asr": True, "normalize": True, "punctuate": True, "correct": True, "numeric": True}
    assert len(llm.requests) == asked
    doc = only_document(results)
    assert "五十个百分点" in "".join(s.text for s in doc.segments)
    assert ("numeric", "十五", "五十") in [(e.stage, e.before, e.after) for s in doc.segments for e in s.edits]


def test_a_transcript_in_another_language_skips_the_numerals_stage(session_kit, worker) -> None:
    _, results, events = run(session_kit, session_kit.yt(), [EN], llm=FakeLLM())
    assert "numerals" not in stored(results[0])
    assert "numerals" not in [e.stage for e in events]
    assert not (session_kit.out / EN / ".stages" / "numerals.json").exists()


def test_a_numerals_rule_change_reruns_only_numerals_and_the_document(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    asked = len(llm.requests)
    monkeypatch.setattr(numerals, "VERSION", numerals.VERSION + "-bumped")
    _, results, events = run(session_kit, yt, llm=llm)
    assert "skip" not in [e.stage for e in events]
    assert stored(results[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": True,
        "correct": True,
        "numeric": True,
        "numerals": False,
    }
    assert len(llm.requests) == asked
    assert len(worker.calls) == 1


def test_correction_reruns_when_its_glossary_changes(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    correct = config.load_options(overrides={"text": {"correct": True}})
    run(session_kit, yt, options=correct, llm=FakeLLM())
    set_info(session_kit, ZH, tags=["估值"])
    store.StageStore(session_kit.out / ZH).discard("context")
    _, results, _ = run(session_kit, yt, options=correct, llm=FakeLLM())
    assert stored(results[0])["correct"] is False
    assert stored(results[0])["punctuate"] is True


NOTE_TEXT = "Correction: the revenue figure is 12 million."
PINNED = {"is_pinned": True, "author_is_uploader": True, "text": NOTE_TEXT, "author": "@synthetic", "like_count": 3}


def note_run(kit, yt, llm: FakeLLM, *, note: bool):
    options = config.load_options(overrides={"text": {"correct": True}, "ytdlp": {"creator_note": note}})
    return run(kit, yt, options=options, llm=llm)


def correct_prompts(llm: FakeLLM) -> list[str]:
    return [r.user for r in llm.requests if "ITEMS:" not in r.user and '"items"' not in r.user]


def test_the_creator_note_reaches_correction_only_when_the_option_is_on(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    yt.comments[ZH] = [PINNED]
    llm = FakeLLM()
    note_run(session_kit, yt, llm, note=False)
    assert correct_prompts(llm)
    assert not any(NOTE_TEXT in prompt for prompt in correct_prompts(llm))

    llm = FakeLLM()
    _, results, _ = note_run(session_kit, yt, llm, note=True)
    assert llm.stages() == ["correct"], "only correction is asked again"
    assert all(NOTE_TEXT in prompt for prompt in correct_prompts(llm))
    assert stored(results[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": True,
        "correct": False,
        "numeric": True,
        "numerals": True,
    }
    assert only_youtube(results).creator_note is not None

    llm = FakeLLM()
    _, results, events = note_run(session_kit, yt, llm, note=True)
    assert llm.requests == []
    assert [e.stage for e in events] == ["skip"]

    _, results, _ = note_run(session_kit, yt, FakeLLM(), note=False)
    assert stored(results[0])["correct"] is False
    assert only_youtube(results).creator_note is None


def test_turning_the_note_on_for_a_video_without_one_reuses_correction(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    note_run(session_kit, yt, FakeLLM(), note=False)
    llm = FakeLLM()
    _, results, _ = note_run(session_kit, yt, llm, note=True)
    assert llm.requests == []
    assert stored(results[0]) == {
        "asr": True,
        "normalize": True,
        "punctuate": True,
        "correct": True,
        "numeric": True,
        "numerals": True,
    }
    assert only_youtube(results).creator_note is None


def test_numbers_without_independent_readings_are_called_unchecked(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    yt.tracks[ZH] = {}
    _, results, _ = run(session_kit, yt, llm=FakeLLM())
    doc = only_document(results)
    (notice,) = results[0].notices
    assert notice.startswith("numbers not cross-checked: no captions in zh and no second opinion")
    assert notice in doc.quality.warnings
    cli = invoke(session_kit, zh_yt(session_kit), monkeypatch, "zhVideo0002")
    assert "! numbers not cross-checked" in cli.output


def test_a_run_without_captions_does_not_count_for_a_run_with_them(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt, captions=False)
    assert yt.labels("captions") == []
    _, results, events = run(session_kit, yt)
    assert "skip" not in [e.stage for e in events]
    assert len(yt.labels("captions")) == 1
    assert any(f.kind == "numeric_disputed" for s in only_document(results).segments for f in s.flags)


def test_llm_usage_is_what_the_document_cost_even_when_every_batch_is_reused(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    _, first, _ = run(session_kit, yt, llm=FakeLLM())
    stages_dir = session_kit.out / ZH / ".stages"
    for name in ("punctuate.json", "numeric.json", "context.json"):
        (stages_dir / name).unlink()
    llm = FakeLLM()
    _, again, _ = run(session_kit, yt, llm=llm)
    assert llm.requests == [] and stored(again[0])["punctuate"] is False, "recomputed from batch checkpoints"
    assert only_document(again).llm_usage == only_document(first).llm_usage != ()


def test_a_resumed_partial_stage_counts_every_batch(session_kit, worker, monkeypatch) -> None:
    monkeypatch.setattr(punctuate_module, "BATCH_CHARS", 8)
    worker.pause = 1.0
    yt = zh_yt(session_kit)
    failing = FakeLLM()
    original = failing.complete

    def second_call_fails(request):
        if len(failing.requests) == 1:
            failing.requests.append(request)
            raise LLMUnavailableError("synthetic outage")
        return original(request)

    failing.complete = second_call_fails  # type: ignore[method-assign]
    report, _, _ = run(session_kit, yt, llm=failing)
    assert report.exit_code == 1
    llm = FakeLLM()
    _, results, _ = run(session_kit, yt, llm=llm)
    batches = len([r for r in llm.requests if '"items"' in r.user]) + 1
    (punctuation,) = [u for u in only_document(results).llm_usage if u.stage == "punctuation"]
    assert batches >= 3
    assert punctuation.calls == batches


def test_a_missing_claude_cli_fails_before_any_request(session_kit, worker, monkeypatch) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _name: None)
    yt = zh_yt(session_kit)
    options = config.load_options(overrides={"llm": {"provider": "claude-cli", "model": "claude-sonnet-5-5"}})
    with pytest.raises(DependencyError, match="claude CLI not found"):
        run(session_kit, yt, options=options)
    assert yt.calls == []
    assert worker.calls == []


def test_the_default_llm_without_a_chatgpt_sign_in_fails_before_any_request(session_kit, worker) -> None:
    (session_kit.home / ".config" / "anyverbatim" / "config.toml").unlink()
    yt = zh_yt(session_kit)
    with pytest.raises(ConfigError, match=r"not signed in.*`anyverbatim login`.*`--llm none`"):
        run(session_kit, yt)
    assert yt.calls == []
    assert worker.calls == []


def test_an_unresolvable_model_family_fails_once_before_any_request(session_kit, worker, monkeypatch) -> None:
    (session_kit.home / ".config" / "anyverbatim" / "config.toml").unlink()
    store_account()
    asked: list[str] = []

    def resolve(model: str, **_kwargs: object) -> str:
        asked.append(model)
        raise ConfigError("the ChatGPT plan lists no 'sol' model")

    monkeypatch.setattr(chatgpt_models, "resolve", resolve)
    yt = zh_yt(session_kit)
    with pytest.raises(ConfigError, match="lists no 'sol' model"):
        run(session_kit, yt, (ZH, EN))
    assert asked == ["gpt-sol"]
    assert yt.calls == []
    assert worker.calls == []


@pytest.mark.parametrize("interactive", [True, False])
def test_interactivity_reaches_the_llm_checks(session_kit, worker, monkeypatch, interactive: bool) -> None:
    (session_kit.home / ".config" / "anyverbatim" / "config.toml").unlink()
    seen: list[tuple[str, object]] = []

    def check_ready(_options: config.LLMOptions, **kwargs: object) -> None:
        seen.append(("check_ready", kwargs["interactive"]))

    def client_from_options(_options: config.LLMOptions, **kwargs: object) -> FakeLLM:
        seen.append(("client", kwargs["interactive"]))
        return FakeLLM()

    monkeypatch.setattr(pipeline, "check_ready", check_ready)
    monkeypatch.setattr(pipeline, "client_from_options", client_from_options)
    run(session_kit, zh_yt(session_kit), interactive=interactive)
    assert seen == [("check_ready", interactive), ("client", interactive)]


def test_cli_default_llm_without_a_sign_in_names_login_and_llm_none(session_kit, worker, monkeypatch) -> None:
    (session_kit.home / ".config" / "anyverbatim" / "config.toml").unlink()
    refused = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH)
    assert refused.exit_code == 1
    assert "anyverbatim login" in refused.output
    assert "--llm none" in refused.output
    without = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "none")
    assert without.exit_code == 0, without.output
    assert worker.calls != []


def test_cli_llm_spec_switches_provider_over_the_configured_one(session_kit, worker, monkeypatch) -> None:
    config_file = session_kit.home / ".config" / "anyverbatim" / "config.toml"
    config_file.write_text('[llm]\nprovider = "openai-compat"\nmodel = "m"\nbase_url = "http://127.0.0.1:8080/v1"\n')
    built: list[config.LLMOptions] = []

    def client_from_options(options: config.LLMOptions, **_kwargs: object) -> FakeLLM:
        built.append(options)
        return FakeLLM()

    monkeypatch.setattr(pipeline, "client_from_options", client_from_options)
    monkeypatch.setattr(pipeline, "check_ready", lambda _options, **_kwargs: None)
    result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "claude-cli:claude-sonnet-5-5")
    assert result.exit_code == 0, result.output
    assert (built[0].provider, built[0].model, built[0].base_url) == ("claude-cli", "claude-sonnet-5-5", None)


def test_a_skipped_video_gets_its_markdown_back_and_an_unloadable_document_is_rebuilt(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    vdir = session_kit.out / ZH
    expected = (vdir / "context.md").read_text(encoding="utf-8")
    (vdir / "context.md").unlink()
    _, _, events = run(session_kit, yt)
    assert [e.stage for e in events] == ["skip"]
    assert (vdir / "context.md").read_text(encoding="utf-8") == expected

    broken = '{"schema_version": "9.0"}'
    (vdir / "context.json").write_text(broken, encoding="utf-8")
    record = json.loads((vdir / ".stages" / "context.json").read_text(encoding="utf-8"))
    record["output"]["sha256"] = __import__("hashlib").sha256(broken.encode()).hexdigest()
    (vdir / ".stages" / "context.json").write_text(json.dumps(record), encoding="utf-8")
    report, results, events = run(session_kit, yt)
    assert report.exit_code == 0
    assert "skip" not in [e.stage for e in events]
    assert load_document(vdir / "context.json") == only_document(results)


def test_captions_mode_says_why_it_cannot_choose_among_manual_tracks(session_kit, worker) -> None:
    yt = session_kit.yt()
    yt.language[EN] = None  # type: ignore[assignment]
    yt.tracks[EN] = {"en": json.dumps(EN_MANUAL), "zh": json.dumps(EN_MANUAL)}
    options = config.load_options(overrides={"asr": {"source": "captions"}})
    _, results, _ = run(session_kit, yt, [EN], options=options)
    assert results[0].notices[0] == (
        "2 manual caption tracks and no declared video language to choose one; transcribed with ASR instead"
    )


def captions_mode() -> config.Options:
    return config.load_options(overrides={"asr": {"source": "captions"}})


def test_an_audio_fetch_that_fails_after_the_captions_fell_through_fails_the_video(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    yt.failures[("audio", False)] = [session_kit.fail(Signal.NO_FORMATS)]
    report, results, events = run(session_kit, yt, options=captions_mode())

    (result,) = results
    assert result.document is None
    assert result.markdown is None
    assert result.outcome.status is session.VideoStatus.ERROR
    assert result.outcome.error == Signal.NO_FORMATS.value
    assert [e.stage for e in events] == ["fetch", "fetch"]
    assert worker.calls == []
    assert report.exit_code == 1
    batch = json.loads((session_kit.out / "batch_report.json").read_text(encoding="utf-8"))
    (recorded,) = batch["videos"]
    assert (recorded["id"], recorded["status"], recorded["error"]) == (ZH, "error", Signal.NO_FORMATS.value)
    with store.lock(session_kit.out / ZH):
        pass


def test_tags_come_from_the_extraction_the_audio_fetch_refreshed(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    extract = yt._info

    def tagged(video_id: str, with_comments: bool = False) -> dict[str, object]:
        return {**extract(video_id, with_comments), "tags": [f"extraction{len(yt.labels('metadata'))}"]}

    monkeypatch.setattr(yt, "_info", tagged)
    # Every saved extraction is too old for a fetch that still has stages to run, so the audio fetch extracts again.
    monkeypatch.setattr(session, "INFO_MAX_AGE_S", -1.0)
    llm = FakeLLM()
    _, results, events = run(session_kit, yt, options=captions_mode(), llm=llm)

    assert len(yt.labels("metadata")) == 2
    assert [e.stage for e in events][:2] == ["fetch", "fetch"]
    assert only_youtube(results).tags == ("extraction2",)
    (correction,) = [r for r, stage in zip(llm.requests, llm.stages(), strict=True) if stage == "correct"]
    assert "extraction2" in correction.system + correction.user
    assert "extraction1" not in correction.system + correction.user
