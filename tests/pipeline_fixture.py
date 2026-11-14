"""A synthetic, committed `run` output: every stage stored, made with a fake yt-dlp, ASR worker and LLM.

`uv run python tests/pipeline_fixture.py` regenerates `tests/fixtures/pipeline/`. Every text here is made up. The
fixture pins the pipeline's stage keys and output bytes, so a refactor that changes either fails
`test_pipeline_fixture.py`; regenerate it only for a deliberate output change.
"""

import contextlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from ytverbatim import config, proc
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import (
    FINGERPRINT_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    ChunkHypothesis,
    RunMetrics,
    WorkerRequest,
)
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.asr.worker import _claim_output as claim_output
from ytverbatim.contract import Word
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.pipeline import VideoResult, process_many
from ytverbatim.store import LOCK_NAME
from ytverbatim.ytdlp import jsruntime, version
from ytverbatim.ytdlp.types import YtDlpResult, YtDlpRunner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "pipeline"
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
DURATION = 40.0
# The fixture's worker folders hold this WAV path in place of the temporary one they were made with.
PLACEHOLDER_WAV = "/fixture/audio.wav"

# Mandarin with English terms, decoded by ASR: the run with a second opinion, a creator note, chapters and tags.
MIXED = "fxMixZhEn01"
# English from a manual caption track: the captions-mode run.
CAPTIONED = "fxCaptEn001"

# Whisper-style: no punctuation, a space at each pause. 十五 is the number the captions and the second opinion
# dispute; GAP is the term correction fixes; 三万五千 and 十五 become digits in the numerals stage.
MIXED_PHRASES = (
    "大家好欢迎回来",
    "今天聊聊ETF和GAP利润",
    "上周涨了十五个百分点",
    "收入三万五千美元",
    "really strong quarter",
)
MIXED_AUTO_PHRASES = ("大家好欢迎回来", "今天聊聊ETF和GAAP利润", "上周涨了五十个百分点", "收入三万五千美元")
CAPTIONED_MANUAL = ("so today we look at fifteen funds", "Markets rose twelve percent.", "Then they fell sharply.")


def options(video_id: str) -> config.Options:
    """The options each fixture video was made with; the rerun must use the same ones."""
    base = config.Options(llm=config.LLMOptions(provider="none"))
    if video_id == MIXED:
        return replace(
            base,
            ytdlp=replace(base.ytdlp, creator_note=True),
            text=replace(base.text, numeric_second_opinion="w-vad"),
        )
    return replace(base, asr=replace(base.asr, source="captions"))


def _json3(phrases: Sequence[str], starts_ms: Sequence[int]) -> str:
    events = [
        {"tStartMs": start, "dDurationMs": 4000, "segs": [{"utf8": phrase}]}
        for phrase, start in zip(phrases, starts_ms, strict=True)
    ]
    return json.dumps({"events": events}, ensure_ascii=False)


_TRACKS: Mapping[str, Mapping[str, str]] = {
    MIXED: {"zh-orig": _json3(MIXED_AUTO_PHRASES, (0, 5000, 10000, 20000))},
    CAPTIONED: {
        "en": _json3(CAPTIONED_MANUAL, (0, 6000, 12000)),
        "en-orig": _json3(CAPTIONED_MANUAL, (0, 6000, 12000)),
    },
}


def _info(video_id: str, *, with_comments: bool) -> dict[str, object]:
    tracks = _TRACKS[video_id]
    info: dict[str, object] = {
        "id": video_id,
        "title": "合成标题 Synthetic title" if video_id == MIXED else "Synthetic captions title",
        "channel": "Synthetic channel",
        "channel_id": "UCxxxxxxxxxxxxxxxxxxxxxx",
        "upload_date": "20291231",
        "duration": DURATION,
        "description": "A made-up description for a test fixture.",
        "language": "zh" if video_id == MIXED else "en",
        "availability": "public",
        "tags": ["GAAP", "ETF", "合成标签"],
        "chapters": [
            {"start_time": 0, "end_time": 16, "title": "开场"},
            {"start_time": 16, "end_time": DURATION, "title": "Numbers"},
        ],
        "automatic_captions": {code: [] for code in tracks if code.endswith("-orig")},
        "subtitles": {code: [] for code in tracks if not code.endswith("-orig")},
    }
    if with_comments:
        info["comments"] = [
            {
                "text": "更正：GAAP 利润以财报为准。",
                "author": "@synthetic-channel",
                "author_is_uploader": True,
                "is_pinned": True,
                "like_count": 7,
                "_time_text": "1 day ago",
            }
        ]
    return info


@dataclass
class FakeYtDlp:
    """Metadata, captions and audio succeed for the fixture videos."""

    labels: list[str] = field(default_factory=list[str])

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        self.labels.append(label)
        video_id = args[-1].rsplit("=", 1)[-1]
        if label == "metadata":
            info = _info(video_id, with_comments="--get-comments" in args)
            return YtDlpResult(0, json.dumps(info, ensure_ascii=False), "", None, (), ())
        if label == "captions":
            target = args[args.index("-o") + 1].removeprefix("subtitle:").replace("%(ext)s", "")
            for code, content in _TRACKS[Path(target).parent.parent.name].items():
                Path(f"{target}{code}.json3").write_text(content, "utf-8")
        elif label == "audio":
            Path(args[args.index("-o") + 1].replace("%(ext)s", "webm")).write_bytes(b"fake audio")
        return YtDlpResult(0, "", "", None, (), ())


def _fake_proc(argv: Sequence[str], **_kwargs: object) -> proc.ProcResult:
    if Path(argv[0]).name == "ffmpeg":
        Path(argv[-1]).write_bytes(b"RIFF fake wav")
        return proc.ProcResult(0, "", "", 0.0)
    return proc.ProcResult(0, f"{DURATION}\n", "", 0.0)


def _phrase_rows(request: WorkerRequest) -> list[ChunkHypothesis]:
    spec = request.engine.spec.id
    file_id = request.files[0].id
    chunks = request.files[0].chunks
    if chunks is not None:
        # The second opinion hears the disputed figure as the captions do.
        return [
            ChunkHypothesis(
                hypothesis=spec,
                file=file_id,
                chunk=index,
                start=chunk.start,
                end=chunk.end,
                text="上周涨了五十个百分点",
                language="zh",
            )
            for index, chunk in enumerate(chunks)
        ]
    rows: list[ChunkHypothesis] = []
    for chunk, phrases in enumerate((MIXED_PHRASES[:3], MIXED_PHRASES[3:])):
        start = 20.0 * chunk
        words = tuple(
            Word(text=phrase, start=start + 5.0 * index, end=start + 5.0 * index + 4.0, conf=0.6 + 0.05 * index)
            for index, phrase in enumerate(phrases)
        )
        rows.append(
            ChunkHypothesis(
                hypothesis=spec,
                file=file_id,
                chunk=chunk,
                start=start,
                end=start + 5.0 * len(phrases),
                text=" ".join(phrases),
                language="zh",
                words=words,
            )
        )
    return rows


@dataclass
class FakeWorker:
    """Claims its output folder as the real worker does, then writes the rows a real decode would."""

    calls: list[WorkerRequest] = field(default_factory=list[WorkerRequest])

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        claim_output(out, request)
        rows = _phrase_rows(request)
        (out / HYPOTHESES_FILE).write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
        metrics = RunMetrics(
            hypothesis=request.engine.spec.id,
            files=1,
            chunks_decoded=len(rows),
            chunks_skipped=0,
            load_s=0.0,
            decode_s=0.0,
            audio_s=DURATION,
            peak_memory_bytes=None,
            max_rss_bytes=0,
        )
        (out / METRICS_FILE).write_text(metrics.model_dump_json(), encoding="utf-8")
        return WorkerResult(out / HYPOTHESES_FILE, metrics, 0, 0.0)


@dataclass
class FakeLLM:
    """Punctuates at the pauses, fixes GAP to GAAP, and takes the last reading of every disputed number."""

    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])

    @property
    def id(self) -> str:
        return "fake:fixture-model"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if "SEGMENTS:" in request.user and '"items"' in request.user:
            items = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
            reply: object = {"items": [{"i": it["i"], "text": _punctuated(it["text"])} for it in items]}
        elif "SEGMENTS:" in request.user:
            items = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
            fixed = [{"i": it["i"], "text": it["text"].replace("GAP", "GAAP")} for it in items if "GAP" in it["text"]]
            reply = {"fixed": fixed}
        else:
            items = json.loads(request.user.split("ITEMS:\n", 1)[1])
            reply = {
                "verdicts": [
                    {"id": it["id"], "value": it["readings"][-1], "confidence": "high", "reason": "context"}
                    for it in items
                ]
            }
        return LLMResponse(text=json.dumps(reply, ensure_ascii=False), usage=Usage(100, 20), elapsed_s=0.0)


def _punctuated(text: str) -> str:
    if text.isascii():
        return text[:1].upper() + text[1:] + "."
    return text.replace(" ", "，") + "。"


@contextlib.contextmanager
def isolated(home: Path) -> Iterator[None]:
    """Point every config, state and auth location at `home`, so no real setting or cooldown leaks in."""
    names = ("HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME", "YTVERBATIM_AUTH")
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    os.environ["HOME"] = str(home)
    try:
        with (
            mock.patch.object(version, "installed", lambda: version.YtDlpVersion("2099.01.01", NOW.date())),
            mock.patch.object(jsruntime, "detect", lambda: jsruntime.JsRuntime("deno", (), None)),
            mock.patch.object(registry, "find_spec", lambda _name: object()),
        ):
            yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run(video_id: str, out: Path, llm: FakeLLM, yt: YtDlpRunner) -> VideoResult:
    results: list[VideoResult] = []
    process_many(
        [video_id],
        options(video_id),
        llm=llm,
        out=out,
        runner_factory=lambda _sleep: yt,
        sleep=lambda _s: None,
        clock=lambda: NOW,
        on_video=results.append,
    )
    (result,) = results
    return result


def fixture_files(vdir: Path) -> Generator[Path]:
    """Every file of a video directory the fixture keeps: not the lock, nor media (which the repo never holds)."""
    for path in sorted(vdir.rglob("*")):
        if path.is_file() and path.name != LOCK_NAME and path.suffix not in (".webm", ".wav"):
            yield path


def without_wav_path(path: Path) -> bytes:
    """A worker fingerprint with its WAV path replaced: a moved directory re-stamps that path, nothing else."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for job in data.get("files", []):
        job["wav"] = PLACEHOLDER_WAV
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def comparable(path: Path) -> bytes:
    return without_wav_path(path) if path.name == FINGERPRINT_FILE else path.read_bytes()


def generate(target: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        home.mkdir()
        out = Path(tmp) / "out"
        with isolated(home), mock.patch.object(proc, "run", _fake_proc):
            worker = FakeWorker()
            with mock.patch.object(asr_runner, "run_worker", worker):
                for video_id in (MIXED, CAPTIONED):
                    result = run(video_id, out, FakeLLM(), FakeYtDlp())
                    if result.document is None:
                        raise SystemExit(f"{video_id} failed: {result.outcome}")
        shutil.rmtree(target, ignore_errors=True)
        for video_id in (MIXED, CAPTIONED):
            vdir = out / video_id
            for path in fixture_files(vdir):
                dest = target / video_id / path.relative_to(vdir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(comparable(path))
        leaked = [p for p in target.rglob("*") if p.is_file() and tmp.encode() in p.read_bytes()]
        if leaked:
            raise SystemExit(f"temporary paths leaked into {leaked}")


if __name__ == "__main__":
    written = Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE_ROOT
    generate(written)
    print(f"wrote {written}")
