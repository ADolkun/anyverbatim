"""Local recordings through `process_files`, `process` and `run`, with fake ffprobe/ffmpeg, worker and LLM."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytverbatim import config, pipeline, proc
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, FileJob, RunMetrics, WorkerRequest
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.cli import main
from ytverbatim.contract import RecordingSource, Word, load_document
from ytverbatim.errors import MediaValidationError
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.pipeline import ProgressEvent, VideoResult, process, process_files

NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
TITLE = "Synthetic weekly sync"
SPOKEN = ("we", "met", "5", "people", "today", "and", "then", "we", "left")


@dataclass
class Media:
    """ffprobe and ffmpeg over synthetic files: a file's first line names its audio packets, so two files sharing it
    are the same audio in two containers. Everything else about a file comes from `tags`."""

    tags: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])
    converted: list[str] = field(default_factory=list[str])
    # Deletes each file once it is hashed, as if it were moved away during the run.
    vanish: bool = False

    def add(self, path: Path, packets: str = "alpha", *, tags: dict[str, str] | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{packets}\n{path.suffix}\n", encoding="utf-8")
        self.tags[path.name] = tags or {}
        return path

    def ffprobe(self, argv: list[str]) -> proc.ProcResult:
        source = Path(argv[-1])
        if "json" not in argv:
            return proc.ProcResult(0, "12.5\n", "", 0.0)
        streams = [] if source.read_text(encoding="utf-8").startswith("silent") else [{"codec_name": "aac"}]
        probe = {"streams": streams, "format": {"duration": "12.5", "tags": self.tags.get(source.name, {})}}
        return proc.ProcResult(0, json.dumps(probe), "", 0.0)

    def ffmpeg(self, argv: list[str]) -> proc.ProcResult:
        source = Path(argv[argv.index("-i") + 1])
        if "hash" in argv:
            packets = source.read_text(encoding="utf-8").splitlines()[0]
            if self.vanish:
                source.unlink()
            return proc.ProcResult(0, f"SHA256={hashlib.sha256(packets.encode()).hexdigest()}\n", "", 0.0)
        assert argv[argv.index("-map") + 1] == "0:a:0"
        self.converted.append(source.name)
        Path(argv[-1]).write_bytes(b"RIFF fake wav")
        return proc.ProcResult(0, "", "", 0.0)


@dataclass
class Worker:
    calls: list[WorkerRequest] = field(default_factory=list[WorkerRequest])

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        words = tuple(Word(text=f" {w}", start=1.0 + i * 0.5, end=1.4 + i * 0.5) for i, w in enumerate(SPOKEN))
        job = request.files[0]
        row = ChunkHypothesis(
            hypothesis=request.engine.spec.id,
            file=job.id,
            chunk=0,
            start=0.0,
            end=12.5,
            text="".join(word.text for word in words),
            language=job.language or "en",
            words=words,
        )
        (out / HYPOTHESES_FILE).write_text(row.model_dump_json() + "\n", encoding="utf-8")
        metrics = RunMetrics(
            hypothesis=request.engine.spec.id,
            files=1,
            chunks_decoded=1,
            chunks_skipped=0,
            load_s=0.0,
            decode_s=0.0,
            audio_s=12.5,
            peak_memory_bytes=None,
            max_rss_bytes=0,
        )
        return WorkerResult(out / HYPOTHESES_FILE, metrics, 0, 0.0)


@dataclass
class FakeLLM:
    """Punctuates by capitalising and adding a full stop; finds nothing to correct."""

    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])

    @property
    def id(self) -> str:
        return "fake:model-1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if '"items"' in request.user:
            items = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
            reply: object = {
                "items": [{"i": it["i"], "text": it["text"][:1].upper() + it["text"][1:] + "."} for it in items]
            }
        else:
            reply = {"fixed": []}
        return LLMResponse(text=json.dumps(reply), usage=Usage(10, 2), elapsed_s=0.0)


@pytest.fixture(autouse=True)
def engine_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "find_spec", lambda _name: object())


@pytest.fixture
def media(session_kit) -> Media:
    fake = Media()
    session_kit.proc.responses["ffprobe"] = fake.ffprobe
    session_kit.proc.responses["ffmpeg"] = fake.ffmpeg
    return fake


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> Worker:
    fake = Worker()
    monkeypatch.setattr(asr_runner, "run_worker", fake)
    return fake


def run_files(kit, files: list[Path], options: config.Options | None = None, **kwargs: object):
    results: list[VideoResult] = []
    events: list[ProgressEvent] = []
    report = process_files(
        files,
        options,
        out=kit.out,
        progress=events.append,
        on_video=results.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return report, results, events


def recording(result: VideoResult) -> RecordingSource:
    assert result.document is not None and isinstance(result.document.source, RecordingSource)
    return result.document.source


def rec_id(packets: str = "alpha") -> str:
    return f"rec-{hashlib.sha256(packets.encode()).hexdigest()[:16]}"


def test_a_recording_end_to_end_gives_a_2_0_document_markdown_and_index(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    path = media.add(tmp_path / "in" / "meeting.m4a", tags={"title": TITLE})
    llm = FakeLLM()
    report, results, events = run_files(session_kit, [path], llm=llm)

    assert report.exit_code == 0
    rdir = session_kit.out / rec_id()
    doc = load_document(rdir / "context.json")
    assert doc.schema_version == "2.0" and results[0].document == doc
    source = recording(results[0])
    assert source.id == rec_id() and source.audio_sha256 == hashlib.sha256(b"alpha").hexdigest()
    assert (source.title, source.file_name, source.origin) == (TITLE, "meeting.m4a", "file")
    assert source.recorded_at is None and source.duration_s == 12.5 and source.language is None
    assert doc.chapters == () and doc.speakers == ()
    assert doc.segments[0].text.startswith("We met 5 people") and doc.transcript.punctuation == "llm"
    assert not any("caption track" in warning for warning in doc.quality.warnings), "recordings have no captions"
    assert any("the source has no captions" in notice for notice in results[0].notices)

    markdown = (rdir / "context.md").read_text(encoding="utf-8")
    assert markdown.startswith(f"# {TITLE}\n") and "meeting.m4a" in markdown and "youtu" not in markdown
    assert not (rdir / "audio.wav").exists() and media.converted == ["meeting.m4a"]
    assert path.read_text(encoding="utf-8") == "alpha\n.m4a\n", "the source file is never modified"

    (request,) = worker.calls
    assert request.engine.spec.id == "w-prod"
    assert request.files[0] == FileJob(id=rec_id(), wav=str(rdir / "audio.wav"), language=None)
    assert [e.stage for e in events] == ["fetch", "asr", "normalize", "punctuate", "correct", "numeric", "write"]
    assert all(TITLE not in r.user + r.system and "meeting" not in r.user + r.system for r in llm.requests)

    batch = json.loads((session_kit.out / "batch_report.json").read_text(encoding="utf-8"))
    assert [(v["id"], v["status"], v["audio_s"]) for v in batch["videos"]] == [(rec_id(), "ok", 12.5)]
    index = (session_kit.out / "index.md").read_text(encoding="utf-8")
    assert f"[{TITLE}]({rec_id()}/context.md)" in index and "| 00:12 |" in index


def test_a_done_file_is_skipped_even_renamed_or_remuxed(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    run_files(session_kit, [media.add(tmp_path / "meeting.m4a")])
    renamed = media.add(tmp_path / "renamed.mov", tags={"title": "Edited"})
    report, results, events = run_files(session_kit, [renamed])
    assert report.exit_code == 0 and len(worker.calls) == 1 and media.converted == ["meeting.m4a"]
    assert results[0].outcome.reused == ("context",) and results[0].name == "renamed.mov"
    assert [e.stage for e in events] == ["skip"]
    assert recording(results[0]).file_name == "meeting.m4a", "the reused document is the first run's"


def test_the_same_audio_under_two_names_is_processed_once(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    first = media.add(tmp_path / "zoom" / "audio.m4a")
    second = media.add(tmp_path / "zoom" / "video.mp4")
    other = media.add(tmp_path / "zoom" / "other.m4a", "beta")
    report, results, events = run_files(session_kit, [first, second, other])
    assert [r.outcome.video_id for r in results] == [rec_id(), rec_id("beta")]
    assert len(worker.calls) == 2 and report.exit_code == 0
    assert ProgressEvent(rec_id(), "skip", "same audio as audio.m4a") in events


def test_a_file_that_cannot_be_read_is_an_error_keyed_by_its_name(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    silent = media.add(tmp_path / "screen.mov", "silent")
    good = media.add(tmp_path / "meeting.m4a")
    report, results, _ = run_files(session_kit, [silent, good])
    assert report.exit_code == 1
    assert [(r.outcome.video_id, r.outcome.status.value) for r in results] == [
        ("screen.mov", "error"),
        (rec_id(), "ok"),
    ]
    assert results[0].outcome.error == "media_invalid" and "no audio stream" in results[0].outcome.detail


def test_a_source_gone_before_its_audio_is_read_asks_for_the_file_again(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    media.vanish = True
    path = media.add(tmp_path / "meeting.m4a")
    _, results, _ = run_files(session_kit, [path])
    assert results[0].outcome.error == "media_invalid" and results[0].outcome.fix == "pass the file again"
    assert worker.calls == []

    path = media.add(tmp_path / "meeting.m4a")
    with pytest.raises(MediaValidationError) as raised:
        process(path, out=session_kit.out)
    assert raised.value.fix == "pass the file again"


def test_process_takes_a_file(session_kit, media: Media, worker: Worker, tmp_path: Path) -> None:
    path = media.add(tmp_path / "meeting.m4a")
    doc = process(str(path), out=session_kit.out)
    assert isinstance(doc.source, RecordingSource) and doc == load_document(session_kit.out / rec_id() / "context.json")
    assert process(path, out=session_kit.out) == doc


@pytest.mark.parametrize("given", [{"auth": "chrome"}, {"captions": False}, {"cookies_first": True}])
def test_youtube_only_arguments_with_a_file_raise(session_kit, tmp_path: Path, given: dict[str, object]) -> None:
    path = tmp_path / "meeting.m4a"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match=f"{next(iter(given))} only apply to YouTube videos"):
        process(path, out=session_kit.out, **given)  # type: ignore[arg-type]


def test_process_refuses_a_folder(session_kit, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="process takes one file"):
        process(tmp_path, out=session_kit.out)


def settings(kit, kind: pipeline.SourceKind, **asr: object) -> pipeline._Settings:
    options = config.load_options(overrides={"asr": asr})
    return pipeline._settings(
        options, None, kit.out, kind=kind, force=False, captions=True, emit=lambda _e: None, clock=lambda: NOW
    )


@pytest.mark.parametrize("kind", ["youtube", "recording"])
def test_an_explicit_engine_wins_for_every_source(session_kit, kind: pipeline.SourceKind) -> None:
    assert settings(session_kit, kind).definition.spec.id == "w-prod"
    assert settings(session_kit, kind, engine="w-vad").definition.spec.id == "w-vad"


def test_an_explicit_engine_reaches_a_recordings_worker(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    options = config.load_options(overrides={"asr": {"engine": "w-vad"}})
    run_files(session_kit, [media.add(tmp_path / "meeting.m4a")], options)
    assert worker.calls[0].engine.spec.id == "w-vad"


def test_youtube_requests_keep_their_fingerprint(session_kit) -> None:
    youtube = settings(session_kit, "youtube", engine="w-runs-lid", languages=("zh", "en"))
    assert youtube.languages == (), "the allowed set is for recordings only"
    request = pipeline._Decoding(youtube, "pubVideo001", session_kit.out / "pubVideo001", "en").asr_request()
    job = request.fingerprint()["files"][0]  # type: ignore[index]
    assert job == {
        "id": "pubVideo001",
        "wav": str(session_kit.out / "pubVideo001" / "audio.wav"),
        "language": "en",
        "context": None,
        "chunks": None,
    }
    plain = settings(session_kit, "youtube")
    assert plain.run_key() == settings(session_kit, "youtube", languages=("zh", "en")).run_key()

    recording_settings = settings(session_kit, "recording", engine="w-runs-lid", languages=("zh-Hans", "en"))
    assert recording_settings.languages == ("zh", "en")
    assert settings(session_kit, "recording", languages=("zh", "en")).languages == (), "w-prod detects nothing"


def test_a_forced_language_reaches_the_request_the_document_and_the_run_key(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    path = media.add(tmp_path / "meeting.m4a")
    run_files(session_kit, [path])
    forced = config.load_options(overrides={"asr": {"language": "zh"}})
    _, results, _ = run_files(session_kit, [path], forced)
    assert len(worker.calls) == 2 and worker.calls[1].files[0].language == "zh"
    assert recording(results[0]).language == "zh"


def test_youtube_documents_are_unaffected_by_recording_only_settings(session_kit) -> None:
    youtube = settings(session_kit, "youtube")
    keyed = youtube.run_key()
    assert settings(session_kit, "youtube", languages=("zh",)).run_key() == keyed
    assert settings(session_kit, "youtube", language="zh").run_key() != keyed, "a forced language is a choice"


def cli(kit, *args: str):
    return CliRunner().invoke(main, ["run", "--out", str(kit.out), "--llm", "none", *args])


def test_cli_runs_a_folder_and_prints_each_context_path(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    media.add(tmp_path / "in" / "b.m4a", "beta")
    media.add(tmp_path / "in" / "sub" / "a.m4a")
    (tmp_path / "in" / "notes.txt").write_text("not media", encoding="utf-8")
    result = cli(session_kit, str(tmp_path / "in"))
    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if line.startswith("✓")]
    assert [line.split()[1] for line in lines] == ["b.m4a", "a.m4a"]
    assert all("context.md" in line and "chapters" not in line for line in lines)
    assert "2 ok, 0 failed" in result.output and "index.md" in result.output


def test_cli_refuses_mixed_youtube_and_local_targets(session_kit, tmp_path: Path) -> None:
    path = tmp_path / "meeting.m4a"
    path.write_bytes(b"x")
    result = cli(session_kit, "pubVideo001", str(path))
    assert result.exit_code == 1 and "run YouTube and local targets separately" in result.output


def test_cli_refuses_youtube_only_options_with_files(session_kit, tmp_path: Path) -> None:
    path = tmp_path / "meeting.m4a"
    path.write_bytes(b"x")
    result = cli(session_kit, "--no-captions", "--auth", "chrome", str(path))
    assert result.exit_code == 1 and "--auth, --captions/--no-captions only apply to YouTube targets" in result.output


def test_cli_refuses_a_target_that_names_nothing(session_kit, tmp_path: Path) -> None:
    result = cli(session_kit, str(tmp_path / "missing.m4a"))
    assert result.exit_code == 1 and "no such file or folder" in result.output
