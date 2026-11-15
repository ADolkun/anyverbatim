"""Local recordings through `process_files`, `process`, `run` and `speakers`, with fake media, workers and LLM."""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from ytverbatim import atomic, config, pipeline, proc, speakers, store
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import (
    FINGERPRINT_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    ChunkHypothesis,
    FileJob,
    RunMetrics,
    WorkerRequest,
)
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.cli import main
from ytverbatim.contract import ContextDocument, RecordingSource, Word, load_document
from ytverbatim.diarize import attribution
from ytverbatim.diarize import protocol as diarize_protocol
from ytverbatim.diarize import registry as diarize_registry
from ytverbatim.diarize import runner as diarize_runner
from ytverbatim.diarize.protocol import DiarizeRequest, FileDiarization, Turn
from ytverbatim.diarize.runner import DiarizeResult
from ytverbatim.errors import ConfigError, ContractError, DependencyError, EngineError, LLMError, MediaValidationError
from ytverbatim.index import INDEX_MD
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.mlx_worker import without_wav
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
    # Fails every probe as ffmpeg does for an unreadable file, echoing the input as it was given.
    unreadable: bool = False

    def add(self, path: Path, packets: str = "alpha", *, tags: dict[str, str] | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{packets}\n{path.suffix}\n", encoding="utf-8")
        self.tags[path.name] = tags or {}
        return path

    def __call__(self, argv: list[str], *, timeout_s: float, cwd: Path | None = None, **_: object) -> proc.ProcResult:
        if argv[0] == "ffprobe" and "json" not in argv:
            return proc.ProcResult(0, "12.5\n", "", 0.0)
        given = argv[argv.index("-i") + 1] if "-i" in argv else argv[-1]
        assert cwd is not None and given.startswith("./") and "/" not in given[2:], "ffmpeg sees only ./<name>"
        return (self.ffprobe if argv[0] == "ffprobe" else self.ffmpeg)(argv, cwd / given)

    def ffprobe(self, argv: list[str], source: Path) -> proc.ProcResult:
        if self.unreadable:
            given = argv[-1]
            return proc.ProcResult(1, "", f"{given}: Invalid data found when processing input", 0.0)
        streams = [] if source.read_text(encoding="utf-8").startswith("silent") else [{"codec_name": "aac"}]
        probe = {"streams": streams, "format": {"duration": "12.5", "tags": self.tags.get(source.name, {})}}
        return proc.ProcResult(0, json.dumps(probe), "", 0.0)

    def ffmpeg(self, argv: list[str], source: Path) -> proc.ProcResult:
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
    error: EngineError | None = None
    spoken: tuple[str, ...] = SPOKEN

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        words = tuple(
            Word(text=f" {w}", start=word_start(i), end=word_start(i) + 0.4) for i, w in enumerate(self.spoken)
        )
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
        atomic.write_json(out / FINGERPRINT_FILE, request.fingerprint())
        atomic.write_json(out / METRICS_FILE, metrics.model_dump(mode="json"))
        return WorkerResult(out / HYPOTHESES_FILE, metrics, 0, 0.0)


def word_start(index: int) -> float:
    """Where the fake worker puts word `index`: 0.4 s words, 0.1 s apart, from 1 s on."""
    return 1.0 + index * 0.5


FRAME_S = 0.01
FRAMES = 1250


@dataclass
class Diarizer:
    """Stands in for the diarization worker: `turns` are (first word, last word, channel), with the channel active
    from the first word's start to the last word's end; leaves what a finished worker leaves."""

    turns: list[tuple[int, int, int]] = field(default_factory=lambda: [(0, len(SPOKEN) - 1, 0)])
    calls: list[DiarizeRequest] = field(default_factory=list[DiarizeRequest])
    error: EngineError | None = None
    # Leaves a probabilities file that disagrees with its turns file.
    truncated: bool = False

    def __call__(self, request: DiarizeRequest, *, python: str | None = None) -> DiarizeResult:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        spec = request.diarizer.spec
        probs = np.zeros((FRAMES, spec.max_speakers), dtype=np.float32)
        turns: list[Turn] = []
        for first, last, channel in self.turns:
            start, end = word_start(first), word_start(last) + 0.4
            probs[round(start / FRAME_S) : round(end / FRAME_S), channel] = 0.9
            turns.append(Turn(start=start, end=end, speaker=channel))
        (job,) = request.files
        atomic.write_bytes(diarize_protocol.probs_path(out, job.id), probs.astype("<f2").tobytes())
        stored = FileDiarization(
            diarizer=spec.id,
            file=job.id,
            duration_s=12.5,
            threshold=request.diarizer.threshold(),
            frame_s=FRAME_S,
            frames=FRAMES,
            channels=spec.max_speakers,
            num_speakers=len({turn.speaker for turn in turns}),
            turns=tuple(turns),
        )
        atomic.write_json(diarize_protocol.turns_path(out, job.id), stored.model_dump(mode="json"))
        atomic.write_json(out / diarize_protocol.FINGERPRINT_FILE, request.fingerprint())
        metrics = diarize_protocol.RunMetrics(
            diarizer=spec.id,
            files=1,
            files_diarized=1,
            files_skipped=0,
            load_s=0.0,
            diarize_s=0.0,
            audio_s=12.5,
            peak_memory_bytes=None,
            max_rss_bytes=0,
        )
        atomic.write_json(out / diarize_protocol.METRICS_FILE, metrics.model_dump(mode="json"))
        if self.truncated:
            atomic.write_bytes(diarize_protocol.probs_path(out, job.id), b"\x00\x00")
        return DiarizeResult(out, metrics, 0, 0.0)


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
    monkeypatch.setattr(diarize_registry, "find_spec", lambda _name: object())


@pytest.fixture(autouse=True)
def diarizer(monkeypatch: pytest.MonkeyPatch) -> Diarizer:
    fake = Diarizer()
    monkeypatch.setattr(diarize_runner, "run_worker", fake)
    return fake


@pytest.fixture
def media(session_kit, monkeypatch: pytest.MonkeyPatch) -> Media:
    fake = Media()
    monkeypatch.setattr(proc, "run", fake)
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
    assert doc.chapters == () and [s.id for s in doc.speakers] == ["S1"], "one voice is still a diarized speaker"
    assert {s.speaker for s in doc.segments} == {"S1"} and doc.transcript.diarization is not None
    assert doc.segments[0].text.startswith("We met 5 people") and doc.transcript.punctuation == "llm"
    assert not any("caption track" in warning for warning in doc.quality.warnings), "recordings have no captions"
    notice = "numbers not cross-checked: recordings have no captions; 1 number(s) rest on the transcript alone"
    assert notice in results[0].notices

    markdown = (rdir / "context.md").read_text(encoding="utf-8")
    assert markdown.startswith(f"# {TITLE}\n") and "meeting.m4a" in markdown and "youtu" not in markdown
    assert "Speaker" not in markdown, "a single speaker needs no label"
    assert not (rdir / "audio.wav").exists() and media.converted == ["meeting.m4a"]
    assert path.read_text(encoding="utf-8") == "alpha\n.m4a\n", "the source file is never modified"

    (request,) = worker.calls
    assert request.engine.spec.id == "w-prod"
    assert request.files[0] == FileJob(id=rec_id(), wav=str(rdir / "audio.wav"), language=None)
    assert [e.stage for e in events] == [
        "fetch",
        "asr",
        "diarize",
        "normalize",
        "punctuate",
        "correct",
        "numeric",
        "write",
    ]
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
    assert [(r.outcome.video_id, r.name) for r in results] == [
        (rec_id(), "audio.m4a"),
        (rec_id(), "video.mp4"),
        (rec_id("beta"), "other.m4a"),
    ]
    assert len(worker.calls) == 2 and report.exit_code == 0
    assert ProgressEvent(rec_id(), "skip", "same audio as audio.m4a") in events
    batch = json.loads((session_kit.out / "batch_report.json").read_text(encoding="utf-8"))
    duplicate = batch["videos"][1]
    assert (duplicate["reused"], duplicate["detail"]) == (["duplicate"], "same audio as audio.m4a")


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
    first = process(str(path), out=session_kit.out)
    doc = first.document
    assert isinstance(doc.source, RecordingSource) and doc == load_document(session_kit.out / rec_id() / "context.json")
    assert process(path, out=session_kit.out).document == doc


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


def test_a_name_with_a_colon_is_a_file_and_errors_name_no_folder(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    folder = tmp_path / "Client Acme Secret"
    path = media.add(folder / "notes:1.m4a")
    _, results, _ = run_files(session_kit, [path])
    assert results[0].outcome.status.value == "ok" and recording(results[0]).file_name == "notes:1.m4a"

    media.unreadable = True
    _, results, _ = run_files(session_kit, [media.add(folder / "broken.m4a", "gamma")])
    report = (session_kit.out / "batch_report.json").read_text(encoding="utf-8")
    assert results[0].outcome.error == "media_invalid" and "./broken.m4a: Invalid data" in report
    assert "Client Acme Secret" not in report and str(tmp_path) not in report


def test_a_missing_ffmpeg_stops_the_batch_once(session_kit, worker: Worker, tmp_path: Path, monkeypatch) -> None:
    def missing(argv: list[str], **_: object) -> proc.ProcResult:
        raise DependencyError(f"required executable not found: {argv[0]}")

    monkeypatch.setattr(proc, "run", missing)
    files = [tmp_path / "a.m4a", tmp_path / "b.m4a"]
    for path in files:
        path.write_bytes(b"x")
    with pytest.raises(DependencyError, match="ffprobe"):
        run_files(session_kit, files)
    assert json.loads((session_kit.out / "batch_report.json").read_text(encoding="utf-8"))["videos"] == []


def test_a_failed_run_leaves_no_copy_of_the_audio(session_kit, media: Media, worker: Worker, tmp_path: Path) -> None:
    worker.error = EngineError("synthetic decode failure")
    _, results, _ = run_files(session_kit, [media.add(tmp_path / "meeting.m4a")])
    assert results[0].outcome.error == "engine"
    assert media.converted == ["meeting.m4a"] and not (session_kit.out / rec_id() / "audio.wav").exists()


def test_control_characters_in_a_file_name_are_dropped(
    session_kit, media: Media, worker: Worker, tmp_path: Path
) -> None:
    _, results, _ = run_files(session_kit, [media.add(tmp_path / "team\x1b[31mcall.m4a")])
    source = recording(results[0])
    assert (source.title, source.file_name) == ("team [31mcall", "team [31mcall.m4a")


@pytest.mark.parametrize(("given", "normalised"), [("zh-Hans", "zh"), ("EN", "en"), ("yue", "yue")])
def test_a_forced_language_is_normalised_up_front(session_kit, given: str, normalised: str) -> None:
    assert settings(session_kit, "recording", language=given).language == normalised


@pytest.mark.parametrize("given", ["klingon", "xx", "z"])
def test_an_unknown_forced_language_is_refused_before_any_work(session_kit, given: str) -> None:
    with pytest.raises(ConfigError, match="is not a language code w-prod knows"):
        settings(session_kit, "youtube", language=given)


def test_cli_reports_a_duplicate_in_the_summary(session_kit, media: Media, worker: Worker, tmp_path: Path) -> None:
    first = media.add(tmp_path / "audio.m4a")
    second = media.add(tmp_path / "video.mp4")
    result = cli(session_kit, str(first), str(second))
    assert result.exit_code == 0, result.output
    assert "= video.mp4  same audio as audio.m4a" in result.output
    assert "1 ok, 0 failed, 1 duplicate" in result.output


def test_cli_names_an_unreadable_folder_with_the_access_fix(session_kit, tmp_path: Path) -> None:
    locked = tmp_path / "in" / "Recordings"
    locked.mkdir(parents=True)
    locked.chmod(0)
    try:
        result = cli(session_kit, str(tmp_path / "in"))
    finally:
        locked.chmod(0o700)
    assert result.exit_code == 1
    assert "cannot read" in result.output and "Recordings: permission denied" in result.output
    assert "Full Disk Access" in result.output and "no audio or video files" not in result.output


def two_voices(diarizer: Diarizer) -> None:
    # "we met 5 people today" is one voice, "and then we left" another, with no pause between them.
    diarizer.turns = [(0, 4, 0), (5, 8, 3)]


def test_a_two_voice_recording_gets_speaker_paragraphs_and_reuses_both_workers(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    two_voices(diarizer)
    path = media.add(tmp_path / "meeting.m4a")
    _, _, events = run_files(session_kit, [path], llm=FakeLLM())

    rdir = session_kit.out / rec_id()
    doc = load_document(rdir / "context.json")
    assert [(s.text, s.speaker) for s in doc.segments] == [
        ("We met 5 people today.", "S1"),
        ("And then we left.", "S2"),
    ]
    assert [(p.speaker, p.first_segment, p.last_segment) for p in doc.paragraphs] == [("S1", 0, 0), ("S2", 1, 1)]
    assert [(s.id, s.name, s.speech_s) for s in doc.speakers] == [("S1", None, 2.4), ("S2", None, 1.9)]
    assert doc.transcript.diarization is not None and doc.transcript.diarization.engine == "d-nemotron-8bit"
    markdown = (rdir / "context.md").read_text(encoding="utf-8")
    assert "[00:01] **Speaker 1:** We met 5 people today.\n\n[00:03] **Speaker 2:** And then we left." in markdown
    assert "Speakers: Speaker 1 (56%) · Speaker 2 (44%)" in markdown

    stages = [e.stage for e in events]
    assert stages.index("asr") < stages.index("diarize") < stages.index("normalize")
    (request,) = diarizer.calls
    assert request.diarizer.spec.id == "d-nemotron-8bit" and request.files[0].wav == str(rdir / "audio.wav")
    assert not (rdir / "audio.wav").exists(), "the WAV goes once both workers' outputs are stored"

    (rdir / ".stages" / "asr.json").unlink()
    (rdir / "context.json").unlink()
    rerun, _, events = run_files(session_kit, [path], llm=FakeLLM())
    assert rerun.exit_code == 0 and len(worker.calls) == 1 and len(diarizer.calls) == 1
    assert media.converted == ["meeting.m4a"], "finished worker folders need no audio"
    assert "diarize" in [e.stage for e in events] and load_document(rdir / "context.json").segments == doc.segments


def test_the_asr_stage_stores_the_assignment_digest(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    two_voices(diarizer)
    run_files(session_kit, [media.add(tmp_path / "meeting.m4a")])
    stored = json.loads((session_kit.out / rec_id() / ".stages" / "asr.json").read_text(encoding="utf-8"))["output"]
    stretches = [[s["start"], s["end"], s["speaker"]] for s in stored["stretches"]]
    assert [speaker for _, _, speaker in stretches] == ["S1", "S2"]
    assert stored["assignment"] == store.digest(stretches)
    assert stored["diarization"]["params"]["attribution"]["smoothing"] is True


def test_overlapping_speech_is_flagged_in_json_only(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    # A second voice talks over "5 people".
    diarizer.turns = [(0, 8, 0), (2, 3, 1)]
    run_files(session_kit, [media.add(tmp_path / "meeting.m4a")])
    rdir = session_kit.out / rec_id()
    doc = load_document(rdir / "context.json")
    (segment,) = doc.segments
    (flag,) = [f for f in segment.flags if f.kind == "overlapping_speech"]
    assert segment.text[flag.char_start : flag.char_end] == "5 people"
    assert segment.speaker == "S1" and [s.id for s in doc.speakers] == ["S1"]
    markdown = (rdir / "context.md").read_text(encoding="utf-8")
    assert "overlapping_speech" not in markdown and "[^" not in markdown


def test_diarization_can_be_turned_off_for_a_recording(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    two_voices(diarizer)
    path = media.add(tmp_path / "meeting.m4a")
    off = config.load_options(overrides={"diarize": {"mode": "off"}})
    _, results, events = run_files(session_kit, [path], off)
    doc = results[0].document
    assert doc is not None and doc.speakers == () and doc.transcript.diarization is None
    assert diarizer.calls == [] and "diarize" not in [e.stage for e in events]
    assert all(s.speaker is None for s in doc.segments)


def test_cli_diarize_flags_set_the_mode(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    two_voices(diarizer)
    path = media.add(tmp_path / "meeting.m4a")
    result = cli(session_kit, "--no-diarize", str(path))
    assert result.exit_code == 0, result.output
    assert diarizer.calls == [] and "speaker" not in result.output
    result = cli(session_kit, "--diarize", str(path))
    assert result.exit_code == 0, result.output
    assert len(diarizer.calls) == 1 and "2 speakers" in result.output


def test_youtube_with_auto_is_not_diarized_and_keeps_its_keys(session_kit) -> None:
    youtube = settings(session_kit, "youtube")
    assert youtube.diarizer is None
    off = pipeline._settings(
        config.load_options(overrides={"diarize": {"mode": "off"}}),
        None,
        session_kit.out,
        kind="youtube",
        force=False,
        captions=True,
        emit=lambda _e: None,
        clock=lambda: NOW,
    )
    assert youtube.run_key() == off.run_key()
    decoding = pipeline._Decoding(youtube, "pubVideo001", session_kit.out / "pubVideo001", "en")
    assert decoding.diarize_request() is None
    request = decoding.asr_request()
    assert decoding.asr_key(request, None) == pipeline._key("asr", {"request": without_wav(request.fingerprint())})

    forced = pipeline._settings(
        config.load_options(overrides={"diarize": {"mode": "on"}}),
        None,
        session_kit.out,
        kind="youtube",
        force=False,
        captions=True,
        emit=lambda _e: None,
        clock=lambda: NOW,
    )
    assert forced.diarizer is not None and forced.run_key() != youtube.run_key()
    assert settings(session_kit, "recording").diarizer is not None


def test_a_failed_diarization_is_an_engine_error_and_a_rerun_keeps_the_decode(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarizer.error = EngineError("synthetic diarizer failure")
    path = media.add(tmp_path / "meeting.m4a")
    _, results, _ = run_files(session_kit, [path])
    (failed,) = results
    assert failed.outcome.error == "engine" and failed.error is not None
    assert isinstance(failed.error, EngineError) and failed.error.stage == "diarize"
    assert not (session_kit.out / rec_id() / "audio.wav").exists()

    diarizer.error = None
    report, results, _ = run_files(session_kit, [path])
    assert report.exit_code == 0 and results[0].document is not None
    assert len(worker.calls) == 1, "the finished decode is not repeated"
    assert len(diarizer.calls) == 2 and [s.id for s in results[0].document.speakers] == ["S1"]


def test_an_attribution_rule_change_rebuilds_the_document_without_either_worker(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two_voices(diarizer)
    path = media.add(tmp_path / "meeting.m4a")
    run_files(session_kit, [path])
    rdir = session_kit.out / rec_id()
    before = json.loads((rdir / ".stages" / "asr.json").read_text(encoding="utf-8"))["input_hash"]

    monkeypatch.setattr(attribution, "GAP_S", 0.4)
    _, results, events = run_files(session_kit, [path])
    stored = json.loads((rdir / ".stages" / "asr.json").read_text(encoding="utf-8"))
    assert stored["input_hash"] != before and stored["output"]["diarization"]["params"]["attribution"]["gap_s"] == 0.4
    assert "skip" not in [e.stage for e in events] and results[0].outcome.reused == ()
    assert (len(worker.calls), len(diarizer.calls), media.converted) == (1, 1, ["meeting.m4a"])


def test_a_moved_output_root_reuses_both_finished_worker_folders(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    two_voices(diarizer)
    path = media.add(tmp_path / "meeting.m4a")
    run_files(session_kit, [path])
    moved = tmp_path / "moved"
    session_kit.out.rename(moved)
    (moved / rec_id() / ".stages" / "asr.json").unlink()
    (moved / rec_id() / "context.json").unlink()

    events: list[ProgressEvent] = []
    report = process_files([path], out=moved, progress=events.append)
    assert report.exit_code == 0 and "asr" in [e.stage for e in events]
    assert (len(worker.calls), len(diarizer.calls), media.converted) == (1, 1, ["meeting.m4a"])
    doc = load_document(moved / rec_id() / "context.json")
    assert [s.id for s in doc.speakers] == ["S1", "S2"]


def test_a_failure_reading_the_diarization_back_is_reported_at_the_asr_stage(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarizer.truncated = True
    _, results, _ = run_files(session_kit, [media.add(tmp_path / "meeting.m4a")])
    error = results[0].error
    assert isinstance(error, EngineError) and error.stage == "asr" and "probs.f16" in str(error)


# Three voices of three words each, with short pauses between them.
THREE_VOICES = [(0, 2, 0), (3, 5, 3), (6, 8, 5)]


def speakers_cli(kit, *args: str):
    return CliRunner().invoke(main, ["speakers", "--out", str(kit.out), *args])


def diarized(kit, media: Media, diarizer: Diarizer, tmp_path: Path, turns=None, **kwargs) -> Path:
    diarizer.turns = turns or THREE_VOICES
    path = media.add(tmp_path / "meeting.m4a")
    run_files(kit, [path], **kwargs)
    return path


def stored_document(kit) -> ContextDocument:
    return load_document(kit.out / rec_id() / "context.json")


def test_list_mode_shows_each_speakers_id_name_talk_time_and_a_short_first_line(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    worker.spoken = tuple(f"word{i}" for i in range(22))
    diarized(session_kit, media, diarizer, tmp_path, turns=[(0, 10, 0), (11, 21, 3)])
    result = speakers_cli(session_kit, rec_id())
    assert result.exit_code == 0, result.output
    first, second = result.output.splitlines()
    assert first.startswith("S1  -  ") and "[0:01] word0 word1" in first and first.endswith("…")
    assert second.startswith("S2  -  ") and "[0:06] word11" in second
    assert all(len(line.split("] ", 1)[1]) <= 60 for line in (first, second))
    assert "  0:05  " in first, "talk time is m:ss"

    named = speakers_cli(session_kit, str(session_kit.out / rec_id() / "context.json"), "S2=Bob")
    assert named.exit_code == 0, named.output
    assert speakers_cli(session_kit, str(session_kit.out / rec_id())).output.splitlines()[1].startswith("S2  Bob  ")


def test_a_rename_rewrites_json_markdown_record_and_index_and_a_rerun_skips(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    before = stored_document(session_kit)
    mark = (rdir / ".stages" / "context.json").read_text(encoding="utf-8")

    result = speakers_cli(session_kit, rec_id(), "S1=Alice", "S2=Bob")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("S1  Alice  ") and lines[1].startswith("S2  Bob    ") and lines[2].startswith("S3  -")
    assert lines[3] == str(rdir / "context.md")

    after = stored_document(session_kit)
    assert [(s.id, s.name, s.speech_s) for s in after.speakers] == [
        ("S1", "Alice", before.speakers[0].speech_s),
        ("S2", "Bob", before.speakers[1].speech_s),
        ("S3", None, before.speakers[2].speech_s),
    ]
    assert after.segments == before.segments and after.created_at == before.created_at
    markdown = (rdir / "context.md").read_text(encoding="utf-8")
    assert "**Alice:**" in markdown and "**Bob:**" in markdown and "**Speaker 3:**" in markdown
    assert "Speakers: Alice (" in markdown
    assert "| Alice, Bob, Speaker 3 |" in (session_kit.out / INDEX_MD).read_text(encoding="utf-8")
    saved = json.loads((rdir / "speakers.json").read_text(encoding="utf-8"))
    assert saved["names"] == {"S1": "Alice", "S2": "Bob"}
    assert (rdir / ".stages" / "context.json").read_text(encoding="utf-8") != mark

    calls = (len(worker.calls), len(diarizer.calls), media.converted)
    report, results, events = run_files(session_kit, [path])
    assert report.exit_code == 0 and results[0].outcome.reused == ("context",)
    assert [e.stage for e in events] == ["skip"]
    assert (len(worker.calls), len(diarizer.calls), media.converted) == calls
    assert stored_document(session_kit) == after


def test_clearing_every_name_removes_the_file_and_restores_the_record_of_a_document_never_named(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    mark = (rdir / ".stages" / "context.json").read_text(encoding="utf-8")
    assert speakers_cli(session_kit, rec_id(), "S1=Alice", "S2=Bob").exit_code == 0
    assert speakers_cli(session_kit, rec_id(), "S1=", "S2=").exit_code == 0
    assert not (rdir / "speakers.json").exists()
    assert (rdir / ".stages" / "context.json").read_text(encoding="utf-8") == mark
    assert all(s.name is None for s in stored_document(session_kit).speakers)
    _, results, _ = run_files(session_kit, [path])
    assert results[0].outcome.reused == ("context",)


def test_sharing_a_name_merges_labels_talk_time_and_consecutive_paragraphs(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    before = stored_document(session_kit)
    assert [p.speaker for p in before.paragraphs] == ["S1", "S2", "S3"]
    talk = {s.id: s.speech_s for s in before.speakers}
    total = sum(talk.values())

    assert speakers_cli(session_kit, rec_id(), "S1=Alice", "S3=Alice").exit_code == 0
    markdown = (rdir / "context.md").read_text(encoding="utf-8")
    share = round(100 * (talk["S1"] + talk["S3"]) / total)
    assert f"Speakers: Alice ({share}%) · Speaker 2 ({100 - share}%)" in markdown
    assert markdown.count("**Alice:**") == 2 and markdown.count("**Speaker 2:**") == 1, "Bob sits between them"
    assert "| Alice, Speaker 2 |" in (session_kit.out / INDEX_MD).read_text(encoding="utf-8")

    assert speakers_cli(session_kit, rec_id(), "S1=Alice", "S2=Bob", "S3=Bob").exit_code == 0
    merged = (rdir / "context.md").read_text(encoding="utf-8")
    body = merged.split("## Transcript\n\n", 1)[1].split("\n\n")
    assert len(body) == 2 and body[0].startswith("[00:01] **Alice:**") and body[1].startswith("[00:02] **Bob:**")
    assert f"Bob ({round(100 * (talk['S2'] + talk['S3']) / total)}%)" in merged
    segments = stored_document(session_kit).segments
    assert [(s.speaker, s.paragraph) for s in segments] == [("S1", 0), ("S2", 1), ("S3", 2)], (
        "the document keeps its speakers and paragraphs; only the render joins them"
    )


def test_one_name_for_everyone_shows_no_labels(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    assert speakers_cli(session_kit, rec_id(), "S1=Alice", "S2=Alice", "S3=Alice").exit_code == 0
    markdown = (session_kit.out / rec_id() / "context.md").read_text(encoding="utf-8")
    assert "**" not in markdown and "Speakers:" not in markdown


def test_a_text_stage_rerun_keeps_the_names_and_sends_none_to_the_llm(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = diarized(session_kit, media, diarizer, tmp_path)
    assert speakers_cli(session_kit, rec_id(), "S1=Zelda", "S2=Quentin").exit_code == 0

    llm = FakeLLM()
    _, results, events = run_files(session_kit, [path], llm=llm)
    stages = [e.stage for e in events]
    assert "punctuate" in stages and "skip" not in stages
    assert llm.requests, "the text stages ran with the LLM"
    for request in llm.requests:
        assert "Zelda" not in repr(request) and "Quentin" not in repr(request)
    doc = results[0].document
    assert doc is not None
    assert [(s.id, s.name) for s in doc.speakers] == [("S1", "Zelda"), ("S2", "Quentin"), ("S3", None)]
    assert "**Zelda:**" in (session_kit.out / rec_id() / "context.md").read_text(encoding="utf-8")
    assert len(worker.calls) == 1 and len(diarizer.calls) == 1, "no worker ran again"

    _, results, _ = run_files(session_kit, [path], llm=FakeLLM())
    assert results[0].outcome.reused == ("context",)


def test_a_changed_assignment_drops_the_names_with_a_warning_and_they_can_be_set_again(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    assert speakers_cli(session_kit, rec_id(), "S1=Alice", "S2=Bob").exit_code == 0
    old = json.loads((rdir / "speakers.json").read_text(encoding="utf-8"))["assignment"]

    diarizer.turns = [(0, 3, 0), (4, 8, 3)]
    _, results, _ = run_files(session_kit, [path], force=True)
    (result,) = results
    assert result.document is not None and speakers.STALE_WARNING in result.notices
    assert [s.name for s in result.document.speakers] == [None, None]
    assert speakers.STALE_WARNING in result.document.quality.warnings
    markdown = (rdir / "context.md").read_text(encoding="utf-8")
    assert "Alice" not in markdown and "Bob" not in markdown and "**Speaker 1:**" in markdown
    assert json.loads((rdir / "speakers.json").read_text(encoding="utf-8"))["assignment"] == old

    _, results, _ = run_files(session_kit, [path])
    assert results[0].outcome.reused == ("context",), "the stale file is a stable input, not a reason to rebuild"

    listed = speakers_cli(session_kit, rec_id())
    assert listed.exit_code == 0 and "were not applied" in listed.output and "Alice" not in listed.output

    assert speakers_cli(session_kit, rec_id(), "S1=Carol").exit_code == 0
    fixed = stored_document(session_kit)
    assert [s.name for s in fixed.speakers] == ["Carol", None]
    assert speakers.STALE_WARNING not in fixed.quality.warnings
    assert json.loads((rdir / "speakers.json").read_text(encoding="utf-8"))["assignment"] != old


def test_an_unknown_id_fails_and_lists_the_valid_ones_without_touching_anything(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    before = (rdir / "context.json").read_bytes()
    result = speakers_cli(session_kit, rec_id(), "S1=Alice", "S9=Bob")
    assert result.exit_code == 1
    assert "no speaker S9; this recording has S1, S2, S3" in result.output
    assert (rdir / "context.json").read_bytes() == before and not (rdir / "speakers.json").exists()


def test_a_name_that_is_another_ids_default_label_is_refused(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    result = speakers_cli(session_kit, rec_id(), "S1=Speaker 2")
    assert result.exit_code == 1 and "default label of S2" in result.output
    assert not (session_kit.out / rec_id() / "speakers.json").exists()
    assert speakers_cli(session_kit, rec_id(), "S1=Speaker 1").exit_code == 0, "an id's own default label is fine"


@pytest.mark.parametrize("argument", ["Alice", "=Alice", "X1=Alice", "S0=Alice"])
def test_an_assignment_without_a_speaker_id_is_refused(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path, argument: str
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    result = speakers_cli(session_kit, rec_id(), argument)
    assert result.exit_code == 1 and "is not ID=NAME" in result.output


def test_a_recording_that_was_not_diarized_has_no_speakers_to_name(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = media.add(tmp_path / "meeting.m4a")
    off = config.load_options(overrides={"diarize": {"mode": "off"}})
    run_files(session_kit, [path], off)
    result = speakers_cli(session_kit, rec_id(), "S1=Alice")
    assert result.exit_code == 1 and "was not diarized" in result.output
    assert speakers_cli(session_kit, rec_id()).exit_code == 1


def test_a_garbage_names_file_fails_the_run_loudly(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = diarized(session_kit, media, diarizer, tmp_path)
    (session_kit.out / rec_id() / "speakers.json").write_text('{"names": 1}', encoding="utf-8")
    _, results, _ = run_files(session_kit, [path])
    assert results[0].outcome.error is not None and "speaker names file" in results[0].outcome.detail


@dataclass
class BrokenLLM(FakeLLM):
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMError("synthetic LLM outage")


def untouched(rdir: Path) -> tuple[bytes, bytes, bool]:
    return (rdir / "context.json").read_bytes(), (rdir / "context.md").read_bytes(), (rdir / "speakers.json").exists()


def test_a_rename_keeps_the_settings_the_document_was_made_with(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = diarized(session_kit, media, diarizer, tmp_path, llm=FakeLLM())
    assert speakers_cli(session_kit, rec_id(), "S1=Alice").exit_code == 0

    _, results, events = run_files(session_kit, [path], llm=FakeLLM())
    assert [e.stage for e in events] == ["skip"] and results[0].outcome.reused == ("context",)

    _, results, events = run_files(session_kit, [path])
    assert "skip" not in [e.stage for e in events], "the config file's settings are not the ones the document had"
    doc = results[0].document
    assert doc is not None and doc.transcript.correction_llm is None and doc.speakers[0].name == "Alice"
    assert (len(worker.calls), len(diarizer.calls)) == (1, 1)

    _, results, _ = run_files(session_kit, [path])
    assert results[0].outcome.reused == ("context",)


def test_a_rename_binds_to_the_attribution_of_the_document_not_of_a_newer_asr_stage(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    path = diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    shown = json.loads((rdir / ".stages" / "context.json").read_text(encoding="utf-8"))["output"]["assignment"]
    before = untouched(rdir)

    # Another diarization redoes the asr stage, and the run then fails before it writes a document.
    (rdir / ".stages" / "asr.json").unlink()
    for folder in rdir.glob(".stages/**/d-*"):
        shutil.rmtree(folder)
    diarizer.turns = [(0, 3, 0), (4, 8, 3)]
    _, results, _ = run_files(session_kit, [path], llm=BrokenLLM())
    assert results[0].document is None and len(diarizer.calls) == 2
    newer = json.loads((rdir / ".stages" / "asr.json").read_text(encoding="utf-8"))["output"]["assignment"]
    assert newer != shown and untouched(rdir) == before

    assert speakers_cli(session_kit, rec_id(), "S1=Alice").exit_code == 0
    assert json.loads((rdir / "speakers.json").read_text(encoding="utf-8"))["assignment"] == shown

    _, results, _ = run_files(session_kit, [path], llm=FakeLLM())
    doc = results[0].document
    assert doc is not None and [s.name for s in doc.speakers] == [None, None]
    assert speakers.STALE_WARNING in doc.quality.warnings


def test_a_rename_refuses_a_document_without_a_record_that_vouches_for_it(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    record = rdir / ".stages" / "context.json"
    kept = record.read_bytes()

    record.unlink()
    before = untouched(rdir)
    result = speakers_cli(session_kit, rec_id(), "S1=Alice")
    assert result.exit_code == 1 and "was interrupted; run `ytverbatim run` on it" in result.output
    assert untouched(rdir) == before and not record.exists()

    record.write_bytes(kept)
    (rdir / "context.json").write_bytes(before[0] + b"\n")
    before = untouched(rdir)
    result = speakers_cli(session_kit, rec_id(), "S1=Alice")
    assert result.exit_code == 1 and "was interrupted" in result.output
    assert untouched(rdir) == before and record.read_bytes() == kept


def test_a_source_never_transcribed_fails_with_exit_1_and_leaves_no_folder(session_kit) -> None:
    never = rec_id("never")
    for args in ([never], [never, "S1=Alice"]):
        result = speakers_cli(session_kit, *args)
        assert result.exit_code == 1 and "does not exist" in result.output
    assert not (session_kit.out / never).exists()
    assert speakers_cli(session_kit, "neither-a-folder-nor-an-id").exit_code == 2


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("S1=Alice", "S1=Bob"), "S1 is given twice"),
        (("s1=Alice", "S1=Bob"), "S1 is given twice"),
        (("S1=Al\nice",), "control character U+000A"),
        (("S1=Al\x07ice",), "control character U+0007"),
        (("S1=Al ice",), "control character U+2028"),
        (("S1=speaker  2",), "default label of S2"),
    ],
)
def test_a_repeated_id_a_control_character_or_another_ids_default_label_is_refused(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path, arguments, message: str
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    before = untouched(rdir)
    result = speakers_cli(session_kit, rec_id(), *arguments)
    assert result.exit_code == 1 and message in result.output
    assert untouched(rdir) == before


def test_a_hand_edited_name_with_a_control_character_is_caught(tmp_path: Path) -> None:
    (tmp_path / "speakers.json").write_text('{"assignment": "x", "names": {"S1": "A\\u0000B"}}', encoding="utf-8")
    with pytest.raises(ContractError, match="control character U\\+0000"):
        speakers.read(tmp_path)


def test_the_contract_refuses_another_ids_default_label_in_any_case_or_spacing(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    data = stored_document(session_kit).model_dump(mode="json")
    data["speakers"][0]["name"] = " SPEAKER   2 "
    with pytest.raises(ValidationError, match="the default label of S2"):
        ContextDocument.model_validate(data)
    data["speakers"][0]["name"] = "speaker 1"
    ContextDocument.model_validate(data)


def test_lowercase_ids_work_and_a_new_id_merges_onto_an_existing_name(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    assert speakers_cli(session_kit, rec_id(), "s1=Alice").exit_code == 0
    assert speakers_cli(session_kit, rec_id(), "S3=Alice").exit_code == 0
    assert [s.name for s in stored_document(session_kit).speakers] == ["Alice", None, "Alice"]
    markdown = (session_kit.out / rec_id() / "context.md").read_text(encoding="utf-8")
    assert "Speakers: Alice (" in markdown and markdown.count("Alice (") == 1


def test_a_rename_while_a_run_holds_the_source_fails_at_once(
    session_kit, media: Media, worker: Worker, diarizer: Diarizer, tmp_path: Path
) -> None:
    diarized(session_kit, media, diarizer, tmp_path)
    rdir = session_kit.out / rec_id()
    before = untouched(rdir)
    with store.lock(rdir):
        result = speakers_cli(session_kit, rec_id(), "S1=Alice")
    assert result.exit_code == 1 and "another ytverbatim run is processing" in result.output
    assert untouched(rdir) == before


@dataclass
class LidWorker(Worker):
    """A detecting engine's worker: one row per (text, language), one after another, without word timings."""

    rows: tuple[tuple[str, str], ...] = ()

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        result = super().__call__(request, python=python)
        job = request.files[0]
        lines = [
            ChunkHypothesis(
                hypothesis=request.engine.spec.id,
                file=job.id,
                chunk=index,
                start=4.0 * index,
                end=4.0 * index + 3.5,
                text=text,
                language=language,
            ).model_dump_json()
            for index, (text, language) in enumerate(self.rows)
        ]
        (Path(request.out_dir) / HYPOTHESES_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result


LID_OPTIONS = config.load_options(
    overrides={"asr": {"engine": "w-runs-lid"}, "text": {"fillers": "remove"}, "diarize": {"mode": "off"}}
)


def lid_run(kit, media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *rows: tuple[str, str]):
    monkeypatch.setattr(asr_runner, "run_worker", LidWorker(rows=rows))
    llm = FakeLLM()
    _, results, _ = run_files(kit, [media.add(tmp_path / "meeting.m4a")], LID_OPTIONS, llm=llm)
    doc = results[0].document
    assert doc is not None
    return doc, llm


def punctuation_system(llm: FakeLLM, text: str) -> str:
    """The system prompt of the punctuation batch whose segment, not just a neighbour's edge, is `text`."""
    for request in llm.requests:
        if "SEGMENTS:" in request.user and '"items"' in request.user:
            segments = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
            if any(item["text"] == text for item in segments):
                return request.system
    raise AssertionError(f"no punctuation batch holds {text!r}")


def test_a_mandarin_stretch_in_an_english_recording_gets_mandarin_text_rules(
    session_kit, media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc, llm = lid_run(
        session_kit,
        media,
        tmp_path,
        monkeypatch,
        ("so we met today and talked about the budget for next year", "en"),
        ("我們開會了 嗯 三個人都在", "zh"),
    )
    assert doc.transcript.language == "en"
    assert [s.text for s in doc.segments] == [
        "So we met today and talked about the budget for next year.",
        "我们开会了 3个人都在.",
    ]
    assert "full-width Chinese punctuation" in punctuation_system(llm, "我们开会了 三个人都在")
    assert "Use ASCII punctuation" in punctuation_system(
        llm, "so we met today and talked about the budget for next year"
    )
    (correction,) = [r for r in llm.requests if '"fixed"' in r.user]
    assert "The transcript is mainly en, with zh mixed in; keep every word in the language it was spoken." in (
        correction.user
    )


def test_an_english_stretch_in_a_mandarin_recording_keeps_english_text_rules(
    session_kit, media: Media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc, llm = lid_run(
        session_kit,
        media,
        tmp_path,
        monkeypatch,
        ("今天我们开会讨论明年的预算大家都来了", "zh"),
        ("um I think so", "en"),
    )
    assert doc.transcript.language == "zh"
    assert [s.text for s in doc.segments][1] == "I think so."
    assert "Use ASCII punctuation" in punctuation_system(llm, "I think so")


def test_stretch_languages_change_only_the_asr_key_of_a_recording_with_a_chunked_engine(session_kit) -> None:
    def asr_key(kind: pipeline.SourceKind, engine: str) -> tuple[str, str]:
        decoding = pipeline._Decoding(settings(session_kit, kind, engine=engine), "src", session_kit.out / "src", None)
        request = decoding.asr_request()
        return decoding.asr_key(request, None), pipeline._key("asr", {"request": without_wav(request.fingerprint())})

    for kind, engine in (("youtube", "w-runs-lid"), ("youtube", "w-prod"), ("recording", "w-prod")):
        key, plain = asr_key(kind, engine)  # type: ignore[arg-type]
        assert key == plain, (kind, engine)
    key, plain = asr_key("recording", "w-runs-lid")
    assert key != plain


def test_an_empty_glossary_keeps_the_youtube_run_key_and_a_glossary_changes_it(session_kit) -> None:
    def run_key(**text: object) -> str:
        options = config.load_options(overrides={"text": text})
        return pipeline._settings(
            options,
            None,
            session_kit.out,
            kind="youtube",
            force=False,
            captions=True,
            emit=lambda _e: None,
            clock=lambda: NOW,
        ).run_key()

    # Computed on main before the glossary option existed.
    assert run_key() == "c9f9638ad6b4b5006329444a05747d4f763a2570f2bbff3c80aa148c59fa98a7"
    assert run_key(glossary=()) == run_key(glossary=(" ", "")) == run_key()
    assert run_key(glossary=("Acme",)) != run_key()


def test_the_glossary_is_trimmed_deduplicated_and_capped(session_kit) -> None:
    given = (" Acme ", "", "Acme", "Kubernetes")
    options = config.load_options(overrides={"text": {"glossary": given}})
    built = pipeline._settings(
        options,
        None,
        session_kit.out,
        kind="recording",
        force=False,
        captions=True,
        emit=lambda _e: None,
        clock=lambda: NOW,
    )
    assert built.text.glossary == ("Acme", "Kubernetes")
    many = config.load_options(overrides={"text": {"glossary": tuple(f"term{n}" for n in range(21))}})
    with pytest.raises(ConfigError, match="at most 20"):
        pipeline._settings(
            many,
            None,
            session_kit.out,
            kind="youtube",
            force=False,
            captions=True,
            emit=lambda _e: None,
            clock=lambda: NOW,
        )
