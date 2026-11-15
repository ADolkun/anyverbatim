"""`ytverbatim run`: transcribe, refine the text and write `context.json` plus `context.md` per video or recording.

Stages, per video under its directory lock: metadata, captions and audio (the fetch session's) → asr (the worker
subprocess, which also plans the VAD chunks) or manual captions → normalize → punctuate → correct → numeric →
numerals (Mandarin only) → segment, chapters and assemble (`stages.build_document`) → render. Every stage from asr to
numerals is checkpointed in the video's `StageStore`, keyed by what it read, so a rerun with changed options
recomputes only the stages those options reach, and an interrupted run resumes where it stopped. The 16 kHz WAV is
deleted once the ASR output is stored; the compressed audio stays.

`_YouTubeRun` is the YouTube adapter: it fetches, then hands a `_Prepared` source to `_Transcription`, which runs
everything from asr to the written files and knows nothing of where the source came from. `_FileRun` is the adapter
for a local recording, whose id is the hash of its audio (`acquire.local`). Both run inside `_run_source`, which
holds the source's lock, skips a current document and records a failure as the source's outcome. YouTube batches are
the fetch session's, with cooldowns; local files get the plain loop in `process_files`.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from ytverbatim import atomic, config, ids, index, paths, stages, store
from ytverbatim.acquire import local
from ytverbatim.acquire.audio import WAV_NAME
from ytverbatim.acquire.captions import CaptionKind, CaptionTrack
from ytverbatim.acquire.metadata import VideoMetadata, info_path, load_metadata
from ytverbatim.acquire.session import (
    REPORT_FILE,
    BatchReport,
    FetchOptions,
    FetchSession,
    PlannedVideo,
    VideoOutcome,
    VideoStatus,
    default_runner,
    explain,
    fetch,
    in_stage,
    saved_captions,
)
from ytverbatim.acquire.video import ready_video
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import METRICS_FILE, ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.auth.cooldown import utcnow
from ytverbatim.auth.source import CookieSource, resolve_source
from ytverbatim.contract import (
    Access,
    ContextDocument,
    LLMUsage,
    RecordingSource,
    YouTubeSource,
    dump_document,
    load_document,
)
from ytverbatim.errors import (
    BatchStopError,
    BusyError,
    ConfigError,
    ContractError,
    DependencyError,
    EngineError,
    EngineMemoryError,
    LLMError,
    MediaValidationError,
    ProcTimeout,
    YtDlpError,
    YtverbatimError,
)
from ytverbatim.llm import check_ready, client_from_options
from ytverbatim.llm.protocol import LLMClient
from ytverbatim.mlx_worker import without_wav
from ytverbatim.render import render_markdown
from ytverbatim.stages import (
    SourceInfo,
    Stretch,
    Transcribed,
    build_document,
    stretch_flag,
    with_results,
)
from ytverbatim.store import StageStore, input_hash
from ytverbatim.store import lock as video_lock
from ytverbatim.targets import classify as classify_target
from ytverbatim.text import clean, guards, numbers, numerals, numeric
from ytverbatim.text import correct as correct_module
from ytverbatim.text import punctuate as punctuate_module
from ytverbatim.text import segment as seg
from ytverbatim.text.correct import CorrectionContext, correct_segments
from ytverbatim.text.punctuate import needs_punctuation, punctuate
from ytverbatim.ytdlp.runner import redact
from ytverbatim.ytdlp.types import YtDlpRunner

log = logging.getLogger(__name__)

CONTEXT_JSON = "context.json"
CONTEXT_MD = "context.md"

# The pipeline's own part of each stage's version: bump it when this module changes what a stage outputs for the same
# input. The text modules' versions are folded in by `stage_versions`, so a prompt, tokenizer, guard or rule change
# there redoes that stage, everything after it, and the context record. LLM checkpoints are keyed by the request alone
# (`PROMPT_VERSION`, client, prompt), so a `numbers` or `guards` bump re-applies the new rules to the stored replies
# and re-asks only where a request changes: a numeric batch whose spans the tokenizer added, split or joined, or a
# different second correction pass. Worker folders are keyed by the decode request alone, so an `asr` or
# `second_opinion` bump re-reads the decoded rows they keep instead of decoding the audio again.
_PIPELINE_VERSIONS: dict[str, int] = {
    "asr": 2,
    "captions": 1,
    "second_opinion": 1,
    "normalize": 1,
    "punctuate": 2,
    "correct": 1,
    "numeric": 1,
    "numerals": 1,
    "context": 4,
}
_GLOSSARY_MAX_TERMS = 20


def stage_versions() -> dict[str, str]:
    """Each stored stage's version: the pipeline's own, plus the versions of the modules it runs."""
    modules = {
        "asr": [stages.VERSION],
        "captions": [stages.VERSION],
        "second_opinion": [numeric.PROMPT_VERSION, numbers.VERSION],
        "normalize": [clean.VERSION, stages.VERSION],
        "punctuate": [punctuate_module.PROMPT_VERSION, stages.VERSION],
        "correct": [correct_module.PROMPT_VERSION, stages.VERSION, numbers.VERSION, guards.VERSION],
        "numeric": [numeric.PROMPT_VERSION, stages.VERSION, numbers.VERSION, guards.VERSION],
        "numerals": [numerals.VERSION, numbers.VERSION, stages.VERSION],
        "context": [stages.VERSION],
    }
    return {stage: "+".join([str(version), *modules[stage]]) for stage, version in _PIPELINE_VERSIONS.items()}


__all__ = [
    "CONTEXT_JSON",
    "CONTEXT_MD",
    "ProgressEvent",
    "Result",
    "StageTiming",
    "VideoResult",
    "check_dependencies",
    "process",
    "process_files",
    "process_many",
    "video_lock",
]

SourceKind = Literal["youtube", "recording"]
# The `reused` mark of a file whose audio already ran in the same batch under another name.
DUPLICATE = "duplicate"
# The `reused` mark of a file whose source another run holds the lock of: it is skipped, not failed.
BUSY = "busy"
# The engine a source gets when `[asr] engine` is unset; recordings move to the language policy once it is chosen.
_DEFAULT_ENGINE: dict[SourceKind, str] = {"youtube": "w-prod", "recording": "w-prod"}
_WHISPER_FACTORY = "ytverbatim.asr.engines.whisper_mlx."


@dataclass(frozen=True)
class ProgressEvent:
    """One step of a source's run: `id` is the video or recording id.

    `stage` is one of `fetch`, `skip`, `asr`, `diarize`, `second_opinion`, `normalize`, `punctuate`, `correct`,
    `numeric`, `numerals` or `write`; these names are part of the public API.
    """

    id: str
    stage: str
    detail: str = ""


@dataclass(frozen=True)
class StageTiming:
    stage: str
    seconds: float
    reused: bool


@dataclass(frozen=True)
class VideoResult:
    outcome: VideoOutcome
    document: ContextDocument | None = None
    markdown: Path | None = None
    elapsed_s: float = 0.0
    # The exception behind a failed outcome when the pipeline caught one, so `process` can raise it unchanged.
    error: Exception | None = None
    # Stages that could not run as asked (no LLM for punctuation, no manual captions); also in `quality.warnings`.
    notices: tuple[str, ...] = ()
    timings: tuple[StageTiming, ...] = ()
    # A local recording's file name, which says more to a reader than its id; None for a video.
    name: str | None = None
    # The exact `context.json` that `document` was read from, taken under the source's lock; None without a document.
    json_bytes: bytes | None = None
    # The validated `video.mp4`, probed under the same lock; None unless the video track was requested.
    video_path: Path | None = None


@dataclass(frozen=True)
class Result:
    """What `process` made of one video or recording.

    `json_bytes` is the exact `context.json` on disk, read while the source's lock was held, so it is the file
    `document` was loaded from even when another process rewrites it a moment later. `video_path` is the validated
    `video.mp4`, set only when the video track was requested (`ytdlp.video`). `reused` names what an earlier run
    already had (`context` when the whole document was current, and `info`, `captions`, `audio`, `wav` or `video`
    for the fetched parts); `notices` say which stages could not run as asked.
    """

    id: str
    kind: SourceKind
    document: ContextDocument
    json_bytes: bytes
    directory: Path
    markdown_path: Path
    video_path: Path | None
    reused: tuple[str, ...]
    notices: tuple[str, ...]


def process(
    target: str | os.PathLike[str],
    options: config.Options | None = None,
    *,
    llm: LLMClient | None = None,
    out: Path | None = None,
    auth: str | None = None,
    captions: bool | None = None,
    force: bool = False,
    cookies_first: bool | None = None,
    interactive: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> Result:
    """Produce the context document for one video or local file, reusing it when nothing it depends on has changed.

    `target` is a YouTube video URL or id, or a local audio or video file (a path object is always a file).
    `llm` is the client for punctuation, correction and numeric verdicts; without it the configured `[llm]`
    provider applies, and with none configured those stages degrade as `quality.warnings` records. Raises the
    source's own error (`AuthError`, `EngineError`, `LLMError`, `ContractError`, `MediaValidationError`, ...), or
    `BatchStopError` when YouTube asked us to back off before the video was done. Every such error has its `fix` set
    (empty when there is no specific remedy) and, for a failure inside a stage, its `stage`. `interactive` lets the
    configured chatgpt LLM open a browser to sign in again when its sign-in is missing or lapses; off by default,
    because a library must not open a browser on its own.

    YouTube only, and a `ValueError` when given for a file: `auth` is an auth spec as for `--auth`; without it the
    configured source applies. `captions` (default on) downloads the caption tracks. `cookies_first` sends the video's
    first request with cookies, for a video known to need a login such as one from a members playlist; without it the
    first request is anonymous and cookies follow only when YouTube asks for them.
    """
    found = classify_target(target)
    if isinstance(found, Path):
        youtube_only = {"auth": auth, "captions": captions, "cookies_first": cookies_first}
        given = [name for name, value in youtube_only.items() if value is not None]
        if given:
            raise ValueError(f"{', '.join(given)} only apply to YouTube videos, not to the local file {found.name}")
        if found.is_dir():
            raise ValueError(f"process takes one file, not the folder {found.name}; call it once per file")
        return _process_file(found, options, llm=llm, out=out, force=force, interactive=interactive, progress=progress)
    if not isinstance(found, ids.YouTubeRef) or found.kind is not ids.RefKind.VIDEO:
        raise ValueError(
            f"process takes one YouTube video or local file, not {target!r}; use process_many for playlists and "
            "channels"
        )
    options = options or config.load_options()
    source: CookieSource | None = None
    try:
        source = resolve_source(auth)
        results: list[VideoResult] = []
        report = process_many(
            [found.url],
            options,
            llm=llm,
            out=out,
            source=source,
            captions=True if captions is None else captions,
            force=force,
            cookies_first=bool(cookies_first),
            interactive=interactive,
            progress=progress,
            on_video=results.append,
            # Looked up on every call, so a test can swap the yt-dlp runner without a parameter on the public API.
            runner_factory=default_runner,
        )
        (result,) = results
        if result.document is not None:
            if report.stopped is not None:
                # The cooldown file is already written, so later runs wait; this video itself finished.
                log.warning("%s is done, but YouTube asked to back off: %s", found.id, report.stopped)
            return _finished(result, "youtube", out)
        if report.stopped is not None:
            raise report.stopped
        _raise(result)
    except YtverbatimError as exc:
        explain(exc, source)
        raise


def _process_file(
    path: Path,
    options: config.Options | None,
    *,
    llm: LLMClient | None,
    out: Path | None,
    force: bool,
    interactive: bool,
    progress: Callable[[ProgressEvent], None] | None,
) -> Result:
    results: list[VideoResult] = []
    process_files(
        [path],
        options,
        llm=llm,
        out=out,
        force=force,
        interactive=interactive,
        progress=progress,
        on_video=results.append,
    )
    (result,) = results
    if result.document is not None:
        return _finished(result, "recording", out)
    _raise(result)


def _finished(result: VideoResult, kind: SourceKind, out: Path | None) -> Result:
    if result.document is None or result.json_bytes is None:
        raise YtverbatimError(f"{result.outcome.video_id} finished without a document")
    source_id = result.outcome.video_id
    directory = paths.source_dir(source_id, out)
    return Result(
        id=source_id,
        kind=kind,
        document=result.document,
        json_bytes=result.json_bytes,
        directory=directory,
        markdown_path=directory / CONTEXT_MD,
        video_path=result.video_path,
        reused=result.outcome.reused,
        notices=result.notices,
    )


def check_dependencies(options: config.Options | None = None, *, source: SourceKind = "youtube") -> None:
    """Raise `DependencyError` naming the extra to install when an engine a run of this source needs is missing.

    Checks the ASR engine `options` select for `source` (`w-prod` unless `[asr] engine` is set), and the numeric
    second-opinion engine when one is on; each names the extra its library comes from (`whisper` or `asr`). Nothing
    is imported or downloaded. An unknown engine is a `ConfigError`, an unknown `source` a `ValueError`. `process`
    runs the same check before any request.
    """
    if source not in _DEFAULT_ENGINE:
        raise ValueError(f"source must be one of {', '.join(_DEFAULT_ENGINE)}, not {source!r}")
    options = options or config.load_options()
    registry.check_dependencies(registry.get(options.asr.engine or _DEFAULT_ENGINE[source]))
    if options.text.numeric_second_opinion != "off":
        registry.check_dependencies(registry.get(options.text.numeric_second_opinion))


def _raise(result: VideoResult) -> NoReturn:
    if result.error is not None:
        raise result.error
    raise _error_for(result.outcome)


def process_many(  # noqa: PLR0913
    targets: Sequence[str],
    options: config.Options | None = None,
    *,
    llm: LLMClient | None = None,
    out: Path | None = None,
    source: CookieSource | None = None,
    captions: bool = True,
    force: bool = False,
    ignore_cooldown: bool = False,
    cookies_first: bool = False,
    interactive: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
    on_video: Callable[[VideoResult], None] | None = None,
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = default_runner,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = utcnow,
) -> BatchReport:
    """Run every video behind `targets` into `<out>/<id>/` and write `<out>/batch_report.json`.

    Raises `ConfigError` for options that cannot run (an unknown engine, correction without an LLM),
    `DependencyError` when an engine's extra is missing, and refuses a cooldown (`BatchStopError`), all before any
    request; the batch itself is `fetch`'s, with this pipeline as its per-video hook. Per-video failures land in the
    report; a batch stop mid-run ends the batch and is recorded as well. An ASR worker killed by the memory watch
    raises `EngineMemoryError`, after the report of the videos before it is written: every later video would run the
    same engine into the same limit. `interactive` is as for `process`.
    """
    options = options or config.load_options()
    out = out if out is not None else paths.default_output_root()
    captions = captions or options.asr.source == "captions"
    settings = _settings(
        options,
        llm,
        out,
        kind="youtube",
        force=force,
        captions=captions,
        interactive=interactive,
        emit=progress or _no_progress,
        clock=clock,
    )
    fetch_options = FetchOptions(
        policy=options.auth.policy,
        captions=captions,
        cookie_channels=options.auth.cookie_channels,
        max_videos=options.ytdlp.max_videos_per_run,
        ignore_cooldown=ignore_cooldown,
        cookies_first=cookies_first,
        creator_note=options.ytdlp.creator_note,
        video=options.ytdlp.video,
    )
    hook = _RunHook(settings)

    def emit(outcome: VideoOutcome) -> None:
        result = hook.results.pop(outcome.video_id, None)
        if on_video is not None:
            on_video(result if result is not None else VideoResult(outcome))

    return fetch(
        targets,
        source=source,
        out=out,
        options=fetch_options,
        runner_factory=runner_factory,
        sleep=sleep,
        clock=clock,
        on_video=emit,
        hook=hook,
    )


def process_files(
    files: Sequence[Path],
    options: config.Options | None = None,
    *,
    llm: LLMClient | None = None,
    out: Path | None = None,
    force: bool = False,
    interactive: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
    on_video: Callable[[VideoResult], None] | None = None,
    clock: Callable[[], datetime] = utcnow,
    provenance: Mapping[Path, local.Provenance] | None = None,
) -> BatchReport:
    """Run every local audio or video file into `<out>/<rec-id>/`; write `<out>/batch_report.json` and `index.md`.

    A recording's id is the hash of its audio, so the same audio under two names, or in another container, runs once.
    Option errors are raised before any file is read, as for `process_many`; a file that cannot be read or
    transcribed is an error in the report, keyed by its file name until it has an id. `EngineMemoryError` ends the
    batch after the report is written.
    """
    options = options or config.load_options()
    out = out if out is not None else paths.recordings_root()
    settings = _settings(
        options,
        llm,
        out,
        kind="recording",
        force=force,
        captions=False,
        interactive=interactive,
        emit=progress or _no_progress,
        clock=clock,
    )
    started = clock()
    outcomes: list[VideoOutcome] = []
    seen: dict[str, str] = {}
    listed = index.Index.scan(out)
    try:
        for path in files:
            result = _run_file(path, settings, seen, (provenance or {}).get(path))
            outcomes.append(result.outcome)
            if result.document is not None and "context" not in result.outcome.reused:
                listed.add(result.document)
                listed.write()
            if on_video is not None:
                on_video(result)
    finally:
        report = BatchReport(tuple(outcomes), started_at=started, finished_at=clock())
        out.mkdir(parents=True, exist_ok=True)
        atomic.write_json(out / REPORT_FILE, report.to_json())
    index.write_index(out)
    return report


def _run_file(
    path: Path, settings: "_Settings", seen: dict[str, str], provenance: local.Provenance | None
) -> VideoResult:
    """Run one file; a file whose audio already ran in this batch under another name is reported as a duplicate."""
    began = time.monotonic()
    try:
        recording = local.open_recording(path, provenance)
    except (DependencyError, EngineMemoryError):
        # A missing ffprobe or ffmpeg would fail every file the same way.
        raise
    except (YtverbatimError, ValueError, OSError) as exc:
        return replace(_failed(path.name, exc), name=path.name)
    if recording.id in seen:
        detail = f"same audio as {seen[recording.id]}"
        settings.emit(ProgressEvent(recording.id, "skip", detail))
        outcome = VideoOutcome(recording.id, VideoStatus.OK, reused=(DUPLICATE,), detail=detail)
        return VideoResult(outcome, name=recording.file_name)
    seen[recording.id] = recording.file_name
    sdir = paths.source_dir(recording.id, settings.out)
    adapter = _FileRun(recording, sdir, settings)

    def current(document: ContextDocument) -> bool:
        # Caller-given metadata changes (a renamed memo) must reach the document; a plain file keeps the first run's.
        return provenance is None or _same_record(document.source, recording.source(settings.language))

    result = _run_source(recording.id, sdir, settings, reuse=adapter.reuse, run=adapter.run, current=current)
    if isinstance(result.error, BusyError):
        outcome = VideoOutcome(recording.id, VideoStatus.OK, reused=(BUSY,), detail="being transcribed by another run")
        return VideoResult(outcome, name=recording.file_name)
    return replace(result, elapsed_s=time.monotonic() - began, name=recording.file_name)


def _same_record(stored: object, wanted: RecordingSource) -> bool:
    return isinstance(stored, RecordingSource) and (
        stored.title,
        stored.recorded_at,
        stored.origin,
        stored.file_name,
    ) == (wanted.title, wanted.recorded_at, wanted.origin, wanted.file_name)


def _no_progress(_event: ProgressEvent) -> None:
    return None


@dataclass(frozen=True)
class _Settings:
    """One batch's settings; a batch runs one kind of source, and every source default is resolved for it."""

    kind: SourceKind
    text: config.TextOptions
    source: Literal["asr", "captions"]
    definition: EngineDef
    second_opinion: EngineDef | None
    llm: LLMClient | None
    memory_cap_bytes: int | None
    out: Path
    force: bool
    captions: bool
    creator_note: bool
    video: bool
    emit: Callable[[ProgressEvent], None]
    clock: Callable[[], datetime]
    # The language forced on every source, else None.
    language: str | None = None
    # The allowed set handed to a detecting engine; empty unless the batch's sources and engine use it.
    languages: tuple[str, ...] = ()

    @property
    def llm_id(self) -> str | None:
        return self.llm.id if self.llm is not None else None

    def run_key(self) -> str:
        """Everything a finished document depends on besides the source itself; a change means a rerun."""
        versions = stage_versions()
        inputs: dict[str, JsonValue] = {
            "versions": dict[str, JsonValue](versions),
            "engine": _engine_key(self.definition),
            "second_opinion": _engine_key(self.second_opinion) if self.second_opinion else None,
            "text": _plain(asdict(self.text)),
            "llm": self.llm_id,
        }
        if self.kind == "youtube":
            inputs |= {"source": self.source, "captions": self.captions}
        # Each added only when set, so the keys of documents made before the option existed stay valid.
        if self.creator_note:
            inputs["creator_note"] = True
        if self.language is not None:
            inputs["language"] = self.language
        if self.languages:
            inputs["languages"] = list(self.languages)
        return input_hash("context", versions["context"], inputs)


def _settings(
    options: config.Options,
    llm: LLMClient | None,
    out: Path,
    *,
    kind: SourceKind,
    force: bool,
    captions: bool,
    emit: Callable[[ProgressEvent], None],
    clock: Callable[[], datetime],
    interactive: bool = False,
) -> _Settings:
    check_dependencies(options, source=kind)
    definition = registry.get(options.asr.engine or _DEFAULT_ENGINE[kind])
    languages = _allowed_languages(options.asr.languages, definition) if kind == "recording" else ()
    second: EngineDef | None = None
    if options.text.numeric_second_opinion != "off":
        second = registry.get(options.text.numeric_second_opinion)
        if second.spec.max_chunk_s is None:
            raise ConfigError(
                f"text.numeric_second_opinion = {second.spec.id!r} decodes whole files; it needs a chunked engine "
                "that can re-decode just the numeric spans"
            )
        if second.decodes_runs():
            raise ConfigError(
                f"text.numeric_second_opinion = {second.spec.id!r} merges same-language spans into runs, so it would "
                "decode the audio between the numeric spans too; choose a chunk-by-chunk engine such as w-vad"
            )
    if llm is None and options.llm.provider != "none":
        # A missing `claude` CLI or base URL must fail now, not after an hour of ASR.
        check_ready(options.llm, interactive=interactive)
        llm = client_from_options(options.llm, interactive=interactive)
    if options.text.correct and llm is None:
        raise ConfigError("correction needs an LLM; pass --llm PROVIDER:MODEL or set [llm], or use --no-correct")
    text = replace(options.text, correct=llm is not None if options.text.correct is None else options.text.correct)
    memory_cap = asr_runner.memory_cap_bytes(options.asr.memory_cap_gib)
    youtube = kind == "youtube"
    return _Settings(
        kind=kind,
        text=text,
        source=options.asr.source,
        definition=definition,
        second_opinion=second,
        llm=llm,
        memory_cap_bytes=memory_cap,
        out=out,
        force=force,
        captions=captions,
        creator_note=youtube and options.ytdlp.creator_note,
        video=youtube and options.ytdlp.video,
        emit=emit,
        clock=clock,
        language=_forced_language(options.asr.language, definition),
        languages=languages,
    )


def _forced_language(code: str | None, definition: EngineDef) -> str | None:
    """Return the forced language as the primary code the engines take, or raise for one the engine cannot take.

    The Whisper adapter quietly auto-detects an unknown code, so a mistyped `--language` would go unnoticed.
    """
    if code is None:
        return None
    primary = re.split(r"[-_]", code.strip(), maxsplit=1)[0].lower()
    whisper = definition.factory.startswith(_WHISPER_FACTORY)
    if not re.fullmatch(r"[a-z]{2,3}", primary) or (whisper and primary not in registry.WHISPER_LANGUAGES):
        raise ConfigError(f"asr.language = {code!r} is not a language code {definition.spec.id} knows (zh, en, ...)")
    return primary


def _allowed_languages(languages: tuple[str, ...], definition: EngineDef) -> tuple[str, ...]:
    """Return the allowed set as the worker takes it; an engine that does not detect languages has no use for one."""
    if not languages or definition.language_policy() != "detect":
        return ()
    try:
        job = FileJob(id="check", wav="", languages=languages)
        WorkerRequest(engine=definition, out_dir="", files=(job,))
    except ValidationError as exc:
        raise ConfigError(f"asr.languages = {list(languages)} cannot run with {definition.spec.id}:\n{exc}") from exc
    return job.languages


def _engine_key(definition: EngineDef) -> JsonValue:
    return _plain(asdict(definition))


def _plain(value: object) -> JsonValue:
    return json.loads(json.dumps(value))


class _ContextMark(BaseModel):
    """Written after `context.json`: the document there was made with these settings and is this exact file."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: str


@dataclass
class _RunHook:
    """`fetch`'s per-video hook for `run`; keeps each video's full result for `process_many`'s `on_video`."""

    settings: _Settings
    results: dict[str, VideoResult] = field(default_factory=dict[str, VideoResult])

    def done(self, video_id: str) -> bool:
        try:
            vdir = paths.video_dir(video_id, self.settings.out)
            if _reusable(vdir, self.settings) is None:
                return False
            return not self.settings.video or ready_video(vdir) is not None
        except (OSError, YtverbatimError):
            # A failed probe must not abort the batch from its pre-check; the video's own run records it.
            return False

    def __call__(self, session: FetchSession, planned: PlannedVideo) -> VideoOutcome:
        began = time.monotonic()
        result = _run_video(session, planned, self.settings)
        self.results[planned.video_id] = replace(result, elapsed_s=time.monotonic() - began)
        return result.outcome


def _reusable(vdir: Path, settings: _Settings) -> ContextDocument | None:
    """Return the finished document, unless `--force` is set or it was made with other settings."""
    context = vdir / CONTEXT_JSON
    if settings.force or not context.is_file():
        return None
    mark = StageStore(vdir).load("context", stage_versions()["context"], settings.run_key(), _ContextMark)
    if mark is None:
        return None
    data = context.read_bytes()
    if hashlib.sha256(data).hexdigest() != mark.sha256:
        return None
    try:
        return load_document(data.decode("utf-8"))
    except ContractError:
        # A document this reader can no longer load (a contract change) is rebuilt from the stored stages.
        return None


def _run_source(
    source_id: str,
    sdir: Path,
    settings: _Settings,
    *,
    reuse: Callable[[ContextDocument], VideoResult],
    run: Callable[["_Trace"], VideoResult],
    current: Callable[[ContextDocument], bool] | None = None,
) -> VideoResult:
    """Run one source of any kind under its directory lock, and record its own failure as its outcome.

    A current document is reused, its Markdown written again when missing, and handed to `reuse`; otherwise `run`
    makes it, from a cleared stage store under `--force`. `current` can reject a reusable document whose source
    record is out of date: `run` then rebuilds it from the stored stages. Only what ends the batch is raised.
    """
    trace = _Trace(source_id, settings.emit)
    try:
        with video_lock(sdir):
            document = _reusable(sdir, settings)
            if document is not None and current is not None and not current(document):
                document = None
            if document is not None:
                settings.emit(ProgressEvent(source_id, "skip", f"{CONTEXT_JSON} exists"))
                if not (sdir / CONTEXT_MD).is_file():
                    atomic.write_text(sdir / CONTEXT_MD, render_markdown(document))
                result = reuse(document)
            else:
                if settings.force:
                    StageStore(sdir).clear()
                result = run(trace)
            if result.document is None:
                return result
            return replace(
                result, json_bytes=(sdir / CONTEXT_JSON).read_bytes(), video_path=_requested_video(sdir, settings)
            )
    except YtverbatimError as exc:
        if exc.stage is None:
            exc.stage = trace.stage
        if isinstance(exc, BatchStopError | DependencyError | EngineMemoryError):
            raise
        return _failed(source_id, exc)
    except (ValueError, OSError) as exc:
        return _failed(source_id, exc)


def _requested_video(sdir: Path, settings: _Settings) -> Path | None:
    """Return the validated video file when the video track was requested; a missing or unprobeable one is an error."""
    if not settings.video:
        return None
    try:
        path = ready_video(sdir)
    except ProcTimeout as exc:
        error = MediaValidationError(f"the video file of {sdir.name} could not be probed: {exc}")
        error.stage = "video"
        raise error from exc
    if path is None:
        error = MediaValidationError(f"{sdir.name} finished with the video track requested but no valid video file")
        error.stage = "video"
        raise error
    return path


def _failed(source_id: str, exc: YtverbatimError | ValueError | OSError) -> VideoResult:
    detail = str(exc) if isinstance(exc, YtverbatimError) else f"{type(exc).__name__}: {exc}"
    outcome = VideoOutcome(
        source_id, VideoStatus.ERROR, error=_error_kind(exc), detail=redact(detail), fix=getattr(exc, "fix", "")
    )
    return VideoResult(outcome, error=exc)


def _run_video(session: FetchSession, planned: PlannedVideo, settings: _Settings) -> VideoResult:
    vdir = paths.video_dir(planned.video_id, settings.out)

    def reuse(document: ContextDocument) -> VideoResult:
        return _reused_video(session, planned, vdir, settings, document)

    def run(trace: _Trace) -> VideoResult:
        return _YouTubeRun(session, planned, vdir, settings, trace).run()

    return _run_source(planned.video_id, vdir, settings, reuse=reuse, run=run)


def _reused_video(
    session: FetchSession, planned: PlannedVideo, vdir: Path, settings: _Settings, document: ContextDocument
) -> VideoResult:
    video_id = planned.video_id
    access = document.source.access if isinstance(document.source, YouTubeSource) else None
    chapters = len(document.chapters)
    outcome = VideoOutcome(video_id, VideoStatus.OK, access=access, chapters=chapters, reused=("context",))
    timings: tuple[StageTiming, ...] = ()
    if settings.video and ready_video(vdir) is None:
        # The video is not part of the run key, so a current document says nothing about it. The saved info is
        # usually too old to reuse, so the earlier auth decision spares a refused anonymous try.
        settings.emit(ProgressEvent(video_id, "fetch"))
        began = time.monotonic()
        cookies_first = planned.cookies_first or (access is not None and access.cookies_used)
        fetched = session.fetch_video(video_id, cookies_first=cookies_first, audio=False, captions=False)
        if fetched.status is not VideoStatus.OK:
            if fetched.cause is not None:
                # Whatever of this fetch failed, the video is what the caller was waiting for.
                fetched.cause.stage = "video"
            return VideoResult(fetched)
        outcome = replace(outcome, reused=("context", *fetched.reused))
        timings = (StageTiming("fetch", time.monotonic() - began, reused=False),)
    return VideoResult(outcome, document, vdir / CONTEXT_MD, timings=timings)


def _error_kind(exc: Exception) -> str:
    match exc:
        case BusyError():
            return "busy"
        case EngineError():
            return "engine"
        case LLMError():
            return "llm"
        case ContractError():
            return "contract"
        case MediaValidationError():
            return "media_invalid"
        case ConfigError():
            return "config"
        case OSError():
            return "os_error"
        case ValueError():
            return "invalid_value"
        case _:
            return "error"


def _error_for(outcome: VideoOutcome) -> YtverbatimError:
    """Return the error of a failure the fetch session recorded as an outcome, as raised and with its fix."""
    error = outcome.cause or YtverbatimError(
        f"{outcome.video_id}: {outcome.error}: {outcome.detail or outcome.error or 'failed'}"
    )
    error.fix = outcome.fix or error.fix
    return error


class _Stretches(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stretches: tuple[Stretch, ...]


class _TextResult(BaseModel):
    """One text stage's output; `notices` say what the stage could not do (and are also quality warnings).

    `usage` is what producing this output cost, batches reused from checkpoints included.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    stretches: tuple[Stretch, ...]
    usage: tuple[LLMUsage, ...] = ()
    notices: tuple[str, ...] = ()
    llm_punctuated: bool = False


class _Opinion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    opinion: numeric.SecondOpinion | None


class _Unavailable(Exception):  # noqa: N818
    """The source could not give the audio a stage still needs; `outcome` is the adapter's record of why."""

    def __init__(self, outcome: VideoOutcome) -> None:
        super().__init__(outcome.video_id)
        self.outcome = outcome


@dataclass
class _Trace:
    """One source's progress events and stage timings, shared by its adapter and its transcription."""

    source_id: str
    emit_event: Callable[[ProgressEvent], None]
    timings: list[StageTiming] = field(default_factory=list[StageTiming])
    # The text or ASR stage that last started, named on an error that escapes it. "fetch" is left out: the fetch
    # session names its own finer stages (`info`, `audio`, ...) on the errors it records.
    stage: str | None = None

    def emit(self, stage: str, detail: str = "") -> None:
        if stage != "fetch":
            self.stage = stage
        self.emit_event(ProgressEvent(self.source_id, stage, detail))


@dataclass(frozen=True)
class _Prepared:
    """A source ready to transcribe, whatever it came from.

    `ensure_wav` puts the 16 kHz WAV at `dir / WAV_NAME`, or raises an error its adapter handles; it is called only when
    a stage still has audio to decode, so a rerun over stored stages needs no audio at all. `correction` is read only
    when correction runs, after any `ensure_wav`, since getting the audio may refresh what the context is made from.
    `captions` is None for a source that never has captions, so nothing reports them missing.
    """

    info: SourceInfo
    dir: Path
    language: str | None
    captions: tuple[CaptionTrack, ...] | None
    correction: Callable[[], CorrectionContext]
    ensure_wav: Callable[[], None]

    @property
    def id(self) -> str:
        return self.info.id


def _key(stage: str, inputs: dict[str, JsonValue]) -> str:
    return input_hash(stage, stage_versions()[stage], inputs)


@dataclass(frozen=True)
class _Decoding:
    """The worker requests a source's audio would be decoded with, and whether their output is stored already.

    Separate from `_Transcription`, because an adapter asks `needs_audio` before it has prepared the source: the YouTube
    adapter then fetches the audio in the same request as the metadata.
    """

    settings: _Settings
    source_id: str
    dir: Path
    language: str | None

    @property
    def store(self) -> StageStore:
        return StageStore(self.dir)

    def placed(self, stage: str, request: WorkerRequest) -> WorkerRequest:
        """Point `request` at its worker folder in `stage`'s working directory.

        The folder is keyed by what the decode depends on and never by a stage version: a change to what the pipeline
        makes of the rows must not decode the audio again. The worker refuses a folder holding another request's
        output, and an interrupted run of the same request resumes in the same folder.
        """
        decode = without_wav(request.fingerprint())
        out_dir = self.store.work_dir(stage) / request.engine.spec.id / store.digest(decode)[:16]
        _adopt(out_dir, decode)
        placed = request.model_copy(update={"out_dir": str(out_dir)})
        asr_runner.follow_moved_audio(placed)
        return placed

    def asr_request(self) -> WorkerRequest:
        request = WorkerRequest(
            engine=self.settings.definition,
            out_dir=str(self.store.work_dir("asr")),
            files=(
                FileJob(
                    id=self.source_id,
                    wav=str(self.dir / WAV_NAME),
                    language=self.language,
                    languages=self.settings.languages,
                ),
            ),
            memory_cap_bytes=self.settings.memory_cap_bytes,
        )
        return self.placed("asr", request)

    def asr_key(self, request: WorkerRequest) -> str:
        # The worker's own fingerprint, minus the WAV path: the source id already names the audio.
        return _key("asr", {"request": without_wav(request.fingerprint())})

    def transcribed(self, hypotheses: Path) -> Transcribed:
        rows = _read_rows(hypotheses, self.source_id)
        return stages.from_asr_rows(rows, self.settings.definition, self.language)

    def needs_audio(self) -> bool:
        """Whether an ASR stage still has audio to decode; stored stages and finished worker folders need none."""
        request = self.asr_request()
        source = self.store.load("asr", stage_versions()["asr"], self.asr_key(request), Transcribed)
        if source is None:
            hypotheses = asr_runner.finished(request)
            if hypotheses is None:
                return True
            source = self.transcribed(hypotheses)
        if self.settings.second_opinion is None:
            return False
        key = _key("second_opinion", self.second_inputs(source))
        if self.store.load("second_opinion", stage_versions()["second_opinion"], key, _Opinion) is not None:
            return False
        second = self.second_request(source)
        return second is not None and asr_runner.finished(second) is None

    def second_inputs(self, source: Transcribed) -> dict[str, JsonValue]:
        second = self.settings.second_opinion
        return {
            "source": store.digest(source),
            "engine": _engine_key(second) if second else None,
            "language": self.language,
        }

    def second_request(self, source: Transcribed) -> WorkerRequest | None:
        second = self.settings.second_opinion
        if second is None:
            return None
        request = numeric.second_opinion_request(
            _timed(source.stretches),
            engine=second,
            file_id=self.source_id,
            wav=self.dir / WAV_NAME,
            out_dir=self.store.work_dir("second_opinion"),
            language=self.language,
            words=[word for stretch in source.stretches for word in stretch.words],
        )
        if request is None:
            return None
        request = request.model_copy(update={"memory_cap_bytes": self.settings.memory_cap_bytes})
        return self.placed("second_opinion", request)


@dataclass
class _YouTubeRun:
    """One video's run: fetch its metadata, captions and audio, then hand the prepared video to `_Transcription`."""

    session: FetchSession
    planned: PlannedVideo
    vdir: Path
    settings: _Settings
    trace: _Trace
    # The latest successful fetch: the document records its access, and the run's outcome is it.
    fetched: tuple[VideoOutcome, Access] | None = None
    # Read again after a fetch, which may have refreshed the saved extraction they come from.
    tags: tuple[str, ...] = ()

    @property
    def video_id(self) -> str:
        return self.planned.video_id

    def run(self) -> VideoResult:
        settings = self.settings
        cached = self._local_metadata()
        need_audio = settings.source == "asr" and (
            cached is None
            or _Decoding(settings, self.video_id, self.vdir, settings.language or cached.language).needs_audio()
        )
        outcome = self._fetch(audio=need_audio)
        if not self._accept(outcome):
            return VideoResult(outcome)
        metadata = load_metadata(self.vdir)
        if not settings.creator_note:
            # A note saved while the option was on must not reach the document or the correction prompt once it is off.
            metadata = replace(metadata, creator_note=None)

        def describe(stretches: Sequence[Stretch]) -> YouTubeSource:
            return stages.youtube_source(metadata, self.tags, self._latest()[1], stretches)

        prepared = _Prepared(
            info=SourceInfo(id=self.video_id, chapters=metadata.chapters, describe=describe),
            dir=self.vdir,
            language=settings.language or metadata.language,
            captions=saved_captions(self.vdir),
            correction=lambda: correction_context(metadata, self.tags),
            ensure_wav=self._ensure_wav,
        )
        transcription = _Transcription(prepared, settings, self.trace)
        try:
            document = transcription.run()
        except _Unavailable as exc:
            return VideoResult(exc.outcome)
        return VideoResult(
            self._latest()[0],
            document,
            self.vdir / CONTEXT_MD,
            notices=tuple(transcription.notices),
            timings=tuple(self.trace.timings),
        )

    def _accept(self, outcome: VideoOutcome) -> bool:
        if outcome.status is not VideoStatus.OK or outcome.access is None:
            return False
        self.fetched = (outcome, outcome.access)
        self.tags = _tags(self.vdir)
        return True

    def _latest(self) -> tuple[VideoOutcome, Access]:
        # `run` prepares the source only after accepting a fetch, and a failed later fetch never clears it.
        assert self.fetched is not None
        return self.fetched

    def _ensure_wav(self) -> None:
        if (self.vdir / WAV_NAME).is_file():
            return
        outcome = self._fetch(audio=True)
        if not self._accept(outcome):
            raise _Unavailable(outcome)

    def _fetch(self, *, audio: bool) -> VideoOutcome:
        self.trace.emit("fetch")
        began = time.monotonic()
        outcome = self.session.fetch_video(self.video_id, cookies_first=self.planned.cookies_first, audio=audio)
        self.trace.timings.append(StageTiming("fetch", time.monotonic() - began, reused=False))
        return outcome

    def _local_metadata(self) -> VideoMetadata | None:
        try:
            return load_metadata(self.vdir)
        except (OSError, ValueError, YtDlpError):
            return None


@dataclass(frozen=True)
class _FileRun:
    """One local recording's run: its audio comes from the file, and nothing else is fetched."""

    recording: local.LocalRecording
    dir: Path
    settings: _Settings

    def outcome(self, *, reused: tuple[str, ...] = ()) -> VideoOutcome:
        return VideoOutcome(self.recording.id, VideoStatus.OK, audio_s=self.recording.duration_s, reused=reused)

    def reuse(self, document: ContextDocument) -> VideoResult:
        return VideoResult(self.outcome(reused=("context",)), document, self.dir / CONTEXT_MD)

    def run(self, trace: _Trace) -> VideoResult:
        recording = self.recording
        language = self.settings.language

        def describe(_stretches: Sequence[Stretch]) -> RecordingSource:
            return recording.source(language)

        def ensure_wav() -> None:
            wav = self.dir / WAV_NAME
            if wav.is_file():
                return
            trace.emit("fetch")
            began = time.monotonic()
            with in_stage("audio"):
                local.to_wav(recording, wav)
            trace.timings.append(StageTiming("fetch", time.monotonic() - began, reused=False))

        prepared = _Prepared(
            info=SourceInfo(id=recording.id, chapters=(), describe=describe),
            dir=self.dir,
            language=language,
            captions=None,
            correction=CorrectionContext,
            ensure_wav=ensure_wav,
        )
        transcription = _Transcription(prepared, self.settings, trace)
        try:
            document = transcription.run()
        except BaseException:
            # The source stays where it is and the WAV is quick to make again, so no copy of its audio is left.
            (self.dir / WAV_NAME).unlink(missing_ok=True)
            raise
        return VideoResult(
            self.outcome(),
            document,
            self.dir / CONTEXT_MD,
            notices=tuple(transcription.notices),
            timings=tuple(trace.timings),
        )


@dataclass
class _Transcription:
    """The source-agnostic part of a run, from the audio or a manual caption track to `context.json` and `context.md`.

    asr (or manual captions) → second opinion → normalize → punctuate → correct → numeric → numerals → build → write.
    """

    prepared: _Prepared
    settings: _Settings
    trace: _Trace
    notices: list[str] = field(default_factory=list[str])

    @property
    def store(self) -> StageStore:
        return StageStore(self.prepared.dir)

    @property
    def decoding(self) -> _Decoding:
        return _Decoding(self.settings, self.prepared.id, self.prepared.dir, self.prepared.language)

    def run(self) -> ContextDocument:
        settings = self.settings
        prepared = self.prepared
        source: Transcribed | None = None
        if settings.source == "captions" and prepared.captions is not None:
            source = self._from_captions(prepared.captions)
        if source is None:
            if self.decoding.needs_audio():
                prepared.ensure_wav()
            source = self._asr()
        second = self._second_opinion(source) if source.run.kind == "asr" else None
        (prepared.dir / WAV_NAME).unlink(missing_ok=True)

        language = source.language
        normalized = self._normalize(source)
        punctuated = self._punctuate(normalized, language)
        corrected, correction_llm = self._correct(punctuated, language)
        verified = self._numeric(corrected, source, second)
        displayed = self._numerals(verified.stretches, language)
        self.notices += [*punctuated.notices, *verified.notices]

        self.trace.emit("write")
        began = time.monotonic()
        document = build_document(
            info=prepared.info,
            source=source,
            stretches=displayed,
            chapters=settings.text.chapters != "none",
            created_at=settings.clock(),
            correction_llm=correction_llm,
            llm_punctuated=punctuated.llm_punctuated,
            warnings=self.notices,
            llm_usage=(*punctuated.usage, *corrected.usage, *verified.usage),
        )
        dumped = dump_document(document)
        # The record goes first and comes back last, so it never vouches for a half-written pair of files.
        self.store.discard("context")
        atomic.write_text(prepared.dir / CONTEXT_MD, render_markdown(document))
        atomic.write_text(prepared.dir / CONTEXT_JSON, dumped)
        sha = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
        self.store.save("context", stage_versions()["context"], settings.run_key(), _ContextMark(sha256=sha))
        self.trace.timings.append(StageTiming("write", time.monotonic() - began, reused=False))
        return document

    def _stage[M: BaseModel](
        self,
        stage: str,
        key: str,
        model: type[M],
        compute: Callable[[], M],
        detail: str = "",
        *,
        keep: Path | None = None,
    ) -> M:
        """Run or reuse one stage; `keep` is its worker folder, and the stage's other folders go once it is stored."""
        began = time.monotonic()

        def announced() -> M:
            self.trace.emit(stage, detail)
            return compute()

        output, reused = self.store.run(stage, stage_versions()[stage], key, model, announced)
        if keep is not None and not reused:
            _prune(self.store.work_dir(stage), keep)
        self.trace.timings.append(StageTiming(stage, time.monotonic() - began, reused))
        return output

    def _asr(self) -> Transcribed:
        decoding = self.decoding
        request = decoding.asr_request()

        def compute() -> Transcribed:
            hypotheses = asr_runner.finished(request) or asr_runner.run_worker(request).hypotheses
            return decoding.transcribed(hypotheses)

        engine = request.engine.spec.id
        return self._stage("asr", decoding.asr_key(request), Transcribed, compute, engine, keep=Path(request.out_dir))

    def _from_captions(self, captions: Sequence[CaptionTrack]) -> Transcribed | None:
        language = self.prepared.language
        track, missing = _manual_track(captions, language)
        if track is None:
            self.notices.append(f"{missing}; transcribed with ASR instead")
            return None
        key = _key("captions", {"track": _track_key(track), "language": language})
        return self._stage("captions", key, Transcribed, lambda: stages.from_captions(track, language), track.language)

    def _second_opinion(self, source: Transcribed) -> numeric.SecondOpinion | None:
        second = self.settings.second_opinion
        if second is None:
            return None
        decoding = self.decoding
        key = _key("second_opinion", decoding.second_inputs(source))
        request = decoding.second_request(source)

        def compute() -> _Opinion:
            if request is None:
                # Nothing to re-decode, so no folder of this engine can be resumed again.
                _remove(self.store.work_dir("second_opinion") / second.spec.id)
                return _Opinion(opinion=None)
            hypotheses = asr_runner.finished(request) or asr_runner.run_worker(request).hypotheses
            rows = _read_rows(hypotheses, self.prepared.id)
            return _Opinion(opinion=numeric.SecondOpinion.from_hypotheses(rows))

        keep = Path(request.out_dir) if request is not None else None
        return self._stage("second_opinion", key, _Opinion, compute, second.spec.id, keep=keep).opinion

    def _normalize(self, source: Transcribed) -> _Stretches:
        text = self.settings.text
        key = _key("normalize", {"source": store.digest(source), "script": text.script, "fillers": text.fillers})

        def compute() -> _Stretches:
            normalized = stages.normalize(source.stretches, source.language, script=text.script, fillers=text.fillers)
            return _Stretches(stretches=normalized)

        return self._stage("normalize", key, _Stretches, compute)

    def _punctuate(self, previous: _Stretches, language: str) -> _TextResult:
        llm = self.settings.llm
        key = _key("punctuate", {"text": _text_key(previous.stretches), "llm": self.settings.llm_id})

        def compute() -> _TextResult:
            stretches = previous.stretches
            texts = [stretch.text for stretch in stretches]
            # Per stretch: Whisper often leaves only the opening minute unpunctuated in an otherwise punctuated video.
            if not any(needs_punctuation(text, language) for text in texts):
                return _TextResult(stretches=stretches)
            if llm is None:
                notice = (
                    "punctuation not restored: some stretches have few sentence marks and no LLM is configured "
                    "(correction is skipped without one too)"
                )
                return _TextResult(stretches=stretches, notices=(notice,))
            result = punctuate(texts, language=language, client=llm, checkpoint_dir=self.store.work_dir("punctuate"))
            failed = set(result.failed)
            note = "the LLM's punctuation changed the words twice, so this stretch was kept as transcribed"
            flags = [
                [stretch_flag(stretch, "punctuation_failed", note)] if index in failed else []
                for index, stretch in enumerate(stretches)
            ]
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, flags, record=False),
                usage=result.usage,
                llm_punctuated=any(result.edits),
            )

        return self._stage("punctuate", key, _TextResult, compute, self.settings.llm_id or "")

    def _correct(self, previous: _TextResult, language: str) -> tuple[_TextResult, str | None]:
        llm = self.settings.llm if self.settings.text.correct else None
        if llm is None:
            return _TextResult(stretches=previous.stretches), None
        context = self.prepared.correction()
        inputs: dict[str, JsonValue] = {
            "text": _text_key(previous.stretches),
            "llm": self.settings.llm_id,
            "glossary": list(context.glossary),
            "creator_note": context.creator_note,
        }

        def compute() -> _TextResult:
            stretches = previous.stretches
            result = correct_segments(
                [stretch.text for stretch in stretches],
                client=llm,
                language=language,
                context=context,
                checkpoint_dir=self.store.work_dir("correct"),
                confidence=stages.word_confidence(stretches),
            )
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, result.flags), usage=result.usage
            )

        return self._stage("correct", _key("correct", inputs), _TextResult, compute, llm.id), llm.id

    def _numeric(self, previous: _TextResult, source: Transcribed, second: numeric.SecondOpinion | None) -> _TextResult:
        llm = self.settings.llm
        captions = self.prepared.captions
        evidence_tracks = _evidence_tracks(captions or (), source)
        inputs: dict[str, JsonValue] = {
            "text": _text_key(previous.stretches),
            "captions": [_track_key(track) for track in evidence_tracks],
            "second_opinion": store.digest(_Opinion(opinion=second)),
            "llm": self.settings.llm_id,
        }

        def compute() -> _TextResult:
            stretches = previous.stretches
            primary = source.run.id if source.run.kind == "asr" else stages.CAPTION_MANUAL
            evidence = numeric.Evidence(
                primary=primary,
                words=tuple(word for stretch in stretches for word in stretch.words),
                captions=tuple(evidence_tracks),
                second_opinion=second,
            )
            result = numeric.verify_numbers(
                _timed(stretches),
                language=source.language,
                evidence=evidence,
                client=llm,
                checkpoint_dir=self.store.work_dir("numeric"),
            )
            notices: list[str] = []
            if not evidence_tracks and second is None and result.stats.single_source:
                missing = f"no captions in {source.language}" if captions is not None else "the source has no captions"
                notices.append(
                    f"numbers not cross-checked: {missing} and no second opinion; "
                    f"{result.stats.single_source} number(s) rest on the transcript alone"
                )
            if llm is None and result.stats.disputed:
                notices.append(
                    f"{result.stats.disputed} disputed number(s) left as transcribed and flagged: "
                    "no LLM is configured to choose among the readings"
                )
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, result.flags),
                usage=result.usage,
                notices=tuple(notices),
            )

        return self._stage("numeric", _key("numeric", inputs), _TextResult, compute, self.settings.llm_id or "")

    def _numerals(self, previous: Sequence[Stretch], language: str) -> tuple[Stretch, ...]:
        # Display only: the stages before it read the text as recognized, so a rule change never repeats an LLM call.
        style = self.settings.text.numerals
        if style == "keep" or not seg.is_zh(language):
            return tuple(previous)
        key = _key("numerals", {"text": _text_key(previous), "numerals": style})

        def compute() -> _Stretches:
            return _Stretches(stretches=stages.numerals(previous, language))

        return self._stage("numerals", key, _Stretches, compute).stretches


def _adopt(out_dir: Path, decode: JsonValue) -> None:
    """Move a folder holding this request's output, left under another name, to `out_dir`.

    Folders were once keyed by the stage version, so a version bump would otherwise decode the audio again. That
    scheme can leave a complete folder and an interrupted one of the same request; the complete one is taken, since
    the other is pruned once the stage is stored.
    """
    if out_dir.exists() or not out_dir.parent.is_dir():
        return
    matches = [
        entry
        for entry in out_dir.parent.iterdir()
        if (stored := asr_runner.stored_fingerprint(entry)) is not None and without_wav(stored) == decode
    ]
    if matches:
        min(matches, key=lambda entry: (not (entry / METRICS_FILE).is_file(), entry.name)).rename(out_dir)


def _prune(root: Path, keep: Path) -> None:
    """Remove the stale requests of `keep`'s engine; its other entries can never be resumed again.

    Other engines' folders stay: switching back to one resumes its finished chunks instead of decoding again.
    """
    for entry in keep.parent.iterdir():
        if entry != keep:
            _remove(entry)
    for entry in root.iterdir():
        if not entry.is_dir():
            _remove(entry)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _text_key(stretches: Sequence[Stretch]) -> str:
    # Only the text a stage reads; what the previous stage cost or noticed does not change this stage's output.
    return store.digest(_Stretches(stretches=tuple(stretches)))


def _timed(stretches: Sequence[Stretch]) -> list[numeric.TimedText]:
    return [numeric.TimedText(stretch.text, stretch.start, stretch.end) for stretch in stretches]


def _manual_track(tracks: Sequence[CaptionTrack], language: str | None) -> tuple[CaptionTrack | None, str]:
    """Pick the manual track to transcribe from; without one, say why there is none."""
    manual = [track for track in tracks if track.kind is CaptionKind.MANUAL]
    if language is None:
        if len(manual) == 1:
            return manual[0], ""
        if manual:
            return None, f"{len(manual)} manual caption tracks and no declared video language to choose one"
        return None, "no manual caption track"
    track = next((track for track in manual if _base(track.language) == _base(language)), None)
    return track, "" if track is not None else f"no manual caption track in {language}"


def _evidence_tracks(tracks: Sequence[CaptionTrack], source: Transcribed) -> list[CaptionTrack]:
    """Caption tracks that are independent readings of the transcript's numbers."""
    # A track in another language is a translation; it would only make every number look unheard. The manual track a
    # captions-mode transcript was built from is the transcript itself.
    return [
        track
        for track in tracks
        if _base(track.language) == _base(source.language)
        and not (source.run.kind != "asr" and track.kind is CaptionKind.MANUAL)
    ]


def _base(code: str) -> str:
    return code.split("-")[0].lower()


def _track_key(track: CaptionTrack) -> str:
    return store.digest(_plain(asdict(track)))


def _read_rows(path: Path, video_id: str) -> list[ChunkHypothesis]:
    rows: list[ChunkHypothesis] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(ChunkHypothesis.model_validate_json(line))
        except ValidationError as exc:
            raise ContractError(f"{path.name} line {number} is not a valid hypothesis row:\n{exc}") from exc
    return sorted((row for row in rows if row.file == video_id), key=lambda row: row.chunk)


def correction_context(metadata: VideoMetadata, tags: Sequence[str]) -> CorrectionContext:
    note = metadata.creator_note.text if metadata.creator_note is not None else None
    return CorrectionContext(glossary=tuple(tags[:_GLOSSARY_MAX_TERMS]), creator_note=note)


def info_tags(raw: object) -> tuple[str, ...]:
    tags = raw.get("tags") if isinstance(raw, dict) else None
    return tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()


def _tags(vdir: Path) -> tuple[str, ...]:
    # VideoMetadata does not keep tags, but the raw extraction does.
    try:
        raw: object = json.loads(info_path(vdir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    return info_tags(raw)
