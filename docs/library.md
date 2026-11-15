# Using ytverbatim as a library

`ytverbatim` is also a Python library. This page lists what is stable, how it is versioned, and what each release breaks. `tests/test_public_api.py` pins the names, the signatures and the dataclass fields below, and the error attributes an application reads, so a change to them is always deliberate.

## The stable surface

Everything in `ytverbatim.__all__`, plus the modules `ytverbatim.errors`, `ytverbatim.contract` and `ytverbatim.config`. Every other module, function and attribute is internal and may change in any release, a patch included.

| Name | What it is |
|---|---|
| `process(target, options=None, *, llm=None, out=None, auth=None, captions=None, force=False, cookies_first=None, interactive=False, progress=None) -> Result` | Transcribe one YouTube video or one local audio or video file. |
| `Result` | A frozen dataclass: what `process` made. |
| `expand(target, *, limit=0, auth=None, options=None) -> list[str]` | The video ids behind a video, playlist or channel URL. |
| `video_file(video_id, out=None) -> Path \| None` | The validated `video.mp4` of a video, or `None`. It never downloads. |
| `parse_youtube(text) -> YouTubeRef \| None`, `YouTubeRef`, `RefKind`, `video_url(video_id)` | Classify a YouTube URL or id as a video, playlist or channel. |
| `check_dependencies(options=None, *, source="youtube") -> None` | Raise `DependencyError` naming the missing extra, before any work. |
| `load_document`, `ContextDocument` | Read and model `context.json` (contract 2.0); the sub-models are in `ytverbatim.contract`. |
| `Options`, `AuthOptions`, `YtdlpOptions`, `AsrOptions`, `TextOptions`, `DiarizeOptions`, `LLMOptions` | The options; also in `ytverbatim.config`. |
| `LLMClient`, `LLMRequest`, `LLMResponse`, `Usage` | The protocol an injected LLM implements. |
| `ProgressEvent` | What the `progress` callback receives: `id`, `stage`, `detail`. |
| `quantity_tokens`, `QToken`, `QKind` | The quantity tokenizer that numeric verification uses. |
| `SUPPORTED_BROWSERS`, `__version__` | The browsers `--auth` accepts, and the installed version. |
| `ytverbatim.errors` | Every error class, plus `AuthState` and `Signal`. |

`import ytverbatim` loads nothing heavy: each name is imported on first use.

### `process`

`target` is a YouTube video URL or id, or a path to a local file (a `PathLike` is always a file). `options` defaults to the configured options (`config.toml`). `out` is the output root; by default `~/.local/share/ytverbatim/videos` for a video and `.../recordings` for a recording.

`auth`, `captions` and `cookies_first` apply to YouTube only, and `None` means the default (`captions=True`, `cookies_first=False`, the configured auth source). Giving any of them for a file raises `ValueError`.

`llm` is any `LLMClient`. Without one, the configured `[llm]` provider applies; with `provider = "none"` the LLM stages are skipped and `Result.notices` says which. `interactive=True` lets the ChatGPT provider open a browser to sign in again; the library never does so otherwise.

`progress` is called with a `ProgressEvent` as each stage starts. `id` is the video or recording id, and `stage` is one of the names below. These names are API.

| Stage | When |
|---|---|
| `fetch` | The video's metadata, captions, audio or video are being downloaded (a recording's audio is being converted). |
| `skip` | A current `context.json` already exists, or the file is the same audio as another in the batch. |
| `asr`, `diarize`, `second_opinion` | Speech recognition; speaker diarization, which runs after the decode for a diarized source (see `DiarizeOptions`); and the second engine's reading of numbers. |
| `normalize`, `punctuate`, `correct`, `numeric`, `numerals` | The text stages. |
| `write` | The document is being built and written. |

### `Result`

| Field | Meaning |
|---|---|
| `id` | The video id, or the recording id (`rec-` plus 16 hex digits). |
| `kind` | `"youtube"` or `"recording"`. Matches `document.source.kind`. |
| `document` | The `ContextDocument`. |
| `json_bytes` | The exact bytes of `context.json`, read while the source's lock was held. Store these, not a re-dump of `document`: a document of an older contract minor re-dumps differently. |
| `directory` | The source's folder, holding `context.json`, `context.md` and the downloaded media. |
| `markdown_path` | `directory / "context.md"`. |
| `video_path` | The validated `video.mp4` when the video track was requested (`ytdlp.video`), else `None`. It is probed under the same lock as `json_bytes`; a video that is missing or cannot be probed fails the call with `stage == "video"`. |
| `reused` | What an earlier run already had: `context` when the whole document was current, otherwise any of `info`, `captions`, `audio`, `wav`, `video`. |
| `notices` | The stages that could not run as asked (for example punctuation without an LLM). They are also in `document.quality.warnings`. |

### Errors

Every error `process` and `expand` raise on purpose subclasses `YtverbatimError`, and carries:

- `fix`: a one-line remedy to show the user, or `""` when there is nothing specific to say. It is set for every `BatchStopError`, `AuthError` and `YtDlpError` they raise.
- `stage`: where the failure happened, or `None` when unknown. While fetching a video it is `info` (the metadata request), `captions`, `audio` or `video`; after the fetch it is the stage that was running (`asr`, `punctuate`, `correct`, ...). A failed video download is `stage == "video"`, and so is every failure of the video-only fetch that a finished document needs when the video track is requested later, so a caller can retry without the video track.

`BatchStopError` (`signal`, `until`) means YouTube asked to back off; the cooldown is already written, and every later request refuses until it ends. `AuthError` carries an `AuthState`, and `YtDlpError` its primary `Signal`; both enums are importable from `ytverbatim.errors`. Bad input raises `ValueError`.

### Dependencies

`check_dependencies(options, source="youtube")` checks that the ASR engine the options select for that source (`"youtube"` or `"recording"`) can be imported, plus the diarizer when that source is diarized and the numeric second-opinion engine when one is on, and names the extra that is missing: `whisper` for `w-prod`, the default, and `asr` for the other engines and the diarizer. With the default options a recording therefore needs both, `ytverbatim[whisper,asr]`. `process` runs the same check before any request. Nothing is imported or downloaded; an unknown engine is a `ConfigError`, and an unknown source a `ValueError`.

### Diarization

`DiarizeOptions(mode="auto")`, the `[diarize]` section of `config.toml`, decides which sources get speaker labels. `"auto"` diarizes recordings and leaves YouTube videos alone; `"on"` diarizes every source transcribed with ASR, and `"off"` none. A diarized document lists its speakers in `document.speakers` (`S1`, `S2`, … numbered by first appearance when attributed, listed in id order, each with `speech_s`; a later text stage never renumbers them, so ids may have gaps and need not appear in order), names one on every segment and paragraph, and describes the diarizer in `document.transcript.diarization`; an undiarized one has no speakers, `speaker` is `None` everywhere and `transcript.diarization` is `None`. An application that wants no diarization, whatever the source, sets `mode="off"` explicitly.

## Versioning

- 0.1.0 is the first release with a declared surface.
- Before 1.0, a minor release may break the surface. Every break is listed below and in the release notes. A patch release never breaks it.
- 1.0 follows once an application has run on this surface and the recordings milestone has settled.
- The contract version is independent of the package version: it is 2.0 now, and later optional fields become 2.x. See the README's contract section.

## Breaking changes

### 0.1.0

1. `process` returns a `Result`, not a `ContextDocument`. Read `result.document`.
2. Contract 2.0: `document.video` is `document.source` (check `source.kind == "youtube"`), and `document.access` is `document.source.access`. `load_document` still upgrades 1.0 and 1.1 documents in memory.
3. Internal modules are no longer API. Of the modules an application may have imported, `ids`, `paths`, `pipeline`, `acquire.*`, `auth.*`, `asr.*`, `ytdlp.*` and `llm.*` are internal; the names below replace what they were used for.
4. `BatchStopError.fix` is set by `process` as well as `expand`, and every error from either has a `stage` attribute.
5. `AsrOptions.engine` defaults to `None`, meaning the source's default engine (YouTube still gets `w-prod`).
6. A new `[diarize]` section (`DiarizeOptions`, a new `Options.diarize` field); `auto` means no change for YouTube.
7. `ProgressEvent.video_id` is now `ProgressEvent.id`, since it also names recordings.
8. `process` no longer takes `runner_factory`.

If you imported internals, the replacements are:

| Before | Now |
|---|---|
| `ids.extract_video_id`, `ids.classify`, `ids.RefKind`, `ids.video_url` | `parse_youtube` (read `.id` and `.kind`), `RefKind`, `video_url` |
| `acquire.session.MEMBERS_PLAYLIST_PREFIX` | `parse_youtube(url).members_only` |
| `paths.video_dir`, `pipeline.video_lock`, `pipeline.CONTEXT_JSON`, reading `context.json` under the lock | `result.directory`, `result.json_bytes` |
| `acquire.video.video_file` after `process` | `result.video_path` |
| `acquire.audio.WAV_NAME` and inferring the failed stage from progress events | `error.stage == "video"` |
| `asr.registry.get` and `check_dependencies` | `check_dependencies(options)` |
| `auth.cooldown.stop_fix`, `auth.fixes.fix_for` | `error.fix`, always set by `process` and `expand` |
| `auth.state.AuthState`, `ytdlp.types.Signal` | `ytverbatim.errors.AuthState`, `ytverbatim.errors.Signal` |
| `auth.chromium.SUPPORTED_BROWSERS` | `ytverbatim.SUPPORTED_BROWSERS` |
| `pipeline.ProgressEvent`, `ytverbatim.llm.LLMClient` and its request types | `ytverbatim.ProgressEvent`, `LLMClient`, `LLMRequest`, `LLMResponse`, `Usage` |
| `pipeline.VideoResult`, `BatchReport`, `VideoOutcome` in a test double | a fake `process` that returns a `Result` |

## Not in the library yet

Voice Memos listing, speaker renaming, folder expansion, a recording type with caller-given metadata, and batch reports stay CLI features. They can be added without breaking the surface.
