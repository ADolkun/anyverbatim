# Using anyverbatim as a library

`anyverbatim` is also a Python library. This page lists what is stable, how it is versioned, and what each release breaks. `tests/test_public_api.py` pins the names, the signatures and the dataclass fields below, and the error attributes an application reads, so a change to them is always deliberate.

## The stable surface

Everything in `anyverbatim.__all__`, plus the modules `anyverbatim.errors`, `anyverbatim.contract` and `anyverbatim.config`. Every other module, function and attribute is internal and may change in any release, a patch included.

| Name | What it is |
|---|---|
| `process(target, options=None, *, llm=None, out=None, auth=None, captions=None, force=False, cookies_first=None, interactive=False, progress=None) -> Result` | Transcribe one YouTube video or one local audio or video file. |
| `Result` | A frozen dataclass: what `process` made. |
| `expand(target, *, limit=0, auth=None, options=None) -> list[str]` | The video ids behind a video, playlist or channel URL. |
| `video_file(video_id, out=None) -> Path \| None` | The validated `video.mp4` of a video, or `None`. It never downloads. |
| `parse_youtube(text) -> YouTubeRef \| None`, `YouTubeRef`, `RefKind`, `video_url(video_id)` | Classify a YouTube URL or id as a video, playlist or channel. |
| `check_dependencies(options=None, *, source="youtube") -> None` | Raise `DependencyError` naming the missing extra, before any work. |
| `load_document`, `ContextDocument` | Read and model `context.json` (contract 2.0); the sub-models are in `anyverbatim.contract`. |
| `Options`, `AuthOptions`, `YtdlpOptions`, `AsrOptions`, `TextOptions`, `DiarizeOptions`, `LLMOptions` | The options; also in `anyverbatim.config`. |
| `LLMClient`, `LLMRequest`, `LLMResponse`, `Usage` | The protocol an injected LLM implements. |
| `ProgressEvent` | What the `progress` callback receives: `id`, `stage`, `detail` and `retry`. |
| `STAGES`, `ProgressStage`, `LLM_STAGES` | The stage names in run order, their type, and the frozenset of stages that may call your `LLMClient`. |
| `quantity_tokens`, `QToken`, `QKind` | The quantity tokenizer that numeric verification uses. |
| `SUPPORTED_BROWSERS`, `__version__` | The browsers `--auth` accepts, and the installed version. |
| `anyverbatim.errors` | Every error class, plus `AuthState` and `Signal`. |

`import anyverbatim` loads nothing heavy: each name is imported on first use.

### `process`

`target` is a YouTube video URL or id, or a path to a local audio or video file (a `PathLike` is always a file). It takes one source: a playlist or channel URL, or a folder, raises `ValueError`, so a caller loops over files itself, and `expand` lists a playlist or channel. `options` defaults to the configured options (`config.toml`). `out` is the output root; by default `~/.local/share/anyverbatim/videos` for a video and `.../recordings` for a recording (both under `$ANYVERBATIM_HOME` when it is set).

A recording gets the id `rec-` plus 16 hex digits of the hash of its audio, and a folder of that name holding `context.json`, `context.md` and the stored stages. Its title and recording date come from the file's own tags, and nothing in the library sets them. As `anyverbatim run` does, `process` also rewrites the recordings root's `index.md` and `batch_report.json`. See [docs/recordings.md](recordings.md#the-recording-id).

`auth`, `captions` and `cookies_first` apply to YouTube only, and `None` means the default (`captions=True`, `cookies_first=False`, the configured auth source). Giving any of them for a file raises `ValueError`.

`llm` is any `LLMClient`. Without one, the configured `[llm]` provider applies, ChatGPT by default, and `process` raises `ConfigError` before any work until `anyverbatim login` has run once. The default model, `gpt-sol`, is resolved at that point to the newest Sol model the ChatGPT plan lists (one catalog request); `LLMOptions(model="gpt-astra")` picks the newest of another family, and a versioned model such as `"gpt-6.1-sol"` pins it with no catalog request. If the catalog cannot be read or lists no model of the family, `ConfigError` is raised with a `fix` that says how to pin a version. `interactive=True` lets the ChatGPT provider open a browser to sign in again, for an application with a person at the screen; the library never does so otherwise.

To run without an LLM, pass `Options(llm=LLMOptions(provider="none"))` (both from `anyverbatim.config`). The LLM stages are then skipped and `Result.notices` says which. `anyverbatim.process(url, options, llm=client)` takes any object that satisfies `LLMClient`.

`progress` is called with a `ProgressEvent` as each stage starts. It is also called with `retry=True` when an LLM call of that stage failed transiently and will be repeated; `detail` then reads like `LLM unavailable (server_is_overloaded); retry 1/4 in 17s`. Such an event is not a stage starting, so a consumer that counts stages skips it. `id` is the video or recording id, and `stage` is one of the names below. `anyverbatim.STAGES` lists the stage names in run order (typed as `anyverbatim.ProgressStage`), and `anyverbatim.LLM_STAGES` is the frozenset of those that may call your `LLMClient` (`punctuate`, `correct`, `numeric`); both are API.

| Stage | When |
|---|---|
| `fetch` | The video's metadata, captions, audio or video are being downloaded (a recording's audio is being converted). |
| `skip` | A current `context.json` already exists. |
| `captions` | A source with manual captions reads them instead of running `asr`. |
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

### `expand`

`expand(target, *, limit=0, auth=None, options=None)` returns the video ids behind one video, playlist or channel URL, in order and without repeats. A video id costs no request, and `limit` caps the listing (0 lists everything; the `max_videos_per_run` cap belongs to `run`). Like every request, it refuses to start during a cooldown and goes through the same stop guard, so a bot check or rate limit starts the cooldown and raises.

Only a members playlist (`UUMO...`) is listed with cookies, from `auth` or the configured source. With none (or with `auth.policy = "always"` and none), `expand` raises `AuthError` instead of listing anonymously. The ids carry no memory of where they came from, so for ids from a members playlist call `process(video_id, cookies_first=True)`: the video's first request then uses cookies, where otherwise each one is requested anonymously, refused, and only then retried with cookies.

### Video

The video track is opt-in, for callers that need frames (screenshots, say). With `[ytdlp] video = true` (or `Options(ytdlp=YtdlpOptions(video=True))`), `run`, `process` and `fetch` also save `<out>/<video id>/video.mp4` after the audio: the best video-only stream up to 1080p, remuxed to MP4, requested on the same cookie path as the audio, so a members-only or Premium video needs nothing extra.

- It is validated like the audio: it decodes and covers at least 90% of the video's duration. A file that fails is downloaded again, and a valid one is reused.
- A video that is already transcribed is not transcribed again for this. `process` only fetches the missing file, which is one `video` download plus a fresh metadata extraction when the saved one is over an hour old (its signed URLs have expired), made with cookies if the earlier run used them.
- With the option off, the default, no video is ever requested. A failed video download fails the video.
- `anyverbatim.video_file(video_id, out=None)` returns the path of the validated file under `out` (default output root), or `None` when it is missing or fails validation. It never downloads. A probe that times out counts as absent; a missing `ffprobe` raises `DependencyError`, and a malformed id `ValueError`.

### Quantities

`anyverbatim.quantity_tokens(text)` returns the numbers, currency amounts, percentages, tickers, Chinese numerals and spelled-out English numbers in `text` as `anyverbatim.QToken`s in order. It is the tokenizer that numeric verification and the correction guards use.

- `char_start` and `char_end` are half-open offsets into `text`; `surface` is the text matched.
- `value_itn` is the normalized value two spellings share: `一百五十期` and `150期` are both `150期`, `$1.5B` is `$1500000000`, `twenty-five` is `25`.
- `kind` is a `anyverbatim.QKind`: `number`, `currency`, `percent`, `ticker` or `cjk_quantity`.

A Chinese numeral is a quantity when it has two or more numeral characters (`三十`), follows `第` or `百分之`, or has a measure word: a currency, 个, 年, 次, a count classifier such as 家, 种 or 位, a time or measurement unit such as 分钟, 岁 or 公里, or a value unit such as 倍 or 个基点.

- 一 after a figure in 万 or 亿 starts a per-unit phrase, so `三万一个月` is `30000` and `1个月`.
- 点 before tens is a clock time: `三点四十分` is `3点` and `40`.
- Ranges of adjacent digits (`三四个`), idioms (`万一`, `一样`, `十万八千里`) and weekdays are not quantities.
- A fraction's `value_itn` is over/under: `三分之一` and `1/3` are `1/3`.
- The start of a range linked by 到, 至, ~ or - shares its end's 万 or 亿 and currency sign, in either script: `两到3亿美元` and `2到3亿美元` are `$200000000` and `$300000000`, `三到五万` is `30000` and `50000`. A start never takes a scale that would put it at or above its end (`两千到三万` keeps `2000`). A one-digit start may take the end's leading place as well: it does when only that reading fits (`两到三亿`), takes the magnitude alone when only that fits (`五到十二万` is `50000`), and takes the place when both fit only if the end is one digit and a 百 or 千 (`两到三百万` is `2000000`); otherwise it stays unscaled (`两到三十万`, `2到30万`). Chains share the scale in either script (`3到五到七万`). A start with its own magnitude, percent, ordinal or unit, a magnitude on the left (`三万到五`), 千 or 百 alone (`两到三千` is `2` and `3000`) and a list with 、 share nothing. The token's `surface` stays the start as written (`两`, `2`).
- A rough figure with 几 or 数 carries `~`: `二十几岁` and `20几岁` are `~20`, and `数十亿` is `~1000000000`. Each agrees with its other spelling but never with an exact value.

### Errors

Every error `process` and `expand` raise on purpose subclasses `AnyverbatimError`, and carries:

- `fix`: a one-line remedy to show the user, or `""` when there is nothing specific to say. It is set for every `BatchStopError`, `AuthError` and `YtDlpError` they raise.
- `stage`: where the failure happened, or `None` when unknown. While fetching a video it is `info` (the metadata request), `captions`, `audio` or `video`; after the fetch it is the stage that was running (`asr`, `punctuate`, `correct`, ...). A failed video download is `stage == "video"`, and so is every failure of the video-only fetch that a finished document needs when the video track is requested later, so a caller can retry without the video track.

`process` raises the source's own error as it happened: `AuthError`, `UnavailableError`, `YtDlpError`, `MediaValidationError` (a recording that cannot be read, for one), `ProcTimeout`, `EngineError`, `LLMError` or `ContractError`, with `fix` set where a remedy exists.

`LLMUnavailableError` has a `transient` attribute, `False` by default. A provider sets it for a server-side overload or error that passes by itself (the ChatGPT `server_is_overloaded` and HTTP 500, 502, 503 and 504, for one), and `process` then repeats that LLM call up to four more times, waiting about 15, 30, 60 and 120 seconds with jitter, before it raises the original error with the number of attempts in its message. An `LLMClient` you pass as `llm=` opts in by raising `LLMUnavailableError(..., transient=True)`; every other error, `LLMOutputError` included, is raised on the first attempt. Each retry repeats a call that stored nothing on the provider's side, so only raise it for a failure that leaves the call safe to send again.

`BatchStopError` (`signal`, `until`) means YouTube asked to back off; the cooldown is already written, and every later request refuses until it ends. `process` and `expand` both raise it. `AuthError` carries an `AuthState`, and `YtDlpError` its primary `Signal`; both enums are importable from `anyverbatim.errors`. Bad input raises `ValueError`, and a missing engine extra `DependencyError`.

### Dependencies

`check_dependencies(options, source="youtube")` checks that the ASR engine the options select for that source (`"youtube"` or `"recording"`) can be imported, plus the diarizer when that source is diarized and the numeric second-opinion engine when one is on, and names the extra that is missing: `whisper` for `w-prod`, the default, and `asr` for the other engines and the diarizer. A recording with `AsrOptions.languages` set defaults to `w-runs-lid`, which needs both extras. A recording also needs both with the default engine, because of its diarizer: `anyverbatim[whisper,asr]`. `process` runs the same check before any request. Nothing is imported or downloaded; an unknown engine is a `ConfigError`, and an unknown source a `ValueError`.

### Diarization

`DiarizeOptions(mode="auto")`, the `[diarize]` section of `config.toml`, decides which sources get speaker labels. See [docs/recordings.md](recordings.md#speakers) for how speakers are found, named and merged. `"auto"` diarizes recordings and leaves YouTube videos alone; `"on"` diarizes every source transcribed with ASR, and `"off"` none. A diarized document lists its speakers in `document.speakers` (`S1`, `S2`, … numbered by first appearance when attributed, listed in id order, each with `speech_s`; a later text stage never renumbers them, so ids may have gaps and need not appear in order), names one on every segment and paragraph, and describes the diarizer in `document.transcript.diarization`; an undiarized one has no speakers, `speaker` is `None` everywhere and `transcript.diarization` is `None`. An application that wants no diarization, whatever the source, sets `mode="off"` explicitly.

## Versioning

- 0.1.0 is the first release with a declared surface.
- Before 1.0, a minor release may break the surface. Every break is listed below and in the release notes. A patch release never breaks it.
- 1.0 follows once an application has run on this surface and the recordings milestone has settled.
- The contract version is independent of the package version: it is 2.0 now, and later optional fields become 2.x. See the README's [contract section](../README.md#contract-20).

## Breaking changes

### Next release: renamed to anyverbatim

The project was renamed from ytverbatim to anyverbatim, since it transcribes recordings as well as YouTube videos. The surface is otherwise unchanged.

1. The import path: the package was renamed from `ytverbatim` to `anyverbatim`, so `from ytverbatim import process` is now `from anyverbatim import process`, and `ytverbatim.errors|contract|config` are `anyverbatim.errors|contract|config`.
2. The distribution was renamed too: depend on `anyverbatim[whisper,asr] @ git+https://github.com/ADolkun/anyverbatim`, and drop the old ytverbatim requirement.
3. The base error `YtverbatimError` was renamed to `AnyverbatimError`.
4. Environment variables were renamed from `YTVERBATIM_*` to `ANYVERBATIM_*` (`ANYVERBATIM_HOME`, `ANYVERBATIM_AUTH`, `ANYVERBATIM_JS_RUNTIMES`). An old variable set without its new counterpart is a `ConfigError` whose `fix` names the new one.
5. A new document's `generator.name` is `anyverbatim`.

Stored work carries over. On first use, each directory left from before the project was renamed (`~/.local/share/ytverbatim`, `~/.config/ytverbatim`, `~/.local/state/ytverbatim`, `~/Library/Caches/ytverbatim`) is moved to its `anyverbatim` counterpart in one rename, and a notice goes to stderr. If both exist, the call raises `ConfigError` and moves nothing. A move to another filesystem is never copied: it raises `ConfigError` too. A home set with `ANYVERBATIM_HOME` is never moved. Transcripts, diarization and LLM results made before the rename stay current, so the first run after it rebuilds only `context.json` and `context.md`.

### 0.1.0

1. `process` returns a `Result`, not a `ContextDocument`. Read `result.document`.
2. Contract 2.0: `document.video` is `document.source` (check `source.kind == "youtube"`), and `document.access` is `document.source.access`. `load_document` still upgrades 1.0 and 1.1 documents in memory.
3. Internal modules are no longer API. Of the modules an application may have imported, `ids`, `paths`, `pipeline`, `acquire.*`, `auth.*`, `asr.*`, `ytdlp.*` and `llm.*` are internal; the names below replace what they were used for.
4. `BatchStopError.fix` is set by `process` as well as `expand`, and every error from either has a `stage` attribute.
5. `AsrOptions.engine` defaults to `None`, meaning the source's default engine. That is `w-prod` for YouTube, and for a recording too unless `AsrOptions.languages` is non-empty, which makes it `w-runs-lid`. An engine you set wins for every source. Since `w-runs-lid` checks the languages it is given, an unknown code in `languages` is now a `ConfigError` for a recording, where `w-prod` ignored it.
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
| `auth.state.AuthState`, `ytdlp.types.Signal` | `anyverbatim.errors.AuthState`, `anyverbatim.errors.Signal` |
| `auth.chromium.SUPPORTED_BROWSERS` | `anyverbatim.SUPPORTED_BROWSERS` |
| `pipeline.ProgressEvent`, `anyverbatim.llm.LLMClient` and its request types | `anyverbatim.ProgressEvent`, `LLMClient`, `LLMRequest`, `LLMResponse`, `Usage` |
| `pipeline.VideoResult`, `BatchReport`, `VideoOutcome` in a test double | a fake `process` that returns a `Result` |

## Not in the library yet

Voice Memos listing, speaker renaming (`anyverbatim speakers`), folder expansion, a recording type with caller-given metadata, and batch reports stay CLI features. They can be added without breaking the surface.
