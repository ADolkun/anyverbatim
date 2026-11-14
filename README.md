# ytverbatim

Turn YouTube URLs into verified, timestamped transcripts for LLM context.

Language models need transcripts they can cite. `ytverbatim` aims to produce text where every line carries a timestamp, so a quote can be traced back to the exact moment in the video. It works with your own logged-in browser session, so Premium and members-only videos you can already watch are covered. Speech recognition runs locally on Apple Silicon (MLX), with a focus on Mandarin and English.

**Status: pre-alpha.** `ytverbatim run` turns a video into a transcript with local Whisper and checks numbers against independent readings where there are any (the video's own captions, or an optional second engine). With an LLM (ChatGPT's gpt-6.1-sol by default, or one you choose), it also restores missing punctuation and corrects misheard words. An engine evaluation kept mlx-whisper large-v3 as the default ([result](evals/results/asr-2026-11-13.md)).

## Usage

```
ytverbatim run URL... [--out DIR] [--auth SPEC] [--engine ID] [--source asr|captions] [--correct/--no-correct]
                      [--chapters auto|none] [--llm SPEC] [--captions/--no-captions] [--force] [--ignore-cooldown]
ytverbatim render CONTEXT.json [--stdout]
ytverbatim login [--switch] [--no-browser]
ytverbatim logout
ytverbatim status [--json] [--refresh]
ytverbatim llm test [--provider P] [--model M] [--effort E]
```

Each video (playlists and channels are expanded) gets a folder `<out>/<video id>/` holding `context.json` (the versioned contract), `context.md` (the same transcript as markdown with a timestamped link per paragraph and YouTube chapters as headings) and the downloaded audio (plus `video.mp4` when [video](#library) is on). Exit codes: 0 ok, 1 some videos failed, 2 auth error, 3 batch stopped. `ytverbatim fetch` downloads without transcribing; `ytverbatim auth` sets up your browser login for YouTube (see [docs/auth.md](docs/auth.md)); `ytverbatim status` shows the YouTube login, the ChatGPT sign-in and the LLM in use on one screen, without an LLM call, and exits 1 when a line needs attention (no browser login is fine: YouTube is then limited to public videos); `ytverbatim render` rewrites the markdown from a `context.json`.

The stages run in this order: metadata, captions and audio; ASR (or, with `--source captions`, the manual caption track, falling back to ASR with a warning when there is none); normalize (Simplified Chinese by default, `[text] script = "keep"` to leave the script alone; fillers are kept unless `[text] fillers = "remove"`); punctuation, only where the transcript has too few sentence marks; correction, whenever the text stages have an LLM (`--no-correct` or `[text] correct = false` turns it off); numeric verification against captions in the transcript's language (without such captions or a second opinion the numbers rest on the transcript alone, and a warning says so); then sentences, paragraphs, chapters and the documents. Every stage is stored under `<video>/.stages/`, keyed by what it read: a rerun with other options recomputes only the stages those options reach, an interrupted run resumes, and a video whose `context.json` is current is skipped. `--force` discards the stored stages and recomputes every one, including the paid LLM calls. The 16 kHz WAV is deleted once the transcript is stored; the compressed audio stays.

`--llm PROVIDER:MODEL` picks the text stages' LLM for one run and `--llm none` runs without one (see [LLM](#llm)). A provider other than the configured one starts from its own defaults, so a configured `base_url` or `effort` does not get in the way. The LLM's effort and endpoint host are part of its identity, so changing them redoes the LLM stages. Without one, punctuation is left as transcribed, correction is skipped and disputed numbers stay flagged with every reading; punctuation and numbers are recorded in `quality.warnings` and printed as a notice. With an LLM, correction runs unless switched off; `--correct` (or `[text] correct = true`) without an LLM is an error. `llm_usage` is what the document cost in LLM tokens, per stage: batches reused from earlier runs count too, so a rerun that reuses everything reports the same totals. It is not what one run spent. As a library, `ytverbatim.process(url, options, llm=client)` takes any object that satisfies `ytverbatim.LLMClient` (see [Library](#library) for running without one).

When YouTube answers with a bot check or a rate limit, the batch stops and a cooldown starts (1 hour, doubling on repeats up to 24 hours). Until it ends, `run` and `fetch` refuse to start; only a `run` whose targets are all single videos it has already finished goes ahead, because it requests nothing. `--ignore-cooldown` starts anyway, at the risk of your account being flagged; use it only when you know the block has lifted.

The default engine, `w-prod`, is mlx-whisper large-v3 and needs the `whisper` extra (`uv sync --extra whisper`); the other engines need the `asr` extra. `run` checks for the engine's extra before it downloads anything.

ASR runs in a separate worker process under a memory watch, so a runaway model cannot take down the Mac. The worker is killed if its physical footprint passes `memory_cap_gib` (default: half of physical memory) or if the system's available memory falls below 10%. The batch then stops with a fix for whichever one fired, and the next run resumes from the chunks already finished. On an 8 GiB Mac the default cap of 4 GiB may be too tight for `w-prod`; raise `asr.memory_cap_gib` (at most 90% of physical memory). To change the cap, set it in `~/.config/ytverbatim/config.toml`:

```toml
[asr]
memory_cap_gib = 8
```

## LLM

Punctuation, correction and numeric verdicts use an LLM. The default is ChatGPT's gpt-6.1-sol at medium reasoning effort, which needs a ChatGPT plan and one sign-in:

```
ytverbatim login         # opens OpenAI's sign-in page and waits for it; --no-browser only prints the URL
ytverbatim login --switch  # signs in with another account; the current one stays until the new one is verified
ytverbatim status        # the YouTube login, the ChatGPT sign-in and the LLM in use; no LLM call
ytverbatim llm test      # one tiny live call to the configured LLM (--provider/--model/--effort override it)
ytverbatim logout        # revokes the sign-in and removes its tokens from this machine
```

In the [correction evaluation](evals/results/correction-2026-11-13.md) gpt-6.1-sol led the screen, removing about 57.6 net errors per hour against 42.7 for Claude Sonnet, at a fix precision of 0.929. Medium effort matched high (57.6 against 57.8 net errors removed per hour, with overlapping intervals) in about 0.6 times the time, so medium is the default. The sign-in belongs to ytverbatim alone: its tokens stay in an owner-only file under `~/.local/state/ytverbatim/llm/chatgpt/` and are never printed. It is kept alive automatically: short-lived access tokens are renewed at the first moment OpenAI allows, both before a run starts and before each call, so one sign-in lasts as long as OpenAI keeps renewing it. The sign-in stays until you sign out: ytverbatim never signs you out on its own. If OpenAI refuses a renewal, the stored sign-in is kept and its access token is used until 30 seconds before it expires, but it is not renewed again: `ytverbatim status` says a new sign-in is needed, and `ytverbatim login` signs in again with your account preselected. When the sign-in is missing, refused or lapsed, an interactive `run` (a terminal on both stdin and stderr) opens the browser to sign in again and then continues; a ChatGPT request refused as unauthorized mid-run does the same, once. A non-interactive run (no terminal, cron, the library) never opens a browser: it stops before any download and tells you to run `ytverbatim login`.

The alternatives are `--llm claude-cli:claude-sonnet-5-5`, which runs through the `claude` CLI on your claude.ai subscription, and `--llm openai-compat:MODEL` with `[llm] base_url` set, for any OpenAI-compatible endpoint such as a local `mlx_lm.server`. `--llm none` skips the LLM stages and keeps everything on your machine. To change the default for every run, set it in `~/.config/ytverbatim/config.toml`; `effort` takes low, medium, high, xhigh or max for chatgpt and claude-cli:

```toml
[llm]
provider = "claude-cli"
model = "claude-sonnet-5-5"
```

The LLM stages send transcript text, plus the video's tags and the uploader's pinned note when there is one, to the provider you chose. Your cookies never leave your machine.

## Library

```python
import ytverbatim

ids = ytverbatim.expand("https://www.youtube.com/playlist?list=...", limit=20)
document = ytverbatim.process(ids[0], llm=client)
```

Without `llm=`, `process` uses the configured LLM, ChatGPT by default, and raises `ConfigError` before any work until `ytverbatim login` has run once (`process(..., interactive=True)` opens the browser to sign in instead, for an application with a person at the screen; the library never does so on its own); `process(url, Options(llm=LLMOptions(provider="none")))` (both from `ytverbatim.config`) runs without an LLM instead.

`expand(target, *, limit=0, auth=None, options=None)` returns the video ids behind one video, playlist or channel URL, in order and without repeats. A video id costs no request, and `limit` caps the listing (0 lists everything; the `max_videos_per_run` cap belongs to `run`). Like every request, it refuses to start during a cooldown and goes through the same stop guard, so a bot check or rate limit starts the cooldown and raises. Only a members playlist (`UUMO...`) is listed with cookies, from `auth` or the configured source; with none (or with `auth.policy = "always"` and none), it raises `AuthError` instead of listing anonymously. The ids carry no memory of where they came from, so for ids from a members playlist call `process(video_id, cookies_first=True)`: the video's first request then uses cookies, where otherwise each one is requested anonymously, refused, and only then retried with cookies.

**Video.** The video track is opt-in, for callers that need frames (screenshots, say). With `[ytdlp] video = true` (or `Options(ytdlp=YtdlpOptions(video=True))`), `run`, `process` and `fetch` also save `<out>/<video id>/video.mp4` after the audio: the best video-only stream up to 1080p, remuxed to MP4, requested on the same cookie path as the audio, so a members-only or Premium video needs nothing extra. It is validated like the audio (it decodes and covers at least 90% of the video's duration), a file that fails is downloaded again, and a valid one is reused. A video that is already transcribed is not transcribed again for this: `process` only fetches the missing file, which is one `video` download plus a fresh metadata extraction when the saved one is over an hour old (its signed URLs have expired), made with cookies if the earlier run used them; with the option off (the default) no video is ever requested. A failed video download fails the video. `ytverbatim.video_file(video_id, out=None)` returns the path of the validated file under `out` (default output root), or `None` when it is missing or fails validation; it never downloads. A probe that times out counts as absent; a missing `ffprobe` raises `DependencyError`, and a malformed id `ValueError`.

**Quantities.** `ytverbatim.quantity_tokens(text)` returns the numbers, currency amounts, percentages, tickers, Chinese numerals and spelled-out English numbers in `text` as `ytverbatim.QToken`s in order: `char_start` and `char_end` (half-open offsets into `text`), `surface`, `value_itn` (the normalized value two spellings share: `一百五十期` and `150期` are both `150期`, `$1.5B` is `$1500000000`, `twenty-five` is `25`) and `kind` (`ytverbatim.QKind`: `number`, `currency`, `percent`, `ticker` or `cjk_quantity`). A Chinese numeral is a quantity when it has two or more numeral characters (`三十`), follows `第` or `百分之`, or has a measure word: a currency, 个, 年, 次, a count classifier such as 家, 种 or 位, a time or measurement unit such as 分钟, 岁 or 公里, or a value unit such as 倍 or 个基点. 一 after a figure in 万 or 亿 starts a per-unit phrase, so `三万一个月` is `30000` and `1个月`, and 点 before tens is a clock time (`三点四十分` is `3点` and `40`). Ranges of adjacent digits (`三四个`), idioms (`万一`, `一样`, `十万八千里`) and weekdays are not quantities; fractions (`三分之一`) and rough figures with 几 or 数 (`二十几岁`, `数十亿`) are tokens whose `value_itn` is their surface, so they are still compared but never read as an exact value. It is the tokenizer numeric verification and the correction guards use.

**Errors.** Every error the package raises on purpose subclasses `ytverbatim.errors.YtverbatimError`, whose `fix` attribute is a one-line remedy to show the user (empty when there is none). `process` raises the video's own error as it happened: `AuthError` (with its `state`), `UnavailableError`, `YtDlpError` (with its `primary` signal), `MediaValidationError`, `ProcTimeout`, `EngineError`, `LLMError` or `ContractError`, and `fix` is set where a remedy exists. A bot check or rate limit raises `BatchStopError` (`signal`, `until`) after starting the cooldown, from `process` and `expand` alike; `expand` also sets its `fix`, `process` leaves it empty. Bad input raises `ValueError`, and a missing engine extra `DependencyError`.

**Contract 1.1.** `context.json` carries `schema_version` `"1.1"`; the bundled schema, `ytverbatim/contract/context.v1.schema.json`, describes the current minor version (1.1) only, while `load_document` also reads 1.0. Minor versions only add optional fields, so a reader loads any `1.x` document: `load_document` accepts 1.0 documents (the new fields read as `None`) and a newer minor version with the fields it does not know dropped. A document with another major version is refused with `ContractError`. Version 1.1 adds `video.creator_note`: `{text, author, like_count, published}`, the uploader's own pinned comment (often an erratum), or `None`. It is only filled when `[ytdlp] creator_note = true`, which adds a bounded page of comments to the video's one metadata request; turning it on for a video fetched without it extracts its metadata once more and rebuilds the document from the stored stages. With the option off, which is the default, no extra request is made.

## Planned for v1

- Auth status and a `doctor` command that explain problems with clear fixes
- Local ASR by default, with YouTube captions used as cross-checking evidence
- Timestamps on every segment, word-level where the engine provides them
- A versioned JSON output plus a markdown rendering

## Requirements

- macOS on Apple Silicon
- `ffmpeg`
- A JavaScript runtime for yt-dlp: Deno or Node 22 or newer
- `curl_cffi` (installed with ytverbatim, MIT): yt-dlp asks for browser impersonation on caption downloads, and without it YouTube rate-limits them (HTTP 429) more often and yt-dlp warns that no impersonate target is available

## Responsible use

Use `ytverbatim` only on content you can already access with your own account. It does not circumvent DRM. You are responsible for complying with YouTube's Terms of Service and with copyright law. Your cookies never leave your machine; transcript text reaches an LLM provider only through the LLM stages (see [LLM](#llm)).

## License

[MIT](LICENSE)
