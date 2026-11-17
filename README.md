# ytverbatim

Turn YouTube videos and your own recordings (meetings, iPhone Voice Memos) into verified, timestamped transcripts for LLM context.

Language models need transcripts they can cite. `ytverbatim` aims to produce text where every line carries a timestamp, so a quote can be traced back to the exact moment in a video or a meeting. Speech recognition runs locally on Apple Silicon (MLX), with a focus on Mandarin and English.

- **YouTube** videos, playlists and channels. It works with your own logged-in browser session, so Premium and members-only videos you can already watch are covered.
- **Local recordings:** audio or video files, folders of them, and iPhone Voice Memos synced to your Mac. Conversations get speaker labels.

**Status: pre-alpha.** `ytverbatim run` turns a source into a transcript with local Whisper and checks numbers against independent readings where there are any (a video's own captions, or an optional second engine). With an LLM (ChatGPT's gpt-6.1-sol by default, or one you choose), it also restores missing punctuation and corrects misheard words. An engine evaluation kept mlx-whisper large-v3 as the default ([result](evals/results/asr-2026-11-13.md)). Recordings support is new and is still being checked on real meetings and memos.

## Install

Requirements:

- macOS on Apple Silicon
- `ffmpeg`, with `ffprobe`
- For YouTube: a JavaScript runtime for yt-dlp, Deno or Node 22 or newer
- `curl_cffi` (installed with ytverbatim, MIT): yt-dlp asks for browser impersonation on caption downloads, and without it YouTube rate-limits them (HTTP 429) more often and yt-dlp warns that no impersonate target is available

Install the tool with [uv](https://docs.astral.sh/uv/). This puts `ytverbatim` on your `PATH`:

```
uv tool install "ytverbatim[whisper,asr] @ git+https://github.com/ADolkun/ytverbatim"
uv tool upgrade ytverbatim            # to update later
```

There is no PyPI release yet, so the install comes from the repository. That installs the newest `main`; to pin a release, add its tag, as in `…/ytverbatim@v0.1.0`. The part in brackets names two optional extras, which hold the speech engines. `run` checks for the extra it needs before it downloads anything, and its error names the extra that is missing.

| You want | Extras |
|---|---|
| YouTube with the default engine (`w-prod`, mlx-whisper large-v3) | `whisper` |
| Recordings: ASR plus speaker diarization, which labels who speaks when | `whisper` and `asr` |
| The chunked Whisper engines `w-vad`, `w-vad-lid` and `w-runs-lid` (the default for recordings once `[asr] languages` is set), or diarizing YouTube videos with the default engine | `whisper` and `asr` |
| The other engines (`q-base`, `q-ctx`, `f-aed`, `m-4bit`) | `asr` |

To work on ytverbatim itself, use a checkout and run it through uv:

```
uv sync --extra whisper --extra asr
uv run ytverbatim --help
```

## Quick start

The commands assume `ytverbatim` is on your `PATH`; from a checkout, write `uv run ytverbatim …`.

```
ytverbatim login                                     # once, for the default LLM
ytverbatim run https://www.youtube.com/watch?v=VIDEO_ID
ytverbatim run meeting.m4a                           # a recording; folders work too
ytverbatim run memos:new                             # new iPhone Voice Memos
ytverbatim speakers rec-0123456789abcdef S1=Alice S2=Bob
```

No ChatGPT plan? Skip `login` and add `--llm none` to each `run`, or pick another provider (see [LLM](#llm)).

| Command | What it does |
|---|---|
| `run TARGET...` | Transcribe YouTube URLs or ids, files, folders, or `memos:` selectors. |
| `memos` | List the iPhone Voice Memos synced to this Mac. |
| `speakers TARGET [ID=NAME]...` | List, name or merge a recording's speakers. |
| `render CONTEXT.json [--stdout]` | Rewrite the markdown from a `context.json`, or print it with `--stdout`. |
| `fetch URL...` | Download a video's metadata, captions and audio without transcribing. |
| `auth` | Set up your browser login for YouTube (see [docs/auth.md](docs/auth.md)). |
| `login`, `logout` | Sign in to or out of ChatGPT, the default LLM. |
| `status` | Show the YouTube login, the ChatGPT sign-in, the LLM in use and Voice Memos access, with no LLM call. |
| `llm test` | One tiny live call to the configured LLM. |

`status` exits 1 when a line needs attention. No browser login is fine: YouTube is then limited to public videos. The Voice Memos line is information only and never changes the exit code.

A run takes one kind of target: YouTube, or local (files, folders and memos, which can be mixed). Exit codes: 0 ok, 1 some sources failed, 2 auth error, 3 batch stopped (a YouTube cooldown).

## YouTube

```
ytverbatim run URL... [--out DIR] [--auth SPEC] [--engine ID] [--source asr|captions] [--correct/--no-correct]
                      [--chapters auto|none] [--diarize/--no-diarize] [--language CODE] [--glossary TERMS]
                      [--llm SPEC] [--captions/--no-captions] [--force] [--ignore-cooldown]
```

Each video (playlists and channels are expanded) gets a folder `<out>/<video id>/` under `~/.local/share/ytverbatim/videos/`. It holds:

- `context.json`, the versioned [contract](#contract-20)
- `context.md`, the same transcript as markdown with a timestamped link per paragraph and YouTube chapters as headings
- the downloaded audio, plus `video.mp4` when [video](docs/library.md#video) is on

`--source captions` trusts the video's manual caption track and falls back to ASR with a warning when there is none.

When YouTube answers with a bot check or a rate limit, the batch stops and a cooldown starts (1 hour, doubling on repeats up to 24 hours). Until it ends, `run` and `fetch` refuse to start. Only a `run` whose targets are all single videos it has already finished goes ahead, because it requests nothing. `--ignore-cooldown` starts anyway, at the risk of your account being flagged; use it only when you know the block has lifted.

## Recordings

```
ytverbatim run meeting.m4a standup.mp4 ~/Recordings/2026-11/
```

Files and folders (searched recursively) work like YouTube videos, with no download and no cooldown. YouTube-only options (`--auth`, `--source`, `--captions`, `--ignore-cooldown`) are refused for them, and `--quiet` is refused for YouTube.

**Recording other people needs their consent.** See [Privacy and consent](#privacy-and-responsible-use) before you do.

- **Id.** A recording's id is `rec-` plus 16 hex digits: the first 16 of the SHA-256 of the transcribed audio stream's encoded packets. Renaming the file, editing its tags or remuxing it keeps the id. The same audio twice in a run is transcribed once.
- **Folder.** `~/.local/share/ytverbatim/recordings/rec-0123456789abcdef/` holds `context.json`, `context.md` and, once you name speakers, `speakers.json`. Your file is only read, never copied or changed.
- **`index.md`.** The recordings folder has an `index.md`: every recording by date, title, duration and speakers, newest first, each linking to its `context.md`. It is the one file to hand to another tool.
- **Reruns.** A recording whose `context.json` is current is skipped. A rerun with other options redoes only the stages they reach. `--force` redoes everything, including paid LLM calls.
- **Title and date** come from the file's tags, or its name, and never from its modification time.

Everything else, including the folder layout, the markdown format and the file types read from folders, is in [docs/recordings.md](docs/recordings.md).

## Voice Memos

```
ytverbatim memos [--all] [--json]
ytverbatim run memos:new|latest|all|ID [--quiet] ...
```

- `memos` lists your memos, newest 20 or `--all`: date, duration, title, status and a short id. Status is `transcribed`, `new`, `not downloaded` or `unsupported`.
- `memos:new` transcribes the memos that have no document yet, and hashes only the ones it has not seen before. `memos:latest` and `memos:all` do what they say. `memos:ID` takes the start of an id as `memos` lists it (8 characters). Every `run` option applies.
- The memo's title and recording date from Voice Memos become the document's `title` and `recorded_at`. If you rename a memo, or first transcribed its shared file as a plain file, the next `memos:` run that selects it updates the record from the stored stages, with no ASR or LLM call.

Setup:

- Turn on Voice Memos in iCloud settings on the iPhone and the Mac, and open Voice Memos on the Mac once, so memos sync.
- Give the app you run ytverbatim from Full Disk Access (System Settings → Privacy & Security → Full Disk Access), then restart it. ytverbatim reads a private copy of the Voice Memos database, so nothing is written into Apple's folder; the audio is read in place.
- A memo that is not downloaded to this Mac is skipped with a notice. Open it once in Voice Memos to download it.

To transcribe new memos on a schedule, see [Automation](docs/recordings.md#automation): `run memos:new --quiet`, which prints nothing when idle, skips memos another run holds, needs Full Disk Access for the program the job runs, and needs `ytverbatim login` to have run once.

## How a run works

The stages run in this order:

1. Metadata, captions and audio for YouTube, or the file's audio for a recording.
2. ASR (or, with `--source captions`, the manual caption track).
3. For a diarized source, speaker diarization (see [Speakers](#speakers)).
4. Normalize: Simplified Chinese by default, with ASCII marks next to Chinese characters written full-width (`[text] script = "keep"` leaves both alone). Both work on the characters, so Chinese inside an English stretch is converted too, whatever language the stretch was decoded in. Fillers are kept unless `[text] fillers = "remove"`, and they follow the stretch's language.
5. Punctuation, only where the transcript has too few sentence marks.
6. Correction, whenever the text stages have an LLM (`--no-correct` or `[text] correct = false` turns it off). A name the LLM changed in Latin script is flagged `entity_corrected`, as tickers and acronyms are.
7. Numeric verification against captions in the transcript's language. Without such captions or a second opinion the numbers rest on the transcript alone, and a warning says so (recordings have no captions).
8. For Mandarin, Chinese numerals written as digits (see [Numerals](#numerals)).
9. Sentences, paragraphs, chapters and the documents.

Every stage is stored under `.stages/` in the source's folder (the numerals stage only when it runs), keyed by what it read. A rerun with other options recomputes only the stages those options reach, an interrupted run resumes, and a source whose `context.json` is current is skipped. `--force` discards the stored stages and recomputes every one, including the paid LLM calls. The 16 kHz WAV is deleted once the transcript and the speaker turns are stored; the compressed audio of a video stays.

**Engines and memory.** The default engine, `w-prod`, is mlx-whisper large-v3. ASR and diarization run in separate worker processes under a memory watch, so a runaway model cannot take down the Mac. The worker is killed if its physical footprint passes `memory_cap_gib` (default: half of physical memory) or if the system's available memory falls below 10%. The batch then stops with a fix for whichever one fired, and the next run resumes from the chunks already finished. On an 8 GiB Mac the default cap of 4 GiB may be too tight for `w-prod`; raise it (at most 90% of physical memory) in `~/.config/ytverbatim/config.toml`:

```toml
[asr]
memory_cap_gib = 8
```

## Speakers

Recordings are diarized by default: a local model (Nemotron-3-Diarization, 8-bit, about 107 MB, downloaded once; needs the `asr` extra) finds who speaks when, up to 8 speakers. Paragraphs break at every change of speaker, and `context.md` labels each one `**Speaker 1:**`, `**Speaker 2:**`, … with each speaker's share of the talk time in the header. A recording with one speaker shows no labels. `[diarize] mode` is `"auto"` (recordings only, the default), `"on"` or `"off"`, and `--diarize/--no-diarize` overrides it for one run.

```
ytverbatim speakers rec-0123456789abcdef                    # list: id, name, talk time, first line
ytverbatim speakers rec-0123456789abcdef S1=Alice S2=Bob    # name
ytverbatim speakers rec-0123456789abcdef S2=                # clear a name
ytverbatim speakers rec-0123456789abcdef S1=Alice S3=Alice  # merge: one name, one person
```

Nothing is transcribed again and names are never sent to an LLM. Re-diarizing drops saved names with a warning. Details: [docs/recordings.md](docs/recordings.md#speakers).

## Languages

`--language CODE` forces one spoken language for any source (`zh`, `en`, …). `[asr] languages = ["zh", "en"]` limits the languages a language-detecting engine may choose from, for recordings only.

A recording whose languages you name gets per-stretch languages: with `[asr] languages` set (for mixed Mandarin and English, `languages = ["zh", "en"]`), the default engine for recordings is `w-runs-lid`, which detects each stretch's language among those and decodes it with that language. Each stretch is then normalized, punctuated and given digit numerals by its own language's rules. Without `[asr] languages` the default is `w-prod`, which decodes the whole file and detects one language for it, from the start of the file. YouTube always defaults to `w-prod`. `--engine ID` picks the engine for every source; the ids and the evidence are in [docs/recordings.md](docs/recordings.md#languages).

## Glossary

To steer correction toward names and jargon it would otherwise mishear, set `[text] glossary = ["Acme", "Kubernetes"]` (at most 20 terms) or pass `--glossary "Acme, Kubernetes"` for one run. For a YouTube video the configured terms come first and the video's tags follow, up to 20 in all. The terms are sent to the LLM. Details: [docs/recordings.md](docs/recordings.md#glossary).

## Numerals

Mandarin transcripts write amounts as digits, following GB/T 15835-2011 (出版物上数字用法). For recordings this applies to the Mandarin stretches. `[text] numerals = "keep"` leaves the numerals as recognized.

| Kind | Example |
|---|---|
| Percentages | `百分之八十` → `80%` |
| Currency | `六百五十六美元` → `656美元` |
| Values in 万 or 亿, in the speaker's unit | `三万五` → `3.5万`, `一亿五千三` → `1.53亿` |
| 千 and 百, which never stay beside digits | `4千` → `4000` |
| Years | `二零二六年` → `2026年` |
| Counts of two and up | `三个` → `3个`, `两年` → `2年` |
| Dates | `一月一日` → `1月1日` |
| Clock times | `下午四点四十分` → `下午4点40分`, `三点半` → `3点半`, `两点零五分` → `2点05分` |
| Index points | `涨了三点` → `涨了3点`, `指数两千八百点` → `指数2800点` |
| Bare numbers | `三点五` → `3.5` |
| Ranges, whole | `三到五个` → `3到5个`, `下午两点到四点` → `下午2点到4点`, `两到三亿` → `2到3亿`, `两到三百万` → `200到300万` |

X点 up to 10 is written in digits only as a clock time (minutes, 半 or a time of day such as 下午 before it) or after a move such as 涨, 下跌 or 收涨, so `两点最重要` and `有三点需要注意` stay.

These stay Chinese:

- approximations (`三四个`, `几十`, `二十几岁`)
- idioms and set words (`一样`, `万一`, `三大`, `十二生肖`)
- ordinals (`第二十五期`, `二季度`)
- 一 before a count word (`一个`, `一次`)
- lunar dates (`腊月二十三`, `八月十五`)
- fractions (`三分之一`)
- spoken two-digit years (`九七年`)
- figures a following numeral continues (`三块五`, `一米八`)
- ranges that share a percent between their ends (`百分之三到五`), carry a magnitude on the left (`三万到五`) or cannot be rewritten whole
- 一到 meaning "as soon as" (`一到三月份就涨`)

Only figures the [quantity tokenizer](docs/library.md#quantities) reads are rewritten, and a rewrite never changes a figure's value. The stage runs after numeric verification, so punctuation, correction and verification read the text as recognized, and a change to these rules never repeats an LLM call. Word timings, flags and edits follow the rewritten text; a figure its timed words only partly match stays as said.

## LLM

Punctuation, correction and numeric verdicts use an LLM. The default is ChatGPT's gpt-6.1-sol at medium reasoning effort, which needs a ChatGPT plan and one sign-in:

```
ytverbatim login           # opens OpenAI's sign-in page and waits for it; --no-browser only prints the URL
ytverbatim login --switch  # signs in with another account; the current one stays until the new one is verified
ytverbatim status          # the YouTube login, the ChatGPT sign-in, the LLM in use; no LLM call
ytverbatim llm test        # one tiny live call to the configured LLM (--provider/--model/--effort override it)
ytverbatim logout          # revokes the sign-in and removes its tokens from this machine
```

In the [correction evaluation](evals/results/correction-2026-11-13.md) gpt-6.1-sol led the screen, removing about 57.6 net errors per hour against 42.7 for Claude Sonnet, at a fix precision of 0.929. Medium effort matched high (57.6 against 57.8 net errors removed per hour, with overlapping intervals) in about 0.6 times the time, so medium is the default.

**The sign-in.**

- It belongs to ytverbatim alone. Its tokens stay in an owner-only file under `~/.local/state/ytverbatim/llm/chatgpt/` and are never printed.
- It is kept alive automatically: short-lived access tokens are renewed at the first moment OpenAI allows, both before a run starts and before each call, so one sign-in lasts as long as OpenAI keeps renewing it.
- It stays until you sign out: ytverbatim never signs you out on its own.
- If OpenAI refuses a renewal, the stored sign-in is kept and its access token is used until 30 seconds before it expires, but it is not renewed again. `ytverbatim status` says a new sign-in is needed, and `ytverbatim login` signs in again with your account preselected.
- When the sign-in is missing, refused or lapsed, an interactive `run` (a terminal on both stdin and stderr) opens the browser to sign in again and then continues. A ChatGPT request refused as unauthorized mid-run does the same, once.
- A non-interactive run (no terminal, cron, the library) never opens a browser: it stops before any download and tells you to run `ytverbatim login`.

**Choosing another LLM.** `--llm PROVIDER:MODEL` picks the text stages' LLM for one run:

- `--llm claude-cli:claude-sonnet-5-5` runs through the `claude` CLI on your claude.ai subscription.
- `--llm openai-compat:MODEL`, with `[llm] base_url` set, uses any OpenAI-compatible endpoint, such as a local `mlx_lm.server`.
- `--llm none` runs without an LLM and keeps everything on your machine.

A provider other than the configured one starts from its own defaults, so a configured `base_url` or `effort` does not get in the way. The LLM's effort and endpoint host are part of its identity, so changing them redoes the LLM stages. To change the default for every run, set it in `~/.config/ytverbatim/config.toml`; `effort` takes low, medium, high, xhigh or max for chatgpt and claude-cli:

```toml
[llm]
provider = "claude-cli"
model = "claude-sonnet-5-5"
```

**Without an LLM,** punctuation is left as transcribed, correction is skipped and disputed numbers stay flagged with every reading. Punctuation and numbers are recorded in `quality.warnings` and printed as a notice. With an LLM, correction runs unless switched off; `--correct` (or `[text] correct = true`) without an LLM is an error.

`llm_usage` is what the document cost in LLM tokens, per stage. Batches reused from earlier runs count too, so a rerun that reuses everything reports the same totals: it is not what one run spent.

## Privacy and responsible use

**Recording other people.** Get everyone's consent before you record a conversation. In many places recording without it is illegal, and some require every participant's consent. ytverbatim does not check.

- **Stays on your Mac:** audio, your files, your Voice Memos library, speech recognition and diarization. Your YouTube cookies never leave your machine.
- **Sent to the LLM provider** (ChatGPT, so OpenAI, by default): transcript text, speaker ids (`S1`, `S2`, …) and glossary terms. For a YouTube video also its tags, and the uploader's pinned note only when `[ytdlp] creator_note = true` (off by default).
- **Never sent:** a recording's title, file name, recording date and speaker names.
- **Opt out** with `--llm none` (or `[llm] provider = "none"`): no transcript text is sent anywhere.

Whether your ChatGPT plan's data controls apply to this sign-in is unverified; see [docs/recordings.md](docs/recordings.md#privacy-and-consent) for that and the rest.

**YouTube.** Use `ytverbatim` only on content you can already access with your own account. It does not circumvent DRM. You are responsible for complying with YouTube's Terms of Service and with copyright law.

## Library

```python
import ytverbatim

ids = ytverbatim.expand("https://www.youtube.com/playlist?list=...", limit=20)
result = ytverbatim.process(ids[0], llm=client)  # or process("meeting.m4a", llm=client)
document = result.document
```

The stable surface is every name in `ytverbatim.__all__`, plus the modules `ytverbatim.errors`, `ytverbatim.contract` and `ytverbatim.config`; everything else is internal. [docs/library.md](docs/library.md) lists it, with the versioning policy and the breaking changes of each release.

`process` takes a YouTube video or a local audio or video file and returns a `Result`: the `document`, the exact `context.json` bytes (`json_bytes`), its `directory`, the `markdown_path`, the `video_path` when the video track was requested, and what was `reused`. It takes any object that satisfies `ytverbatim.LLMClient` as `llm=`. Without one it uses the configured LLM, ChatGPT by default, and raises `ConfigError` until `ytverbatim login` has run once; `process(url, Options(llm=LLMOptions(provider="none")))` runs without an LLM. The library never opens a browser unless you pass `interactive=True`.

The default engine, `w-prod`, needs `ytverbatim[whisper]`, and the other engines and the diarizer need `ytverbatim[asr]`, so recordings, which are diarized by default, need `ytverbatim[whisper,asr]`. `ytverbatim.check_dependencies(options, source="youtube")` (or `"recording"`) raises `DependencyError` naming the missing extra, and `process` runs the same check before any request.

Also in the library, all described in [docs/library.md](docs/library.md): `expand` for playlist and channel ids, the opt-in `video.mp4` track, `quantity_tokens`, and the errors, each with a `fix` and a `stage`.

## Contract 2.0

`context.json` carries `schema_version` `"2.0"`; the bundled schema, `ytverbatim/contract/context.v2.schema.json`, describes it.

- **`source`** says where the transcript came from, as a union on `source.kind`:
  - `"youtube"` holds the video's `id`, `url`, `title`, `channel`, `channel_id`, `upload_date`, `duration_s`, `description`, `language`, `tags`, `creator_note` and `access` (how it was fetched: auth source, cookies, login, Premium, membership, availability, yt-dlp version, warnings).
  - `"recording"` holds a local recording's `id` (`rec-` plus the first 16 hex digits of `audio_sha256`, the hash of its audio stream's encoded packets), `title`, `file_name` (a basename, never a path), `origin` (`file` or `voice_memos`), `recorded_at` (or `None` when unknown), `duration_s` and `language`.
- **`speakers`** lists each diarized speaker (`id` `S1`, `S2`, …; `name`; `speech_s`) and is empty when the transcript was not diarized. Ids are numbered by first appearance when the speech is attributed and are never renumbered, so a later text change can leave a gap or let a higher id speak first. The list is in id order, each id once. Each segment and paragraph then names its `speaker`, and a paragraph never spans two.
- **`transcript.languages`** lists every language the transcript was decoded in, the document's own first and the others by the speech they cover. `transcript.diarization` describes the diarizer run, or is `None`.
- **Flags.** `overlapping_speech` marks speech under another speaker; like `timing_interpolated` it stays out of the markdown. `entity_corrected` marks a ticker or acronym the LLM changed, and a Latin-script name it changed: in a Chinese segment any Latin-script word, elsewhere a word with a capital after its first letter or a capitalized word inside a sentence (a name that opens a sentence is not flagged). A change only of case, spaces or punctuation inside the name is not flagged. Its note reads `before → after`, where `before` is the text that stood in its place, Chinese characters included, and is empty for a name inserted into unchanged text. Names written in Chinese are not flagged.
- **`creator_note`** (added in 1.1) is `{text, author, like_count, published}`, the uploader's own pinned comment (often an erratum), or `None`. It is only filled when `[ytdlp] creator_note = true`, which adds a bounded page of comments to the video's one metadata request. Turning it on for a video fetched without it extracts its metadata once more and rebuilds the document from the stored stages. With the option off, the default, no extra request is made.

**Versions.** Minor versions only add optional fields. `load_document` reads any `2.x` document, dropping the fields a newer minor version adds, and upgrades a 1.0 or 1.1 document in memory (`video` and `access` become a YouTube `source`, with no speakers and its one language as `transcript.languages`). Any other version is refused with `ContractError`, whose `fix` says to upgrade ytverbatim when the document is newer. Readers refuse enum values they do not know, so a new flag kind, `origin` or `source.kind` comes with a new major version, never a minor one.

**Upgrading from contract 1.1.** Every YouTube `context.json` becomes 2.0, and each video's `context.md` is written again. Its text changes only where two segments of a Mandarin transcript meet in Latin script on both sides, which is rare: an English sentence that follows another no longer runs into it ("end.Next" becomes "end. Next"), while a boundary at any Chinese character or punctuation still takes no space. The next `run` or `process` of each video rebuilds only the document from its stored stages: no download, no Whisper run and no LLM call. A reader of `document.video` reads `document.source` (after checking `source.kind == "youtube"`), and `document.access` is now `document.source.access`.

**Upgrading to character-level script conversion.** With `[text] script = "zh-Hans"`, Chinese inside English stretches is now converted to Simplified and its marks written full-width, as it already was in Chinese stretches. The next `run` or `process` of each document normalizes it again once, then rebuilds the context from the stored stages. Only two things change: Chinese characters in a non-Chinese stretch with the marks next to them, and an ASCII mark that follows a full-width one with no Chinese character beside it (`Well…, ok`), which no longer turns full-width. Every other stretch keeps its text byte for byte. Stages after normalization are redone only where their input text changed, so a document with no such text makes no new LLM call.

**Upgrading to name flags.** The contract stays 2.0: no field, flag kind or edit stage is new. The next `run` or `process` of each document applies the correction guards again to the stored LLM replies, so correction makes no new call, then redoes numeric verification and the document, and the numerals stage where its input changed. Numeric verification asks the LLM again only for a disputed batch whose prompt changed, which the caption cue fix does when a mostly-Chinese caption track's cues meet in Latin script or digits (they are now read as separate words, not one). Punctuation and every earlier stage are kept. More segments carry an `entity_corrected` flag, most of all in Mandarin transcripts with Latin-script names; the flags change no text.

## What ships, and what is next

Shipped:

- YouTube and local recordings, each with local ASR, timestamps on every segment (word-level where the engine provides them) and number checks
- Speaker diarization, with naming and merging
- Voice Memos, with scheduled runs
- A versioned JSON contract plus a markdown rendering
- LLM punctuation and correction, with a glossary, and per-stretch language handling for mixed-language recordings
- Login and status commands that explain problems with a fix (`login`, `status`, `auth doctor`)

Not yet:

- Separate tracks for each participant of a call, and importing the transcripts meeting apps produce (with their speaker names)
- Chapters for recordings
- A speaker count hint for the diarizer
- Voice Memos listing, speaker naming and folder expansion in the library; they stay CLI features
- A fix for short stretches of a mixed Mandarin and English recording that come out translated: decoding each run with both language tokens [did not clear its pre-registered rule](evals/results/two-token-2026-11-16.md)

## License

[MIT](LICENSE)
