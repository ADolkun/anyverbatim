# Recordings

`ytverbatim run` also transcribes your own recordings: audio or video files, folders of them, and iPhone Voice Memos synced to your Mac. A recording gets the same `context.json` and `context.md` as a YouTube video, plus speaker labels when more than one person talks.

This page is the full reference. The [README](../README.md) has the short version.

- [Before you record people](#before-you-record-people)
- [Files and folders](#files-and-folders)
- [The recording id](#the-recording-id)
- [Output](#output)
- [Reruns, skips and `--force`](#reruns-skips-and---force)
- [Voice Memos](#voice-memos)
- [Meeting sources](#meeting-sources)
- [Speakers](#speakers)
- [Languages](#languages)
- [Glossary](#glossary)
- [Privacy and consent](#privacy-and-consent)
- [Automation](#automation)
- [Settings](#settings)
- [Common problems](#common-problems)

## Before you record people

Get everyone's consent before you record a conversation. In many places recording without it is illegal, and some require every participant's consent. ytverbatim does not check. See [Privacy and consent](#privacy-and-consent) for what leaves your Mac.

## Files and folders

```
ytverbatim run meeting.m4a standup.mp4
ytverbatim run ~/Recordings/2026-11/
```

A target is a file, a folder, or a [Voice Memos selector](#voice-memos). Files, folders and `memos:` selectors can be mixed in one run. YouTube targets cannot: run them separately. A bare name that is both a file and an 11-character video id is refused; write `./NAME` for the file or the watch URL for the video.

- **Folders** are searched recursively and sorted by path. Hidden files and folders are skipped, and so are ytverbatim's own `rec-…` folders and the output folder.
- **Extensions** read from a folder: `m4a`, `mp3`, `wav`, `aac`, `flac`, `ogg`, `opus`, `mp4`, `m4v`, `mov`, `mkv`, `webm`, `qta`, `caf`, `aif`, `aiff`, `amr` and `3gp`. A file you name yourself is read whatever its extension is, as long as ffmpeg can read it.
- **No audio stream** is an error for that file, never a silent skip. The other files still run.
- **Several audio tracks:** one is transcribed, the first AAC track, or else the first track ffmpeg can decode. The others are ignored. This is what makes Voice Memos' spatial `.qta` files work.
- **Title and date** come from the file's own tags: the `title` tag, or else the file name without its extension, and the recording date from the container's creation-date tags. Only a tag that carries a time zone counts; one without gives no date. The file's modification time is never used, because copying and syncing change it. A recording with no usable date has none: the header leaves it out and `index.md` lists it last.
- **The same audio twice** in one run (the same file under two names, or in two containers) is transcribed once; the second is reported as a duplicate.

A few options belong to YouTube, and a run with local targets refuses them: `--auth`, `--source`, `--captions/--no-captions` and `--ignore-cooldown`. The reverse holds for `--quiet`, which a YouTube run refuses. Everything else applies to recordings, including `--engine`, `--language`, `--diarize/--no-diarize`, `--glossary`, `--llm`, `--correct/--no-correct`, `--force` and `--out`.

Your file is only read. It is never copied, moved or changed, and nothing remembers where it lives. When a rerun needs the audio again, for example after you change the engine, pass the file again.

## The recording id

A recording's id is `rec-` plus 16 hex digits, for example `rec-0123456789abcdef`. They are the first 16 digits of the SHA-256 of the transcribed audio stream's encoded packets, not of the file. The full hash is kept in `context.json` as `source.audio_sha256`.

So the id follows the audio:

- Renaming the file, editing its tags, or remuxing it (for example `m4a` to `mp4` or `mov`) keeps the id.
- Re-encoding the audio gives a new id.
- A raw `.aac` file frames the same audio differently and gets its own id.

An id is never 11 characters, so it cannot be mistaken for a YouTube video id.

## Output

Recordings go to `~/.local/share/ytverbatim/recordings/`, or to `recordings/` under `$YTVERBATIM_HOME`, or to `--out DIR`.

```
recordings/
  index.md                     every recording, newest first
  batch_report.json            what the last run did, per file
  rec-0123456789abcdef/
    context.json               the contract (see the README)
    context.md                 the transcript as markdown
    speakers.json              only once you have named speakers
    .stages/                   stored results of each stage, for reruns
```

`index.md` is a table of recorded date, title, duration and speakers, newest first, each title linking to its `context.md`. It is rebuilt from the `context.json` files in the folder after each recording, so it is complete even when two runs overlap. It is the one file to give another tool.

`run` prints one line per recording:

```
✓ weekly-sync.m4a  ~/.local/share/ytverbatim/recordings/rec-0123456789abcdef/context.md  (412 segments · 96 paragraphs · 3 speakers · 2 flags · punctuation: llm · 214s)
```

Exit codes: 0 when every source finished, 1 when some failed. The codes 2 and 3 are YouTube's, for an auth error and a cooldown.

`context.md` of a recording, with synthetic content:

```markdown
# Weekly sync

Recorded 2026-11-03 14:05 +08:00 · 47:12 · Voice Memos · Speakers: Speaker 1 (48%) · Speaker 2 (37%) · Speaker 3 (15%)

Transcript: w-runs-lid · languages: zh, en · punctuation: llm · corrected by: chatgpt:gpt-6.1-sol · 1 flag

## Transcript

[00:00] **Speaker 1:** 好，我们开始吧。First item is the Q3 budget.

[00:09] **Speaker 2:** 我这边预算还是 120 万，没有变。[^1]

[00:15] **Speaker 1:** OK, so we keep it flat.

[^1]: 00:09 — low_confidence: engine confidence 0.31
```

- Timestamps are plain `[mm:ss]`, or `[h:mm:ss]` once the recording passes an hour. There are no links, since there is no video to link to.
- Every paragraph starts with its time and, when several people talk, its speaker, so any paragraph still makes sense when you cut the file into pieces for an LLM. A long turn repeats the label.
- A new paragraph starts at every change of speaker.
- Footnotes mark doubts, and the changes that correction and number checks made, with the text as it was ("was X"). Overlapping speech is recorded in `context.json` only.
- The `Recorded` part is left out when the date is unknown. The origin is `Voice Memos`, or else the file name.
- There are no chapters for recordings.

Numbers in a recording rest on the transcript alone, because a recording has no captions to check them against, unless you set `[text] numeric_second_opinion` (see [Settings](#settings)). A notice says so, and words the engine was unsure of are flagged in the footnotes.

## Reruns, skips and `--force`

Every stage is stored under the recording's `.stages/` folder, keyed by what it read. Then:

- A recording whose `context.json` is current is skipped. `run` says so (`already done`).
- A rerun with other options redoes only the stages those options reach. Changing the glossary or the LLM redoes the text stages but not the speech recognition.
- An interrupted run resumes where it stopped.
- `--force` throws away every stored stage and recomputes all of them, including the LLM calls you pay for.
- The 16 kHz WAV that ASR works from is deleted as soon as the transcript and the speaker turns are stored. If a run fails or stops, it is deleted too.
- A recording another run is transcribing right now is skipped, not failed: it counts as busy and the exit code stays 0.

A plain file keeps the title and date from its first run. A Voice Memo is different. If you rename it in Voice Memos, or if you first transcribed its shared file as a plain file, the next `memos:` run that selects it updates its record to the Voice Memos title and date from the stored stages, with no ASR and no LLM call.

`ytverbatim render CONTEXT.json` rewrites `context.md` beside a `context.json`, from the stored document alone, and touches nothing else. `--stdout` prints the markdown instead of writing it.

## Voice Memos

Memos you record on the iPhone sync to the Mac's Voice Memos app through iCloud. ytverbatim reads them from there.

### Set up

1. Turn on Voice Memos in iCloud settings on the iPhone and on the Mac.
2. Open Voice Memos on the Mac once, so it creates the library.
3. Give the app you run ytverbatim from Full Disk Access: System Settings → Privacy & Security → Full Disk Access, then restart that app. For a scheduled job, see [Automation](#automation).

`ytverbatim status` has a `Voice Memos` line that tells you where you stand: `ok · N memos`, `no access (Full Disk Access)`, `not syncing`, or `unreadable` with the reason. It is only information and never changes the exit code.

ytverbatim reads only the library location macOS uses since Sonoma, `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings`. Apple does not document that layout, so a future macOS may change it. Without Full Disk Access, share or export the memo from the Voice Memos app as a file and pass the file.

### List memos

```
ytverbatim memos [--all] [--json] [--out DIR]
```

Lists the newest 20 memos, or every memo with `--all`, one line each: date, duration, title, status and a short id (the first 8 characters). `--json` prints every memo as one JSON array with `unique_id`, `title`, `recorded_at`, `duration_s`, `status` and `recording_id`. `--out` is the recordings folder to look in for finished transcripts.

| Status | Meaning |
|---|---|
| `transcribed` | A document for this memo exists in the output folder. |
| `new` | The memo is on this Mac and has no document yet. |
| `not downloaded` | iCloud has not downloaded the audio. Open the memo once in Voice Memos. |
| `unsupported` | ytverbatim cannot read it: a `.composition` folder, or a file outside the library. |

Listing never hashes audio. It reads a short-lived private copy of the Voice Memos database, so nothing is written into Apple's folder, and it remembers each memo's id in `voicememos.json` in ytverbatim's state folder. A memo ytverbatim has never hashed shows as `new`.

### Transcribe memos

```
ytverbatim run memos:new
ytverbatim run memos:latest
ytverbatim run memos:all
ytverbatim run memos:3f9a2c1d
```

| Selector | Transcribes |
|---|---|
| `memos:new` | Every downloaded memo that has no document yet. It may select nothing. |
| `memos:latest` | The newest downloaded memo. |
| `memos:all` | Every downloaded memo. |
| `memos:ID` | The one memo whose id starts with `ID`, as `memos` lists it. An unknown prefix is refused, with a pointer to `memos --all`. A prefix that fits several memos is refused with the longer prefixes that tell them apart. |

Selectors are case-insensitive. Give several and each memo runs once, newest first. Every `run` option applies.

- `memos:new` means "no document yet", so changing options never makes it redo old memos. It hashes only the memos it has not seen before, and remembers the rest. To redo one, name it and force it: `memos:ID --force`, or `memos:all --force` for every memo. `memos:new --force` redoes nothing, because a memo that has a document is not new.
- The first `memos:new` transcribes every memo you already have. Run it by hand once before you schedule it.
- `new`, `all` and `latest` skip memos that are not downloaded or unsupported, with a notice, and never hash them. Open a memo once in Voice Memos, then run again. `latest` skips newer memos that are not downloaded and takes the newest downloaded one.
- `memos:ID` on a memo that is not downloaded or unsupported is an error. So are `all` and `latest` when no memo is downloaded.
- The memo's title and recording date from Voice Memos become the document's `title` and `recorded_at`, ahead of the file's own tags, and `source.origin` is `voice_memos`.
- If Voice Memos is not syncing, or ytverbatim cannot read its library, the error says how to fix it.
- `--quiet` prints nothing when there was nothing to do. See [Automation](#automation).

## Meeting sources

ytverbatim reads files. Any audio or video file that ffmpeg can read works, so the task is to get your meeting into a file.

- **A recording your meeting app saves on your Mac:** pass the file or its folder. If a meeting's folder holds a video file and an audio-only file of the same meeting, pass the audio file. When both carry the same audio, a folder run transcribes it once, because the id follows the audio. When they do not, it is transcribed twice.
- **A recording kept in your meeting app's cloud:** download it from the app and pass the file. Whether you can download one depends on the app and your plan.
- **Your phone:** record in Voice Memos and use the [Voice Memos](#voice-memos) selectors, or export or share the recording from whatever app made it and pass the file.
- **A call you record on your Mac yourself:** a recording made with a microphone only captures what the microphone hears, so the other people may be faint or missing. Use a tool that records the call's audio too.

Not supported yet: separate tracks for each participant, and importing the transcripts that meeting apps produce, with their speaker names. Rename speakers yourself with [`ytverbatim speakers`](#speakers).

## Speakers

Recordings are diarized by default: a local model finds who speaks when. The model is Nemotron-3-Diarization (8-bit, through mlx-audio, about 107 MB). It downloads once into the Hugging Face cache on first use. It needs the `asr` extra.

- Each word goes to the speaker most active over it. Very short runs between two stretches of the same speaker are smoothed into it.
- The model tells apart up to 8 speakers. It does not take a speaker count.
- Sentences and paragraphs break at every change of speaker.
- Speakers are numbered `S1`, `S2`, … in order of first appearance, and `context.md` calls them `Speaker 1`, `Speaker 2`, … with each one's share of the talk time in the header. The ids are never renumbered, so after a later change one can be missing, or a higher id can speak first.
- A recording with one speaker records it in `context.json` but shows no labels.
- Speech where a second speaker talks at the same time is flagged `overlapping_speech` in `context.json`. The markdown does not show it.

`[diarize] mode` decides which sources are diarized:

| Mode | Diarizes |
|---|---|
| `"auto"` (default) | Recordings only. |
| `"on"` | Every source transcribed with ASR, YouTube videos too. |
| `"off"` | Nothing. |

`--diarize` and `--no-diarize` set it for one run. A diarized run needs `ytverbatim[asr]`; recordings with the default engine need `ytverbatim[whisper,asr]`.

### Naming speakers

```
ytverbatim speakers rec-0123456789abcdef
ytverbatim speakers rec-0123456789abcdef S1=Alice S2=Bob
ytverbatim speakers rec-0123456789abcdef S2=
ytverbatim speakers rec-0123456789abcdef S1=Alice S3=Alice
```

`TARGET` is the recording's folder, its `context.json`, or its `rec-…` id. `--out DIR` says where to look an id up.

- **List:** without assignments it prints one line per speaker: id, name (`-` when unnamed), talk time and the speaker's first line with its timestamp. Use it to hear who is who.
- **Name:** `S1=Alice S2=Bob`. Ids are not case-sensitive, and a name is one line of plain text. A name cannot be another speaker's default label (`Speaker 2`).
- **Clear:** `S2=` removes a name.
- **Merge:** give two ids one name (`S1=Alice S3=Alice`) when the model split one person in two. They become one person: one label, their talk time added up in the header, and consecutive paragraphs joined.

Naming rewrites `context.json`, `context.md` and `index.md`. No ASR worker and no LLM runs, so it takes a moment.

The names are saved in `speakers.json` beside `context.json`. They belong to the speaker attribution they were given for:

- A rerun of the text stages (a new glossary, another LLM) keeps them.
- A new diarization, or any change to how words were attributed or recognised, drops them, with a warning in `quality.warnings` and a note when you list speakers. You set them again. A speaker is never given the wrong name.

Speaker names are never sent to an LLM. The text stages only see `S1`, `S2`, …, and only the final document is built with names.

## Languages

- **`--language CODE`** forces one spoken language for any source, a recording or a YouTube video: `zh`, `en` and the other Whisper language codes. A region suffix such as `zh-CN` is read as `zh`. An unknown code is an error. Force a language only when the whole file is in it: Whisper is known to translate speech in another language into the forced one, instead of transcribing it.
- **`[asr] languages`** is the set of languages an engine that detects languages may choose from, for recordings only. `languages = ["zh", "en"]` limits detection to those two. The default, an empty list, allows any.
- **`--engine ID`** (or `[asr] engine`) picks the engine for every source. The ids are listed below.

The default engine for recordings is `w-prod`, the same Whisper large-v3 as for YouTube. It decodes the whole file at once and detects one language for it. So with the defaults a recording has one language, and `[asr] languages` has no effect.

| Engine | What it is | Needs |
|---|---|---|
| `w-prod` | Whisper large-v3, the whole file at once. The default. | `whisper` |
| `w-vad` | Whisper large-v3 in 30-second chunks; Whisper detects each chunk's language unless you force one. | `whisper`, `asr` |
| `w-vad-lid` | Whisper large-v3 in chunks, each decoded with its own detected language. | `whisper`, `asr` |
| `w-runs-lid` | Whisper large-v3, neighbouring same-language chunks decoded together. | `whisper`, `asr` |
| `q-base`, `q-ctx` | Qwen3-ASR 1.7B (8-bit), without and with context. | `asr` |
| `f-aed` | FireRedASR2 (AED). | `asr` |
| `m-4bit` | MiMo-V2.5-ASR (4-bit). | `asr` |

The [engine evaluation](../evals/results/asr-2026-11-13.md) compared `w-prod`, `w-vad`, `q-base`, `q-ctx`, `f-aed` and `m-4bit` on YouTube videos and kept `w-prod` as the default; `w-vad-lid` and `w-runs-lid` are still being evaluated on meeting and mixed-language audio. `[asr] languages` applies only to `w-vad-lid` and `w-runs-lid`. Diarizing needs `asr` as well, so a diarized recording with the default engine needs both extras.

Per-stretch languages need an engine that detects the language of each stretch:

```toml
[asr]
engine = "w-runs-lid"
languages = ["zh", "en"]
```

- `w-runs-lid` detects the language of each chunk, merges neighbouring chunks of the same language into runs, and decodes each run whole with its own language.
- `w-vad-lid` decodes each chunk with its own detected language.

Both need `ytverbatim[whisper,asr]`.

When a recording has stretches in more than one language, each stretch is treated by its own language:

- Simplified-Chinese conversion and Chinese fillers apply to the Chinese stretches.
- Punctuation follows each language's own style.
- [Chinese numerals as digits](../README.md#numerals) apply to the Chinese stretches.
- The correction and number-check prompts say the transcript mixes languages and that every word stays in the language it was spoken in.

`transcript.languages` in `context.json` lists the languages the transcript was decoded in: the document's own first, then the others by how much speech they cover. For recordings the markdown header shows them too. YouTube videos read their text by the document's language alone.

## Glossary

Names and jargon that Whisper mishears can be listed so that the correction step prefers them:

```toml
[text]
glossary = ["Acme", "Kubernetes"]
```

or `--glossary "Acme, Kubernetes"` for one run, which replaces the configured list. At most 20 terms; more is an error that asks you to shorten the list. Blank and repeated terms are dropped.

A recording has no tags, so the glossary is its only steer. For a YouTube video the configured terms come first and the video's tags follow, 20 terms in all. The terms are sent to the LLM.

## Privacy and consent

Recording other people needs their consent, and local law may require it. Anyone you record should know that the audio is transcribed and that the text goes to an LLM provider.

**Stays on your Mac:**

- The audio, your files and your Voice Memos library. They are read in place, never copied or uploaded.
- Speech recognition and diarization.
- The titles, file names, recording dates and speaker names you see in the output. The transcript files contain them, so check before you share a `context.json` or `context.md`.

**Sent to the LLM provider:**

- The transcript text, in batches, for punctuation, correction and number checks.
- For a diarized recording, the speaker ids (`S1`, `S2`, …), and a line saying the text is a conversation.
- Your glossary terms.

**Never sent:** the title, file name, recording date and speaker names.

The provider is whichever the LLM setting names. By default that is ChatGPT through your own sign-in, so the text goes to OpenAI. `claude-cli` sends it to Anthropic through the `claude` command. `openai-compat` sends it to the endpoint you set, which can be a model on your own Mac.

Models are downloaded from Hugging Face the first time they are used. Only the download goes over the network, not your audio or text.

**To opt out of sending any text,** use `--llm none` for a run or set it for good:

```toml
[llm]
provider = "none"
```

Without an LLM, punctuation is left as the engine wrote it, correction is skipped, and numbers that disagree stay flagged with every reading. A notice and `quality.warnings` say so. `--no-correct` alone is not an opt-out, because punctuation still uses the LLM. A local `openai-compat` endpoint also keeps everything on your Mac.

With the default ChatGPT provider, check your ChatGPT plan's data controls, the setting about whether conversations may be used to improve models, before you transcribe business meetings. Whether that setting applies to the sign-in ytverbatim uses is unverified.

## Automation

`run memos:new --quiet` is meant for a scheduled job. It transcribes the memos that are new, and prints nothing when there is nothing to do, so cron sends no mail. Errors and finished work still print.

What the job needs:

- **`ytverbatim login` once,** by hand in a terminal. A job has no terminal, so it never opens a browser. If the ChatGPT sign-in is missing or has lapsed, the run stops before it does any work, prints the fix and exits 1. Check with `ytverbatim status`.
- **Full Disk Access for the program the job runs,** not for your terminal. For cron that is `cron` itself. For a launchd agent it is the executable in `ProgramArguments`. This attribution is unverified: if the job reports that it cannot read the Voice Memos library, add the program the error names or the one you scheduled, then try again.
- **A `PATH` that finds `ffmpeg`.** Scheduled jobs start with a short `PATH`. The example uses the Homebrew folder of an Apple Silicon Mac; use the folder `command -v ffmpeg` prints.
- **The full path to `ytverbatim`.** `command -v ytverbatim` prints it from your shell. After `uv tool install` it is `~/.local/bin/ytverbatim` by default, and `uv tool dir --bin` shows the folder. Use that folder in `PATH` too, if the job needs it.
- **A backlog cleared by hand.** The first `memos:new` transcribes every memo you have, which can take hours. Run it once yourself first.
- **iCloud downloads.** Memos that are not downloaded yet are skipped and picked up by a later run once Voice Memos has downloaded them.

Runs may overlap. A memo that another run is transcribing is skipped as busy, and the exit code stays 0. ASR is heavy, so do not schedule jobs close together.

### cron

`crontab -e`, every 30 minutes:

```
PATH=/opt/homebrew/bin:/usr/bin:/bin
*/30 * * * * /full/path/to/ytverbatim run memos:new --quiet
```

### launchd

Save as `~/Library/LaunchAgents/com.example.ytverbatim-memos.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.example.ytverbatim-memos</string>
  <key>ProgramArguments</key>
  <array>
    <string>/full/path/to/ytverbatim</string>
    <string>run</string>
    <string>memos:new</string>
    <string>--quiet</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  <key>StartInterval</key>
  <integer>1800</integer>
  <key>StandardErrorPath</key>
  <string>/tmp/ytverbatim-memos.log</string>
</dict>
</plist>
```

Then load it:

```
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.example.ytverbatim-memos.plist
```

To stop it, run `launchctl bootout gui/$(id -u)/com.example.ytverbatim-memos`.

## Settings

Set these in `~/.config/ytverbatim/config.toml`. A command-line option wins for one run.

| Key | Default | What it does |
|---|---|---|
| `[diarize] mode` | `"auto"` | `auto`, `on` or `off`. `--diarize/--no-diarize`. |
| `[asr] engine` | none | The engine for every source. Without it, `w-prod`. `--engine`. |
| `[asr] language` | none | Force one language for every source. `--language`. |
| `[asr] languages` | `[]` | The languages a detecting engine may choose from, for recordings. |
| `[asr] memory_cap_gib` | half of physical memory | The workers are stopped above this footprint. |
| `[text] glossary` | `[]` | Terms for correction to prefer, at most 20. `--glossary`. |
| `[text] numeric_second_opinion` | `"off"` | An engine id whose reading of the numbers is checked against the first. It must decode chunk by chunk and re-decode single spans, such as `w-vad`; `w-prod` and `w-runs-lid` are refused. |
| `[text] correct` | on with an LLM | `--correct/--no-correct`. |
| `[text] script` | `"zh-Hans"` | `"keep"` leaves the script as recognised. |
| `[text] fillers` | `"keep"` | `"remove"` drops fillers. |
| `[text] numerals` | `"digits"` | `"keep"` leaves Chinese numerals as recognised. |
| `[llm] provider`, `model`, `effort` | `chatgpt` | `--llm PROVIDER:MODEL`, or `--llm none`. |

`YTVERBATIM_HOME` moves the output root (default `~/.local/share/ytverbatim`; recordings live in its `recordings/` folder and videos in `videos/`).

## Common problems

| You see | What to do |
|---|---|
| `Voice Memos is not syncing to this Mac` | Turn on Voice Memos in iCloud settings on the iPhone and the Mac, then open Voice Memos once. |
| `cannot read the Voice Memos library` | Give the app you run ytverbatim from Full Disk Access, then restart it. |
| Memos listed as `not downloaded` | Open the memo once in Voice Memos. |
| `has no audio stream` | The file has no sound track. Nothing to transcribe. |
| `ffmpeg cannot decode any audio stream` | Convert the file to `m4a` or `wav` with ffmpeg and pass that. |
| `install ytverbatim[…]` | Install the extra it names. Recordings need `whisper` and `asr`. |
| `numbers not cross-checked: recordings have no captions` | Expected for recordings. Check the flagged numbers in the footnotes yourself. |
| Speaker names disappeared | You re-diarized, or something changed how words were attributed. Set them again with `ytverbatim speakers`. |
| `is no longer at the path it was read from` | The audio is needed again. Pass the file again. |
