# Checking correction against manual captions (CCE)

Run 2026-11-17. Aggregates only; files appear as generic labels, and no transcript, caption, reference text, name or
video id appears. The decision rule in [Decision](#decision) was fixed before any case was labelled and is applied
as written. Rule C ships without its caption exemption and with refinements made after the evaluation; [What
ships](#what-ships) measures both.

## What was tested

Three deterministic rules, each built on the manual caption track in the transcript's language: the uploader's
captions, not YouTube's speech recognition.

| rule | what it does |
|---|---|
| A | A correction may not change a run of at least 2 words or characters on which the manual caption agrees with the original. The change is vetoed, and any other change in the same proposal is lost with it. |
| B | In a Chinese segment, a Latin-script word that the manual caption reads differently, with agreed text on both sides, is replaced by the caption's reading. |
| C | Every Latin-script name the LLM changed is flagged `entity_corrected` with the note `before → after`, as tickers and acronyms already were. A change on which the manual caption agrees with the final text is not flagged. |

The rules were replayed over stored correction runs through the production `correct_segments`, with each run's
stored proposals standing in for the model. No model was asked again.

## Data

| files | language | hours | manual track | candidates replayed |
|---|---|---|---|---|
| en-01 to en-03 | en | 0.97 | none (auto only) | c-haiku, c-sonnet, l-qwen35-9b, x-gpt61-high, x-gpt61-medium |
| zh-01 to zh-04, zh-06, zh-07 | zh | 3.51 | none | the same five |
| zh-05 | zh | 0.23 | yes | the same five |
| zh-08 to zh-11 | zh | 1.46 | yes | o-gpt61-medium |

- The first ten files and their five candidates are the ones in [correction-2026-11-13.md](correction-2026-11-13.md).
- zh-08 to zh-11 are public Mandarin videos with manual captions, added for this evaluation. Their bases were built by
  the same `base` command. They were corrected only by `o-gpt61-medium`, the default correction model (`gpt-6.1-sol` at
  medium effort through the native `chatgpt` provider).
- Manual-caption audio is 1.69 h in total: 0.23 h for each of the five older candidates and 1.46 h for
  `o-gpt61-medium`.

## Cases and labels

A case is one spot: file, segment, original reading and proposed reading, merged across candidates. Cases come from
the 8 files whose captions were fetched. Auto captions never decide, so the vetoes they would have made
(`auto_veto`) are reported for information and were not labelled.

| kind | rule | cases | answers |
|---|---|---|---|
| veto | A | 24 | proposal right 19, original right 2, both fine 1, both wrong 2 |
| lost co-edit | A | 4 | proposal right 3, original right 1 |
| caption reading | B | 31 | caption reading right 25, original right 6 |
| name flag | C | 13 | name or term 13 |
| auto veto | (A, information) | 19 | not labelled |

The owner set two standing policies:

- A Latin-script name takes the real spelling of the entity meant, ignoring case.
- Chinese takes standard usage.

The evaluator applied the policies by reading each case in the context of its neighbouring segments: 24 of the 28
Rule A cases and 20 of the 31 Rule B cases. The other 4 Rule A and 11 Rule B cases, and all 13 Rule C pairs, went to
the owner as questions, each with a link to its moment in the video.

## Rates per candidate

Vetoes, lost co-edits and Rule B replacements are per hour of manual-caption audio. Name flags are per hour of all the
audio the candidate ran on, with Rules A and B applied and Rule C's caption exemption on.

| candidate | manual h | vetoes (/h) | lost co-edits (/h) | replacements (/h) | auto vetoes | all h | ticker flags (/h) | ticker and name flags (/h) |
|---|---|---|---|---|---|---|---|---|
| c-haiku | 0.23 | 4 (17.64) | 1 (4.41) | 6 (26.45) | 4 | 4.71 | 15 (3.19) | 36 (7.65) |
| c-sonnet | 0.23 | 3 (13.23) | 0 (0.00) | 4 (17.64) | 9 | 4.71 | 16 (3.40) | 40 (8.50) |
| l-qwen35-9b | 0.23 | 1 (4.41) | 1 (4.41) | 8 (35.27) | 6 | 4.71 | 8 (1.70) | 24 (5.10) |
| o-gpt61-medium | 1.46 | 17 (11.61) | 2 (1.37) | 20 (13.66) | 0 | 1.46 | 0 (0.00) | 3 (2.05) |
| x-gpt61-high | 0.23 | 2 (8.82) | 0 (0.00) | 2 (8.82) | 5 | 4.71 | 16 (3.40) | 46 (9.77) |
| x-gpt61-medium | 0.23 | 2 (8.82) | 0 (0.00) | 2 (8.82) | 4 | 4.71 | 17 (3.61) | 46 (9.77) |

## Decision

| rule | pre-registered check | result |
|---|---|---|
| A | (a) at least 10 distinct vetoes, all decided | pass: 24, none undecided |
| A | (b) at most half of the vetoes block a right proposal | **fail**: 19 of 24 (0.792) |
| A | (c) `o-gpt61-medium` loses at most 2 right edits per hour of manual-caption audio | **fail**: 17 right edits lost (15 vetoed, 2 co-vetoed) over 1.46 h = 11.61/h |
| B | every replacement decided, none judged wrong | **fail**: 6 of 31 wrong |
| C | at least 90% of distinct new flag pairs are names or terms | pass: 13 of 13 |
| all | number violations do not rise | pass: 0 with the rules, 0 without, over 54 replays |
| all | a file without a manual caption differs only by Rule C flags | pass: 0 of 45 replays differ (2 segments left out, see [Limits](#limits)) |

**Rule A stays off. Rule B stays off. Rule C ships.** A failed (c) reopens deciding a veto hunk by hunk rather than
for the whole proposal, but that would not rescue Rule A, since (b) fails on its own. Of `o-gpt61-medium`'s 17
vetoes, the 2 not counted as lost had both readings fine (1) or neither right (1).

Rules A and B fail for one reason. In 19 of the 24 vetoes, the manual caption held the same reading as the
recognizer at a spot where the LLM's fix was right. And 6 of the 31 readings Rule B took from the captions were wrong.
These uploaders' manual tracks repeat the recognizer's homophone errors, so on this data they are not independent
evidence. The plan had expected that of auto captions only.

## What ships

Rule C ships without its caption exemption. The exemption rests on the same manual captions, and 6 of the 31 readings
Rule B took from them were wrong. An exemption could therefore hide the flag exactly where a reader needs it. Dropping
it also keeps manual captions out of correction altogether. The shipped configuration was measured by replaying the
same stored runs with no caption evidence, so with no Rule A, no Rule B and no exemption.

**Refinements.** After the evaluation, a review of the rule led to five changes:

- A note's `before` is the original text in the name's place, from a character alignment of the two texts, where the
  evaluated rule took the original Latin words. A deleted letter at the name's edge belongs to it; a deleted word
  beside it does not.
- A change of spaces, apostrophes, ampersands, periods or hyphens alone is format-only, like a change of case.
- A word that a ticker change overlaps no longer suppresses a name changed beside it.
- Accented Latin letters belong to the word.
- In English, a capitalized word after a title such as Dr. counts as a name.

Over all 14 files and the six candidates, the shipped rule raises 211 name flags against the evaluated rule's 221:

- None is added.
- 10 are dropped as format-only. In 8, the LLM only put punctuation between a name's letters; in 2, it only split a
  number from the ticker after it.
- 9 notes changed. In 8, the name had replaced non-Latin characters, and the note had named nothing (6), the wrong
  word (1) or only the Latin part (1); each now names those characters. In 1, the note no longer includes words the
  LLM dropped after the name.

**Gate C, as shipped.** On the 8 files with captions there are 76 flag cases instead of 13.

- Of the owner's 13 cases, 10 are still raised with the readings the owner labelled. 2 were re-punctuations and are
  format-only now, and 1 merged with another case once its note stopped including the dropped words.
- The evaluator labelled the 66 new ones: 62 are names or terms, and 4 are ordinary English words in a Chinese
  segment.
- 72 of 76 (0.947) are names or terms, against the 0.9 limit: pass.

**Volume, as shipped.** `entity_corrected` flags per hour of all the audio each candidate ran on:

| candidate | all h | ticker flags (/h) | name flags (/h) | all flags (/h) |
|---|---|---|---|---|
| c-haiku | 4.71 | 15 (3.19) | 28 (5.95) | 43 (9.14) |
| c-sonnet | 4.71 | 16 (3.40) | 29 (6.16) | 45 (9.56) |
| l-qwen35-9b | 4.71 | 8 (1.70) | 17 (3.61) | 25 (5.31) |
| o-gpt61-medium | 1.46 | 0 (0.00) | 64 (43.72) | 64 (43.72) |
| x-gpt61-high | 4.71 | 16 (3.40) | 37 (7.86) | 53 (11.26) |
| x-gpt61-medium | 4.71 | 17 (3.61) | 36 (7.65) | 53 (11.26) |

`o-gpt61-medium`'s four files are talks dense with Latin-script names, and most of its flags are corrections of
those names. Flags change no text; each is a footnote that shows what correction replaced.

## Also fixed

A Chinese caption track without word timings joined two cues that met in Latin script into one word. A name ending one
cue and starting the next read as the name written twice with no space, which gave Rule B wrong readings. Cues are now
word boundaries. The fix ships too, because numeric verification reads caption tracks the same way.

## Limits

- The rules were replayed over stored proposals; no model was asked again. A second-pass proposal was made against
  first-pass text that a rule may have changed.
- 2 segments were left out of the caption-less check. In each, a guard added after the run was stored now refuses an
  edit the run had applied.
- The owner decided 28 of the 72 labelled cases as questions. The evaluator decided the rest under the owner's two
  policies, and alone the 66 cases the shipped configuration adds.
- The refinements were measured on the same runs they were reviewed against, not on new data.
- Five files have manual captions. The finding about their independence is about these uploaders' tracks, not
  captions in general, such as those made by a professional captioner.

## How it was produced

From a worktree of the corroboration branch, which is not merged: the rules it rejected and the eval command stayed
on it.

```bash
export ANYVERBATIM_EVAL_DIR=~/.local/share/anyverbatim/eval
uv run python -m evals.run_asr asr --hyp w-prod --files ID,ID,ID,ID            # the four added files
uv run python -m evals.run_correction base --hyp w-prod --files ID,ID,ID,ID
uv run python -m evals.run_correction correct --candidate o-gpt61-medium --files ID,ID,ID,ID
uv run python -m evals.run_correction corroboration cases     # replay every stored run, write the cases
uv run python -m evals.run_correction corroboration show      # per rule: link to the moment, context, both readings
uv run python -m evals.run_correction corroboration label --case ID --answer proposal --method policy
uv run python -m evals.run_correction corroboration report
```

The shipped configuration was measured by replaying every stored run with no caption evidence
(`corroboration.replay(base, run, None)`) and flagging the replayed text with this release's `name_changes` and
`symbol_changes`.
