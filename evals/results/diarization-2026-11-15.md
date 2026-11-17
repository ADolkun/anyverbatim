# Diarization eval (E1)

Run 2026-11-15. Aggregates only, on public corpora; no transcript or reference text. The owner's recordings are not in
yet, so the owner-evidence parts of the rule are pending and the decision rests on public data.

## Candidates

| id | what it is |
|---|---|
| `d-nemotron-8bit` | the incumbent: Nemotron speaker diarization, 8-bit MLX weights |
| `d-nemotron-bf16` | the same model in bf16 |
| pyannote community-1 | not run: its weights are gated behind a token (owner decision #9) |

## Data and metrics

- AMI SDM (6 meetings, 3.1 h), AliMeeting far-field channel 1 (8 meetings, 4.2 h), ASCEND long-form sessions (4, 1.2 h).
- DER is computed at collar 0 with overlap scored, and again at a 0.25 s collar. The primary metric is the macro DER
  over AMI and AliMeeting at collar 0.
- WDER is the share of eligible hypothesis words given the wrong speaker. It uses `w-runs-lid`'s words after the loop
  repair, with and without smoothing. Smoothing gives a run of fewer than 2 words, or shorter than 0.5 s, to the
  surrounding speaker when both neighbours are the same speaker and the run is not overlap-flagged.

## Results

| diarizer | corpus | DER collar 0 | DER collar 0.25 | WDER raw | WDER smoothed | speaker-count error | RTF | peak GiB |
|---|---|---|---|---|---|---|---|---|
| 8-bit | AMI | 27.22% | 25.00% | 0.539% | 0.587% | 0.17 | 0.0028 | 1.14 |
| 8-bit | AliMeeting | 17.22% | 9.85% | 0.189% | 0.216% | 0.00 | 0.0028 | 1.12 |
| 8-bit | ASCEND | 19.83% | 13.00% | 7.151% | 7.131% | 0.50 | 0.0029 | 0.92 |
| bf16 | AMI | 26.97% | 24.78% | 0.423% | 0.465% | 0.17 | 0.0026 | 1.23 |
| bf16 | AliMeeting | 17.20% | 9.92% | 0.183% | 0.216% | 0.00 | 0.0026 | 1.20 |
| bf16 | ASCEND | 19.68% | 12.90% | 7.131% | 7.111% | 0.75 | 0.0026 | 1.00 |

- **Primary macro DER:** 8-bit 21.50%, bf16 21.39%. Paired per-recording bootstrap, bf16 − 8-bit: −0.11 points,
  95% CI [−0.27, +0.03].
- ASCEND's high WDER is partly an artifact of how the sessions are rebuilt from separate utterances: stitched audio
  with silence where the other speaker's lines are missing.

## Smoothing

Paired per-recording bootstrap of smoothed − raw WDER, in points (8-bit):

| corpus | difference [95% CI] | speaker changes per hour, raw → smoothed | words relabelled |
|---|---|---|---|
| all 18 recordings | **+0.030 [+0.009, +0.051]** | — | — |
| AMI | **+0.048 [+0.019, +0.080]** | 474 → 350 (−26%) | 2.07% |
| AliMeeting | +0.027 [+0.000, +0.056] | 754 → 639 (−15%) | 0.56% |
| ASCEND | −0.020 [−0.057, +0.000] | 424 → 347 (−18%) | 0.65% |

With bf16, all 18 recordings give +0.031 [+0.007, +0.055]. Smoothing removes a sixth to a quarter of the speaker
changes, and the words it moves are slightly more often right before than after.

## Pre-registered rules

| rule | result | outcome |
|---|---|---|
| 1–3. A challenger replaces 8-bit only with macro DER lower by more than 2.0 points (CI excluding 0) and owner evidence that does not favour 8-bit | bf16: −0.11 [−0.27, +0.03]; pyannote not run | **8-bit stays** |
| 4. bf16 replaces 8-bit only if its DER is better by more than 1.0 point | −0.11 | 8-bit stays |
| 5. Smoothing ships unless smoothed WDER is worse than raw, with a paired CI excluding 0 | +0.030 [+0.009, +0.051] | the rule says no smoothing; **the owner kept it** (below) |

## Decision

- `d-nemotron-8bit` stays the diarizer.
- **Smoothing stays on, by the owner's decision (2026-11-16), against rule 5.** The rule had no minimum effect size,
  and the effect it found is about 3 words in 10,000 eligible words, mostly one-word backchannels. Smoothing is there
  for readability, which WDER does not measure: it removes 15–26% of the speaker changes. The owner's recordings
  recheck it when they arrive.
