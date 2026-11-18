"""Give every ASR word a speaker from a stored diarization, then renumber speakers by first appearance.

Rules (plan §2.5):
- A word goes to the channel with the largest summed probability over its frames, among the channels active (above
  the diarizer's threshold) in at least one of them.
- A word with no active channel takes the speaker of the nearest active frame within `GAP_S`, else the previous
  word's speaker; leading words with neither take the first attributed word's speaker.
- A word is overlapped when one other channel is active together with the chosen one over at least half of its
  frames.
- Smoothing: a run of same-speaker words is short when it lasts under `SMOOTH_S` or has fewer than `SMOOTH_WORDS`
  words, and protected when one of its words is overlapped. A short, unprotected run takes its neighbours' speaker
  when both neighbouring runs share one and neither is itself short and unprotected. Judging neighbours too keeps a
  string of alternating blips from flipping into new blips, and makes a second pass change nothing.
- Overlap spans merge consecutive overlapped words less than `GAP_S` apart.
"""

import math
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from anyverbatim.contract.models import Word
from anyverbatim.diarize.protocol import Diarization

# Part of every key that depends on what attribution decides: bump it on any change to the rules below.
VERSION = "attribution-v1"
GAP_S = 0.5
OVERLAP_SHARE = 0.5
SMOOTH_S = 0.5
SMOOTH_WORDS = 2
# What the pipeline runs; evals compare both settings.
SMOOTHING = True
_FRAME_DIGITS = 6


class Span(NamedTuple):
    start: float
    end: float


class Attribution(NamedTuple):
    """`speakers[i]` is word i's speaker (`S1`, `S2`, … by first appearance), None only when nobody ever speaks.

    `overlapped[i]` says whether another speaker talks over word i; `overlaps` are the merged spans of consecutive
    overlapped words.
    """

    speakers: tuple[str | None, ...]
    overlaps: tuple[Span, ...]
    overlapped: tuple[bool, ...]


class _Scored(NamedTuple):
    channel: int | None
    overlapped: bool


def assign(words: Sequence[Word], diarization: Diarization, *, smoothing: bool = SMOOTHING) -> Attribution:
    scored = [_score(word, diarization) for word in words]
    channels = _fill_gaps(words, [s.channel for s in scored], diarization)
    overlapped = [s.overlapped for s in scored]
    if smoothing:
        channels = _smooth(words, channels, overlapped)
    return Attribution(_renumber(channels), _overlap_spans(words, overlapped), tuple(overlapped))


def speech_seconds(segments: Iterable[tuple[float, float, str | None]]) -> dict[str, float]:
    """Talk time per speaker in first-appearance order: the sum of `end - start` over their segments."""
    totals: dict[str, float] = {}
    for start, end, speaker in segments:
        if speaker is not None:
            totals[speaker] = totals.get(speaker, 0.0) + (end - start)
    return totals


def _frames(start: float, end: float, d: Diarization) -> range:
    """Frames overlapping [start, end); a zero-length word gets the frame it falls in. Clipped to the stored frames."""
    # Rounded first: 0.29 / 0.01 is 28.999..., which would add a frame that ends where the word starts.
    first = max(0, math.floor(round(start / d.frame_s, _FRAME_DIGITS)))
    last = max(first + 1, math.ceil(round(end / d.frame_s, _FRAME_DIGITS)))
    return range(min(first, d.frames), min(last, d.frames))


def _active(d: Diarization, frame: int) -> list[int]:
    return [c for c in range(d.channels) if d.prob(frame, c) > d.threshold]


def _score(word: Word, d: Diarization) -> _Scored:
    frames = _frames(word.start, word.end, d)
    sums = [0.0] * d.channels
    active_frames = [0] * d.channels
    for frame in frames:
        for c in range(d.channels):
            p = d.prob(frame, c)
            sums[c] += p
            if p > d.threshold:
                active_frames[c] += 1
    candidates = [c for c in range(d.channels) if active_frames[c]]
    if not candidates:
        return _Scored(None, False)
    best = max(candidates, key=lambda c: (sums[c], -c))
    together = [0] * d.channels
    for frame in frames:
        active = _active(d, frame)
        if best in active:
            for c in active:
                together[c] += 1
    together[best] = 0
    return _Scored(best, max(together) >= OVERLAP_SHARE * len(frames))


def _nearest(word: Word, d: Diarization) -> int | None:
    """Return the most probable active channel at the active frame nearest the word, if within `GAP_S`."""
    reach = math.ceil(GAP_S / d.frame_s)
    span = _frames(word.start, word.end, d)
    for step in range(1, reach + 1):
        for frame in (span.start - step, span.stop - 1 + step):
            if 0 <= frame < d.frames and _distance(word, frame, d) <= GAP_S:
                active = _active(d, frame)
                if active:
                    return max(active, key=lambda c: (d.prob(frame, c), -c))
    return None


def _distance(word: Word, frame: int, d: Diarization) -> float:
    frame_start, frame_end = frame * d.frame_s, (frame + 1) * d.frame_s
    return max(0.0, frame_start - word.end, word.start - frame_end)


def _fill_gaps(words: Sequence[Word], channels: list[int | None], d: Diarization) -> list[int | None]:
    filled: list[int | None] = []
    previous: int | None = None
    for word, channel in zip(words, channels, strict=True):
        chosen = channel if channel is not None else _nearest(word, d)
        if chosen is None:
            chosen = previous
        filled.append(chosen)
        if chosen is not None:
            previous = chosen
    first = next((c for c in filled if c is not None), None)
    return [first if c is None else c for c in filled]


def _runs(channels: Sequence[int | None]) -> list[range]:
    runs: list[range] = []
    start = 0
    for index in range(1, len(channels) + 1):
        if index == len(channels) or channels[index] != channels[start]:
            runs.append(range(start, index))
            start = index
    return runs


def _smooth(words: Sequence[Word], channels: list[int | None], overlapped: Sequence[bool]) -> list[int | None]:
    # Decided on the unsmoothed runs, so the outcome does not depend on the order runs are visited in.
    runs = _runs(channels)
    movable = [_movable(words, run, overlapped) for run in runs]
    smoothed = list(channels)
    for index in range(1, len(runs) - 1):
        before, run, after = runs[index - 1], runs[index], runs[index + 1]
        neighbour = channels[before.start]
        if not movable[index] or movable[index - 1] or movable[index + 1] or neighbour != channels[after.start]:
            continue
        for i in run:
            smoothed[i] = neighbour
    return smoothed


def _movable(words: Sequence[Word], run: range, overlapped: Sequence[bool]) -> bool:
    if any(overlapped[i] for i in run):
        return False
    duration = words[run.stop - 1].end - words[run.start].start
    return duration < SMOOTH_S or len(run) < SMOOTH_WORDS


def _renumber(channels: Sequence[int | None]) -> tuple[str | None, ...]:
    labels: dict[int, str] = {}
    for channel in channels:
        if channel is not None and channel not in labels:
            labels[channel] = f"S{len(labels) + 1}"
    return tuple(None if c is None else labels[c] for c in channels)


def _overlap_spans(words: Sequence[Word], overlapped: Sequence[bool]) -> tuple[Span, ...]:
    spans: list[Span] = []
    previous = False
    for word, flagged in zip(words, overlapped, strict=True):
        if flagged:
            if previous and word.start - spans[-1].end < GAP_S:
                spans[-1] = Span(spans[-1].start, max(spans[-1].end, word.end))
            else:
                spans.append(Span(word.start, word.end))
        previous = flagged
    return tuple(spans)
