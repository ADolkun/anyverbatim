"""The default quality guard: spot a failed decode and decode the failed part again once, in two halves.

A chunk is one contiguous slice of audio, so its replacement is a whole chunk too: the guard never splices a
re-decode into a chunk's original text, and there is no anchor to check.

A whole-call decode (a whole file, or a same-language run) is a sequence of the engine's own decode windows, and a
decoder loop stays inside the window it started in. When the whole call fails, each window that fails on its own is
decoded again like a chunk, without the prompt and in the call's language, and spliced back by the windows' word
counts. A window's edges are cuts the decoder already made, so the splice adds none. A window still failing after
that is reported as an anomaly span; a failure no single window explains flags the whole decode, as before.
"""

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate, pairwise
from typing import Literal

from anyverbatim.asr.audio import SAMPLE_RATE
from anyverbatim.asr.chunks import FRAME_S
from anyverbatim.asr.protocol import ChunkFlag, ChunkInput, Engine, GuardedChunk, RawChunk, Window, WindowDecoder
from anyverbatim.asr.quality import (
    COMPRESSION_RATIO_THRESHOLD,
    Unit,
    compression_ratio,
    cross_segment_repetitions,
    decoder_anomalies,
    has_lexical_content,
    is_nonlexical_failure,
    nonlexical_runs,
)
from anyverbatim.contract.models import Word
from anyverbatim.errors import DependencyError, EngineError

try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError as exc:
    raise DependencyError("install anyverbatim[asr]") from exc

Trigger = Literal[
    "token_cap",
    "repetition_loop",
    "compression_ratio",
    "empty_output",
    "nonlexical_output",
    "word_timing_collapse",
    "replacement_character",
]

# Below this a half would be under a second, too little for an engine to decode reliably, and a speech chunk this
# short with no text is more likely noise than a failed decode.
MIN_REDECODE_S = 2.0
# Whisper applies its compression threshold to one 30 s window; over a longer text, ordinary speech repeats enough
# to cross it, so a longer decode is measured per window of its words.
COMPRESSION_WINDOW_S = 30.0
_SPLIT_FROM = 0.3
_SPLIT_TO = 0.7


def triggers(raw: RawChunk, *, duration: float, capped: bool | None = None) -> tuple[Trigger, ...]:
    """Why a decode of `duration` seconds looks failed; empty when it looks sound.

    `capped` overrides the token-cap check for a merged decode, whose token counts no longer say which half hit it.
    """
    found: list[Trigger] = []
    if _hit_cap(raw) if capped is None else capped:
        found.append("token_cap")
    words = tuple(Unit(word.start, word.end, word.text) for word in raw.words)
    kinds = {a.kind for a in decoder_anomalies([Unit(0.0, duration, raw.text, raw.words)])}
    kinds.update(a.kind for a in cross_segment_repetitions(words))
    if kinds & {"intra_segment_repetition", "cross_segment_repetition"}:
        found.append("repetition_loop")
    if _max_compression_ratio(raw, duration) > COMPRESSION_RATIO_THRESHOLD:
        found.append("compression_ratio")
    if duration >= MIN_REDECODE_S and not has_lexical_content(raw.text):
        found.append("empty_output")
    if any(is_nonlexical_failure(run) for run in nonlexical_runs(words)):
        found.append("nonlexical_output")
    if "word_timing_collapse" in kinds:
        found.append("word_timing_collapse")
    if "replacement_character" in kinds:
        found.append("replacement_character")
    return tuple(found)


class QualityGuard:
    """Re-decodes a failed chunk as two halves cut at the quietest point near its middle.

    At most one re-decode happens per chunk. The halves replace the whole chunk when the result passes; when it
    still fails the better of the two attempts is kept and flagged: `empty_output` when it has no text,
    `decoder_anomaly` for any other failure.

    With `whole_call=True` the decode is a whole file or run: the same re-decode applies to each of its windows that
    fails on its own, and a decode without windows is only flagged.
    """

    def __init__(self, *, whole_call: bool = False) -> None:
        self._whole_call = whole_call

    def __call__(self, engine: Engine, chunk: ChunkInput, raw: RawChunk) -> GuardedChunk:
        duration = len(chunk.audio) / SAMPLE_RATE
        first = _Attempt(raw, _hit_cap(raw), triggers(raw, duration=duration))
        if not first.triggers:
            return GuardedChunk(raw)
        if self._whole_call:
            return _repair_windows(engine, chunk, first, duration)
        kept = _redecode(
            chunk.audio,
            first,
            lambda audio: engine.transcribe(audio, language=chunk.language, context=chunk.context),
            _join,
        )
        if kept is None:
            return _verdict(first)
        verdict = _verdict(kept)
        # Marked whichever attempt is kept, so trigger rates stay measurable with the guard on.
        return GuardedChunk(verdict.raw, ("redecoded", *verdict.flags))


@dataclass(frozen=True, slots=True)
class _Attempt:
    raw: RawChunk
    capped: bool
    triggers: tuple[Trigger, ...]
    # Index into `raw.words` where the second half starts, for an attempt merged from two halves.
    seam: int | None = None


def _redecode(
    audio: NDArray[np.float32],
    first: _Attempt,
    decode: Callable[[NDArray[np.float32]], RawChunk],
    join: Callable[[str, str], str],
) -> _Attempt | None:
    """Decode `audio` again as two halves and return the better attempt; None when it is too short to split."""
    cut = _split_point(audio)
    if cut is None:
        return None
    duration = len(audio) / SAMPLE_RATE
    halves = (decode(audio[:cut]), decode(audio[cut:]))
    merged = _merge(
        *halves, offset_s=cut / SAMPLE_RATE, duration=duration, fallback_language=first.raw.language, join=join
    )
    capped = any(_hit_cap(half) for half in halves)
    again = _Attempt(merged, capped, triggers(merged, duration=duration, capped=capped), len(halves[0].words))
    # On a tie the re-decode wins: its damage is bounded by half the audio.
    return min((again, first), key=_badness)


def _repair_windows(engine: Engine, chunk: ChunkInput, first: _Attempt, duration: float) -> GuardedChunk:
    raw = first.raw
    if not _fails(first) or not raw.windows:
        return _verdict(first)
    if not isinstance(engine, WindowDecoder):
        raise EngineError("the engine reports decode windows but cannot decode one again (no transcribe_window)")
    bounds = list(accumulate((window.words for window in raw.windows), initial=0))
    if bounds[-1] != len(raw.words):
        raise EngineError(f"the engine's decode windows hold {bounds[-1]} words, but it returned {len(raw.words)}")
    windows = [(window, raw.words[lo:hi]) for window, (lo, hi) in zip(raw.windows, pairwise(bounds), strict=True)]
    own = [_window_attempt(window, words, raw.language) for window, words in windows]
    if not any(_fails(attempt) for attempt in own):
        return _verdict(first)

    def decode(audio: NDArray[np.float32]) -> RawChunk:
        # The call's language even when it had no hint: the call detected it once, and a window must not re-detect.
        half = engine.transcribe_window(audio, language=raw.language)
        # Judged on its words, as the original window is, because the words are what the splice writes.
        return replace(half, text=_text_of(half.words))

    pieces: list[list[Word]] = [[]]
    outside: list[Word] = []
    spans: list[tuple[float, float]] = []
    redecoded = False
    for (window, words), attempt in zip(windows, own, strict=True):
        kept = attempt
        if _fails(attempt):
            audio = chunk.audio[round(window.start * SAMPLE_RATE) : round(window.end * SAMPLE_RATE)]
            again = _redecode(audio, attempt, decode, lambda left, right: _seam(left, right).strip())
            redecoded = redecoded or again is not None
            kept = attempt if again is None else again
        placed = words if kept.seam is None else _shifted(kept.raw.words, window.start)
        if _fails(kept):
            spans.append((window.start, window.end))
        else:
            outside.extend(placed)
        if kept.seam is None:
            pieces[-1].extend(placed)
        else:
            pieces.extend([list(placed[: kept.seam]), list(placed[kept.seam :]), []])
    windows_fail = bool(spans)
    rest = RawChunk(_text_of(outside).strip(), words=tuple(outside))
    if spans and _failing(triggers(rest, duration=duration)):
        # Something outside the failing windows trips the checks too, so they do not say where the failure lies.
        spans = []
    if not redecoded:
        return GuardedChunk(raw, _verdict(first).flags, tuple(spans))
    final = _spliced(raw, pieces) if len(pieces) > 1 else raw
    verdict = _verdict(_Attempt(final, _hit_cap(final), triggers(final, duration=duration)))
    anomaly: tuple[ChunkFlag, ...] = ("decoder_anomaly",) if windows_fail else ()
    return GuardedChunk(final, tuple(dict.fromkeys(("redecoded", *anomaly, *verdict.flags))), tuple(spans))


def _window_attempt(window: Window, words: Sequence[Word], language: str | None) -> _Attempt:
    local = RawChunk(text=_text_of(words).strip(), language=language, words=_shifted(words, -window.start))
    return _Attempt(local, capped=False, triggers=triggers(local, duration=window.end - window.start))


def _shifted(words: Sequence[Word], offset: float) -> tuple[Word, ...]:
    # mlx-whisper may move a window's first word to start up to about 1.4 s before the window's seek (its long-word
    # truncation in timing.py), and a time must not go negative.
    return tuple(
        Word(text=w.text, start=max(0.0, w.start + offset), end=max(0.0, w.end + offset), conf=w.conf) for w in words
    )


def _spliced(raw: RawChunk, pieces: Sequence[Sequence[Word]]) -> RawChunk:
    """`raw` with its words replaced by `pieces` and its text rebuilt from their texts, spaced at each seam."""
    words = tuple(word for piece in pieces for word in piece)
    probabilities = [w.conf for w in words if w.conf is not None]
    return replace(
        raw,
        text=reduce(_seam, (_text_of(piece) for piece in pieces), "").strip(),
        confidence=sum(probabilities) / len(probabilities) if probabilities else None,
        words=words,
        # The counts and windows described the original decode, not the splice.
        generated_tokens=None,
        max_tokens=None,
        windows=(),
    )


def _text_of(words: Sequence[Word]) -> str:
    return "".join(word.text for word in words)


def _fails(attempt: _Attempt) -> bool:
    return _failing(attempt.triggers)


def _failing(found: Sequence[Trigger]) -> bool:
    # Empty output on speech may be music or noise the VAD let through, so it is counted apart from decoder failures.
    return bool(set(found) - {"empty_output"})


def _badness(attempt: _Attempt) -> tuple[bool, int]:
    # Text of any kind beats none, whatever else is wrong with it.
    return (not has_lexical_content(attempt.raw.text), len(attempt.triggers))


def _verdict(attempt: _Attempt) -> GuardedChunk:
    flags: list[ChunkFlag] = []
    if _fails(attempt):
        flags.append("decoder_anomaly")
    if attempt.capped:
        flags.append("token_cap_hit")
    if "empty_output" in attempt.triggers:
        flags.append("empty_output")
    return GuardedChunk(attempt.raw, tuple(flags))


def _hit_cap(raw: RawChunk) -> bool:
    return raw.generated_tokens is not None and raw.max_tokens is not None and raw.generated_tokens >= raw.max_tokens


def _max_compression_ratio(raw: RawChunk, duration: float) -> float:
    if duration <= COMPRESSION_WINDOW_S:
        return compression_ratio(raw.text)
    # Without word times a long text cannot be windowed, and measured whole it would flag ordinary speech.
    windows: list[list[str]] = []
    window_end = -1.0
    for word in raw.words:
        if word.start >= window_end:
            windows.append([])
            window_end = word.start + COMPRESSION_WINDOW_S
        windows[-1].append(word.text)
    return max((compression_ratio("".join(texts)) for texts in windows), default=0.0)


def _split_point(audio: NDArray[np.float32]) -> int | None:
    """Sample index of the quietest 100 ms frame in the middle 40% of the audio, the one nearest the middle on ties."""
    if len(audio) < MIN_REDECODE_S * SAMPLE_RATE:
        return None
    frame = round(FRAME_S * SAMPLE_RATE)
    first = round(_SPLIT_FROM * len(audio))
    count = (round(_SPLIT_TO * len(audio)) - first) // frame
    frames = audio[first : first + count * frame].reshape(count, frame)
    energy = np.mean(frames * frames, axis=1)
    centers = first + frame * (np.arange(count) + 0.5)
    # lexsort's last key is the primary one.
    best = np.lexsort((np.abs(centers - len(audio) / 2), energy))[0]
    return int(centers[best])


def _merge(
    first: RawChunk,
    second: RawChunk,
    *,
    offset_s: float,
    duration: float,
    fallback_language: str | None,
    join: Callable[[str, str], str],
) -> RawChunk:
    shifted = tuple(
        Word(text=w.text, start=w.start + offset_s, end=w.end + offset_s, conf=w.conf) for w in second.words
    )
    confidence = None
    if first.confidence is not None and second.confidence is not None:
        confidence = (first.confidence * offset_s + second.confidence * (duration - offset_s)) / duration
    return RawChunk(
        text=join(first.text, second.text),
        language=first.language or second.language or fallback_language,
        generated_tokens=_add(first.generated_tokens, second.generated_tokens),
        # Budgets add up with the counts, so the worker's cap check on the merged row fires only when both halves hit.
        max_tokens=_add(first.max_tokens, second.max_tokens),
        confidence=confidence,
        words=(*first.words, *shifted),
    )


def _add(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


def _join(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left or not right:
        return left or right
    return left + right if _wide(left[-1]) or _wide(right[0]) else f"{left} {right}"


def _seam(left: str, right: str) -> str:
    """Join texts that keep Whisper's own spacing, adding one space only where neither side has any."""
    if not left or not right or left[-1].isspace() or right[0].isspace() or _wide(left[-1]) or _wide(right[0]):
        return left + right
    return f"{left} {right}"


def _wide(character: str) -> bool:
    return unicodedata.east_asian_width(character) in "WF"
