"""Align the transcript with an independent transcript of the same audio, such as a caption track.

Both sides are cut into units (a whole quantity token, a CJK character or a Latin word) and matched with Levenshtein
opcodes. A caption window is only trusted as the same speech when enough of the transcript's plain units match it.
"""

import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein, Opcodes

from ytverbatim.acquire.captions import CaptionKind, CaptionTrack
from ytverbatim.text.numbers import QToken, quantity_tokens

# Caption and second-engine timings drift from the transcript's by up to a second or so; untimed cues and chunks are
# interpolated, which drifts more. A wider window costs only some extra unmatched units around the segment.
WINDOW_PAD_S = 2.0
# Below this share of matching plain units the window is not the same speech, so pairing its numbers would be noise.
_MIN_ANCHOR_RATIO = 0.4
_MIN_ANCHOR_UNITS = 4

CJK = re.compile(r"[一-鿿]")
_PLAIN_UNIT = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[一-鿿]")
_SPACE_BY_CJK = re.compile(r"\s+(?=[一-鿿])|(?<=[一-鿿])\s+")
_ALNUM = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True, slots=True)
class TimedText:
    """Text with absolute start and end times in seconds: a transcript segment, a decoded chunk or a word."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class Unit:
    """An alignment unit: a whole quantity token, a CJK character or a Latin word. Offsets index the cleaned text."""

    key: str
    start: int
    end: int
    token: QToken | None


@dataclass(frozen=True, slots=True)
class Timeline:
    """An independent transcript as units with a time span and a confidence each; ``starts`` never decrease."""

    source: str
    text: str
    units: tuple[Unit, ...]
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    confs: tuple[float | None, ...]

    def window(self, start: float, end: float) -> range:
        return range(bisect_left(self.starts, start - WINDOW_PAD_S), bisect_right(self.starts, end + WINDOW_PAD_S))


@dataclass(frozen=True, slots=True)
class Alignment:
    """Opcodes from ``mine`` to ``theirs`` and whether ``theirs`` is trusted as the same speech.

    The opcodes are returned for an unanchored window too, but their equal runs are coincidences and mean nothing.
    """

    opcodes: Opcodes
    anchored: bool


def clean(text: str) -> tuple[str, list[int]]:
    """Drop whitespace next to CJK (150 期 reads as 150期) and map each kept character to its original offset."""
    kept: list[int] = []
    cursor = 0
    for match in _SPACE_BY_CJK.finditer(text):
        kept.extend(range(cursor, match.start()))
        cursor = match.end()
    kept.extend(range(cursor, len(text)))
    return "".join(text[i] for i in kept), kept


def units(text: str) -> list[Unit]:
    result: list[Unit] = []
    cursor = 0

    def plain(stop: int) -> None:
        for match in _PLAIN_UNIT.finditer(text, cursor, stop):
            result.append(Unit(match[0].casefold(), match.start(), match.end(), None))

    for token in quantity_tokens(text):
        plain(token.char_start)
        # Keyed by value, so 一百五十期 aligns with 150期 as the same unit.
        result.append(Unit(f"\x00{token.value_itn}", token.char_start, token.char_end, token))
        cursor = token.char_end
    plain(len(text))
    return result


def mostly_cjk(text: str) -> bool:
    non_space = sum(1 for ch in text if not ch.isspace())
    return non_space > 0 and len(CJK.findall(text)) >= 0.5 * non_space


def timeline(source: str, pieces: Iterable[tuple[str, float, float, float | None]], *, cues: bool = False) -> Timeline:
    kept = sorted(
        ((piece, start, end, conf) for raw, start, end, conf in pieces if (piece := clean(raw)[0]).strip()),
        key=lambda item: item[1],
    )
    joined = "".join(piece for piece, *_ in kept)
    # Whisper words carry their own leading space and Chinese is written without spaces, so only stripped Latin pieces
    # (caption words, chunk texts) need one; inserting it elsewhere would split 15 + 00 into two numbers.
    add_spaces = not any(piece[0].isspace() for piece, *_ in kept[1:]) and not mostly_cjk(joined)
    text: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    confs: list[float | None] = []
    # Rolling caption cues overlap; clamping keeps character times in order so windows can bisect them.
    floor = -math.inf
    for piece, start, end, conf in kept:
        if text and (
            (add_spaces and not (text[-1][-1].isspace() or CJK.match(text[-1][-1]) or CJK.match(piece[0])))
            # A cue ends where its text ends, so even in Chinese two cues meeting in Latin script or digits are two
            # words (收购了Zorbix | Zorbix也), never one.
            or (cues and _ALNUM.match(text[-1][-1]) is not None and _ALNUM.match(piece[0]) is not None)
        ):
            text.append(" ")
            starts.append(max(start, floor))
            ends.append(starts[-1])
            confs.append(None)
        step = max(end - start, 0.0) / len(piece)
        for k in range(len(piece)):
            floor = max(start + step * k, floor)
            starts.append(floor)
            ends.append(max(start + step * (k + 1), floor))
        text.append(piece)
        confs.extend(conf for _ in piece)
    full = "".join(text)
    parsed = units(full)
    return Timeline(
        source=source,
        text=full,
        units=tuple(parsed),
        starts=tuple(starts[u.start] for u in parsed),
        ends=tuple(ends[u.end - 1] for u in parsed),
        confs=tuple(min((c for c in confs[u.start : u.end] if c is not None), default=None) for u in parsed),
    )


def caption_timeline(track: CaptionTrack) -> Timeline:
    """Return the track as a timeline tagged ``caption_manual`` or ``caption_auto``, from its words or else its cues."""
    source, pieces = caption_pieces(track)
    return timeline(
        source,
        ((piece.text, piece.start, piece.end, None) for piece in pieces),
        cues=not any(cue.words for cue in track.segments),
    )


def caption_pieces(track: CaptionTrack) -> tuple[str, tuple[TimedText, ...]]:
    """Return the track's source tag (``caption_manual`` or ``caption_auto``) and its words, or cues without words."""
    source = "caption_manual" if track.kind is CaptionKind.MANUAL else "caption_auto"
    pieces: list[TimedText] = []
    for cue in track.segments:
        if cue.words:
            pieces.extend(TimedText(w.text, w.start, w.end) for w in cue.words)
        else:
            pieces.append(TimedText(cue.text, cue.start, cue.end))
    return source, tuple(pieces)


def align(mine: Sequence[Unit], theirs: Sequence[Unit]) -> Alignment:
    """Match ``mine`` against a window of another transcript, anchored when the window is the same speech.

    A window is refused when it has no matching unit at all, or when ``mine`` holds at least ``_MIN_ANCHOR_UNITS``
    plain units and fewer than ``_MIN_ANCHOR_RATIO`` of them match. Matching quantity units never count toward the
    ratio: they are what callers pair, so they cannot also vouch for the window.
    """
    opcodes = Levenshtein.opcodes([u.key for u in mine], [u.key for u in theirs])
    matching = [i for op in opcodes if op.tag == "equal" for i in range(op.src_start, op.src_end)]
    if not matching:
        return Alignment(opcodes, False)
    plain = sum(1 for u in mine if u.token is None)
    anchors = sum(1 for i in matching if mine[i].token is None)
    return Alignment(opcodes, plain < _MIN_ANCHOR_UNITS or anchors >= _MIN_ANCHOR_RATIO * plain)
