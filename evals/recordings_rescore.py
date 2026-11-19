"""Compare one recording's ASR rows from before and after the whole-call loop repair, as aggregates and spans only.

The guard's repair replaces a looped decode window inside a whole-file or run row, but the row records only the
windows it could not repair (``anomaly_spans``); the repaired ones leave no trace except the ``redecoded`` flag and the
words themselves. So a repaired window is recovered from the two sets of rows:

1. Align the words of all rows before and after by normalized text. Each non-identical stretch of the alignment is a
   difference; differences closer than ``MERGE_GAP_S`` of identical speech merge into one while the merged stretch stays
   within one decode window (``WINDOW_S``). A stretch longer than a window is never split, so it may hold several.
2. A difference is a **repaired window** when it removed a failure of the decode and a row flagged ``redecoded``
   after overlaps it. Removed means one of two things:
   a. The guard's own repetition detectors find a loop in the old words of the difference and none in its new
      words. A loop is a strong pattern, so this holds however short the slice.
   b. Otherwise the guard's failure test (``guard.triggers``, ignoring ``empty_output``) fails on the old words of a
      30 s context around the difference (padded evenly, since the guard judges a whole window and not the slice
      that happened to differ) and passes on the new words of it, and that context holds no localized loop, which
      would belong to a neighbouring window or to speech that really repeats.
   Both contexts' triggers are reported. For ``w-prod`` one row is the whole file, so the row condition always
   holds and this test does the work.
3. Every other difference is reported with its triggers (``other_differences``), so a missed repair is visible. Most
   are different decodes of windows that never looped, since Whisper's temperature fallback is not reproducible; they
   count against the identity share of the words outside the repaired windows.
4. A repaired window's span is the union of its old and new words' times. It is narrowest-differing, so a head or
   tail word the re-decode reproduced is not in it. Overlap of two spans is half-open: windows that touch at a seam
   do not overlap.
5. A seam duplicate: the last new word of a repaired window equals (normalized) the first word after it, which
   starts at most ``SEAM_GAP_S`` after it ends.
6. A repaired window is still flagged when an after row flagged ``decoder_anomaly`` has a span overlapping it
   (``still_flagged``). A row flagged with no spans cannot say where, so a window only it overlaps is
   ``still_flagged_unlocalized`` and the pre-registered rule 2 leaves it out.
7. Looped windows are the repaired windows plus the after-row anomaly spans that overlap none (a window the repair
   kept unchanged, which the alignment cannot see).

The rows hold no window starts, so a loop's offset inside its Whisper window is unavailable; ``loop_offset_s`` is its
offset inside the repaired span. A loop is localized by the guard's own repetition detectors; one they only see
through compression has no ``loop``. No transcript text is ever part of the report.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict

from anyverbatim.asr.guard import triggers
from anyverbatim.asr.models import ChunkHypothesis
from anyverbatim.asr.protocol import RawChunk
from anyverbatim.asr.quality import (
    Unit,
    cross_segment_repetitions,
    is_cjk_text,
    normalize_for_repetition,
    repetition_spans,
)
from anyverbatim.contract.models import Word
from evals.recordings_scoring import Interval, PairedResult, WindowScore, words_from_rows

WINDOW_S = 30.0
MERGE_GAP_S = 5.0
SEAM_GAP_S = 1.0
MAX_STILL_FLAGGED_SHARE = 0.5
MAX_SEAM_SHARE = 0.25

UNAVAILABLE = (
    "window start: rows do not store Whisper's decode windows, so a loop's offset inside its window is not known; "
    "loop_offset_s is relative to the repaired span",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Span(_Frozen):
    start: float
    end: float


class Errors(_Frozen):
    errors: int
    reference_units: int
    rate: float | None

    @classmethod
    def of(cls, windows: Sequence[WindowScore]) -> Errors:
        errors = sum(w.counts.errors for w in windows)
        units = sum(w.counts.reference_units for w in windows)
        return cls(errors=errors, reference_units=units, rate=errors / units if units else None)


class AnomalyRow(_Frozen):
    chunk: int
    start: float
    end: float
    localized: bool
    spans: tuple[Span, ...]


class RowsSummary(_Frozen):
    rows: int
    redecoded: int
    decoder_anomaly: tuple[AnomalyRow, ...]


class OtherDifference(_Frozen):
    span: Span
    before_words: int
    after_words: int
    triggers: tuple[str, ...]
    triggers_after: tuple[str, ...]
    in_redecoded_row: bool


class RepairedWindow(_Frozen):
    span: Span
    before_words: int
    after_words: int
    triggers: tuple[str, ...]
    loop: Span | None
    loop_offset_s: float | None
    still_flagged: bool
    still_flagged_unlocalized: bool
    seam_duplicate: bool | None


class Outside(_Frozen):
    words: int
    identical: int
    share: float | None


class FileRescore(_Frozen):
    item: str
    before: Errors
    after: Errors
    before_rows: RowsSummary
    after_rows: RowsSummary
    repaired_windows: tuple[RepairedWindow, ...]
    unrepaired_spans: tuple[Span, ...]
    other_differences: tuple[OtherDifference, ...]
    outside: Outside


class Repair(_Frozen):
    repaired_windows: int
    looped_windows: int
    still_flagged: int
    still_flagged_unlocalized: int
    seams_checked: int
    seam_duplicates: int
    other_differences: int
    outside: Outside


class Rule1(_Frozen):
    """Ships unless the pooled error is worse after with the window-level CI of after - before entirely above 0."""

    before_rate: float | None
    after_rate: float | None
    difference: float
    window_ci: Interval
    worse_with_ci_above_zero: bool
    ships: bool


class Rule2(_Frozen):
    """Stop and re-plan when more than half of the looped windows are still flagged; only localized flags count."""

    looped_windows: int
    still_flagged_unlocalized: int
    judged_windows: int
    still_flagged: int
    share: float | None
    stop: bool


class Rule3(_Frozen):
    """Plan seam dedupe when more than a quarter of the repaired windows have a seam duplicate."""

    repaired_windows: int
    seam_duplicates: int
    share: float | None
    plan_dedupe: bool


class UnlocalizedRow(_Frozen):
    item: str
    chunk: int
    start: float
    end: float


class Rule4(_Frozen):
    """Rows of the new decode flagged decoder_anomaly with no span, by name."""

    unlocalized_rows: tuple[UnlocalizedRow, ...]


class Rules(_Frozen):
    rule_1: Rule1
    rule_2: Rule2
    rule_3: Rule3
    rule_4: Rule4


class CorpusRescore(_Frozen):
    hypothesis: str
    corpus: str
    metric: str
    view: str
    difference_is_after_minus_before: Literal[True] = True
    before: Errors
    after: Errors
    paired: PairedResult
    rules: Rules
    repair: Repair
    files: tuple[FileRescore, ...]
    unavailable: tuple[str, ...] = UNAVAILABLE


@dataclass(frozen=True, slots=True)
class _Difference:
    old: tuple[int, int]
    new: tuple[int, int]
    start: float
    end: float


def _key(word: Word) -> str:
    return normalize_for_repetition(word.text) or word.text.strip()


def _times(words: Sequence[Word]) -> tuple[float, float]:
    return min(w.start for w in words), max(w.end for w in words)


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _differences(old: Sequence[Word], new: Sequence[Word]) -> tuple[list[_Difference], list[bool]]:
    """Merge the differences between two word lists; also mark which new words are identical to an old one."""
    # Not autojunk: it would drop common words from the alignment on a long file.
    matcher = difflib.SequenceMatcher(None, [_key(w) for w in old], [_key(w) for w in new], autojunk=False)
    identical = [False] * len(new)
    found: list[_Difference] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            identical[j1:j2] = [True] * (j2 - j1)
            continue
        start, end = _times([*old[i1:i2], *new[j1:j2]])
        last = found[-1] if found else None
        if last and start - last.end < MERGE_GAP_S and end - last.start <= WINDOW_S:
            found[-1] = replace(last, old=(last.old[0], i2), new=(last.new[0], j2), end=max(last.end, end))
        else:
            found.append(_Difference((i1, i2), (j1, j2), start, end))
    return found, identical


def _context(words: Sequence[Word], span: tuple[float, float]) -> tuple[tuple[Word, ...], float]:
    """Return the words of a 30 s stretch centred on ``span`` (all of ``span`` when longer) and its length."""
    padding = max(WINDOW_S - (span[1] - span[0]), 0.0) / 2
    low, high = span[0] - padding, span[1] + padding
    return tuple(w for w in words if w.end > low and w.start < high), high - low


def _triggers(words: Sequence[Word], duration: float) -> tuple[str, ...]:
    local = RawChunk(text="".join(w.text for w in words).strip(), words=tuple(words))
    return tuple(t for t in triggers(local, duration=duration) if t != "empty_output")


def _loops(words: Sequence[Word]) -> list[tuple[float, float]]:
    """Find the times of every repetition loop the guard's detectors see in these words."""
    found = [(a.start, a.end) for a in cross_segment_repetitions([Unit(w.start, w.end, w.text) for w in words])]
    keys = [normalize_for_repetition(w.text) for w in words]
    separator = "" if is_cjk_text("".join(keys)) else " "
    owners = [i for i, key in enumerate(keys) for _ in range(len(key) + len(separator))]
    text = "".join(key + separator for key in keys)
    for start, end, _ in repetition_spans(text):
        # A run can open on the separator after the previous word.
        first = start + len(text[start:end]) - len(text[start:end].lstrip())
        found.append((words[owners[first]].start, words[owners[end - 1]].end))
    return sorted(found)


def _judge(
    old: Sequence[Word], new: Sequence[Word], changed: tuple[Sequence[Word], Sequence[Word]], span: tuple[float, float]
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Decide whether a difference removed a failure; also return the triggers of its 30 s context, old then new.

    A loop is a strong pattern even in a slice, so a loop in the old changed words that the new changed words do not
    keep is a repair. Otherwise the guard's failure on the old context, absent from the new, decides, but only when
    that context holds no localized loop: one in a neighbouring window, or in legitimate repetition, would be taken
    for this difference's.
    """
    before, duration = _context(old, span)
    after, _ = _context(new, span)
    found, still = _triggers(before, duration), _triggers(after, duration)
    if _loops(changed[0]):
        return not _loops(changed[1]), found, still
    return bool(found) and not still and not _loops(before), found, still


def _summary(rows: Sequence[ChunkHypothesis]) -> RowsSummary:
    flagged = [r for r in rows if "decoder_anomaly" in r.flags]
    return RowsSummary(
        rows=len(rows),
        redecoded=sum("redecoded" in r.flags for r in rows),
        decoder_anomaly=tuple(
            AnomalyRow(
                chunk=r.chunk,
                start=r.start,
                end=r.end,
                localized=bool(r.anomaly_spans),
                spans=tuple(Span(start=s.start, end=s.end) for s in r.anomaly_spans),
            )
            for r in flagged
        ),
    )


def _flagged_by(rows: Sequence[ChunkHypothesis], span: tuple[float, float]) -> tuple[bool, bool]:
    """Whether a localized span, and whether only a row without spans, of the flagged rows overlaps ``span``."""
    flagged = [r for r in rows if "decoder_anomaly" in r.flags]
    localized = any(_overlaps(span, (s.start, s.end)) for r in flagged for s in r.anomaly_spans)
    whole = any(not r.anomaly_spans and _overlaps(span, (r.start, r.end)) for r in flagged)
    return localized, whole and not localized


def _seam_duplicate(new_words: Sequence[Word], following: Word | None) -> bool | None:
    if not new_words or following is None:
        return None
    return _key(new_words[-1]) == _key(following) and following.start - new_words[-1].end <= SEAM_GAP_S


def rescore_item(
    item: str,
    before_rows: Sequence[ChunkHypothesis],
    after_rows: Sequence[ChunkHypothesis],
    before_windows: Sequence[WindowScore],
    after_windows: Sequence[WindowScore],
) -> FileRescore:
    old, new = words_from_rows(before_rows), words_from_rows(after_rows)
    differences, identical = _differences(old, new)
    redecoded = [(r.start, r.end) for r in after_rows if "redecoded" in r.flags]
    repaired: list[RepairedWindow] = []
    other: list[OtherDifference] = []
    inside = [False] * len(new)
    for d in differences:
        old_words, new_words = old[d.old[0] : d.old[1]], new[d.new[0] : d.new[1]]
        span = (d.start, d.end)
        removed, found, still = _judge(old, new, (old_words, new_words), span)
        in_redecoded = any(_overlaps(span, r) for r in redecoded)
        if not (old_words and removed and in_redecoded):
            other.append(
                OtherDifference(
                    span=Span(start=span[0], end=span[1]),
                    before_words=len(old_words),
                    after_words=len(new_words),
                    triggers=found,
                    triggers_after=still,
                    in_redecoded_row=in_redecoded,
                )
            )
            continue
        inside[d.new[0] : d.new[1]] = [True] * len(new_words)
        loops = _loops(old_words)
        loop = loops[0] if loops else None
        localized, unlocalized = _flagged_by(after_rows, span)
        repaired.append(
            RepairedWindow(
                span=Span(start=span[0], end=span[1]),
                before_words=len(old_words),
                after_words=len(new_words),
                triggers=found,
                loop=Span(start=loop[0], end=loop[1]) if loop else None,
                loop_offset_s=loop[0] - span[0] if loop else None,
                still_flagged=localized,
                still_flagged_unlocalized=unlocalized,
                seam_duplicate=_seam_duplicate(new_words, new[d.new[1]] if d.new[1] < len(new) else None),
            )
        )
    after_summary = _summary(after_rows)
    unrepaired = tuple(
        s
        for row in after_summary.decoder_anomaly
        for s in row.spans
        if not any(_overlaps((s.start, s.end), (w.span.start, w.span.end)) for w in repaired)
    )
    outside = [i for i, is_in in enumerate(inside) if not is_in]
    kept = sum(identical[i] for i in outside)
    return FileRescore(
        item=item,
        before=Errors.of(before_windows),
        after=Errors.of(after_windows),
        before_rows=_summary(before_rows),
        after_rows=after_summary,
        repaired_windows=tuple(repaired),
        unrepaired_spans=unrepaired,
        other_differences=tuple(other),
        outside=Outside(words=len(outside), identical=kept, share=kept / len(outside) if outside else None),
    )


def total_repair(files: Sequence[FileRescore]) -> Repair:
    windows = [w for f in files for w in f.repaired_windows]
    unrepaired = sum(len(f.unrepaired_spans) for f in files)
    seams = [w.seam_duplicate for w in windows if w.seam_duplicate is not None]
    words = sum(f.outside.words for f in files)
    kept = sum(f.outside.identical for f in files)
    return Repair(
        repaired_windows=len(windows),
        looped_windows=len(windows) + unrepaired,
        still_flagged=sum(w.still_flagged for w in windows) + unrepaired,
        still_flagged_unlocalized=sum(w.still_flagged_unlocalized for w in windows),
        seams_checked=len(seams),
        seam_duplicates=sum(seams),
        other_differences=sum(len(f.other_differences) for f in files),
        outside=Outside(words=words, identical=kept, share=kept / words if words else None),
    )


def build_rules(
    before: Errors, after: Errors, comparison: PairedResult, repair: Repair, files: Sequence[FileRescore]
) -> Rules:
    worse = comparison.difference > 0 and comparison.by_window.low > 0
    judged = repair.looped_windows - repair.still_flagged_unlocalized
    flagged_share = repair.still_flagged / judged if judged else None
    seam_share = repair.seam_duplicates / repair.repaired_windows if repair.repaired_windows else None
    return Rules(
        rule_1=Rule1(
            before_rate=before.rate,
            after_rate=after.rate,
            difference=comparison.difference,
            window_ci=comparison.by_window,
            worse_with_ci_above_zero=worse,
            ships=not worse,
        ),
        rule_2=Rule2(
            looped_windows=repair.looped_windows,
            still_flagged_unlocalized=repair.still_flagged_unlocalized,
            judged_windows=judged,
            still_flagged=repair.still_flagged,
            share=flagged_share,
            stop=flagged_share is not None and flagged_share > MAX_STILL_FLAGGED_SHARE,
        ),
        rule_3=Rule3(
            repaired_windows=repair.repaired_windows,
            seam_duplicates=repair.seam_duplicates,
            share=seam_share,
            plan_dedupe=seam_share is not None and seam_share > MAX_SEAM_SHARE,
        ),
        rule_4=Rule4(
            unlocalized_rows=tuple(
                UnlocalizedRow(item=f.item, chunk=r.chunk, start=r.start, end=r.end)
                for f in files
                for r in f.after_rows.decoder_anomaly
                if not r.localized
            )
        ),
    )
