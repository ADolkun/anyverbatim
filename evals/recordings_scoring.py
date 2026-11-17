"""Score ASR runs over the recordings corpora, pair two hypotheses, and write the words WDER needs.

A recording with timed reference lines is cut into windows at pauses in the *reference*, so every hypothesis is scored
over exactly the same windows and any two can be paired. The reference lines and each hypothesis row are normalized
once, over their whole text, and the resulting units are assigned to a window by the midpoint of their time; a clip
with only a reference text is one window. Everything here is pure computation on rows and references.

Two translation-incident counts come out of the same run:

- per window (``CorpusScore.window_incidents``): a 30 s stretch is mostly one script in the reference and mostly the
  other in the hypothesis. A translated phrase inside a mostly-Chinese window does not show here;
- per chunk (``CorpusScore.chunk_incidents``): the same test on each reference line (a clip is one chunk), against the
  hypothesis units timed inside the line. The plan's rule 3, "translation incidents in more than 1% of mixed
  chunks", reads ``rate_in_mixed_recordings``: incidents over the chunks whose reference is at least 70% one script
  inside recordings whose reference holds both scripts. Overlapping lines each count the units timed inside them.

The rule 3 count stays as defined, so runs can be compared with earlier ones, but on a recording of overlapping
speakers it mostly measures another speaker's words: a line another reference line overlaps is charged the words timed
inside it, whoever said them. ``clean_*`` is the same count over the lines no other reference line overlaps, where an
incident is the line's own speech decoded in the other script. ``paired_incidents`` compares two hypotheses on those
lines.
"""

from __future__ import annotations

import bisect
import itertools
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.consensus import TimedUnit
from evals.metrics import (
    MIN_INCIDENT_UNITS,
    ErrorCounts,
    paired_bootstrap,
    script_mix,
    sign_test,
    translation_incident,
    unit_error_counts,
)
from evals.normalize import View, join_units, to_units
from evals.recordings_corpus import FOLDER_CORPORA, Item, Metric
from evals.recordings_data import WORDS_FILE, RefUtterance, ordered_text
from evals.store import hypothesis_units, row_text, row_units
from ytverbatim import atomic
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError

WINDOW_TARGET_S = 30.0
# A cut in a shorter pause could split a phrase the two sides time slightly differently.
MIN_CUT_GAP_S = 0.75
MEETINGS = "meetings"
ASCEND_POOLED = "ascend-pooled"
SPECIAL_TOKEN_MARK = "<|"
MIN_CLUSTERS = 2


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True, slots=True)
class WindowScore:
    item: str
    window: int
    counts: ErrorCounts
    # The reference is long enough to judge a translation (see ``translation_incident``).
    judged: bool
    mixed: bool
    incident: bool


@dataclass(frozen=True, slots=True)
class ChunkScore:
    item: str
    index: int
    judged: bool
    # At least 70% of the reference's units are in one script.
    dominated: bool
    # The recording's whole reference holds both scripts.
    mixed_recording: bool
    # Another reference line overlaps this one in time; always false for a clip.
    overlapped: bool
    incident: bool

    @property
    def clean(self) -> bool:
        """A judged line mostly in one script, in a mixed recording, that no other reference line overlaps."""
        return self.judged and self.dominated and self.mixed_recording and not self.overlapped


@dataclass(frozen=True, slots=True)
class Scored:
    windows: list[WindowScore]
    chunks: list[ChunkScore]


class Incidents(_Frozen):
    chunks: int
    mixed_chunks: int
    incidents: int
    incidents_in_mixed: int
    rate: float | None
    rate_in_mixed: float | None


class ChunkIncidents(_Frozen):
    chunks: int
    incidents: int
    rate: float | None
    # Chunks at least 70% one script inside recordings whose reference holds both scripts: the rule 3 denominator.
    dominated_in_mixed_recordings: int
    incidents_in_mixed_recordings: int
    rate_in_mixed_recordings: float | None
    # The same, over the lines no other reference line overlaps; always equal to rule 3's count for clips.
    clean_in_mixed_recordings: int
    clean_incidents_in_mixed_recordings: int
    clean_rate_in_mixed_recordings: float | None


class CorpusScore(_Frozen):
    corpus: str
    metric: Metric
    view: View
    items: int
    audio_hours: float
    windows: int
    reference_units: int
    errors: int
    rate: float | None
    window_incidents: Incidents
    chunk_incidents: ChunkIncidents


class PairedIncidents(_Frozen):
    """``a`` against ``b`` over the clean lines: ``fixed`` are incidents of ``b`` that ``a`` avoids."""

    lines: int
    a_incidents: int
    b_incidents: int
    fixed: int
    introduced: int
    # One-sided exact sign test that ``a`` fixes more lines than it introduces.
    p: float


class Interval(_Frozen):
    low: float
    high: float
    excludes_zero: bool


class PairedResult(_Frozen):
    windows: int
    recordings: int
    a_rate: float
    b_rate: float
    difference: float
    # Resamples recordings and sums their windows: windows of one recording are correlated, so this is the figure to
    # decide on. None below two recordings, where there is nothing to resample.
    decision: Interval | None
    # Resamples windows as if independent; narrower than ``decision``, shown for comparison only.
    by_window: Interval


def reference_cuts(
    utterances: Sequence[RefUtterance], target_s: float = WINDOW_TARGET_S, min_gap_s: float = MIN_CUT_GAP_S
) -> list[float]:
    """Window start times: 0, then the middle of the first reference pause of ``min_gap_s`` after each ``target_s``.

    A window therefore never splits a reference line or a short pause, and with no such pause it simply runs long.
    """
    spans: list[tuple[float, float]] = []
    for u in sorted(utterances, key=lambda u: (u.start, u.end)):
        if spans and u.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], u.end))
        else:
            spans.append((u.start, u.end))
    cuts = [0.0]
    for (_, end), (start, _) in itertools.pairwise(spans):
        middle = (end + start) / 2
        if start - end >= min_gap_s and middle - cuts[-1] >= target_s:
            cuts.append(middle)
    return cuts


def _by_window(units: Sequence[TimedUnit], starts: Sequence[float]) -> list[list[str]]:
    grouped: list[list[str]] = [[] for _ in starts]
    for unit in units:
        grouped[max(bisect.bisect_right(starts, unit.midpoint_s) - 1, 0)].append(unit.text)
    return grouped


def count_units(metric: Metric, reference: Sequence[str], hypothesis: Sequence[str]) -> ErrorCounts:
    """Count edits over normalized units; CER compares their characters. Without a reference every unit is an insertion.

    ``mer``, ``cer`` and ``wer`` of ``evals.metrics`` do the same on a text (WER is the unit count for English).
    """
    ref, hyp = (
        (list("".join(units)) for units in (reference, hypothesis)) if metric == "cer" else (reference, hypothesis)
    )
    if not ref:
        return ErrorCounts(0, 0, len(hyp), 0)
    return unit_error_counts(ref, hyp)


def _window(item: str, index: int, reference: Sequence[str], hypothesis: Sequence[str], metric: Metric) -> WindowScore:
    ref_mix, hyp_mix = script_mix(join_units(list(reference))), script_mix(join_units(list(hypothesis)))
    judged = ref_mix.total >= MIN_INCIDENT_UNITS
    return WindowScore(
        item,
        index,
        count_units(metric, reference, hypothesis),
        judged=judged,
        mixed=judged and ref_mix.mixed,
        incident=translation_incident(ref_mix, hyp_mix),
    )


def _chunk(
    item: str, index: int, reference: str, hypothesis: str, mixed_recording: bool, overlapped: bool = False
) -> ChunkScore:
    ref_mix, hyp_mix = script_mix(reference), script_mix(hypothesis)
    return ChunkScore(
        item,
        index,
        judged=ref_mix.total >= MIN_INCIDENT_UNITS,
        dominated=ref_mix.mostly("han") or ref_mix.mostly("latin"),
        mixed_recording=mixed_recording,
        overlapped=overlapped,
        incident=translation_incident(ref_mix, hyp_mix),
    )


def overlapped_lines(lines: Sequence[RefUtterance]) -> list[bool]:
    """Per line, whether another line starts before it ends and ends after it starts; ``lines`` sorted by start."""
    flags: list[bool] = []
    latest_end = float("-inf")
    for i, line in enumerate(lines):
        follows = i + 1 < len(lines) and lines[i + 1].start < line.end
        flags.append(latest_end > line.start or follows)
        latest_end = max(latest_end, line.end)
    return flags


def _reference_units(item: Item, lines: Sequence[RefUtterance], view: View) -> list[TimedUnit]:
    """Normalize the whole reference once and time each unit by the line it came from.

    This is how a hypothesis row is read (``row_units``), so both sides are normalized over a whole text, not line by
    line against row by row.
    """
    row = ChunkHypothesis(
        hypothesis="reference",
        file=item.id,
        chunk=0,
        start=min(u.start for u in lines),
        end=max(u.end for u in lines),
        text=ordered_text(lines),
        words=tuple(Word(text=u.text, start=u.start, end=u.end) for u in lines),
    )
    return row_units(row, item.language, view)


def score_item(item: Item, rows: Sequence[ChunkHypothesis], metric: Metric, view: View) -> Scored:
    reference = item.reference
    if reference is None:
        raise ConfigError(f"{item.corpus}/{item.id} has no reference to score against")
    if not reference.utterances:
        wanted, heard = (to_units(t, item.language, view) for t in (reference.text, row_text(rows)))
        mixed = script_mix(reference.text).mixed
        return Scored(
            [_window(item.id, 0, wanted, heard, metric)],
            [_chunk(item.id, 0, reference.text, row_text(rows), mixed)],
        )
    lines = sorted((u for u in reference.utterances if u.text.strip()), key=lambda u: (u.start, u.end, u.speaker))
    starts = reference_cuts(reference.utterances)
    ref_units = _reference_units(item, lines, view) if lines else []
    hyp_units = hypothesis_units(rows, item.language, view)
    windows = [
        _window(item.id, i, ref, hyp, metric)
        for i, (ref, hyp) in enumerate(zip(_by_window(ref_units, starts), _by_window(hyp_units, starts), strict=True))
    ]
    ordered = sorted(hyp_units, key=lambda u: u.midpoint_s)
    midpoints = [u.midpoint_s for u in ordered]
    mixed = script_mix(" ".join(u.text for u in lines)).mixed
    overlapped = overlapped_lines(lines)
    chunks = [
        _chunk(
            item.id,
            i,
            line.text,
            join_units(
                [
                    u.text
                    for u in ordered[
                        bisect.bisect_left(midpoints, line.start) : bisect.bisect_left(midpoints, line.end)
                    ]
                ]
            ),
            mixed,
            overlapped[i],
        )
        for i, line in enumerate(lines)
    ]
    return Scored(windows, chunks)


def score_items(
    items: Sequence[Item], rows: Mapping[str, Sequence[ChunkHypothesis]], metric: Metric, view: View
) -> Scored:
    scored = [score_item(item, rows.get(item.id, ()), metric, view) for item in items]
    return Scored([w for s in scored for w in s.windows], [c for s in scored for c in s.chunks])


def summarize_corpus(corpus: str, metric: Metric, view: View, items: Sequence[Item], scored: Scored) -> CorpusScore:
    windows = scored.windows
    units = sum(w.counts.reference_units for w in windows)
    errors = sum(w.counts.errors for w in windows)
    judged = [w for w in windows if w.judged]
    mixed = [w for w in judged if w.mixed]
    found = sum(w.incident for w in judged)
    found_mixed = sum(w.incident for w in mixed)
    chunks = [c for c in scored.chunks if c.judged]
    eligible = [c for c in chunks if c.dominated and c.mixed_recording]
    chunk_found = sum(c.incident for c in chunks)
    eligible_found = sum(c.incident for c in eligible)
    clean = [c for c in chunks if c.clean]
    clean_found = sum(c.incident for c in clean)
    return CorpusScore(
        corpus=corpus,
        metric=metric,
        view=view,
        items=len(items),
        audio_hours=sum(i.duration_s for i in items) / 3600,
        windows=len(windows),
        reference_units=units,
        errors=errors,
        rate=errors / units if units else None,
        window_incidents=Incidents(
            chunks=len(judged),
            mixed_chunks=len(mixed),
            incidents=found,
            incidents_in_mixed=found_mixed,
            rate=found / len(judged) if judged else None,
            rate_in_mixed=found_mixed / len(mixed) if mixed else None,
        ),
        chunk_incidents=ChunkIncidents(
            chunks=len(chunks),
            incidents=chunk_found,
            rate=chunk_found / len(chunks) if chunks else None,
            dominated_in_mixed_recordings=len(eligible),
            incidents_in_mixed_recordings=eligible_found,
            rate_in_mixed_recordings=eligible_found / len(eligible) if eligible else None,
            clean_in_mixed_recordings=len(clean),
            clean_incidents_in_mixed_recordings=clean_found,
            clean_rate_in_mixed_recordings=clean_found / len(clean) if clean else None,
        ),
    )


def _total(counts: Sequence[ErrorCounts]) -> ErrorCounts:
    return ErrorCounts(
        sum(c.substitutions for c in counts),
        sum(c.deletions for c in counts),
        sum(c.insertions for c in counts),
        sum(c.hits for c in counts),
    )


def paired(a: Sequence[WindowScore], b: Sequence[WindowScore], seed: int = 0) -> PairedResult:
    """Bootstrap the pooled error-rate difference ``a - b`` over windows both hypotheses were scored on.

    The decision interval resamples whole recordings; the window-level one is reported beside it.
    """
    by_a = {(w.item, w.window): w for w in a}
    by_b = {(w.item, w.window): w for w in b}
    if by_a.keys() != by_b.keys():
        raise ConfigError("the two hypotheses were scored over different windows; finish both runs first")
    keys = sorted(by_a)
    left = [by_a[k].counts for k in keys]
    right = [by_b[k].counts for k in keys]
    if _total(left).reference_units == 0:
        raise ConfigError("the reference has no units to score against")
    recordings = sorted({item for item, _ in keys})
    clustered: Interval | None = None
    if len(recordings) >= MIN_CLUSTERS:
        grouped = [
            (_total([by_a[k].counts for k in keys if k[0] == r]), _total([by_b[k].counts for k in keys if k[0] == r]))
            for r in recordings
        ]
        result = paired_bootstrap([x for x, _ in grouped], [y for _, y in grouped], seed)
        clustered = Interval(low=result.low, high=result.high, excludes_zero=result.excludes_zero)
    windowed = paired_bootstrap(left, right, seed)
    return PairedResult(
        windows=len(keys),
        recordings=len(recordings),
        a_rate=_total(left).rate,
        b_rate=_total(right).rate,
        difference=windowed.difference,
        decision=clustered,
        by_window=Interval(low=windowed.low, high=windowed.high, excludes_zero=windowed.excludes_zero),
    )


def paired_incidents(a: Sequence[ChunkScore], b: Sequence[ChunkScore]) -> PairedIncidents:
    """Compare the translation incidents of ``a`` and ``b`` on the clean lines both were scored on."""
    by_a = {(c.item, c.index): c for c in a}
    by_b = {(c.item, c.index): c for c in b}
    if by_a.keys() != by_b.keys():
        raise ConfigError("the two hypotheses were scored over different lines; finish both runs first")
    keys = [k for k in sorted(by_a) if by_a[k].clean]
    fixed = sum(by_b[k].incident and not by_a[k].incident for k in keys)
    introduced = sum(by_a[k].incident and not by_b[k].incident for k in keys)
    return PairedIncidents(
        lines=len(keys),
        a_incidents=sum(by_a[k].incident for k in keys),
        b_incidents=sum(by_b[k].incident for k in keys),
        fixed=fixed,
        introduced=introduced,
        p=sign_test(fixed, introduced),
    )


def pool(scored: Mapping[str, Scored], corpora: Sequence[str]) -> Scored:
    """Merge corpora into one ``Scored``, prefixing each item with its corpus so recordings never collide."""
    return Scored(
        [replace(w, item=f"{c}/{w.item}") for c in corpora for w in scored[c].windows],
        [replace(k, item=f"{c}/{k.item}") for c in corpora for k in scored[c].chunks],
    )


def special_token_rows(rows: Iterable[ChunkHypothesis]) -> int:
    """Rows whose text still holds a Whisper special token such as ``<|en|>``."""
    return sum(SPECIAL_TOKEN_MARK in row.text for row in rows)


def words_from_rows(rows: Sequence[ChunkHypothesis]) -> list[Word]:
    """Every word of the rows in time order; a row with text but no word times cannot feed WDER."""
    for row in rows:
        if row.text.strip() and not row.words:
            raise ConfigError(f"{row.hypothesis} gave no word times for {row.file}; WDER needs a Whisper hypothesis")
    return sorted((w for row in rows for w in row.words), key=lambda w: (w.start, w.end))


def write_words(path: Path, words: Sequence[Word]) -> None:
    """Write ``words.json`` in the format ``evals.run_diarization.load_words`` reads."""
    records = [{"text": w.text, "start": w.start, "end": w.end} for w in words]
    atomic.write_text(path, json.dumps(records, ensure_ascii=False) + "\n")


def write_words_for(items: Sequence[Item], rows: Mapping[str, Sequence[ChunkHypothesis]]) -> int:
    """Write ``words.json`` next to each item's audio; return how many were written."""
    if not all(i.corpus in FOLDER_CORPORA for i in items):
        raise ConfigError("words.json sits next to a recording's own audio; the ASCEND clips share one folder")
    written = 0
    for item in items:
        if item.id in rows:
            write_words(item.wav.parent / WORDS_FILE, words_from_rows(rows[item.id]))
            written += 1
    return written


def language_seconds(rows: Sequence[ChunkHypothesis]) -> dict[str, float]:
    """Seconds decoded per language the engine reported ('none' where it reported none)."""
    seconds: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        seconds[row.language or "none"] += row.end - row.start
    return dict(seconds)
