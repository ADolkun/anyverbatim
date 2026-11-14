"""Score ASR runs over the recordings corpora, pair two hypotheses, and write the words WDER needs.

A recording with timed reference lines is cut into windows at pauses in the *reference*, so every hypothesis is scored
over exactly the same windows and any two can be paired. Each window's reference and hypothesis units are the ones
whose midpoint lies in it; a clip with only a reference text is one window. Everything here is pure computation on
rows and references.
"""

from __future__ import annotations

import bisect
import itertools
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from evals.consensus import Segment, TimedUnit, timed_units
from evals.metrics import (
    MIN_INCIDENT_UNITS,
    Bootstrap,
    ErrorCounts,
    cer,
    mer,
    paired_bootstrap,
    script_mix,
    translation_incident,
    wer,
)
from evals.normalize import Language, View, join_units, normalize, to_units
from evals.recordings_corpus import FOLDER_CORPORA, Item, Metric
from evals.recordings_data import WORDS_FILE, RefUtterance
from evals.store import hypothesis_units, row_text
from ytverbatim import atomic
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError

WINDOW_TARGET_S = 30.0
MEETINGS = "meetings"

_METRICS: Mapping[Metric, Callable[[str, str, Language, View], ErrorCounts]] = {"wer": wer, "cer": cer, "mer": mer}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True, slots=True)
class WindowScore:
    item: str
    window: int
    counts: ErrorCounts
    # The reference is long enough to judge a translation (see ``translation_incident``).
    chunk: bool
    mixed: bool
    incident: bool


class Incidents(_Frozen):
    chunks: int
    mixed_chunks: int
    incidents: int
    incidents_in_mixed: int
    rate: float | None
    rate_in_mixed: float | None


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
    incidents: Incidents


class PairedResult(_Frozen):
    windows: int
    a_rate: float
    b_rate: float
    difference: float
    low: float
    high: float
    excludes_zero: bool


def reference_cuts(utterances: Sequence[RefUtterance], target_s: float = WINDOW_TARGET_S) -> list[float]:
    """Window start times: 0, then the middle of the first pause in the reference after each ``target_s`` of speech.

    A window therefore never splits a reference line, and with no pause it simply runs long.
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
        if middle - cuts[-1] >= target_s:
            cuts.append(middle)
    return cuts


def _by_window(units: Sequence[TimedUnit], starts: Sequence[float]) -> list[list[str]]:
    grouped: list[list[str]] = [[] for _ in starts]
    for unit in units:
        grouped[max(bisect.bisect_right(starts, unit.midpoint_s) - 1, 0)].append(unit.text)
    return grouped


def _size(metric: Metric, text: str, language: Language, view: View) -> int:
    if metric == "wer":
        return len(normalize(text, language, view).split())
    units = to_units(text, language, view)
    return len(units) if metric == "mer" else len("".join(units))


def count_errors(metric: Metric, reference: str, hypothesis: str, language: Language, view: View) -> ErrorCounts:
    """Edit counts under ``metric``; a reference with nothing to say makes every hypothesis unit an insertion."""
    if _size(metric, reference, language, view) == 0:
        return ErrorCounts(0, 0, _size(metric, hypothesis, language, view), 0)
    return _METRICS[metric](reference, hypothesis, language, view)


def _window(
    item: str, index: int, reference: str, hypothesis: str, metric: Metric, language: Language, view: View
) -> WindowScore:
    ref_mix, hyp_mix = script_mix(reference), script_mix(hypothesis)
    chunk = ref_mix.total >= MIN_INCIDENT_UNITS
    return WindowScore(
        item,
        index,
        count_errors(metric, reference, hypothesis, language, view),
        chunk=chunk,
        mixed=chunk and ref_mix.mixed,
        incident=translation_incident(ref_mix, hyp_mix),
    )


def score_item(item: Item, rows: Sequence[ChunkHypothesis], metric: Metric, view: View) -> list[WindowScore]:
    reference = item.reference
    if reference is None:
        raise ConfigError(f"{item.corpus}/{item.id} has no reference to score against")
    if not reference.utterances:
        return [_window(item.id, 0, reference.text, row_text(rows), metric, item.language, view)]
    lines = [Segment(u.text, u.start, u.end) for u in reference.utterances if u.text.strip()]
    starts = reference_cuts(reference.utterances)
    # Units are split in the raw view and joined per window, so the view's normalization then reads the reference and
    # the hypothesis over the same stretch of speech; normalizing a whole row against per-line text would not.
    wanted = _by_window(timed_units(lines, item.language, "raw"), starts)
    heard = _by_window(hypothesis_units(rows, item.language, "raw"), starts)
    return [
        _window(item.id, i, join_units(ref), join_units(hyp), metric, item.language, view)
        for i, (ref, hyp) in enumerate(zip(wanted, heard, strict=True))
    ]


def score_items(
    items: Sequence[Item], rows: Mapping[str, Sequence[ChunkHypothesis]], metric: Metric, view: View
) -> list[WindowScore]:
    return [w for item in items for w in score_item(item, rows.get(item.id, ()), metric, view)]


def summarize_corpus(
    corpus: str, metric: Metric, view: View, items: Sequence[Item], windows: Sequence[WindowScore]
) -> CorpusScore:
    units = sum(w.counts.reference_units for w in windows)
    errors = sum(w.counts.errors for w in windows)
    chunks = [w for w in windows if w.chunk]
    mixed = [w for w in chunks if w.mixed]
    found = sum(w.incident for w in chunks)
    found_mixed = sum(w.incident for w in mixed)
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
        incidents=Incidents(
            chunks=len(chunks),
            mixed_chunks=len(mixed),
            incidents=found,
            incidents_in_mixed=found_mixed,
            rate=found / len(chunks) if chunks else None,
            rate_in_mixed=found_mixed / len(mixed) if mixed else None,
        ),
    )


def paired(a: Sequence[WindowScore], b: Sequence[WindowScore], seed: int = 0) -> PairedResult:
    """Bootstrap the pooled error-rate difference ``a - b`` over windows both hypotheses were scored on."""
    by_a = {(w.item, w.window): w for w in a}
    by_b = {(w.item, w.window): w for w in b}
    if by_a.keys() != by_b.keys():
        raise ConfigError("the two hypotheses were scored over different windows; finish both runs first")
    keys = sorted(by_a)
    left = [by_a[k].counts for k in keys]
    right = [by_b[k].counts for k in keys]
    result: Bootstrap = paired_bootstrap(left, right, seed)
    return PairedResult(
        windows=len(keys),
        a_rate=sum(c.errors for c in left) / sum(c.reference_units for c in left),
        b_rate=sum(c.errors for c in right) / sum(c.reference_units for c in right),
        difference=result.difference,
        low=result.low,
        high=result.high,
        excludes_zero=result.excludes_zero,
    )


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
