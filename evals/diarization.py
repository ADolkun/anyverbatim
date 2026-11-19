"""Diarization scoring: RTTM I/O, DER through ``pyannote.metrics``, and word-level speaker error (WDER).

Everything here is pure computation on turns and timed words; nothing runs a diarizer.
"""

from __future__ import annotations

import bisect
import itertools
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pyannote.core import Annotation, Segment, Timeline

from anyverbatim import atomic
from anyverbatim.contract.models import Word
from anyverbatim.errors import ConfigError
from evals.metrics import percentile_index

if TYPE_CHECKING:
    from pyannote.metrics.diarization import DiarizationErrorRate

# One-sided forgiveness around each reference boundary, the NIST/dscore reading of "collar 0.25". pyannote.metrics
# counts the full width (half before, half after a boundary), so it is handed twice this.
COLLARS_S: tuple[float, ...] = (0.0, 0.25)
MIN_WORD_COVERAGE = 0.5
MIN_CANDIDATES = 2
_MIN_SPAN_S = 1e-3
_RTTM_FIELDS = 9
_SPEAKER_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Turn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_rttm(text: str) -> dict[str, list[Turn]]:
    """Read ``SPEAKER`` lines into turns per recording id, in file order; other record types are ignored."""
    recordings: dict[str, list[Turn]] = defaultdict(list)
    for number, line in enumerate(text.splitlines(), 1):
        fields = _SPEAKER_RE.split(line.strip())
        if not fields[0] or fields[0].startswith(";") or fields[0] != "SPEAKER":
            continue
        if len(fields) < _RTTM_FIELDS:
            raise ConfigError(f"RTTM line {number} has {len(fields)} fields, expected at least {_RTTM_FIELDS}")
        try:
            start, duration = float(fields[3]), float(fields[4])
        except ValueError:
            raise ConfigError(f"RTTM line {number} has a non-numeric time") from None
        if duration < 0:
            raise ConfigError(f"RTTM line {number} has a negative duration")
        recordings[fields[1]].append(Turn(start, start + duration, fields[7]))
    return dict(recordings)


def read_rttm(path: Path) -> dict[str, list[Turn]]:
    return parse_rttm(path.read_text(encoding="utf-8"))


def format_rttm(uri: str, turns: Iterable[Turn]) -> str:
    """Render ``SPEAKER`` lines; a label or id with whitespace would split into extra fields, so it is refused."""
    turns = list(turns)
    for label in {uri, *(t.speaker for t in turns)}:
        if not label or _SPEAKER_RE.search(label):
            raise ConfigError(f"RTTM label {label!r} is empty or contains whitespace")
    return "".join(
        f"SPEAKER {uri} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>\n"
        for t in sorted(turns, key=lambda t: (t.start, t.end, t.speaker))
        if t.duration > 0
    )


def write_rttm(path: Path, uri: str, turns: Iterable[Turn]) -> None:
    atomic.write_text(path, format_rttm(uri, turns))


def speaker_count(turns: Iterable[Turn]) -> int:
    return len({t.speaker for t in turns})


def speaker_count_error(reference: Sequence[Turn], hypothesis: Sequence[Turn]) -> int:
    """Hypothesis speakers minus reference speakers: positive over-counts."""
    return speaker_count(hypothesis) - speaker_count(reference)


def _der_metric(*, collar: float = 0.0, skip_overlap: bool = False) -> DiarizationErrorRate:
    # Imported on first use: pyannote.metrics pulls in scipy and pandas, seconds on a cold start, and every test worker
    # would pay that just to collect tests.
    from pyannote.metrics.diarization import DiarizationErrorRate  # noqa: PLC0415

    return DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)


def _annotation(turns: Iterable[Turn]) -> Annotation:
    annotation = Annotation()
    # The index keeps two speakers with identical boundaries apart; keyed by segment alone, the second would replace
    # the first.
    for index, turn in enumerate(turns):
        if turn.duration > 0:
            annotation[Segment(turn.start, turn.end), index] = turn.speaker
    return annotation


@dataclass(frozen=True, slots=True)
class DerScore:
    """Error time in seconds; ``total`` is the scored reference speech."""

    total: float
    missed: float
    false_alarm: float
    confusion: float

    @property
    def errors(self) -> float:
        return self.missed + self.false_alarm + self.confusion

    @property
    def rate(self) -> float:
        return self.errors / self.total if self.total else float("nan")


def der(
    reference: Sequence[Turn],
    hypothesis: Sequence[Turn],
    *,
    collar_s: float = 0.0,
    skip_overlap: bool = False,
) -> DerScore:
    """Score one recording over the union of the reference and hypothesis extents."""
    if not reference:
        raise ConfigError("the reference has no turns")
    truth, guess = _annotation(reference), _annotation(hypothesis)
    uem = Timeline([truth.get_timeline().extent() | guess.get_timeline().extent()])
    metric = _der_metric(collar=2 * collar_s, skip_overlap=skip_overlap)
    detail = cast(Mapping[str, float], metric(truth, guess, uem=uem, detailed=True))
    return DerScore(
        total=detail["total"],
        missed=detail["missed detection"],
        false_alarm=detail["false alarm"],
        confusion=detail["confusion"],
    )


def macro_rate(scores: Sequence[DerScore]) -> float:
    """Mean of the per-recording rates: every recording counts once, whatever its length."""
    if not scores:
        raise ValueError("no scores")
    return sum(s.rate for s in scores) / len(scores)


def pooled_rate(scores: Sequence[DerScore]) -> float:
    """Total error time over total reference speech."""
    total = sum(s.total for s in scores)
    return sum(s.errors for s in scores) / total if total else float("nan")


@dataclass(frozen=True, slots=True)
class MacroBootstrap:
    """Mean per-recording rate difference (a minus b) with its percentile interval."""

    difference: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_macro_bootstrap(
    a: Sequence[float], b: Sequence[float], seed: int, resamples: int = 2000, confidence: float = 0.95
) -> MacroBootstrap:
    """Resample recordings as pairs and average their rate differences, so every recording counts once.

    The pooled bootstrap in ``evals.metrics`` weighs recordings by length, which is not the macro average E1 uses.
    """
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equally long")
    gaps = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(gaps, k=len(gaps))) / len(gaps) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return MacroBootstrap(
        sum(gaps) / len(gaps),
        means[percentile_index(tail, resamples)],
        means[percentile_index(1 - tail, resamples)],
    )


@dataclass(frozen=True, slots=True)
class WderScore:
    errors: int
    eligible: int
    total_words: int

    @property
    def rate(self) -> float:
        return self.errors / self.eligible if self.eligible else float("nan")


def _merged(turns: Iterable[Turn]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for turn in sorted(turns, key=lambda t: t.start):
        if turn.duration <= 0:
            continue
        if spans and turn.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], turn.end))
        else:
            spans.append((turn.start, turn.end))
    return spans


def _covered(spans: Sequence[tuple[float, float]], starts: Sequence[float], start: float, end: float) -> float:
    covered = 0.0
    index = max(bisect.bisect_right(starts, start) - 1, 0)
    for span_start, span_end in spans[index:]:
        if span_start >= end:
            break
        covered += max(0.0, min(end, span_end) - max(start, span_start))
    return covered


def single_speaker_words(reference: Sequence[Turn], words: Sequence[Word]) -> dict[int, str]:
    """Index of every word that lies in a single-speaker reference region, with that speaker.

    A word qualifies when exactly one reference speaker is active anywhere in its span and that speaker covers at
    least half of it, so words in overlap or in reference silence never count for or against a system.
    """
    merged = {
        speaker: _merged(t for t in reference if t.speaker == speaker) for speaker in {t.speaker for t in reference}
    }
    starts = {speaker: [s for s, _ in spans] for speaker, spans in merged.items()}
    eligible: dict[int, str] = {}
    for index, word in enumerate(words):
        span = max(word.end - word.start, _MIN_SPAN_S)
        overlaps = {
            speaker: _covered(merged[speaker], starts[speaker], word.start, word.start + span) for speaker in merged
        }
        active = [speaker for speaker, seconds in overlaps.items() if seconds > 0]
        if len(active) == 1 and overlaps[active[0]] >= MIN_WORD_COVERAGE * span:
            eligible[index] = active[0]
    return eligible


def wder(reference: Sequence[Turn], words: Sequence[Word], assigned: Sequence[str | None]) -> WderScore:
    """Fraction of eligible words whose assigned speaker differs from the reference after the best one-to-one mapping.

    ``assigned[i]`` is the speaker of ``words[i]``, or None when no speaker was given; a None is always wrong and
    takes no part in the mapping. The mapping maximises the eligible word time the two labellings share. A
    hypothesis speaker left without a partner counts as wrong, so extra speakers cost accuracy.
    """
    if len(words) != len(assigned):
        raise ValueError(f"{len(words)} words but {len(assigned)} assigned speakers")
    eligible = single_speaker_words(reference, words)
    if not eligible:
        return WderScore(0, 0, len(words))
    truth = Annotation()
    guess = Annotation()
    for index, speaker in eligible.items():
        label = assigned[index]
        if label is None:
            continue
        word = words[index]
        segment = Segment(word.start, word.start + max(word.end - word.start, _MIN_SPAN_S))
        truth[segment, index] = speaker
        guess[segment, index] = label
    mapping = cast(Mapping[str, str], cast(Any, _der_metric()).optimal_mapping(truth, guess))
    errors = sum(
        assigned[index] is None or mapping.get(cast(str, assigned[index])) != speaker
        for index, speaker in eligible.items()
    )
    return WderScore(errors, len(eligible), len(words))


_Piece = tuple[float, float, dict[str, frozenset[str]]]


@dataclass(frozen=True, slots=True)
class Dispute:
    start: float
    end: float
    # Each candidate's mapped speakers over the longest stretch of the run.
    speakers: Mapping[str, frozenset[str]]

    @property
    def duration(self) -> float:
        return self.end - self.start


def align_labels(candidates: Mapping[str, Sequence[Turn]]) -> dict[str, list[Turn]]:
    """Relabel every candidate's turns in the first candidate's speaker names, by best overlap.

    A speaker with no partner keeps a label of its own (``<candidate>:<speaker>``).
    """
    names = list(candidates)
    base = _annotation(candidates[names[0]])
    mapped: dict[str, list[Turn]] = {names[0]: list(candidates[names[0]])}
    metric = cast(Any, _der_metric())
    for name in names[1:]:
        mapping = cast(Mapping[str, str], metric.optimal_mapping(base, _annotation(candidates[name])))
        mapped[name] = [Turn(t.start, t.end, mapping.get(t.speaker, f"{name}:{t.speaker}")) for t in candidates[name]]
    return mapped


def disagreement_runs(candidates: Mapping[str, Sequence[Turn]], *, min_run_s: float = 2.0) -> list[Dispute]:
    """Find runs of at least ``min_run_s`` where the candidates name different speaker sets, after mapping.

    The first candidate's labels are the common namespace; every other candidate is mapped onto them by overlap. Time
    where only some candidates are silent is not a dispute about who speaks, so a stretch counts only when at least
    two candidates name different non-empty sets.
    """
    if len(candidates) < MIN_CANDIDATES:
        raise ValueError("a dispute needs at least two candidates")
    mapped = align_labels(candidates)
    edges = sorted({edge for turns in mapped.values() for t in turns for edge in (t.start, t.end)})
    runs: list[Dispute] = []
    run: list[_Piece] = []
    for left, right in itertools.pairwise(edges):
        midpoint = (left + right) / 2
        labels = {
            name: frozenset(t.speaker for t in turns if t.start <= midpoint < t.end) for name, turns in mapped.items()
        }
        if len({value for value in labels.values() if value}) > 1:
            run.append((left, right, labels))
            continue
        _flush(runs, run, min_run_s)
        run = []
    _flush(runs, run, min_run_s)
    return runs


def _flush(runs: list[Dispute], run: Sequence[_Piece], min_run_s: float) -> None:
    if run and run[-1][1] - run[0][0] >= min_run_s:
        longest = max(run, key=lambda piece: piece[1] - piece[0])
        runs.append(Dispute(run[0][0], run[-1][1], longest[2]))
