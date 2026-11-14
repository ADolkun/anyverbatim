import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from evals.diarization import Turn
from evals.recordings_data import CorpusIndex, RefUtterance, wav_bytes, write_index, write_recording
from evals.run_diarization import (
    ExternalBackend,
    Recording,
    Run,
    attribute_by_turns,
    load_words,
    make_backend,
    recordings_of,
    run_corpora,
    score_recording,
    summarize,
)
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError

SECONDS = 40
REFERENCE = [RefUtterance(0, 10, "A", "one"), RefUtterance(10, 20, "B", "two"), RefUtterance(20, 30, "A", "three")]


class FakeBackend:
    def __init__(self, turns: Sequence[Turn], smoothed_labels: list[str] | None = None) -> None:
        self.turns = tuple(turns)
        self.calls = 0
        self.smoothed_labels = smoothed_labels

    def diarize(self, recording: Recording, out_dir: Path) -> Run:
        self.calls += 1
        return Run(
            self.turns, diarize_s=recording.duration_s * 0.05, load_s=1.0, peak_footprint_bytes=2**30, handle=None
        )

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str]:
        labels = ["x" if w.start < 10 else "y" if w.start < 20 else "x" for w in words]
        if not smoothing and self.smoothed_labels is not None:
            return self.smoothed_labels
        return labels


def make_corpus(root: Path, corpus: str = "ami", name: str = "m1") -> Recording:
    audio = wav_bytes(np.zeros(16_000 * SECONDS, dtype="<i2"))
    entry = write_recording(root / corpus, name, audio, REFERENCE)
    write_index(root / corpus, CorpusIndex(corpus=corpus, source="s", license="l", recordings=(entry,)))
    return recordings_of(corpus, root=root)[0]


def test_score_recording_scores_der_at_both_collars_count_rtf_and_footprint(tmp_path: Path):
    recording = make_corpus(tmp_path)
    hypothesis = [Turn(0, 10, "x"), Turn(10.2, 20, "y"), Turn(20, 30, "x"), Turn(30, 32, "z")]
    score = score_recording(FakeBackend(hypothesis), recording, tmp_path / "out")
    strict, forgiving = score.der
    assert (strict.collar_s, forgiving.collar_s) == (0.0, 0.25)
    assert strict.total == pytest.approx(30)
    assert strict.missed == pytest.approx(0.2)
    assert strict.false_alarm == pytest.approx(2)
    assert strict.rate == pytest.approx(2.2 / 30)
    assert forgiving.rate < strict.rate
    assert (score.reference_speakers, score.hypothesis_speakers, score.speaker_count_error) == (2, 3, 1)
    assert score.rtf == pytest.approx(0.05)
    assert score.peak_footprint_bytes == 2**30
    assert score.wder_raw is None and score.wder_smoothed is None
    assert (tmp_path / "out" / "hypothesis.rttm").is_file()


def test_wder_is_reported_with_and_without_smoothing_when_words_exist(tmp_path: Path):
    recording = make_corpus(tmp_path)
    words = [{"text": "w", "start": float(t), "end": t + 0.5} for t in (1, 3, 5, 11, 13, 15, 21, 23)]
    recording.words.write_text(json.dumps(words), encoding="utf-8")
    raw = ["x", "x", "y", "y", "y", "y", "x", "x"]
    backend = FakeBackend([Turn(0, 30, "x")], smoothed_labels=raw)
    score = score_recording(backend, recording, tmp_path / "out")
    assert score.wder_smoothed is not None and score.wder_raw is not None
    assert score.wder_smoothed.rate == 0
    assert score.wder_raw.errors == 1
    assert score.wder_raw.eligible == 8


def test_load_words_is_none_without_a_file(tmp_path: Path):
    assert load_words(tmp_path / "words.json") is None


def test_run_corpora_resumes_from_stored_scores(tmp_path: Path):
    recording = make_corpus(tmp_path / "data")
    backend = FakeBackend([Turn(0, 10, "x"), Turn(10, 20, "y"), Turn(20, 30, "x")])
    first = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    second = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    assert backend.calls == 1
    assert first == second
    assert first[0].der[0].rate == 0


def test_summarize_reports_macro_and_pooled_der_by_corpus_without_recording_ids(tmp_path: Path):
    good = make_corpus(tmp_path / "data", "ami", "m1")
    other = make_corpus(tmp_path / "data", "alimeeting", "m2")
    perfect = FakeBackend([Turn(0, 10, "x"), Turn(10, 20, "y"), Turn(20, 30, "x")])
    bad = FakeBackend([Turn(0, 30, "x")])
    scores = [score_recording(perfect, good, tmp_path / "a"), score_recording(bad, other, tmp_path / "b")]
    summary = summarize(scores)
    corpora = summary["corpora"]
    assert isinstance(corpora, dict)
    assert corpora["ami"]["der"]["collar_0.0"]["macro"] == 0
    assert corpora["alimeeting"]["der"]["collar_0.0"]["macro"] == pytest.approx(10 / 30)
    assert corpora["alimeeting"]["mean_signed_speaker_count_error"] == -1
    assert summary["primary_macro_der_collar_0"] == pytest.approx(10 / 30 / 2)
    assert "m1" not in json.dumps(summary) and "m2" not in json.dumps(summary)
    with pytest.raises(ConfigError, match="no recordings"):
        summarize([])


def test_recordings_of_filters_and_rejects_unknown_ids(tmp_path: Path):
    make_corpus(tmp_path, "ascend", "ses1")
    assert [r.id for r in recordings_of("ascend", ["ses1"], root=tmp_path)] == ["ses1"]
    with pytest.raises(ConfigError, match="no recording"):
        recordings_of("ascend", ["nope"], root=tmp_path)


def test_attribute_by_turns_uses_overlap_then_nearest_then_previous():
    turns = [Turn(0, 10, "A"), Turn(10, 20, "B"), Turn(7, 12, "C")]
    words = [
        Word(text="a", start=1, end=2),
        Word(text="b", start=6, end=9.5),
        Word(text="c", start=14, end=15),
        Word(text="d", start=20.3, end=20.6),
        Word(text="e", start=30, end=31),
    ]
    assert attribute_by_turns(words, turns) == ["A", "A", "B", "B", "B"]


def test_external_backend_needs_a_command_and_pyannote_ids_route_to_it():
    with pytest.raises(ConfigError, match="needs a command"):
        make_backend("d-pyannote-c1", None)
    assert isinstance(make_backend("d-pyannote-c1", "uv run python x.py"), ExternalBackend)
