import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from anyverbatim.contract.models import Word
from anyverbatim.diarize import registry as diarize_registry
from anyverbatim.diarize import worker
from anyverbatim.diarize.protocol import DiarizerDef, DiarizeRequest, RawDiarization
from anyverbatim.diarize.protocol import Turn as RawTurn
from anyverbatim.diarize.runner import DiarizeResult
from anyverbatim.errors import ConfigError, EngineError
from evals.diarization import Turn
from evals.recordings_data import CorpusIndex, RefUtterance, wav_bytes, write_index, write_recording
from evals.run_diarization import (
    ExternalBackend,
    ProductionBackend,
    Recording,
    RecordingScore,
    Run,
    _turns_for,
    attribute_by_turns,
    compare,
    der_rates,
    input_digest,
    load_words,
    make_backend,
    recordings_of,
    run_corpora,
    score_recording,
    summarize,
)

SECONDS = 40
REFERENCE = [RefUtterance(0, 10, "A", "one"), RefUtterance(10, 20, "B", "two"), RefUtterance(20, 30, "A", "three")]


class FakeBackend:
    def __init__(self, turns: Sequence[Turn], smoothed_labels: list[str | None] | None = None) -> None:
        self.turns = tuple(turns)
        self.calls = 0
        self.smoothed_labels = smoothed_labels

    def diarize(self, recording: Recording, out_dir: Path) -> Run:
        self.calls += 1
        return Run(
            self.turns, diarize_s=recording.duration_s * 0.05, load_s=1.0, peak_footprint_bytes=2**30, handle=None
        )

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str | None]:
        labels: list[str | None] = ["x" if w.start < 10 else "y" if w.start < 20 else "x" for w in words]
        if not smoothing and self.smoothed_labels is not None:
            return self.smoothed_labels
        return labels


def make_corpus(root: Path, corpus: str = "ami", name: str = "m1") -> Recording:
    audio = wav_bytes(np.zeros(16_000 * SECONDS, dtype="<i2"))
    entry = write_recording(root / corpus, name, audio, REFERENCE)
    write_index(root / corpus, CorpusIndex(corpus=corpus, source="s", license="l", recordings=(entry,)))
    return recordings_of(corpus, root=root)[0]


def test_score_recording_scores_der_at_both_collars_count_rtf_and_footprint(tmp_path: Path) -> None:
    recording = make_corpus(tmp_path)
    hypothesis = [Turn(0, 10, "x"), Turn(10.2, 20, "y"), Turn(20, 30, "x"), Turn(30, 32, "z")]
    score = score_recording(FakeBackend(hypothesis), recording, tmp_path / "out")
    strict, forgiving = score.der
    assert (strict.collar_s, forgiving.collar_s) == (0.0, 0.25)
    assert strict.total == pytest.approx(30)
    assert strict.missed == pytest.approx(0.2)
    assert strict.false_alarm == pytest.approx(2)
    assert strict.rate == pytest.approx(2.2 / 30)
    assert forgiving.rate is not None
    assert strict.rate is not None
    assert forgiving.rate < strict.rate
    assert (score.reference_speakers, score.hypothesis_speakers, score.speaker_count_error) == (2, 3, 1)
    assert score.rtf == pytest.approx(0.05)
    assert score.peak_footprint_bytes == 2**30
    assert score.wder_raw is None
    assert score.wder_smoothed is None
    assert (tmp_path / "out" / "hypothesis.rttm").is_file()


def test_wder_is_reported_with_and_without_smoothing_when_words_exist(tmp_path: Path) -> None:
    recording = make_corpus(tmp_path)
    words = [{"text": "w", "start": float(t), "end": t + 0.5} for t in (1, 3, 5, 11, 13, 15, 21, 23)]
    recording.words.write_text(json.dumps(words), encoding="utf-8")
    raw: list[str | None] = ["x", "x", "y", "y", "y", "y", "x", "x"]
    backend = FakeBackend([Turn(0, 30, "x")], smoothed_labels=raw)
    score = score_recording(backend, recording, tmp_path / "out")
    assert score.wder_smoothed is not None
    assert score.wder_raw is not None
    assert score.wder_smoothed.rate == 0
    assert score.wder_raw.errors == 1
    assert score.wder_raw.eligible == 8


def test_load_words_is_none_without_a_file(tmp_path: Path) -> None:
    assert load_words(tmp_path / "words.json") is None


def test_run_corpora_resumes_from_stored_scores(tmp_path: Path) -> None:
    recording = make_corpus(tmp_path / "data")
    backend = FakeBackend([Turn(0, 10, "x"), Turn(10, 20, "y"), Turn(20, 30, "x")])
    first = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    second = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    assert backend.calls == 1
    assert first == second
    assert first[0].der[0].rate == 0


def test_summarize_reports_macro_and_pooled_der_by_corpus_without_recording_ids(tmp_path: Path) -> None:
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
    assert "m1" not in json.dumps(summary)
    assert "m2" not in json.dumps(summary)
    with pytest.raises(ConfigError, match="no recordings"):
        summarize([])


def test_recordings_of_filters_and_rejects_unknown_ids(tmp_path: Path) -> None:
    make_corpus(tmp_path, "ascend", "ses1")
    assert [r.id for r in recordings_of("ascend", ["ses1"], root=tmp_path)] == ["ses1"]
    with pytest.raises(ConfigError, match="no recording"):
        recordings_of("ascend", ["nope"], root=tmp_path)


def test_attribute_by_turns_uses_overlap_then_nearest_then_previous() -> None:
    turns = [Turn(0, 10, "A"), Turn(10, 20, "B"), Turn(7, 12, "C")]
    words = [
        Word(text="a", start=1, end=2),
        Word(text="b", start=6, end=9.5),
        Word(text="c", start=14, end=15),
        Word(text="d", start=20.3, end=20.6),
        Word(text="e", start=30, end=31),
    ]
    assert attribute_by_turns(words, turns) == ["A", "A", "B", "B", "B"]


def test_external_backend_needs_a_command_and_pyannote_ids_route_to_it() -> None:
    with pytest.raises(ConfigError, match="needs a command"):
        make_backend("d-pyannote-c1", None)
    assert isinstance(make_backend("d-pyannote-c1", "uv run python x.py"), ExternalBackend)


def test_a_nan_rate_is_stored_as_none_and_survives_the_round_trip(tmp_path: Path) -> None:
    recording = make_corpus(tmp_path / "data")
    # Every word sits in the overlap of two reference speakers, so no word is eligible and WDER is undefined.
    both = [RefUtterance(0, 30, "A", "one"), RefUtterance(0, 30, "B", "two")]
    write_recording(tmp_path / "data" / "ami", "m1", wav_bytes(np.zeros(16_000 * SECONDS, dtype="<i2")), both)
    recording.words.write_text(json.dumps([{"text": "w", "start": 1.0, "end": 2.0}]), encoding="utf-8")
    backend = FakeBackend([Turn(0, 30, "x")])
    (first,) = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    assert first.wder_raw is not None
    assert first.wder_raw.rate is None
    (second,) = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    assert second == first
    assert backend.calls == 1


def test_a_cached_score_is_recomputed_when_the_words_appear_but_keeps_its_timing(tmp_path: Path) -> None:
    recording = make_corpus(tmp_path / "data")
    backend = FakeBackend([Turn(0, 10, "x"), Turn(10, 20, "y"), Turn(20, 30, "x")])
    (before,) = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    assert before.wder_raw is None
    words = [{"text": "w", "start": float(t), "end": t + 0.5} for t in (1, 11, 21)]
    recording.words.write_text(json.dumps(words), encoding="utf-8")
    (after,) = run_corpora(backend, "d-fake", [recording], tmp_path / "runs")
    assert after.wder_raw is not None
    assert after.wder_smoothed is not None
    assert backend.calls == 2
    assert (after.diarize_s, after.rtf, after.peak_footprint_bytes) == (
        before.diarize_s,
        before.rtf,
        before.peak_footprint_bytes,
    )
    assert after.inputs != before.inputs == input_digest(make_corpus(tmp_path / "other"))


def test_attributed_speaker_counts_and_unattributed_words_are_reported(tmp_path: Path) -> None:
    recording = make_corpus(tmp_path)
    words = [{"text": "w", "start": float(t), "end": t + 0.5} for t in (1, 3, 11, 13, 21)]
    recording.words.write_text(json.dumps(words), encoding="utf-8")
    labels = ["x", "x", "y", "y", None]
    backend = FakeBackend([Turn(0, 30, "x")], smoothed_labels=labels)
    score = score_recording(backend, recording, tmp_path / "out")
    assert score.wder_raw is not None
    assert score.wder_raw.errors == 1
    assert (score.attributed_speakers_raw, score.attributed_speakers_smoothed) == (2, 2)


def test_primary_macro_needs_both_corpora_and_smoothed_count_error_is_reported(tmp_path: Path) -> None:
    ami = make_corpus(tmp_path / "data", "ami", "m1")
    only = score_recording(FakeBackend([Turn(0, 30, "x")]), ami, tmp_path / "a")
    summary = summarize([only])
    assert summary["primary_macro_der_collar_0"] is None
    corpora = summary["corpora"]
    assert isinstance(corpora, dict)
    assert corpora["ami"]["mean_abs_smoothed_speaker_count_error"] is None
    ami.words.write_text(json.dumps([{"text": "w", "start": 1.0, "end": 2.0}]), encoding="utf-8")
    scored = score_recording(FakeBackend([Turn(0, 30, "x")]), ami, tmp_path / "b")
    again = summarize([scored])["corpora"]
    assert isinstance(again, dict)
    assert again["ami"]["mean_abs_smoothed_speaker_count_error"] == 1


def test_compare_pairs_recordings_across_two_runs(tmp_path: Path) -> None:
    scores: dict[str, list[RecordingScore]] = {"good": [], "bad": []}
    for corpus in ("ami", "alimeeting"):
        for name in ("m1", "m2"):
            recording = make_corpus(tmp_path / "data", corpus, name)
            scores["good"].append(
                score_recording(
                    FakeBackend([Turn(0, 10, "x"), Turn(10, 20, "y"), Turn(20, 30, "x")]),
                    recording,
                    tmp_path / "g" / name,
                )
            )
            scores["bad"].append(score_recording(FakeBackend([Turn(0, 30, "x")]), recording, tmp_path / "b" / name))
    assert set(der_rates(scores["good"])) == {"ami/m1", "ami/m2", "alimeeting/m1", "alimeeting/m2"}
    result = compare(scores["bad"], scores["good"])
    assert result.difference > 0
    with pytest.raises(ConfigError, match="different primary recordings"):
        compare(scores["bad"][:2], scores["good"])


def test_external_rttm_with_the_wrong_uri_is_never_read_as_no_speech(tmp_path: Path) -> None:
    path = tmp_path / "o.rttm"
    path.write_text("SPEAKER audio 1 0.000 5.000 <NA> <NA> S0 <NA> <NA>\n", encoding="utf-8")
    assert _turns_for(path, "R8001_M8004") == (Turn(0.0, 5.0, "S0"),)
    path.write_text(
        "SPEAKER one 1 0.000 5.000 <NA> <NA> S0 <NA> <NA>\nSPEAKER two 1 0.000 5.000 <NA> <NA> S0 <NA> <NA>\n",
        encoding="utf-8",
    )
    with pytest.raises(EngineError, match="none for R8001_M8004"):
        _turns_for(path, "R8001_M8004")
    assert _turns_for(path, "one") == (Turn(0.0, 5.0, "S0"),)
    path.write_text("", encoding="utf-8")
    assert _turns_for(path, "x") == ()


def test_production_backend_reads_what_the_real_diarize_worker_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract check against anyverbatim.diarize with a fake model: no weights, no subprocess."""

    class FakeModel:
        def load(self) -> None:
            return None

        def diarize(self, audio: np.ndarray) -> RawDiarization:
            frames = round(len(audio) / 16_000 / 0.01)
            probs = np.zeros((frames, 8), dtype=np.float32)
            probs[: frames // 2, 0] = 0.9
            probs[frames // 2 :, 1] = 0.8
            half = frames // 2 * 0.01
            turns = (RawTurn(start=0.0, end=half, speaker=0), RawTurn(start=half, end=frames * 0.01, speaker=1))
            return RawDiarization(turns=turns, probs=probs, frame_s=0.01)

    def fake_worker(request: DiarizeRequest) -> DiarizeResult:
        metrics = worker.run_request(request, diarizer=FakeModel())
        return DiarizeResult(Path(request.out_dir), metrics, 0, 0.1)

    def no_dependency_check(definition: DiarizerDef) -> None:
        return None

    monkeypatch.setattr(diarize_registry, "check_dependencies", no_dependency_check)
    recording = make_corpus(tmp_path / "data")
    words = [{"text": "w", "start": float(t), "end": t + 0.5} for t in (1, 3, 11, 13, 21, 23)]
    recording.words.write_text(json.dumps(words), encoding="utf-8")
    backend = ProductionBackend("d-nemotron-8bit", worker=fake_worker)
    score = score_recording(backend, recording, tmp_path / "out")
    assert score.hypothesis_speakers == 2
    assert score.wder_raw is not None
    assert score.wder_smoothed is not None
    assert score.wder_raw.eligible == 6
    assert (score.attributed_speakers_raw, score.attributed_speakers_smoothed) == (2, 2)
