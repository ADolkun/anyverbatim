import json
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError
from recordings_fixture import make_owner, spoken

from anyverbatim.contract.models import Word
from anyverbatim.errors import ConfigError
from evals import disputes
from evals.diarization import Turn
from evals.recordings_corpus import load_items
from evals.recordings_data import INDEX, OWNER, OwnerEntry, load_owner, wav_seconds
from evals.recordings_scoring import write_words_for
from evals.run_diarization import (
    OwnerScore,
    Recording,
    Run,
    input_digest,
    load_words,
    recordings_of,
    run_owner,
    score_owner_recording,
    score_recording,
    summarize,
)


class FakeBackend:
    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns = tuple(turns)
        self.calls = 0

    def diarize(self, recording: Recording, out_dir: Path) -> Run:
        self.calls += 1
        return Run(self.turns, recording.duration_s * 0.05, 1.0, 2**30, None)

    def attribute(self, words: Sequence[Word], run: Run, *, smoothing: bool) -> list[str | None]:
        return ["x" if w.start < 30 else "y" for w in words]


TURNS = [Turn(0, 30, "x"), Turn(30, 60, "y"), Turn(60, 90, "z")]


def test_owner_ids_must_be_generic_labels_and_the_error_does_not_repeat_the_bad_one() -> None:
    assert OwnerEntry(id="rec-01").language == "zh+en"
    for bad in ("Alice-call", "rec-1", "rec-01-final"):
        with pytest.raises(ValidationError, match="generic labels") as raised:
            OwnerEntry(id=bad)
        assert bad not in raised.value.errors()[0]["msg"]


def test_the_owner_index_lists_recordings_with_their_durations_and_optional_speaker_counts(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 3}), ("rec-02", 30, {"language": "en"})])
    first, second = load_owner(tmp_path / OWNER)
    assert (first.entry.id, first.duration_s, first.entry.speakers) == ("rec-01", 90.0, 3)
    assert (second.entry.language, second.entry.speakers) == ("en", None)
    assert first.wav == tmp_path / OWNER / "rec-01" / "audio.wav"
    assert wav_seconds(first.wav) == 90.0


def test_the_owner_index_refuses_missing_files_duplicates_and_other_audio_formats(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no owner index"):
        load_owner(tmp_path / OWNER)
    make_owner(tmp_path, [("rec-01", 5, {})])
    folder = tmp_path / OWNER
    index = folder / INDEX
    index.write_text(json.dumps({"recordings": [{"id": "rec-02"}]}), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"rec-02/audio\.wav"):
        load_owner(folder)
    index.write_text(json.dumps({"recordings": [{"id": "rec-01"}] * 2}), encoding="utf-8")
    with pytest.raises(ConfigError, match="more than once"):
        load_owner(folder)
    (folder / "rec-01" / "audio.wav").write_bytes(b"not a wav")
    index.write_text(json.dumps({"recordings": [{"id": "rec-01"}]}), encoding="utf-8")
    with pytest.raises(ConfigError, match="rec-01 is not a readable WAV"):
        load_owner(folder)


def test_errors_name_the_generic_id_and_never_a_title_or_file_name(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 5, {})])
    folder = tmp_path / OWNER
    for text in (
        {"recordings": [{"id": "rec-01", "audio": "Alice review.m4a"}]},
        {"recordings": [{"id": "Alice-review"}]},
        {"recordings": [{"id": "rec-01", "speakers": "Alice"}]},
    ):
        (folder / INDEX).write_text(json.dumps(text), encoding="utf-8")
        with pytest.raises(ConfigError, match="owner index is invalid") as raised:
            load_owner(folder)
        assert "Alice" not in str(raised.value)
    with wave.open(str(folder / "rec-01" / "audio.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(b"\0\0" * 4)
    (folder / INDEX).write_text(json.dumps({"recordings": [{"id": "rec-01"}]}), encoding="utf-8")
    with pytest.raises(ConfigError, match="rec-01 must be 16 kHz") as raised:
        load_owner(folder)
    assert "audio.wav" not in str(raised.value)


def test_a_stereo_or_wrong_rate_wav_is_refused_with_the_conversion(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(b"\0\0" * 4)
    with pytest.raises(ConfigError, match="ffmpeg"):
        wav_seconds(path)


def test_an_owner_recording_has_no_reference_and_is_scored_by_speaker_count(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 2})])
    (recording,) = recordings_of(OWNER, root=tmp_path)
    assert recording.reference is None
    assert recording.expected_speakers == 2
    score = score_owner_recording(FakeBackend(TURNS), recording, tmp_path / "out")
    assert (score.hypothesis_speakers, score.expected_speakers, score.speaker_count_error) == (3, 2, 1)
    assert score.rtf == pytest.approx(0.05)
    assert score.attributed_speakers_smoothed is None
    assert (tmp_path / "out" / "hypothesis.rttm").is_file()
    with pytest.raises(ConfigError, match="no reference"):
        score_recording(FakeBackend(TURNS), recording, tmp_path / "other")


def test_without_the_owners_count_there_is_no_count_error(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 90, {})])
    (recording,) = recordings_of(OWNER, root=tmp_path)
    score = score_owner_recording(FakeBackend(TURNS), recording, tmp_path / "out")
    assert score.speaker_count_error is None
    assert score.expected_speakers is None


def test_owner_runs_resume_and_rescore_when_words_appear(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 3})])
    (recording,) = recordings_of(OWNER, root=tmp_path)
    backend = FakeBackend(TURNS)
    (first,) = run_owner(backend, "d-fake", [recording], tmp_path / "runs")
    (again,) = run_owner(backend, "d-fake", [recording], tmp_path / "runs")
    assert backend.calls == 1
    assert again == first
    before = input_digest(recording)
    recording.words.write_text(json.dumps([{"text": "a", "start": 1.0, "end": 2.0}]), encoding="utf-8")
    assert input_digest(recording) != before
    (after,) = run_owner(backend, "d-fake", [recording], tmp_path / "runs")
    assert backend.calls == 2
    assert after.attributed_speakers_smoothed == 1
    assert after.diarize_s == first.diarize_s


def test_the_owner_summary_lists_counts_by_generic_label_beside_the_other_corpora(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 3}), ("rec-02", 90, {})])
    scores: list[OwnerScore] = run_owner(
        FakeBackend(TURNS), "d-fake", recordings_of(OWNER, root=tmp_path), tmp_path / "r"
    )
    block = summarize(scores)["corpora"][OWNER]  # type: ignore[index]
    assert block["recordings"] == 2
    assert block["hypothesis_speakers"] == {"rec-01": 3, "rec-02": 3}
    assert block["mean_abs_speaker_count_error"] == 0
    assert summarize(scores)["primary_macro_der_collar_0"] is None


def test_each_owner_recording_has_its_own_words_file(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 40, {}), ("rec-02", 40, {})])
    items = load_items(OWNER, root=tmp_path)
    rows = {
        "rec-01": [spoken("w-runs-lid", "rec-01", [(" first", 1.0, 2.0)])],
        "rec-02": [spoken("w-runs-lid", "rec-02", [(" second", 5.0, 6.0), (" more", 6.0, 7.0)])],
    }
    assert write_words_for(items, rows) == 2
    first = load_words(tmp_path / OWNER / "rec-01" / "words.json")
    second = load_words(tmp_path / OWNER / "rec-02" / "words.json")
    assert first is not None
    assert second is not None
    assert [w.text for w in first] == [" first"]
    assert [w.text for w in second] == [" second", " more"]
    recordings = recordings_of(OWNER, root=tmp_path)
    assert len({r.words for r in recordings}) == 2
    assert [[w.text for w in disputes._words(item)] for item in items] == [[" first"], [" second", " more"]]


def test_each_owner_recording_is_scored_against_its_own_words(tmp_path: Path) -> None:
    make_owner(tmp_path, [("rec-01", 90, {}), ("rec-02", 90, {})])
    recordings = recordings_of(OWNER, root=tmp_path)
    recordings[0].words.write_text(json.dumps([{"text": "a", "start": 1.0, "end": 2.0}]), encoding="utf-8")
    recordings[1].words.write_text(json.dumps([{"text": "b", "start": 40.0, "end": 41.0}]), encoding="utf-8")
    first, second = run_owner(FakeBackend(TURNS), "d-fake", recordings, tmp_path / "runs")
    assert first.attributed_speakers_smoothed == 1
    assert second.attributed_speakers_smoothed == 1
    assert input_digest(recordings[0]) != input_digest(recordings[1])
