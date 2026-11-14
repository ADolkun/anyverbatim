import json
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest
from recordings_fixture import make_owner

from evals.diarization import Turn
from evals.recordings_data import INDEX, OWNER, OwnerEntry, load_owner, wav_seconds
from evals.run_diarization import (
    OwnerScore,
    Recording,
    Run,
    input_digest,
    recordings_of,
    run_owner,
    score_owner_recording,
    score_recording,
    summarize,
)
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError


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


def test_owner_ids_must_be_generic_labels():
    assert OwnerEntry(id="rec-01", audio="a.wav").language == "zh+en"
    for bad in ("Alice-call", "rec-1", "rec-01-final"):
        with pytest.raises(ValueError, match="generic labels"):
            OwnerEntry(id=bad, audio="a.wav")


def test_the_owner_index_lists_wavs_with_their_durations_and_optional_speaker_counts(tmp_path: Path):
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 3}), ("rec-02", 30, {"language": "en"})])
    first, second = load_owner(tmp_path / OWNER)
    assert (first.entry.id, first.duration_s, first.entry.speakers) == ("rec-01", 90.0, 3)
    assert (second.entry.language, second.entry.speakers) == ("en", None)
    assert wav_seconds(first.wav) == 90.0


def test_the_owner_index_refuses_missing_files_duplicates_and_other_audio_formats(tmp_path: Path):
    with pytest.raises(ConfigError, match="no owner index"):
        load_owner(tmp_path / OWNER)
    make_owner(tmp_path, [("rec-01", 5, {})])
    folder = tmp_path / OWNER
    index = folder / INDEX
    index.write_text(json.dumps({"recordings": [{"id": "rec-01", "audio": "gone.wav"}]}), encoding="utf-8")
    with pytest.raises(ConfigError, match="does not exist"):
        load_owner(folder)
    index.write_text(json.dumps({"recordings": [{"id": "rec-01", "audio": "rec-01.wav"}] * 2}), encoding="utf-8")
    with pytest.raises(ConfigError, match="more than once"):
        load_owner(folder)
    (folder / "rec-01.m4a").write_bytes(b"not a wav")
    index.write_text(json.dumps({"recordings": [{"id": "rec-01", "audio": "rec-01.m4a"}]}), encoding="utf-8")
    with pytest.raises(ConfigError, match="not a readable WAV"):
        load_owner(folder)


def test_a_stereo_or_wrong_rate_wav_is_refused_with_the_conversion(tmp_path: Path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(b"\0\0" * 4)
    with pytest.raises(ConfigError, match="ffmpeg"):
        wav_seconds(path)


def test_an_owner_recording_has_no_reference_and_is_scored_by_speaker_count(tmp_path: Path):
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 2})])
    (recording,) = recordings_of(OWNER, root=tmp_path)
    assert recording.reference is None and recording.expected_speakers == 2
    score = score_owner_recording(FakeBackend(TURNS), recording, tmp_path / "out")
    assert (score.hypothesis_speakers, score.expected_speakers, score.speaker_count_error) == (3, 2, 1)
    assert score.rtf == pytest.approx(0.05)
    assert score.attributed_speakers_smoothed is None
    assert (tmp_path / "out" / "hypothesis.rttm").is_file()
    with pytest.raises(ConfigError, match="no reference"):
        score_recording(FakeBackend(TURNS), recording, tmp_path / "other")


def test_without_the_owners_count_there_is_no_count_error(tmp_path: Path):
    make_owner(tmp_path, [("rec-01", 90, {})])
    (recording,) = recordings_of(OWNER, root=tmp_path)
    score = score_owner_recording(FakeBackend(TURNS), recording, tmp_path / "out")
    assert score.speaker_count_error is None and score.expected_speakers is None


def test_owner_runs_resume_and_rescore_when_words_appear(tmp_path: Path):
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 3})])
    (recording,) = recordings_of(OWNER, root=tmp_path)
    backend = FakeBackend(TURNS)
    (first,) = run_owner(backend, "d-fake", [recording], tmp_path / "runs")
    (again,) = run_owner(backend, "d-fake", [recording], tmp_path / "runs")
    assert backend.calls == 1 and again == first
    before = input_digest(recording)
    recording.words.write_text(json.dumps([{"text": "a", "start": 1.0, "end": 2.0}]), encoding="utf-8")
    assert input_digest(recording) != before
    (after,) = run_owner(backend, "d-fake", [recording], tmp_path / "runs")
    assert backend.calls == 2
    assert after.attributed_speakers_smoothed == 1
    assert after.diarize_s == first.diarize_s


def test_the_owner_summary_lists_counts_by_generic_label_beside_the_other_corpora(tmp_path: Path):
    make_owner(tmp_path, [("rec-01", 90, {"speakers": 3}), ("rec-02", 90, {})])
    scores: list[OwnerScore] = run_owner(
        FakeBackend(TURNS), "d-fake", recordings_of(OWNER, root=tmp_path), tmp_path / "r"
    )
    block = summarize(scores)["corpora"][OWNER]  # type: ignore[index]
    assert block["recordings"] == 2
    assert block["hypothesis_speakers"] == {"rec-01": 3, "rec-02": 3}
    assert block["mean_abs_speaker_count_error"] == 0
    assert summarize(scores)["primary_macro_der_collar_0"] is None
