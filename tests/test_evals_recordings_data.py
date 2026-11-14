import urllib.error
from pathlib import Path

import numpy as np
import pytest

from evals import recordings_data
from evals.diarization import Turn, read_rttm
from evals.recordings_data import (
    SAMPLE_RATE,
    AscendUtterance,
    RefUtterance,
    ami_alignment,
    ami_meeting_id,
    check_ascend_offsets,
    check_free_disk,
    clean_text,
    load_index,
    ordered_text,
    parse_ascend_name,
    plan_sessions,
    read_wav,
    render_session,
    tier_utterances,
    wav_bytes,
    write_index,
    write_recording,
)
from ytverbatim.errors import ConfigError


def ascend(session, speaker, line, start, duration, text="x", id=None):
    return AscendUtterance(id or f"{session}-{speaker}-{line}", session, speaker, line, start, duration, "zh", text)


def test_parse_ascend_name_reads_session_speaker_line_offset_and_duration():
    name = "/a/b/ses1_spk17_L3825_16.5740_2.8760.wav"
    assert parse_ascend_name(name) == (1, 17, 3825, 16.574, 2.876)
    assert parse_ascend_name("ses2_spk3_L1_0_5.wav") == (2, 3, 1, 0.0, 5.0)


@pytest.mark.parametrize("name", ["ses1_spk17_L3825_16.5740.wav", "utt_0001.wav", "ses1_spk17_L3_1.0_2.0.mp3"])
def test_parse_ascend_name_rejects_other_shapes(name):
    with pytest.raises(ConfigError, match="unexpected ASCEND file name"):
        parse_ascend_name(name)


def test_offsets_pass_with_cross_speaker_overlap_but_not_with_same_speaker_overlap():
    fine = [ascend(1, 3, 10, 0.0, 5.0), ascend(1, 3, 12, 6.0, 2.0), ascend(1, 17, 1, 2.0, 9.0)]
    check_ascend_offsets(fine)
    with pytest.raises(ConfigError, match="overlap"):
        check_ascend_offsets([ascend(1, 3, 10, 0.0, 5.0, id="a"), ascend(1, 3, 12, 4.0, 2.0, id="b")])
    with pytest.raises(ConfigError, match="line order"):
        check_ascend_offsets([ascend(1, 3, 12, 0.0, 1.0), ascend(1, 3, 10, 5.0, 1.0)])


def test_plan_sessions_orders_by_offset_and_sizes_the_timeline():
    utterances = [
        ascend(2, 3, 5, 4.0, 1.0, "later"),
        ascend(1, 17, 9, 3.0, 2.0, "second"),
        ascend(1, 3, 2, 1.0, 1.5, "first [UNK] one"),
    ]
    plans = plan_sessions(utterances, tail_s=0.5)
    assert [p.session for p in plans] == [1, 2]
    assert [u.line for u in plans[0].utterances] == [2, 9]
    assert plans[0].duration_s == 5.5
    reference = plans[0].reference()
    assert [(r.start, r.end, r.speaker, r.text) for r in reference] == [
        (1.0, 2.5, "spk3", "first one"),
        (3.0, 5.0, "spk17", "second"),
    ]


def test_render_session_places_utterances_at_their_offsets_and_mixes_overlap():
    first, second = ascend(1, 3, 1, 1.0, 1.0), ascend(1, 17, 1, 1.5, 1.0)
    (plan,) = plan_sessions([first, second], tail_s=0.5)
    audio = {first.id: np.full(SAMPLE_RATE, 1000, dtype="<i2"), second.id: np.full(SAMPLE_RATE, 32000, dtype="<i2")}
    mix = render_session(plan, lambda u: audio[u.id])
    assert len(mix) == round(3.0 * SAMPLE_RATE)
    assert mix[:SAMPLE_RATE].max() == 0
    assert mix[int(1.2 * SAMPLE_RATE)] == 1000
    assert mix[int(1.7 * SAMPLE_RATE)] == 32767  # 1000 + 32000 clipped
    assert mix[int(2.2 * SAMPLE_RATE)] == 32000
    assert mix[int(2.7 * SAMPLE_RATE)] == 0


def test_clean_text_and_ordered_text():
    assert clean_text("ok [UNK] 好 [UNK]") == "ok 好"
    utterances = [RefUtterance(5, 6, "b", "later"), RefUtterance(1, 2, "a", "first"), RefUtterance(3, 4, "a", "  ")]
    assert ordered_text(utterances) == "first later"


def test_wav_round_trip_and_format_check(tmp_path):
    samples = np.array([0, 1, -1, 32767], dtype="<i2")
    assert read_wav(wav_bytes(samples)).tolist() == samples.tolist()
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\0\0\0\0")
    with pytest.raises(ConfigError, match="expected 16 kHz mono"):
        read_wav(buffer.getvalue())


def test_ami_meeting_id_and_alignment():
    assert ami_meeting_id("EN2002a.Array1-01.wav") == "EN2002a"
    turns = [Turn(0, 10, "A"), Turn(10, 20, "B")]
    inside = [RefUtterance(1, 9, "A", "x"), RefUtterance(11, 19, "B", "y")]
    assert ami_alignment(turns, inside) == pytest.approx(1.0)
    shifted = [RefUtterance(11, 19, "A", "x"), RefUtterance(1, 9, "B", "y")]
    assert ami_alignment(turns, shifted) == 0.0
    half = [RefUtterance(5, 15, "A", "x"), RefUtterance(10, 20, "B", "y")]
    assert ami_alignment(turns, half) == pytest.approx(15 / 20)
    with pytest.raises(ConfigError, match="different speakers"):
        ami_alignment(turns, [RefUtterance(1, 2, "C", "z")])


TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 30.5
tiers? <exists>
size = 2
item []:
\titem [1]:
\t\tclass = "IntervalTier"
\t\tname = "N_SPK1"
\t\txmin = 0
\t\txmax = 30.5
\t\tintervals: size = 2
\t\tintervals [1]:
\t\t\txmin = 1.5
\t\t\txmax = 3.25
\t\t\ttext = "你好，我是小王。"
\t\tintervals [2]:
\t\t\txmin = 4
\t\t\txmax = 5
\t\t\ttext = ""
\titem [2]:
\t\tclass = "IntervalTier"
\t\tname = "N_SPK2"
\t\txmin = 0
\t\txmax = 30.5
\t\tintervals: size = 1
\t\tintervals [1]:
\t\t\txmin = 2
\t\t\txmax = 6
\t\t\ttext = "嗯 okay"
"""


def test_tier_utterances_reads_non_empty_intervals_per_speaker():
    assert tier_utterances(TEXTGRID) == [
        RefUtterance(1.5, 3.25, "N_SPK1", "你好，我是小王。"),
        RefUtterance(2.0, 6.0, "N_SPK2", "嗯 okay"),
    ]


def test_write_recording_lays_out_audio_references_and_index(tmp_path: Path):
    utterances = [RefUtterance(2, 3, "B", "second"), RefUtterance(0.5, 1.5, "A", "first")]
    audio = wav_bytes(np.zeros(SAMPLE_RATE * 4, dtype="<i2"))
    entry = write_recording(tmp_path, "sessions/ses1", audio, utterances)
    folder = tmp_path / "sessions" / "ses1"
    assert (folder / "reference.txt").read_text(encoding="utf-8") == "first second\n"
    assert read_rttm(folder / "reference.rttm") == {"ses1": [Turn(0.5, 1.5, "A"), Turn(2.0, 3.0, "B")]}
    assert (entry.id, entry.audio, entry.duration_s, entry.speakers) == ("ses1", "sessions/ses1/audio.wav", 4.0, 2)
    own_turns = write_recording(tmp_path, "m1", audio, utterances, turns=[Turn(0, 4, "Z")])
    assert own_turns.speakers == 1
    assert read_rttm(tmp_path / "m1" / "reference.rttm") == {"m1": [Turn(0.0, 4.0, "Z")]}
    from evals.recordings_data import CorpusIndex

    index = CorpusIndex(corpus="c", source="s", license="l", recordings=(entry, own_turns))
    write_index(tmp_path, index)
    assert load_index(tmp_path) == index


def test_load_index_names_the_command_when_data_is_missing(tmp_path: Path):
    with pytest.raises(ConfigError, match="no prepared data"):
        load_index(tmp_path)


def test_check_free_disk_enforces_the_floor(tmp_path: Path):
    assert check_free_disk(tmp_path, free=lambda _: 11 * 10**9) == 11 * 10**9
    with pytest.raises(ConfigError, match="at least 10 GB"):
        check_free_disk(tmp_path, free=lambda _: 9 * 10**9)


def test_tier_names_with_spaces_become_valid_rttm_labels_and_an_empty_grid_is_refused():
    grid = TEXTGRID.replace('name = "N_SPK1"', 'name = "SPK 1"')
    assert tier_utterances(grid)[0].speaker == "SPK_1"
    with pytest.raises(ConfigError, match="no non-empty intervals"):
        tier_utterances(TEXTGRID.replace('text = "嗯 okay"', 'text = ""').replace("你好，我是小王。", ""))


def test_offsets_that_touch_within_a_millisecond_are_not_an_overlap():
    check_ascend_offsets([ascend(1, 3, 1, 0.0, 1.0004, id="a"), ascend(1, 3, 2, 1.0, 1.0, id="b")])


def test_a_failed_download_is_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def refuse(*_: object, **__: object) -> None:
        raise urllib.error.HTTPError("https://example.invalid/x", 503, "unavailable", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(recordings_data.urllib.request, "urlopen", refuse)
    with pytest.raises(ConfigError, match="could not download"):
        recordings_data.download("https://example.invalid/x", tmp_path / "x", free=lambda _: 11 * 10**9)
