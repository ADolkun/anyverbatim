import struct

import pytest

from ytverbatim.contract.models import Word
from ytverbatim.diarize.attribution import Span, assign, speech_seconds
from ytverbatim.diarize.protocol import Diarization

FRAME_S = 0.01
CHANNELS = 8


def diarization(seconds: float, *active: tuple[int, float, float, float]) -> Diarization:
    """Channels `(channel, start, end, probability)`; every other frame and channel is 0."""
    frames = round(seconds / FRAME_S)
    values = [0.0] * (frames * CHANNELS)
    for channel, start, end, p in active:
        for frame in range(round(start / FRAME_S), round(end / FRAME_S)):
            values[frame * CHANNELS + channel] = p
    data = struct.pack(f"<{len(values)}e", *values)
    return Diarization(turns=(), threshold=0.5, frame_s=FRAME_S, frames=frames, channels=CHANNELS, probs=data)


def words(*spans: tuple[float, float]) -> list[Word]:
    return [Word(text=f"w{i}", start=start, end=end) for i, (start, end) in enumerate(spans)]


def test_a_clean_change_splits_words_and_numbers_by_first_appearance() -> None:
    # Channel 3 speaks first, so it becomes S1 whatever its channel number.
    d = diarization(4, (3, 0, 2, 0.9), (1, 2, 4, 0.9))
    result = assign(words((0.2, 0.6), (0.8, 1.6), (2.2, 2.8), (3.0, 3.6)), d)
    assert result.speakers == ("S1", "S1", "S2", "S2")
    assert result.overlaps == ()


def test_a_word_straddling_a_change_goes_to_the_speaker_covering_more_of_it() -> None:
    d = diarization(4, (0, 0, 2, 0.9), (1, 2, 4, 0.9))
    result = assign(words((0.5, 1.5), (1.7, 2.1), (2.5, 3.5)), d, smoothing=False)
    assert result.speakers == ("S1", "S1", "S2")
    assert result.overlaps == ()


def test_the_larger_summed_probability_wins_among_active_channels() -> None:
    d = diarization(2, (0, 0, 1, 0.6), (1, 0, 1, 0.95))
    assert assign(words((0.2, 0.8)), d).speakers == ("S1",)
    assert assign(words((0.2, 0.8)), d).overlaps == (Span(0.2, 0.8),)


def test_a_word_in_a_short_gap_takes_the_nearest_speaker() -> None:
    d = diarization(6, (0, 0, 1, 0.9), (1, 2, 6, 0.9))
    # 1.7-1.9 sits 0.8 s after channel 0 ends and 0.1 s before channel 1 starts.
    result = assign(words((0.2, 0.8), (1.7, 1.9), (2.5, 3.5)), d, smoothing=False)
    assert result.speakers == ("S1", "S2", "S2")


def test_a_word_beyond_half_a_second_of_any_speaker_takes_the_previous_words_speaker() -> None:
    d = diarization(8, (0, 0, 1, 0.9), (1, 5, 8, 0.9))
    result = assign(words((0.2, 0.8), (2.5, 3.0), (5.5, 6.5)), d, smoothing=False)
    assert result.speakers == ("S1", "S1", "S2")


def test_leading_words_with_no_speaker_nearby_take_the_first_attributed_speaker() -> None:
    d = diarization(6, (2, 3, 6, 0.9))
    assert assign(words((0.2, 0.6), (3.5, 4.0)), d).speakers == ("S1", "S1")


def test_a_second_speaker_over_half_of_a_words_frames_flags_overlap() -> None:
    d = diarization(4, (0, 0, 4, 0.9), (1, 1.0, 1.5, 0.7), (1, 2.6, 2.8, 0.7))
    # Word 1 (1.0-1.6): channel 1 over 5 of its 6 tenths. Word 2 (2.5-3.0): only 2 of 5, so not overlapped.
    result = assign(words((0.2, 0.8), (1.0, 1.6), (1.6, 2.0), (2.5, 3.0)), d)
    assert result.speakers == ("S1", "S1", "S1", "S1")
    assert result.overlaps == (Span(1.0, 1.6),)


def test_consecutive_overlapped_words_merge_into_one_span() -> None:
    d = diarization(3, (0, 0, 3, 0.9), (1, 1.0, 2.0, 0.7))
    result = assign(words((0.2, 0.8), (1.0, 1.4), (1.4, 2.0), (2.2, 2.8)), d)
    assert result.overlaps == (Span(1.0, 2.0),)


def test_a_one_word_blip_is_smoothed_into_the_surrounding_speaker() -> None:
    d = diarization(4, (0, 0, 1.5, 0.9), (1, 1.5, 2.3, 0.9), (0, 2.3, 4, 0.9))
    spans = ((0.2, 0.6), (0.7, 1.2), (1.6, 2.2), (2.5, 3.0), (3.1, 3.6))
    assert assign(words(*spans), d, smoothing=False).speakers == ("S1", "S1", "S2", "S1", "S1")
    assert assign(words(*spans), d, smoothing=True).speakers == ("S1",) * 5


def test_a_short_multi_word_run_is_smoothed_but_a_long_one_is_kept() -> None:
    short = diarization(4, (0, 0, 1.5, 0.9), (1, 1.5, 1.9, 0.9), (0, 1.9, 4, 0.9))
    two_quick = words((0.2, 1.2), (1.55, 1.7), (1.7, 1.85), (2.5, 3.5))
    assert assign(two_quick, short).speakers == ("S1",) * 4
    long = diarization(5, (0, 0, 1.5, 0.9), (1, 1.5, 3.0, 0.9), (0, 3.0, 5, 0.9))
    two_slow = words((0.2, 1.2), (1.6, 2.2), (2.3, 2.9), (3.5, 4.5))
    assert assign(two_slow, long).speakers == ("S1", "S2", "S2", "S1")


def test_an_overlap_flagged_blip_is_kept() -> None:
    d = diarization(4, (0, 0, 4, 0.6), (1, 1.6, 2.2, 0.9))
    result = assign(words((0.2, 0.6), (0.7, 1.2), (1.6, 2.2), (2.5, 3.0)), d, smoothing=True)
    assert result.speakers == ("S1", "S1", "S2", "S1")
    assert result.overlaps == (Span(1.6, 2.2),)


def test_a_blip_between_two_different_speakers_is_kept() -> None:
    d = diarization(4, (0, 0, 1.5, 0.9), (1, 1.5, 2.3, 0.9), (2, 2.3, 4, 0.9))
    result = assign(words((0.2, 1.2), (1.6, 2.2), (2.5, 3.5)), d)
    assert result.speakers == ("S1", "S2", "S3")


@pytest.mark.parametrize("smoothing", [True, False])
def test_silence_gives_no_speaker(smoothing: bool) -> None:
    result = assign(words((0.2, 0.6), (1.0, 1.4)), diarization(2), smoothing=smoothing)
    assert result.speakers == (None, None)
    assert result.overlaps == ()


def test_no_words_gives_nothing() -> None:
    assert assign([], diarization(2, (0, 0, 2, 0.9))) == ((), ())


def test_words_past_the_stored_frames_take_the_previous_speaker() -> None:
    d = diarization(1, (0, 0, 1, 0.9))
    assert assign(words((0.2, 0.8), (3.0, 3.5)), d).speakers == ("S1", "S1")


def test_speech_seconds_sums_segments_in_first_appearance_order() -> None:
    totals = speech_seconds([(0.0, 2.0, "S1"), (2.0, 3.5, "S2"), (4.0, 5.0, "S1"), (5.0, 6.0, None)])
    assert list(totals) == ["S1", "S2"]
    assert totals == pytest.approx({"S1": 3.0, "S2": 1.5})
