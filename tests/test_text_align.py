import pytest

from anyverbatim.acquire.captions import CaptionKind, CaptionSegment, CaptionTrack, Word
from anyverbatim.text.align import WINDOW_PAD_S, Timeline, align, caption_pieces, caption_timeline, timeline, units


def equal_runs(mine: str, theirs: str) -> list[tuple[int, int]]:
    result = align(units(mine), units(theirs))
    return [(op.src_start, op.src_end) for op in result.opcodes if op.tag == "equal"]


def words_timeline(*pieces: tuple[str, float, float]) -> Timeline:
    return timeline("test", ((text, start, end, None) for text, start, end in pieces))


def track(kind: CaptionKind, *cues: CaptionSegment) -> CaptionTrack:
    return CaptionTrack(language="zh", kind=kind, source_format="json3", segments=cues)


def test_a_window_that_shares_the_speech_is_anchored_and_reports_its_equal_runs() -> None:
    mine, theirs = "我们今天讨论 Zorbix 的发布", "他说我们今天讨论 Xorbyte 的发布吧"
    result = align(units(mine), units(theirs))

    assert result.anchored
    assert equal_runs(mine, theirs) == [(0, 6), (7, 10)]


def test_a_window_below_the_anchor_ratio_is_refused_but_still_reports_its_opcodes() -> None:
    mine, theirs = "甲乙丙丁戊己庚辛", "子甲丑寅卯辰巳午未申"
    result = align(units(mine), units(theirs))

    assert equal_runs(mine, theirs) == [(0, 1)]
    assert not result.anchored


@pytest.mark.parametrize(
    ("theirs", "anchored"),
    [
        ("甲乙子丑寅", True),
        ("甲子丑寅卯", False),
    ],
)
def test_the_anchor_ratio_is_two_fifths_of_the_plain_units(theirs: str, anchored: bool) -> None:
    assert align(units("甲乙丙丁戊"), units(theirs)).anchored is anchored


def test_fewer_than_four_plain_units_are_anchored_by_a_single_match() -> None:
    assert align(units("甲乙丙"), units("子丑甲寅卯辰")).anchored


@pytest.mark.parametrize("theirs", ["子丑寅卯", ""])
def test_a_window_with_no_equal_op_is_refused(theirs: str) -> None:
    result = align(units("甲乙丙丁"), units(theirs))

    assert not any(op.tag == "equal" for op in result.opcodes)
    assert not result.anchored


def test_an_empty_transcript_is_refused() -> None:
    assert not align(units(""), units("甲乙丙")).anchored


def test_matching_quantities_do_not_vouch_for_the_window() -> None:
    mine, theirs = units("甲乙丙丁戊150"), units("子丑寅卯辰150")
    assert any(u.token is not None for u in mine)

    result = align(mine, theirs)

    assert [(op.src_start, op.src_end) for op in result.opcodes if op.tag == "equal"] == [(5, 6)]
    assert not result.anchored


def test_a_quantity_matches_across_numeral_scripts() -> None:
    assert equal_runs("一百五十期 Zorbix", "150期 Zorbix") == [(0, 2)]


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (20.0, 22.0, ["beta", "gamma", "delta"]),
        (25.0, 26.0, ["delta"]),
        (-5.0, 0.0, ["alpha"]),
        (100.0, 110.0, []),
        (-50.0, -40.0, []),
        (0.0, 100.0, ["alpha", "beta", "gamma", "delta"]),
    ],
)
def test_a_window_holds_the_units_starting_within_the_padded_bounds(
    start: float, end: float, expected: list[str]
) -> None:
    line = words_timeline(("alpha", 1.0, 2.0), ("beta", 18.0, 19.0), ("gamma", 21.0, 22.0), ("delta", 24.0, 25.0))

    assert [line.units[i].key for i in line.window(start, end)] == expected


def test_the_window_pad_is_inclusive_on_both_sides() -> None:
    line = words_timeline(
        ("early", 10.0 - WINDOW_PAD_S - 0.1, 9.0),
        ("first", 10.0 - WINDOW_PAD_S, 9.0),
        ("last", 12.0 + WINDOW_PAD_S, 15.0),
        ("late", 12.0 + WINDOW_PAD_S + 0.1, 16.0),
    )

    assert [line.units[i].key for i in line.window(10.0, 12.0)] == ["first", "last"]


def test_a_window_is_a_range_of_unit_indexes_into_the_timeline() -> None:
    line = words_timeline(("alpha", 1.0, 2.0), ("beta", 18.0, 19.0))

    assert line.window(18.0, 18.0) == range(1, 2)


def test_overlapping_auto_cues_are_ordered_and_clamped_into_non_decreasing_times() -> None:
    rolling = track(
        CaptionKind.AUTO,
        CaptionSegment(start=0.0, end=4.0, text="", words=(Word("Zorbix", 0.0, 1.0), Word("builds", 1.0, 4.0))),
        CaptionSegment(start=2.0, end=6.0, text="", words=(Word("Xorbyte", 2.0, 3.0), Word("tools", 3.0, 6.0))),
    )
    _, pieces = caption_pieces(rolling)

    line = timeline("caption_auto", ((p.text, p.start, p.end, None) for p in reversed(pieces)))

    assert [u.key for u in line.units] == ["zorbix", "builds", "xorbyte", "tools"]
    assert list(line.starts) == sorted(line.starts)
    assert all(start <= end for start, end in zip(line.starts, line.ends, strict=True))
    # "builds" spreads its characters over 1-4 s, so "Xorbyte" (2-3 s) is clamped behind them.
    assert line.starts[2] > 2.0
    assert line.starts[2] >= 1.0 + 3.0 * 5 / 6


def test_a_piece_that_starts_inside_the_previous_one_is_clamped_to_its_last_character() -> None:
    line = words_timeline(("abcdef", 0.0, 6.0), ("xyz", 1.0, 2.0))

    assert line.starts == (0.0, 5.0)


def test_pieces_without_text_are_dropped_and_leave_no_unit() -> None:
    line = words_timeline(("alpha", 0.0, 1.0), ("  ", 1.0, 2.0), ("beta", 2.0, 3.0))

    assert [u.key for u in line.units] == ["alpha", "beta"]
    assert line.text == "alpha beta"


def test_a_manual_track_is_tagged_caption_manual_and_an_auto_track_caption_auto() -> None:
    cue = CaptionSegment(start=0.0, end=2.0, text="我们讨论 Zorbix")

    assert caption_pieces(track(CaptionKind.MANUAL, cue))[0] == "caption_manual"
    assert caption_pieces(track(CaptionKind.AUTO, cue))[0] == "caption_auto"


def test_a_cue_without_words_is_one_piece_and_a_cue_with_words_is_its_words() -> None:
    plain = CaptionSegment(start=0.0, end=2.0, text="我们讨论 Zorbix")
    timed = CaptionSegment(
        start=2.0, end=4.0, text="ignored", words=(Word("Xorbyte", 2.0, 3.0), Word("tools", 3.0, 4.0))
    )

    _, pieces = caption_pieces(track(CaptionKind.MANUAL, plain, timed))

    assert [(p.text, p.start, p.end) for p in pieces] == [
        ("我们讨论 Zorbix", 0.0, 2.0),
        ("Xorbyte", 2.0, 3.0),
        ("tools", 3.0, 4.0),
    ]


def test_a_track_without_cues_has_no_pieces() -> None:
    assert caption_pieces(track(CaptionKind.MANUAL)) == ("caption_manual", ())


def test_chinese_cues_that_meet_in_latin_script_stay_two_words() -> None:
    cues = (
        CaptionSegment(start=0.0, end=2.0, text="他们收购了Zorbix"),
        CaptionSegment(start=2.0, end=4.0, text="Zorbix曾经排第二"),
        CaptionSegment(start=4.0, end=6.0, text="达到了21"),
        CaptionSegment(start=6.0, end=8.0, text="3家公司"),
    )

    built = caption_timeline(track(CaptionKind.MANUAL, *cues))

    assert "Zorbix Zorbix" in built.text
    assert [u.key for u in built.units].count("zorbix") == 2
    assert [u.token.value_itn for u in built.units if u.token is not None] == ["第2", "21", "3家"]


def test_chinese_word_pieces_are_joined_without_spaces() -> None:
    pieces = [
        ("我们", 0.0, 0.5),
        ("的", 0.5, 1.0),
        ("Zor", 1.0, 1.5),
        ("bix", 1.5, 2.0),
        ("公司", 2.0, 2.5),
        ("很好", 2.5, 3.0),
    ]
    cue = CaptionSegment(start=0.0, end=3.0, text="ignored", words=tuple(Word(*p) for p in pieces))

    built = caption_timeline(track(CaptionKind.AUTO, cue))

    assert built.text == "我们的Zorbix公司很好"
    assert built.source == "caption_auto"


def test_english_cues_are_spaced_as_before() -> None:
    cues = (
        CaptionSegment(start=0.0, end=2.0, text="we bought Zorbix"),
        CaptionSegment(start=2.0, end=4.0, text="Zorbix was second"),
    )

    assert caption_timeline(track(CaptionKind.MANUAL, *cues)).text == "we bought Zorbix Zorbix was second"
