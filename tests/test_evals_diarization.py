import math

import pytest

from evals.diarization import (
    COLLARS_S,
    DerScore,
    Turn,
    der,
    disagreement_runs,
    format_rttm,
    macro_rate,
    parse_rttm,
    pooled_rate,
    read_rttm,
    single_speaker_words,
    speaker_count_error,
    wder,
    write_rttm,
)
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError

RTTM = """\
;; a comment
SPEAKER rec-a 1 0.000 10.000 <NA> <NA> A <NA> <NA>
SPEAKER rec-a 1 10.000 10.000 <NA> <NA> B <NA> <NA>
SPEAKER rec-b 1 1.500 2.250 <NA> <NA> X <NA> <NA>
SPACE rec-b 1 0.000 1.500 <NA> <NA> <NA> <NA> <NA>
"""


def turns(*rows: tuple[float, float, str]) -> list[Turn]:
    return [Turn(*row) for row in rows]


def words(*rows: tuple[float, float]) -> list[Word]:
    return [Word(text="w", start=s, end=e) for s, e in rows]


def test_parse_rttm_groups_by_recording_and_skips_other_records():
    parsed = parse_rttm(RTTM)
    assert parsed["rec-a"] == turns((0.0, 10.0, "A"), (10.0, 20.0, "B"))
    assert parsed["rec-b"] == turns((1.5, 3.75, "X"))


def test_parse_rttm_rejects_short_and_negative_lines():
    with pytest.raises(ConfigError, match="line 1 has 4 fields"):
        parse_rttm("SPEAKER rec 1 0.0\n")
    with pytest.raises(ConfigError, match="negative duration"):
        parse_rttm("SPEAKER rec 1 0.0 -1.0 <NA> <NA> A <NA> <NA>\n")
    with pytest.raises(ConfigError, match="non-numeric"):
        parse_rttm("SPEAKER rec 1 zero 1.0 <NA> <NA> A <NA> <NA>\n")


def test_rttm_round_trip_sorts_and_drops_empty_turns(tmp_path):
    path = tmp_path / "out.rttm"
    write_rttm(path, "rec", turns((5.0, 6.0, "B"), (0.0, 2.5, "A"), (7.0, 7.0, "A")))
    assert read_rttm(path) == {"rec": turns((0.0, 2.5, "A"), (5.0, 6.0, "B"))}
    assert format_rttm("rec", []) == ""


def test_der_perfect_and_relabelled_hypothesis_is_zero():
    reference = turns((0, 10, "A"), (10, 20, "B"))
    assert der(reference, turns((0, 10, "A"), (10, 20, "B"))).rate == 0
    assert der(reference, turns((0, 10, "spk1"), (10, 20, "spk0"))).rate == 0


def test_der_components_without_collar():
    reference = turns((0, 10, "A"), (10, 20, "B"))
    score = der(reference, turns((0, 10, "A"), (12, 20, "B")))
    assert (score.total, score.missed, score.false_alarm, score.confusion) == (20, 2, 0, 0)
    assert score.rate == pytest.approx(0.10)

    split = der(reference, turns((0, 10, "A"), (10, 15, "B1"), (15, 20, "B2")))
    assert split.confusion == pytest.approx(5)
    assert split.rate == pytest.approx(0.25)

    spurious = der(reference, turns((0, 10, "A"), (10, 20, "B"), (20, 24, "A")))
    assert spurious.false_alarm == pytest.approx(4)


def test_der_scores_overlapping_speech():
    reference = turns((0, 10, "A"), (5, 15, "B"))
    score = der(reference, turns((0, 10, "A")))
    assert score.total == pytest.approx(20)
    assert score.missed == pytest.approx(10)
    assert score.rate == pytest.approx(0.5)
    assert der(reference, turns((0, 10, "A")), skip_overlap=True).total == pytest.approx(10)


def test_der_collar_forgives_a_misplaced_boundary():
    reference = turns((0, 10, "A"), (10, 20, "B"))
    hypothesis = turns((0, 10.2, "A"), (10.2, 20, "B"))
    strict = der(reference, hypothesis, collar_s=COLLARS_S[0])
    forgiving = der(reference, hypothesis, collar_s=COLLARS_S[1])
    assert strict.rate == pytest.approx(0.2 / 20)
    assert forgiving.errors == 0


def test_der_rejects_an_empty_reference():
    with pytest.raises(ConfigError, match="no turns"):
        der([], turns((0, 1, "A")))


def test_macro_and_pooled_rates_weight_recordings_differently():
    short = DerScore(total=10, missed=5, false_alarm=0, confusion=0)
    long = DerScore(total=90, missed=0, false_alarm=0, confusion=9)
    assert macro_rate([short, long]) == pytest.approx((0.5 + 0.1) / 2)
    assert pooled_rate([short, long]) == pytest.approx(14 / 100)
    assert math.isnan(DerScore(0, 0, 0, 0).rate)
    with pytest.raises(ValueError, match="no scores"):
        macro_rate([])


def test_speaker_count_error_is_signed():
    reference = turns((0, 1, "A"), (1, 2, "B"))
    assert speaker_count_error(reference, turns((0, 1, "x"), (1, 2, "y"), (2, 3, "z"))) == 1
    assert speaker_count_error(reference, turns((0, 2, "x"))) == -1


REFERENCE = turns((0, 10, "A"), (10, 20, "B"), (18, 25, "C"))


def test_single_speaker_words_skip_overlap_silence_and_straddles():
    spoken = words((1, 2), (9.5, 10.5), (19, 19.5), (21, 22), (30, 31), (17, 17.5))
    eligible = single_speaker_words(REFERENCE, spoken)
    # (9.5, 10.5) lies half in A and half in B; (19, 19.5) is in the B/C overlap; (30, 31) is silence.
    assert eligible == {0: "A", 3: "C", 5: "B"}


def test_wder_after_optimal_mapping_ignores_label_names():
    spoken = words((1, 2), (3, 4), (11, 12), (13, 14), (22, 23))
    assert wder(REFERENCE, spoken, ["x", "x", "y", "y", "z"]).rate == 0
    score = wder(REFERENCE, spoken, ["x", "x", "y", "x", "z"])
    assert (score.errors, score.eligible, score.total_words) == (1, 5, 5)
    assert score.rate == pytest.approx(0.2)


def test_wder_counts_words_of_an_unmapped_extra_speaker_as_errors():
    reference = turns((0, 10, "A"), (10, 20, "B"))
    spoken = words((1, 2), (3, 4), (5, 6), (11, 12), (13, 14), (15, 16))
    score = wder(reference, spoken, ["x", "x", "w", "y", "y", "y"])
    assert score.errors == 1
    assert score.eligible == 6


def test_wder_with_no_eligible_words_is_undefined_not_zero():
    score = wder(turns((0, 10, "A"), (0, 10, "B")), words((1, 2)), ["x"])
    assert score.eligible == 0
    assert math.isnan(score.rate)


def test_wder_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="2 words but 1"):
        wder(REFERENCE, words((1, 2), (3, 4)), ["x"])


def test_disagreement_runs_map_labels_and_keep_runs_of_two_seconds():
    first = turns((0, 10, "A"), (10, 20, "B"))
    # Same people under other labels, except 12-15 (3 s) and 17-17.5 (0.5 s) which the second names differently.
    second = turns((0, 10, "p"), (10, 12, "q"), (12, 15, "p"), (15, 17, "q"), (17, 17.5, "p"), (17.5, 20, "q"))
    runs = disagreement_runs({"one": first, "two": second})
    assert [(r.start, r.end) for r in runs] == [(12, 15)]
    assert runs[0].speakers == {"one": frozenset({"B"}), "two": frozenset({"A"})}


def test_disagreement_ignores_silence_against_speech():
    runs = disagreement_runs({"one": turns((0, 10, "A")), "two": turns((0, 4, "p"))})
    assert runs == []
