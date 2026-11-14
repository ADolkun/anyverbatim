import pytest

from evals.metrics import MIN_INCIDENT_UNITS, ScriptMix, script_mix, translation_incident
from evals.normalize import script_counts


def test_script_counts_treat_each_han_character_and_each_latin_run_as_one_unit():
    assert script_counts("我们 meeting 很 long 20 次") == (4, 2)
    assert script_counts("hello world") == (0, 2)
    assert script_counts("12 34 %") == (0, 0)


def test_script_counts_fold_traditional_and_fullwidth_forms():
    assert script_counts("會議ＮＶＤＡ") == (2, 1)


def test_mostly_means_at_least_seventy_percent_exactly():
    assert ScriptMix(7, 3).mostly("han")
    assert not ScriptMix(7, 4).mostly("han")
    assert ScriptMix(3, 7).mostly("latin")
    assert not ScriptMix(0, 0).mostly("han")


def test_a_chinese_reference_heard_as_english_is_an_incident():
    reference = script_mix("我们今天开会讨论预算")
    assert translation_incident(reference, script_mix("we discuss the budget today"))
    assert not translation_incident(reference, script_mix("我们今天开会讨论预算"))


def test_an_english_reference_heard_as_chinese_is_an_incident():
    assert translation_incident(script_mix("we discuss the budget today"), script_mix("我们今天讨论预算"))


def test_the_seventy_percent_boundary_counts_on_both_sides():
    assert script_mix("我们今天开会讨 budget plan now") == ScriptMix(7, 3)
    assert translation_incident(ScriptMix(7, 3), ScriptMix(3, 7))
    assert not translation_incident(ScriptMix(7, 3), ScriptMix(4, 6))
    assert not translation_incident(ScriptMix(6, 4), ScriptMix(0, 10))


def test_a_reference_below_seventy_percent_in_either_script_is_never_an_incident():
    assert not translation_incident(script_mix("我们 meeting 很 long"), script_mix("all english words here"))


def test_a_reference_too_short_to_judge_is_never_an_incident():
    assert MIN_INCIDENT_UNITS == 3
    assert not translation_incident(script_mix("好的"), script_mix("okay"))
    assert translation_incident(script_mix("好的呀"), script_mix("okay then"))


def test_an_empty_or_script_free_hypothesis_is_no_incident():
    assert not translation_incident(script_mix("我们今天开会"), script_mix(""))
    assert not translation_incident(script_mix("我们今天开会"), script_mix("123 456"))


@pytest.mark.parametrize("text", ["", "   ", "…"])
def test_empty_text_has_no_units(text: str):
    assert script_mix(text) == ScriptMix(0, 0)
