import json
from pathlib import Path

import pytest
from recordings_fixture import spoken

from evals.diarization import Turn, wder
from evals.metrics import ErrorCounts
from evals.recordings_corpus import FOLDER_CORPORA, Item, Reference
from evals.recordings_data import RefUtterance
from evals.recordings_scoring import (
    WindowScore,
    count_errors,
    language_seconds,
    paired,
    reference_cuts,
    score_item,
    summarize_corpus,
    words_from_rows,
    write_words,
    write_words_for,
)
from evals.run_diarization import load_words
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.errors import ConfigError


def item(reference: Reference, language="en", corpus="ami", wav="audio.wav") -> Item:
    return Item(corpus, "m1", Path(wav), 100.0, language, reference)


def utterances(*spans: tuple[float, float, str]) -> tuple[RefUtterance, ...]:
    return tuple(RefUtterance(s, e, "A", t) for s, e, t in spans)


def test_cuts_fall_in_the_first_reference_pause_after_the_target():
    lines = utterances((0, 10, "a"), (10.5, 25, "b"), (26, 40, "c"), (45, 60, "d"), (60.2, 70, "e"))
    assert reference_cuts(lines) == [0.0, 42.5]
    assert reference_cuts(lines, target_s=5) == [0.0, 10.25, 25.5, 42.5, 60.1]
    assert reference_cuts([]) == [0.0]


def test_overlapping_reference_lines_never_hide_a_cut_inside_them():
    lines = utterances((0, 50, "a"), (10, 20, "b"), (80, 90, "c"))
    assert reference_cuts(lines) == [0.0, 65.0]


def test_word_error_rate_is_pooled_over_windows_and_pairs_by_window():
    reference = Reference(
        "hello there everyone this is a test now",
        utterances((0, 3, "hello there everyone"), (4, 7, "this is a test"), (70, 73, "now")),
    )
    rows = [
        spoken(
            "w",
            "m1",
            [
                (" hello", 0.5, 1.0),
                (" there", 1.0, 1.5),
                (" everybody", 1.5, 2.5),
                (" this", 4.5, 5.0),
                (" is", 5.0, 5.5),
                (" test", 5.5, 6.5),
                (" now", 70.5, 71.0),
            ],
        )
    ]
    windows = score_item(item(reference), rows, "wer", "itn")
    assert [w.window for w in windows] == [0, 1]
    assert [(w.counts.errors, w.counts.reference_units) for w in windows] == [(2, 7), (0, 1)]
    corpus = summarize_corpus("ami", "wer", "itn", [item(reference)], windows)
    assert (corpus.errors, corpus.reference_units, corpus.rate) == (2, 8, 0.25)
    assert corpus.windows == 2


def test_a_hypothesis_word_goes_to_the_window_holding_its_midpoint():
    reference = Reference("one two", utterances((0, 3, "one"), (70, 73, "two")))
    late = [spoken("w", "m1", [(" one", 0.5, 1.0), (" two", 70.5, 71.0)])]
    early = [spoken("w", "m1", [(" one", 0.5, 1.0), (" two", 2.0, 2.5)])]
    assert [w.counts.errors for w in score_item(item(reference), late, "wer", "itn")] == [0, 0]
    assert [w.counts.errors for w in score_item(item(reference), early, "wer", "itn")] == [1, 1]


def test_mixed_error_rate_scores_cjk_characters_and_latin_words_as_units():
    reference = Reference("我们今天 meeting 很长")
    row = ChunkHypothesis(hypothesis="w", file="m1", chunk=0, start=0, end=3, text="我们今天 meting 很长")
    (window,) = score_item(item(reference, "zh", "ascend-utterances"), [row], "mer", "itn")
    assert (window.counts.errors, window.counts.reference_units) == (1, 7)


def test_character_error_rate_counts_characters():
    assert count_errors("cer", "今天开会", "今天开会啦", "zh", "itn").errors == 1
    assert count_errors("cer", "今天开会", "今天开会啦", "zh", "itn").reference_units == 4


def test_a_window_with_an_empty_reference_counts_every_hypothesis_unit_as_an_insertion():
    assert count_errors("wer", "", "hello there", "en", "itn") == ErrorCounts(0, 0, 2, 0)
    assert count_errors("mer", "…", "你好 ok", "zh", "itn") == ErrorCounts(0, 0, 3, 0)
    assert count_errors("cer", "", "", "zh", "itn") == ErrorCounts(0, 0, 0, 0)


def test_incidents_are_counted_per_window_and_split_by_mixed_references():
    reference = Reference(
        "我们今天开会讨论预算 we shipped the new build yesterday and tested it 我们 meeting 很 long",
        utterances(
            (0, 5, "我们今天开会讨论预算"),
            (70, 75, "we shipped the new build yesterday and tested it"),
            (140, 145, "我们 meeting 很 long"),
        ),
    )
    rows = [
        spoken("w", "m1", [(" we", 1, 2), (" discuss", 2, 3), (" the", 3, 4), (" budget", 4, 5)]),
        spoken("w", "m1", [("我们昨天发布了新版本并且测试了", 71, 74)], chunk=1),
        spoken("w", "m1", [("我们 meeting 很 long", 141, 144)], chunk=2),
    ]
    windows = score_item(item(reference, "zh", "ascend"), rows, "mer", "raw")
    summary = summarize_corpus("ascend", "mer", "raw", [item(reference, "zh", "ascend")], windows)
    assert summary.incidents.chunks == 3
    assert summary.incidents.incidents == 2
    assert summary.incidents.mixed_chunks == 1
    assert summary.incidents.incidents_in_mixed == 0
    assert summary.incidents.rate == pytest.approx(2 / 3)
    assert summary.incidents.rate_in_mixed == 0


def test_an_item_without_a_reference_cannot_be_scored():
    with pytest.raises(ConfigError, match="no reference"):
        score_item(item(None), [], "wer", "itn")  # type: ignore[arg-type]


def window(item_id: str, index: int, errors: int, units: int) -> WindowScore:
    counts = ErrorCounts(errors, 0, 0, units - errors)
    return WindowScore(item_id, index, counts, chunk=True, mixed=False, incident=False)


def test_paired_bootstrap_reports_the_difference_of_pooled_rates():
    a = [window("m", i, 1, 10) for i in range(20)]
    b = [window("m", i, 3, 10) for i in range(20)]
    result = paired(a, b)
    assert result.windows == 20
    assert result.a_rate == pytest.approx(0.1)
    assert result.b_rate == pytest.approx(0.3)
    assert result.difference == pytest.approx(-0.2)
    assert result.excludes_zero
    assert paired(a, a).difference == 0


def test_pairing_needs_the_same_windows():
    with pytest.raises(ConfigError, match="different windows"):
        paired([window("m", 0, 1, 10)], [window("m", 1, 1, 10)])


def test_words_round_trip_into_wder(tmp_path: Path):
    rows = [
        spoken("w", "m1", [(" hi", 11.0, 11.5), (" there", 11.5, 12.0)], chunk=1),
        spoken("w", "m1", [("你", 1.0, 1.4), ("好", 1.4, 2.0)], chunk=0),
    ]
    written = words_from_rows(rows)
    assert [w.text for w in written] == ["你", "好", " hi", " there"]
    path = tmp_path / "words.json"
    write_words(path, written)
    assert json.loads(path.read_text(encoding="utf-8"))[0] == {"text": "你", "start": 1.0, "end": 1.4}
    loaded = load_words(path)
    assert loaded is not None
    assert [(w.text, w.start, w.end) for w in loaded] == [(w.text, w.start, w.end) for w in written]
    reference = [Turn(0, 5, "A"), Turn(10, 15, "B")]
    assert wder(reference, loaded, ["x", "x", "y", "y"]).rate == 0
    assert wder(reference, loaded, ["x", "x", "x", "y"]).rate == 0.25


def test_a_row_with_text_but_no_word_times_cannot_feed_wder():
    bare = ChunkHypothesis(hypothesis="q-base", file="m1", chunk=0, start=0, end=2, text="hello")
    with pytest.raises(ConfigError, match="word times"):
        words_from_rows([bare])
    silent_row = ChunkHypothesis(hypothesis="q-base", file="m1", chunk=0, start=0, end=2, text=" ")
    assert words_from_rows([silent_row]) == []


def test_words_are_written_next_to_each_audio_file_and_never_for_the_clip_corpus(tmp_path: Path):
    wav = tmp_path / "m1" / "audio.wav"
    wav.parent.mkdir()
    rows = {"m1": [spoken("w", "m1", [(" hi", 1.0, 2.0)])]}
    assert write_words_for([Item("ami", "m1", wav, 5.0, "en", None)], rows) == 1
    assert (wav.parent / "words.json").is_file()
    assert "ascend-utterances" not in FOLDER_CORPORA
    with pytest.raises(ConfigError, match="share one folder"):
        write_words_for([Item("ascend-utterances", "u1", wav, 5.0, "zh", None)], rows)


def test_language_seconds_sum_what_each_language_was_decoded_for():
    rows = [
        spoken("w", "m1", [("a", 0, 4)], language="zh"),
        spoken("w", "m1", [("b", 4, 6)], chunk=1, language="en"),
        spoken("w", "m1", [("c", 6, 7)], chunk=2),
    ]
    assert language_seconds(rows) == {"zh": 4.0, "en": 2.0, "none": 1.0}
