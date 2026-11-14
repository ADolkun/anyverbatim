import json

from click.testing import CliRunner

from evals.questions import (
    MAX_ANCHOR_CHARS,
    MAX_OPTION_CHARS,
    MAX_SENTENCES,
    NO_PREFERENCE,
    OTHER,
    SPOT,
    UNCLEAR,
    AttributionDispute,
    AttributionLine,
    TextDispute,
    anchor_text,
    build_questions,
    clock,
    limit_sentences,
    load_disputes,
    option_text,
    pick_anchors,
    redact,
    sentence_count,
    split_sentences,
    spot_excerpt,
    write_questions,
)
from evals.questions import (
    main as build_group,
)


def test_split_sentences_handles_both_scripts_and_decimals():
    assert split_sentences("It costs 3.5 dollars. Really? Yes!") == ["It costs 3.5 dollars.", "Really?", "Yes!"]
    assert split_sentences("你好。我们开始吧！好的") == ["你好。", "我们开始吧！", "好的"]
    assert split_sentences("   ") == []


def test_limit_sentences_keeps_the_first_two():
    assert limit_sentences("One. Two. Three. Four.") == "One. Two."
    assert limit_sentences("Only one") == "Only one"
    assert sentence_count(f"{SPOT} and then. More.") == 2


def test_spot_excerpt_adds_the_previous_sentence():
    text = spot_excerpt("We shipped it. Then the price was ", " dollars. After that we left. Bye.")
    assert text == f"We shipped it. Then the price was {SPOT} dollars."
    assert sentence_count(text) == MAX_SENTENCES


def test_spot_excerpt_falls_back_to_the_next_sentence_at_the_start():
    text = spot_excerpt("Then the price was ", " dollars. After that we left. Bye.")
    assert text == f"Then the price was {SPOT} dollars. After that we left."


def test_spot_excerpt_at_a_sentence_boundary_keeps_the_marker_in_its_own_sentence():
    text = spot_excerpt("First one. Second one. ", "Third one. Fourth.")
    assert text == f"Second one. {SPOT}Third one."
    assert sentence_count(text) <= MAX_SENTENCES


def test_spot_excerpt_never_exceeds_the_limit_whatever_the_context():
    before = "A. B. C. D. e f g "
    after = " h i. J. K. L."
    assert sentence_count(spot_excerpt(before, after)) <= MAX_SENTENCES
    assert sentence_count(spot_excerpt(before, after, limit=1)) == 1
    assert sentence_count(spot_excerpt("你说什么。我觉得", "不对。然后呢。")) <= MAX_SENTENCES


def test_redact_hides_names_by_whole_word_and_in_cjk():
    assert redact("Anna told Joanna and anna's dog", ["Anna"]) == "[name] told Joanna and [name]'s dog"
    assert redact("小明说小明的书", ["小明"]) == "[name]说[name]的书"
    assert redact("nothing here", ["", "  "]) == "nothing here"


def test_clock_and_anchor_text_cut_to_one_short_sentence():
    assert clock(740.9) == "12:20"
    assert anchor_text("Short one. A second sentence.") == "Short one."
    assert anchor_text("w" * 200).endswith("…") and len(anchor_text("w" * 200)) == 81


def line(start, end, text, **speakers):
    return AttributionLine(start=start, end=end, text=text, speakers=speakers)


CANDIDATES = ["n8", "n16"]
LINES = (
    line(10, 14, "I think we should ship it on Friday.", n8="A", n16="A"),
    line(30, 33, "Okay.", n8="A", n16="A"),
    line(40, 44, "My view is the budget is too small.", n8="B", n16="B"),
    line(60, 64, "Nobody agrees about this line at all.", n8="A", n16="B"),
    line(100, 104, "My view is the budget is too small.", n8="B", n16="B"),
    line(120, 124, "Totally different sentence from speaker B.", n8="B", n16="B"),
    line(70, 76, "This one sits inside the disputed stretch.", n8="A", n16="A"),
)


def test_pick_anchors_needs_agreement_distance_length_and_uniqueness():
    chosen = pick_anchors(LINES, ["A", "B"], 70, 76, CANDIDATES)
    assert chosen["A"].start == 10  # nearest agreed line outside the stretch; "Okay." is too short, 70-76 overlaps
    assert chosen["B"].start == 100  # the line at 60 is disputed; 100 is nearer than 40
    assert chosen["B"].text == "My view is the budget is too small."
    # With A's anchor taking the shared sentence, B must pick a different one.
    shared = (
        line(10, 14, "The same long sentence here.", n8="A", n16="A"),
        line(50, 54, "The same long sentence here.", n8="B", n16="B"),
        line(90, 94, "Another long sentence for B.", n8="B", n16="B"),
    )
    again = pick_anchors(shared, ["A", "B"], 60, 70, CANDIDATES)
    assert again["A"].start == 10
    assert again["B"].start == 90


def attribution(**overrides):
    base = {
        "id": "d1",
        "recording": "rec-01",
        "start": 70.0,
        "end": 76.0,
        "speakers": {"n8": "A", "n16": "B"},
        "lines": LINES,
        "clip": "clips/d1.wav",
    }
    return AttributionDispute(**{**base, **overrides})


def test_attribution_question_offers_anchor_lines_and_never_a_speaker_id():
    built = build_questions([attribution()], names=["Friday"])
    (question,) = built.questions
    (example,) = question.examples
    assert question.kind == "attribution"
    assert question.prompt == "Who is speaking between 1:10 and 1:16?"
    assert example.options == (
        "the person who says 'I think we should ship it on [name].' at 0:10",
        "the person who says 'My view is the budget is too small.' at 1:40",
        OTHER,
        UNCLEAR,
    )
    assert built.key[0].options == ((("A",), ("B",), (), ()),)
    assert "n8" not in question.model_dump_json()


def test_attribution_without_two_anchored_speakers_is_skipped_with_a_reason():
    built = build_questions([attribution(lines=LINES[:2])])
    assert built.questions == ()
    assert built.skipped == (("d1", "fewer than two speakers have an anchor line every candidate agrees on"),)


def text_dispute(item, start, group=None, **candidates):
    return TextDispute(
        id=item,
        recording="rec-01",
        start=start,
        end=start + 1,
        before="He paid ",
        after=" for it. Then he left. Done.",
        candidates=candidates or {"p2": "150", "p3": "一百五十", "p0": "150"},
        group=group,
    )


def test_policy_groups_come_first_capped_at_three_examples_with_blind_options():
    disputes = [
        text_dispute("t1", 5),
        text_dispute("t2", 10, group="numerals"),
        text_dispute("t3", 20, group="numerals"),
        text_dispute("t4", 30, group="numerals"),
        text_dispute("t5", 40, group="numerals"),
    ]
    built = build_questions(disputes, seed=1)
    assert [q.kind for q in built.questions] == ["policy", "text"]
    policy = built.questions[0]
    assert policy.group == "numerals"
    assert len(policy.examples) == 3
    assert built.key[0].sources == ("t2", "t3", "t4")
    for example, option_sources in zip(policy.examples, built.key[0].options, strict=True):
        assert example.excerpt.count(SPOT) == 1
        assert sentence_count(example.excerpt) <= MAX_SENTENCES
        assert sorted(example.options[:-2]) == ["150", "一百五十"]
        assert example.options[-2:] == (NO_PREFERENCE, UNCLEAR)
        assert sorted(option_sources[:2]) == [("p0", "p2"), ("p3",)]
    assert build_questions(disputes, seed=1) == built


def test_text_options_are_deduplicated_so_the_owner_sees_each_form_once():
    built = build_questions([text_dispute("t1", 5)])
    assert len(built.questions[0].examples[0].options) == 2 + 2


def test_questions_contain_no_name_from_the_names_list_and_round_trip(tmp_path):
    dispute = TextDispute(
        id="t1",
        recording="rec-01",
        start=1.0,
        end=2.0,
        before="Maria said that ",
        after=" was fine. Maria agreed.",
        candidates={"a": "Maria", "b": "Mario"},
    )
    built = build_questions([dispute], names=["Maria"])
    questions_path, key_path = write_questions(built, tmp_path / "out")
    written = questions_path.read_text(encoding="utf-8")
    assert "Maria" not in written
    assert "[name]" in written
    assert json.loads(key_path.read_text(encoding="utf-8").splitlines()[0])["sources"] == ["t1"]
    disputes_path = tmp_path / "disputes.jsonl"
    disputes_path.write_text(
        dispute.model_dump_json() + "\n" + attribution().model_dump_json() + "\n", encoding="utf-8"
    )
    loaded = load_disputes(disputes_path)
    assert [type(d) for d in loaded] == [TextDispute, AttributionDispute]
    assert loaded[0] == dispute


def test_a_group_with_one_item_is_still_a_policy_question():
    built = build_questions([text_dispute("t1", 5, group="fillers")])
    assert [q.kind for q in built.questions] == ["policy"]


def test_sentence_ends_close_over_quotes_and_brackets():
    assert split_sentences('He said "stop." Then (we left.) Done') == ['He said "stop."', "Then (we left.)", "Done"]
    assert split_sentences("他说“好。”然后走了") == ["他说“好。”", "然后走了"]


def test_option_text_is_redacted_then_limited_in_sentences_and_characters():
    long = "Maria said so. It was fine. Then more words " + "x" * 300
    shown = option_text(long, ["Maria"])
    assert shown == "[name] said so. It was fine."
    assert len(option_text("y" * 500)) <= MAX_OPTION_CHARS + 1


def test_text_options_that_differ_only_beyond_the_limit_show_once():
    built = build_questions(
        [text_dispute("t1", 5, a="Same start. Same second. Third one.", b="Same start. Same second. Other.")]
    )
    assert built.questions[0].examples[0].options[:-2] == ("Same start. Same second.",)
    assert built.key[0].options[0][0] == (("a", "b"))


def test_redaction_happens_before_truncation_so_no_name_fragment_shows():
    name = "Bartholomew"
    text = "x" * (MAX_ANCHOR_CHARS - 4) + f" {name} spoke here"
    shown = anchor_text(text, [name])
    assert "Bartho" not in shown
    assert "[name]" in shown or len(shown) <= MAX_ANCHOR_CHARS + 1


def test_the_cli_needs_names_or_an_explicit_no_names(tmp_path):
    disputes = tmp_path / "d.jsonl"
    disputes.write_text(text_dispute("t1", 5).model_dump_json() + "\n", encoding="utf-8")
    runner = CliRunner()
    args = ["build", str(disputes), "--out", str(tmp_path / "out")]
    assert runner.invoke(build_group, args).exit_code != 0
    names = tmp_path / "names.txt"
    names.write_text("Maria\n", encoding="utf-8")
    assert runner.invoke(build_group, [*args, "--names-file", str(names), "--no-names"]).exit_code != 0
    assert runner.invoke(build_group, [*args, "--no-names"]).exit_code == 0
    assert runner.invoke(build_group, [*args, "--names-file", str(names)]).exit_code == 0
