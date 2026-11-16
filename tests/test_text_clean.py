import pytest

from ytverbatim.contract.models import Flag
from ytverbatim.text.clean import Fillers, Script, normalize, to_simplified
from ytverbatim.text.offsets import remap_flags


def clean_en(text: str) -> str:
    return normalize(text, "en", script="zh-Hans", fillers="remove").text


def clean_zh(text: str, *, fillers: Fillers = "remove", script: Script = "zh-Hans") -> str:
    return normalize(text, "zh", script=script, fillers=fillers).text


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("so um we bought uh more", "so we bought more"),
        ("Um, the market rose.", "the market rose."),
        ("hmm I think, um, that it held", "I think, that it held"),
        ("errr the rate er moved", "the rate moved"),
    ],
)
def test_english_hesitations_go(before: str, after: str) -> None:
    assert clean_en(before) == after


def test_the_word_err_and_all_caps_acronyms_stay() -> None:
    assert clean_en("To err is human.") == "To err is human."
    assert clean_en("UM and ER are hospital units.") == "UM and ER are hospital units."
    assert clean_en("Umbrella sales erupted.") == "Umbrella sales erupted."


def test_chinese_fillers_go_only_at_boundaries_and_repeats_collapse() -> None:
    assert clean_zh("嗯，今天聊聊估值") == "今天聊聊估值"
    assert clean_zh("我们看，啊，这个数据") == "我们看，这个数据"
    assert clean_zh("就是说就是说市场很好") == "就是说市场很好"
    assert clean_zh("然后然后我们买入") == "然后我们买入"
    assert clean_zh("好啊我们走吧") == "好啊我们走吧", "a particle inside a clause is speech, not a filler"
    assert clean_zh("这个那个都看", fillers="remove") == "这个那个都看", "demonstratives are never fillers"


def test_fillers_are_kept_unless_asked_and_punctuation_is_always_tidied() -> None:
    assert normalize("um, so.. we waited,, then", "en", script="zh-Hans", fillers="keep").text == (
        "um, so.. we waited, then"
    )
    assert clean_zh("，，好的。。", fillers="keep") == "好的。"


def test_simplified_script_is_the_default_and_keep_leaves_traditional_text() -> None:
    assert clean_zh("這個軟體很好", fillers="keep") == "这个软体很好"
    assert clean_zh("這個軟體很好", fillers="keep", script="keep") == "這個軟體很好"
    assert to_simplified("Apple 這個") == "Apple 这个"
    assert normalize("這個", "en", script="zh-Hans", fillers="keep").text == "這個", "only zh text is converted"


def test_steps_chain_from_the_input_to_the_output_one_replacement_at_a_time() -> None:
    text = "um the GAAP number um rose"
    result = normalize(text, "en", script="zh-Hans", fillers="remove")
    assert result.text == "the GAAP number rose"
    assert result.steps[0][0] == text and result.steps[-1][1] == result.text
    assert all(a[1] == b[0] for a, b in zip(result.steps, result.steps[1:], strict=False))


def test_a_text_of_only_fillers_is_kept_as_it_was() -> None:
    result = normalize("um uh", "en", script="zh-Hans", fillers="remove")
    assert result.text == "um uh" and result.steps == ()


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("OK,所以我来自福建,所以", "OK，所以我来自福建，所以"),
        ("我们走吧.", "我们走吧。"),
        ("你好?", "你好？"),
        ("价格是3.5万,对吗?", "价格是3.5万，对吗？"),
        ("你好?!", "你好？！"),
        ("好,,我们走", "好，我们走"),
        ("真的吗!我不信;但是:好吧", "真的吗！我不信；但是：好吧"),
        ("我们走吧. 然后呢.好的. OK", "我们走吧。然后呢。好的。OK"),
        ("你好 , 我是 小明 ! 再见", "你好，我是 小明！再见"),
        ("OK, 所以", "OK，所以"),
        ("时间:10:30开会", "时间：10:30开会"),
    ],
)
def test_marks_next_to_chinese_are_written_full_width(before: str, after: str) -> None:
    assert clean_zh(before) == after


@pytest.mark.parametrize(
    "text",
    [
        "10:30开会",
        "1,000美元",
        "比分是3:2",
        "版本v1.2很好",
        "他说 OK, so we go",
        "访问example.com吧",
        "发邮件到a.b@example.com吧",
        "打开report.pdf看看",
        "访问https://example.com/a?b=1,c吧",
        "我在U.S.工作",
        "你好，我是小明。",
        "你好…… 然后",
        "他说好…然后走了",
        "5700...6000",
    ],
)
def test_numbers_latin_stretches_urls_and_ellipses_keep_their_marks(text: str) -> None:
    assert clean_zh(text) == text


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("他说好...然后走了", "他说好……然后走了"),
        ("他说好。。。然后", "他说好……然后"),
        ("好吧... 然后", "好吧……然后"),
        ("OK...然后", "OK……然后"),
        ("然后...OK", "然后……OK"),
        ("他说好......然后", "他说好……然后"),
        ("好...?", "好……？"),
    ],
)
def test_an_ellipsis_next_to_chinese_becomes_a_full_width_one(before: str, after: str) -> None:
    assert clean_zh(before, fillers="keep") == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ('他说:"好" ,然后', '他说："好"，然后'),
        ('他说:  "好"', '他说："好"'),
        ("你好.\n世界", "你好。\n世界"),
        ("你好,\n世界", "你好，\n世界"),
    ],
)
def test_a_converted_mark_drops_the_spaces_around_it_but_never_a_line_break(before: str, after: str) -> None:
    assert clean_zh(before, fillers="keep") == after


def test_full_width_marks_belong_to_the_zh_simplified_rule_only() -> None:
    assert clean_en("我们走吧, OK.") == "我们走吧, OK."
    assert clean_zh("我们走吧, 你好?", script="keep") == "我们走吧, 你好?"
    assert normalize("OK,所以", "en", script="zh-Hans", fillers="keep").text == "OK,所以"


def test_a_converted_mark_lets_the_zh_filler_rules_see_the_boundary() -> None:
    assert clean_zh("嗯,今天聊聊估值") == "今天聊聊估值"


def test_each_converted_mark_is_one_step_and_flag_offsets_follow() -> None:
    text = "OK,所以我来自福建,所以"
    result = normalize(text, "zh", script="zh-Hans", fillers="keep")
    assert [after for _, after in result.steps] == ["OK，所以我来自福建,所以", "OK，所以我来自福建，所以"]
    assert result.steps[0][0] == text and result.steps[-1][1] == result.text
    spaced = normalize("好 , 我", "zh", script="zh-Hans", fillers="keep")
    assert spaced.steps == (("好 , 我", "好，我"),)
    ellipsis = normalize("好吧 ... 然后", "zh", script="zh-Hans", fillers="keep")
    assert ellipsis.steps == (("好吧 ... 然后", "好吧……然后"),)
    quoted = normalize('他说:"好" ,然后', "zh", script="zh-Hans", fillers="keep")
    assert [after for _, after in quoted.steps] == ['他说："好" ,然后', '他说："好"，然后']
    assert normalize("好...?", "zh", script="zh-Hans", fillers="keep").steps == (
        ("好...?", "好……?"),
        ("好……?", "好……？"),
    )


@pytest.mark.parametrize(
    ("text", "word"),
    [('他说:"好" ,然后', "然后"), ("你好.\n世界", "世界")],
)
def test_flag_offsets_follow_steps_that_delete_whitespace(text: str, word: str) -> None:
    result = normalize(text, "zh", script="zh-Hans", fillers="keep")
    start = text.index(word)
    flag = Flag(kind="low_confidence", char_start=start, char_end=start + len(word), note="check")
    (moved,) = remap_flags([flag], result.steps)
    assert result.text[moved.char_start : moved.char_end] == word


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("I think... maybe", "I think... maybe"),
        ("Wait... what?", "Wait... what?"),
        ("Well.. ok", "Well.. ok"),
        ("Wait.....", "Wait..."),
    ],
)
def test_english_ellipses_are_kept_and_only_long_runs_shrink_to_three_dots(before: str, after: str) -> None:
    assert clean_en(before) == after


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("他说好。。。然后", "他说好……然后"),
        ("他说好。。然后", "他说好。然后"),
        ("他说好...然后", "他说好...然后"),
    ],
)
def test_ellipsis_runs_are_tidied_in_keep_mode_too(before: str, after: str) -> None:
    assert clean_zh(before, fillers="keep", script="keep") == after
