import random
import unicodedata
from itertools import product

import pytest

from anyverbatim.text.numbers import itn, quantities, quantity_tokens


def surfaces(text: str) -> list[str]:
    return [t.surface for t in quantity_tokens(text)]


@pytest.mark.parametrize("text", ["一样", "一直", "统一", "不一样", "十字路口", "万事如意", "一般来说"])
def test_lone_numeral_characters_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []


def test_ordinal_follows_di() -> None:
    (token,) = quantity_tokens("第一")
    assert (token.surface, token.value_itn, token.kind) == ("第一", "第1", "cjk_quantity")


@pytest.mark.parametrize(
    ("text", "value", "kind"),
    [
        ("一百五十期", "150期", "cjk_quantity"),
        ("二零二四年", "2024年", "cjk_quantity"),
        ("三十", "30", "cjk_quantity"),
        ("两百块", "200块", "currency"),
        ("兩百", "200", "cjk_quantity"),
        ("廿五", "25", "cjk_quantity"),
        ("三点五", "3.5", "cjk_quantity"),
        ("三点五亿美元", "$350000000", "currency"),
        ("一个", "1个", "cjk_quantity"),
        ("五股", "5股", "cjk_quantity"),
        ("三个百分点", "3个百分点", "cjk_quantity"),
        ("百分之五十", "50%", "percent"),
    ],
)
def test_chinese_quantities_are_converted(text: str, value: str, kind: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn, token.kind) == (text, value, kind)


def test_measure_word_makes_a_single_numeral_a_quantity() -> None:
    assert surfaces("买了三股和五个") == ["三股", "五个"]


def test_unconvertible_run_keeps_its_surface() -> None:
    (token,) = quantity_tokens("千万")
    assert token.value_itn == "千万"


@pytest.mark.parametrize(
    ("text", "surface", "value", "kind"),
    [
        ("1,000", "1,000", "1000", "number"),
        ("3.50", "3.50", "3.50", "number"),
        ("0700", "0700", "0700", "number"),
        ("-5%", "-5%", "-5%", "percent"),
        ("5 %", "5 %", "5%", "percent"),
        ("３.５％", "３.５％", "3.5%", "percent"),
        ("$1,500.50", "$1,500.50", "$1500.50", "currency"),
        ("200元", "200元", "200元", "currency"),
        ("3.5亿", "3.5亿", "350000000", "number"),
        ("3万美元", "3万美元", "$30000", "currency"),
        ("150期", "150期", "150期", "number"),
        ("3.5亿股", "3.5亿股", "350000000股", "number"),
    ],
)
def test_digit_quantities(text: str, surface: str, value: str, kind: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn, token.kind) == (surface, value, kind)


@pytest.mark.parametrize(
    "ticker",
    ["NVDA", "SPY", "Q2", "BRK.B", "0700.HK", "$NVDA", "$F", "NYSE:F"],
)
def test_tickers(ticker: str) -> None:
    (token,) = quantity_tokens(f"看 {ticker} 的")
    assert (token.surface, token.kind) == (ticker, "ticker")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3.35SMH", [("3.35", "number"), ("SMH", "ticker")]),
        ("5%NVDA", [("5%", "percent"), ("NVDA", "ticker")]),
        ("$1.2MSFT", [("$1.2", "currency"), ("MSFT", "ticker")]),
        ("10BABA", [("10", "number"), ("BABA", "ticker")]),
        ("5BRK.B", [("5", "number"), ("BRK.B", "ticker")]),
        ("３.３５SMH", [("３.３５", "number"), ("SMH", "ticker")]),
        ("涨了3.35SMH", [("3.35", "number"), ("SMH", "ticker")]),
    ],
)
def test_a_ticker_written_against_a_figure_is_its_own_token(text: str, expected: list[tuple[str, str]]) -> None:
    assert [(t.surface, t.kind) for t in quantity_tokens(text)] == expected
    spaced = [(t.surface, t.kind) for t in quantity_tokens(text.replace(expected[-1][0], f" {expected[-1][0]}"))]
    assert spaced == expected


@pytest.mark.parametrize(
    ("text", "surface", "value"),
    [
        ("$1.2M", "$1.2M", "$1200000"),
        ("10B", "10B", "10000000000"),
        ("5K", "5K", "5000"),
        ("3bn", "3bn", "3000000000"),
        ("10BN", "10BN", "10000000000"),
        ("1.5BN", "1.5BN", "1500000000"),
        ("$5MM", "$5MM", "$5000000"),
        ("$10MN", "$10MN", "$10000000"),
        ("5mn", "5mn", "5000000"),
        ("2TN", "2TN", "2000000000000"),
        ("$1tn", "$1tn", "$1000000000000"),
        ("$3T", "$3T", "$3000000000000"),
        ("5mm", "5", "5"),
        ("25BPS", "25", "25"),
        ("25bps", "25", "25"),
        ("3x", "3", "3"),
        ("3X", "3", "3"),
    ],
)
def test_figure_suffixes_are_not_tickers(text: str, surface: str, value: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn) == (surface, value)


def test_upper_and_lower_case_magnitudes_share_a_value() -> None:
    assert [t.value_itn for t in quantity_tokens("10BN 10bn 10B 10 billion")] == ["10000000000"] * 4


def test_a_run_that_only_starts_like_a_magnitude_is_a_ticker() -> None:
    assert [(t.surface, t.kind) for t in quantity_tokens("10BNX 10TB")] == [
        ("10", "number"),
        ("BNX", "ticker"),
        ("10", "number"),
        ("TB", "ticker"),
    ]


def test_letters_after_digits_inside_a_word_are_not_a_ticker() -> None:
    assert quantity_tokens("abc1XYZ") == []


def test_share_class_letter_before_gu() -> None:
    assert [(t.surface, t.kind) for t in quantity_tokens("A股")] == [("A", "ticker")]


def test_offsets_index_the_input() -> None:
    text = "SPY涨了3.5%，QQQ跌了一百五十期"
    for token in quantity_tokens(text):
        assert text[token.char_start : token.char_end] == token.surface
    assert surfaces(text) == ["SPY", "3.5%", "QQQ", "一百五十期"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1,000,000", [("1,000,000", "1000000")]),
        ("1,500.5", [("1,500.5", "1500.5")]),
        ("2023,2024", [("2023", "2023"), ("2024", "2024")]),
        ("5,1000", [("5", "5"), ("1000", "1000")]),
        ("５,１０００", [("５", "5"), ("１０００", "1000")]),
        ("1,000,0000", [("1,000", "1000"), ("0000", "0000")]),
    ],
)
def test_a_thousands_group_has_exactly_three_digits(text: str, expected: list[tuple[str, str]]) -> None:
    assert [(t.surface, t.value_itn) for t in quantity_tokens(text)] == expected


def test_full_width_thousands_groups_split_the_same_after_nfkc() -> None:
    assert [t.value_itn for t in quantity_tokens(unicodedata.normalize("NFKC", "５，１０００"))] == ["5", "1000"]


def test_dates_and_identifiers_are_not_split_into_signed_numbers() -> None:
    assert surfaces("2024-10-01 abc123") == ["2024", "10", "01"]


def test_digit_and_chinese_forms_of_a_quantity_share_one_value() -> None:
    forms = ["一百五十期", "150期", "三点五亿股", "3.5亿股"]
    values = [quantity_tokens(f)[0].value_itn for f in forms]
    assert values == ["150期", "150期", "350000000股", "350000000股"]


def test_itn_rewrites_tokens_only() -> None:
    assert itn("一百五十期涨了百分之五，一样 3.5亿") == "150期涨了5%，一样 350000000"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("2万亿美元", "$2000000000000"),
        ("两万亿美元", "$2000000000000"),
        ("1.5万亿", "1500000000000"),
        ("一点五万亿", "1500000000000"),
        ("3.5亿", "350000000"),
        ("三点五亿", "350000000"),
        ("2千万美元", "$20000000"),
        ("5百万", "5000000"),
    ],
)
def test_stacked_magnitudes_are_one_multiplier(text: str, value: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn) == (text, value)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("一亿五", "150000000"),
        ("三亿二", "320000000"),
        ("十一亿三", "1130000000"),
        ("五百亿一", "50010000000"),
        ("十亿五", "1050000000"),
        ("两万亿五", "2500000000000"),
        ("一億五", "150000000"),
        ("一萬億五", "1500000000000"),
        ("廿亿五", "2050000000"),
        ("一万亿五", "1500000000000"),
        ("三亿五美元", "$350000000"),
        ("三亿零五", "300000005"),
        ("一亿二千万", "120000000"),
        ("一万五", "15000"),
        ("十万五", "105000"),
        ("一百十五", "115"),
        ("一千十", "1010"),
        ("两百十", "210"),
        ("十三万六", "136000"),
        ("一百二十五万三", "1253000"),
        ("十五", "15"),
        ("二十五万三", "253000"),
        ("一点五亿", "150000000"),
        ("十点五万亿", "10500000000000"),
        ("一万五千点五", "15000.5"),
        ("一亿五千三", "153000000"),
        ("一亿五千", "150000000"),
        ("三亿五百", "305000000"),
        ("十一亿二千三百", "1123000000"),
        ("一万亿五千", "1500000000000"),
        ("一億五千三", "153000000"),
    ],
)
def test_chinese_shorthand_and_compound_numerals(text: str, value: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn) == (text, value)


@pytest.mark.parametrize(
    "text",
    [
        "七十六十五",
        "十六十四",
        "六十六十六",
        "七百九十九十七",
        "一万十五",
        "零十五",
        "亿三",
        "一亿亿五",
        "三亿二亿",
        "一亿两",
        "一亿五点五",
        "一亿五点五万",
        "一亿五点五亿",
        "一万五点五",
        "一千五点五",
        "一百五点五",
        "一亿五千点五",
    ],
)
def test_runs_that_are_not_one_number_keep_their_surface(text: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn) == (text, text)
    assert itn(f"约{text}左右") == f"约{text}左右"


# 一万亿 reads as 万亿 and 一万万亿 is no number, so a head with 万 only pairs with 亿 when 万 is not last.
_SHORTHAND_CASES = [
    (head, unit, exponent)
    for head in ["一", "三", "十", "十一", "二十", "五百", "一千", "两千", "一万", "三万五千"]
    for unit, exponent in [("亿", 8), ("万亿", 12)]
    if not ((unit == "亿" and head.endswith("万")) or (unit == "万亿" and "万" in head))
]


@pytest.mark.parametrize(("head", "unit", "exponent"), _SHORTHAND_CASES)
def test_digit_after_yi_takes_the_next_lower_unit(head: str, unit: str, exponent: int) -> None:
    (whole,) = quantity_tokens(f"{head}{unit}")
    for digit, name in enumerate("一二三四五六七八九", start=1):
        (token,) = quantity_tokens(f"{head}{unit}{name}")
        assert token.value_itn == str(int(whole.value_itn) + digit * 10 ** (exponent - 1))


_NUMERAL_ALPHABET = (*"零一二两三四五六七八九十百千万亿点几数", "分之", "个", "家", "年", "个月", "分钟")


def test_quantity_tokens_never_raise_on_numeral_runs() -> None:
    exhaustive = ("".join(chars) for length in range(1, 4) for chars in product(_NUMERAL_ALPHABET, repeat=length))
    rng = random.Random(20260101)
    sampled = ("".join(rng.choices(_NUMERAL_ALPHABET, k=rng.randint(4, 8))) for _ in range(5000))
    for text in (*exhaustive, *sampled):
        quantity_tokens(text)
        itn(text)


@pytest.mark.parametrize("text", ["两三个", "三五个", "一一对应", "五五开", "三三两两", "四五十个", "三四百", "九七年"])
def test_adjacent_bare_digits_are_a_range_or_an_idiom_not_one_number(text: str) -> None:
    assert quantity_tokens(text) == []
    assert itn(text) == text


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("二〇一八年", "2018年"),
        ("二零二四年", "2024年"),
        ("一九九七年", "1997年"),
        ("零零后", "00后"),
        ("零点五", "0.5"),
    ],
)
def test_digit_by_digit_years_and_codes(text: str, value: str) -> None:
    assert itn(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "千万不要",
        "千万别",
        "千万记住",
        "万一",
        "万万没想到",
        "一一",
        *(f"周{d}" for d in "一二三四五六日天"),
        *(f"星期{d}" for d in "一二三四五六日天"),
        "礼拜三",
        "一些",
        "一点儿",
        "好一点",
        "一点也不",
        "一下",
        "一定",
        "一起",
        "一直",
        "一样",
        "一般",
        "一致",
        "统一",
        "唯一",
        "一旦",
        "一边",
        "一次性",
        "一成不变",
        "一股脑",
        "一个劲",
        "一天到晚",
        "三天两头",
        "一五一十",
    ],
)
def test_idioms_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []
    assert itn(text) == text


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("每周一次", "每周1次"),
        ("一千万元", "10000000元"),
        ("三万一千", "31000"),
        ("一点五", "1.5"),
        ("一点钟", "1点钟"),
        ("一亿两千万", "120000000"),
        ("百分之三点五", "3.5%"),
        ("一成", "1成"),
    ],
)
def test_quantities_next_to_idiom_lookalikes_still_convert(text: str, value: str) -> None:
    assert itn(text) == value


@pytest.mark.parametrize(
    ("text", "surface", "value", "kind"),
    [
        ("$15 billion", "$15 billion", "$15000000000", "currency"),
        ("15 billion", "15 billion", "15000000000", "number"),
        ("$1.5B", "$1.5B", "$1500000000", "currency"),
        ("2.5 Million", "2.5 Million", "2500000", "number"),
        ("15K", "15K", "15000", "number"),
        ("3bn", "3bn", "3000000000", "number"),
        ("15 percent", "15 percent", "15%", "percent"),
        ("15 dollars", "15 dollars", "$15", "currency"),
        ("15 billion dollars", "15 billion dollars", "$15000000000", "currency"),
    ],
)
def test_english_magnitudes_and_unit_words(text: str, surface: str, value: str, kind: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn, token.kind) == (surface, value, kind)


def test_dollar_amounts_agree_however_they_are_written() -> None:
    values = {
        quantity_tokens(text)[0].value_itn for text in ("$15", "15美元", "十五美元", "15 dollars", "fifteen dollars")
    }
    assert values == {"$15"}
    assert quantity_tokens("$15 billion")[0].value_itn != quantity_tokens("$15 million")[0].value_itn


def test_magnitude_words_need_a_word_boundary() -> None:
    assert surfaces("15 millionaires and 15 percentage points") == ["15", "15"]


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("fifteen", ["15"]),
        ("Twenty-five", ["25"]),
        ("twenty five percent", ["25%"]),
        ("two hundred and ten dollars", ["$210"]),
        ("fifteen hundred", ["1500"]),
        ("one million two hundred thousand", ["1200000"]),
        ("nineteen ninety", ["19", "90"]),
        ("five and six", ["5", "6"]),
        ("one percent", ["1%"]),
    ],
)
def test_spelled_out_english_numbers(text: str, values: list[str]) -> None:
    assert [t.value_itn for t in quantity_tokens(text)] == values


@pytest.mark.parametrize("text", ["no one knows", "one of them", "someone", "the tenant", "a million reasons"])
def test_english_words_that_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("三万一个月", [("三万", "30000"), ("一个月", "1个月")]),
        ("三万一一个月", [("三万一", "31000"), ("一个月", "1个月")]),
        ("五百亿一个", [("五百亿", "50000000000"), ("一个", "1个")]),
        ("三点五万一年", [("三点五万", "35000"), ("一年", "1年")]),
        ("一亿五一股", [("一亿五", "150000000"), ("一股", "1股")]),
        ("三萬一年", [("三萬", "30000"), ("一年", "1年")]),
    ],
)
def test_one_after_a_figure_in_wan_or_yi_starts_a_per_unit_phrase(text: str, expected: list[tuple[str, str]]) -> None:
    assert [(t.surface, t.value_itn) for t in quantity_tokens(text)] == expected


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("十一个", "11个"),
        ("一百一个", "110个"),
        ("两千一个月", "2100个月"),
        ("三万一", "31000"),
        ("五百亿一", "50010000000"),
        ("三万一块", "31000块"),
        ("第三万一个", "第31000个"),
    ],
)
def test_one_stays_part_of_the_number_elsewhere(text: str, value: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn) == (text, value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("三点四十分", [("三点", "3点"), ("四十", "40")]),
        ("十二点三十五", [("十二点", "12点"), ("三十五", "35")]),
        ("下午四点四十分", [("四点", "4点"), ("四十", "40")]),
        ("二十四点五十九", [("二十四点", "24点"), ("五十九", "59")]),
        ("一点十分", [("一点", "1点"), ("十", "10")]),
        ("三点廿", [("三点", "3点"), ("廿", "20")]),
        ("一点五万", [("一点五万", "15000")]),
        ("三点一四", [("三点一四", "3.14")]),
        ("三点零五", [("三点零五", "3.05")]),
    ],
)
def test_tens_after_dian_make_it_a_clock_unit(text: str, expected: list[tuple[str, str]]) -> None:
    assert [(t.surface, t.value_itn) for t in quantity_tokens(text)] == expected


@pytest.mark.parametrize(
    ("text", "value", "kind"),
    [
        ("三家公司", "3家", "cjk_quantity"),
        ("两只股票", "2只", "cjk_quantity"),
        ("五分钟", "5分钟", "cjk_quantity"),
        ("两个小时", "2个", "cjk_quantity"),
        ("三小时", "3小时", "cjk_quantity"),
        ("七十六岁", "76岁", "cjk_quantity"),
        ("一百二十周年", "120周年", "cjk_quantity"),
        ("二十世纪", "20世纪", "cjk_quantity"),
        ("五十个基点", "50个基点", "cjk_quantity"),
        ("二十五基点", "25基点", "cjk_quantity"),
        ("一百平方米", "100平方米", "cjk_quantity"),
        ("一千米", "1000米", "cjk_quantity"),
        ("3家", "3家", "number"),
        ("5分钟", "5分钟", "number"),
    ],
)
def test_count_classifiers_and_units_make_a_quantity(text: str, value: str, kind: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.value_itn, token.kind) == (value, kind)


@pytest.mark.parametrize(
    "text", ["十分重要", "三部曲", "二手房", "二线城市", "六方会谈", "三大银行", "进一步", "一步一步"]
)
def test_words_after_a_numeral_that_are_not_counts(text: str) -> None:
    assert quantity_tokens(text) == []


@pytest.mark.parametrize(
    ("text", "surface", "value", "kind"),
    [
        ("三分之一", "三分之一", "1/3", "cjk_quantity"),
        ("二十分之一", "二十分之一", "1/20", "cjk_quantity"),
        ("千分之十五", "千分之十五", "15/1000", "cjk_quantity"),
        ("万分之一", "万分之一", "1/10000", "cjk_quantity"),
        ("几万分之一", "万分之一", "1/10000", "cjk_quantity"),
        ("一百分之一", "一百分之一", "1/100", "cjk_quantity"),
        ("三四分之一", "三四分之一", "三四分之一", "cjk_quantity"),
        ("1/3", "1/3", "1/3", "number"),
        ("24/7", "24/7", "24/7", "number"),
    ],
)
def test_a_fraction_is_one_token_written_over_under(text: str, surface: str, value: str, kind: str) -> None:
    (quantity,) = quantities(text)
    assert (quantity.token.surface, quantity.token.value_itn, quantity.token.kind) == (surface, value, kind)
    assert quantity.value is None


def test_a_slash_date_is_not_a_fraction() -> None:
    assert surfaces("2024/10/01") == ["2024", "10", "01"]


@pytest.mark.parametrize("text", ["十万八千里", "相差十万八千里", "百度", "百度一下"])
def test_idioms_with_place_words_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []


def test_baifenzhi_on_its_own_is_still_a_percent() -> None:
    assert itn("涨了百分之二十，三分之一") == "涨了20%，1/3"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("二十几岁", [("二十", "~20")]),
        ("20几岁", [("20", "~20")]),
        ("数十个", [("十个", "~10个")]),
        ("几十个", [("十个", "~10个")]),
        ("数十亿", [("十亿", "~1000000000")]),
        ("数10亿", [("10亿", "~1000000000")]),
        ("数十亿美元", [("十亿美元", "~$1000000000")]),
        ("数百万", [("百万", "~1000000")]),
        ("数100万", [("100万", "~1000000")]),
        ("几千", []),
        ("三千几百块", [("三千", "~3000"), ("百块", "~100块")]),
        # After a noun ending in 数 a run of place words alone may be exact or rough; both spellings read it as rough.
        ("参与人数十万", [("十万", "~100000")]),
        ("参与人数10万", [("10万", "~100000")]),
        ("股数十万股", [("十万股", "~100000股")]),
        ("次数十次", [("十次", "~10次")]),
    ],
)
def test_ji_and_shu_make_a_rough_figure(text: str, expected: list[tuple[str, str]]) -> None:
    found = quantities(text)
    assert [(q.token.surface, q.token.value_itn) for q in found] == expected
    assert all(q.value is None for q in found)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("指数十月份", "指数10月份"),
        ("指数10月份", "指数10月份"),
        ("几十点", "几~10点"),
        ("跌了几十点", "跌了几~10点"),
        ("数十点", "数~10点"),
        ("几十号", "几~10号"),
        ("几十日", "几~10日"),
        ("数十日", "数~10日"),
        ("数10日", "数~10日"),
        ("几十月", "几~10月"),
    ],
)
def test_only_a_month_after_shu_is_never_rough(text: str, value: str) -> None:
    assert itn(text) == value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("三到五个", [("三", "3"), ("五个", "5个")]),
        ("三至五年", [("三", "3"), ("五年", "5年")]),
        ("两、三家", [("两", "2"), ("三家", "3家")]),
        ("三到五到七个", [("三", "3"), ("五", "5"), ("七个", "7个")]),
        ("五个到八", [("五个", "5个"), ("八", "8")]),
        ("十~二十个", [("十", "10"), ("二十个", "20个")]),
        ("3到五个", [("3", "3"), ("五个", "5个")]),
        ("三到5个", [("三", "3"), ("5个", "5个")]),
    ],
)
def test_a_lone_numeral_linked_to_a_figure_is_the_other_end_of_its_range(
    text: str, expected: list[tuple[str, str]]
) -> None:
    assert [(t.surface, t.value_itn) for t in quantity_tokens(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "第三到五",
        "几到五个",
        "星期一到五",
        "三到几个",
        "三块五",
        "说到三",
        # A lone end shares the percent of its partner, so read alone it would be a bare figure, and a magnitude on
        # the left is lent to nothing.
        "百分之三到五",
        "三万到五",
        # 、 lists figures as often as it spans them, so it lends no magnitude.
        "三、五万",
        # 一到 before a date or a time is "as soon as".
        "一到三月份就开始涨",
        "一到十二点就睡",
    ],
)
def test_a_lone_numeral_without_a_figure_to_anchor_it_is_no_quantity(text: str) -> None:
    assert all(len(t.surface) > 1 for t in quantity_tokens(text))


@pytest.mark.parametrize(
    ("chinese", "digits", "values"),
    [
        ("两到三亿", "2到3亿", ["200000000", "300000000"]),
        ("两到3亿", "2到3亿", ["200000000", "300000000"]),
        ("两到3亿美元", "2到3亿美元", ["$200000000", "$300000000"]),
        ("两到三亿美元", "2到3亿美元", ["$200000000", "$300000000"]),
        ("三到五万", "3到5万", ["30000", "50000"]),
        ("一到三万", "1到3万", ["10000", "30000"]),
        ("两到3万", "2到3万", ["20000", "30000"]),
        ("三至五万", "3至5万", ["30000", "50000"]),
        ("三~五万", "3~5万", ["30000", "50000"]),
        ("两到三万亿", "2到3万亿", ["2000000000000", "3000000000000"]),
        ("两到三千万", "2到3千万", ["20000000", "30000000"]),
        ("两到三百万", "2到300万", ["2000000", "3000000"]),
        ("五到十万", "5到10万", ["50000", "100000"]),
        ("二十到三十万", "20到30万", ["200000", "300000"]),
        ("一千到两千万", "1000到2000万", ["10000000", "20000000"]),
        ("三点五到四万", "3.5到4万", ["35000", "40000"]),
        ("三到五到七万", "3到5到7万", ["30000", "50000", "70000"]),
        ("三到五到七万", "3到五到七万", ["30000", "50000", "70000"]),
        ("一到二到三亿", "一到2到3亿", ["100000000", "200000000", "300000000"]),
        ("五到十二万", "5到12万", ["50000", "120000"]),
        ("两到十五万", "2到15万", ["20000", "150000"]),
        ("两到十万", "2到10万", ["20000", "100000"]),
        ("两到三点五亿", "2到3.5亿", ["200000000", "350000000"]),
        ("一点五到两百万", "1.5到2百万", ["1500000", "2000000"]),
    ],
)
def test_a_range_start_shares_the_magnitude_and_currency_of_its_end_in_either_script(
    chinese: str, digits: str, values: list[str]
) -> None:
    assert [t.value_itn for t in quantity_tokens(chinese)] == values
    assert [t.value_itn for t in quantity_tokens(digits)] == values


@pytest.mark.parametrize(
    ("text", "values"),
    [
        # The end's measure word stays its own, as in 三到五个.
        ("三到五万人", ["30000", "50000人"]),
        # A start scaled to the end's size or above it is not the start of that range.
        ("五到两百万", ["50000", "2000000"]),
        ("两千到三万", ["2000", "30000"]),
        # 千 and 百 alone are no magnitude to lend.
        ("两到三千", ["2", "3000"]),
        ("2到3千", ["2", "3000"]),
        ("三万到五万", ["30000", "50000"]),
        ("二十几到三十万", ["~20", "300000"]),
        ("2、3万", ["2", "30000"]),
        # One digit could take the end's magnitude or its leading place, and only 两到三百万 says which.
        ("两到三十万", ["300000"]),
        ("两到三千五百万", ["35000000"]),
        ("两到三百五十万", ["3500000"]),
        ("2到30万", ["2", "300000"]),
    ],
)
def test_a_range_start_shares_a_magnitude_only_when_it_has_none_and_stays_below_the_end(
    text: str, values: list[str]
) -> None:
    assert [t.value_itn for t in quantity_tokens(text)] == values


def test_a_scaled_range_start_keeps_its_surface_and_offsets() -> None:
    text = "大概两到3亿美元的收入"
    assert [(t.char_start, t.char_end, t.surface, t.kind) for t in quantity_tokens(text)] == [
        (2, 3, "两", "currency"),
        (4, 8, "3亿美元", "currency"),
    ]


def test_a_rough_figure_never_shares_a_value_with_an_exact_one() -> None:
    assert itn("二十几岁") != itn("二十岁")
    assert itn("二十几") != itn("二十")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("指数两千八百点", "指数2800点"),
        ("人数两万", "人数20000"),
        ("人数十五万人", "人数150000人"),
        ("股数三十万股", "股数300000股"),
        ("二十几乎", "20几乎"),
        ("指数1000点", "指数1000点"),
        ("人数15万", "人数150000"),
    ],
)
def test_a_digit_after_a_word_ending_in_shu_makes_an_exact_figure(text: str, value: str) -> None:
    assert itn(text) == value


@pytest.mark.parametrize(
    ("text", "surface"),
    [
        ("百分之六十支持", "百分之六十"),
        ("从百分之三升到", "百分之三"),
        ("百分之三十人口", "百分之三十"),
        ("百分之五十个", "百分之五十"),
    ],
)
def test_a_spoken_percent_takes_no_measure_word(text: str, surface: str) -> None:
    (token,) = quantity_tokens(text)
    assert token.surface == surface
    assert token.kind == "percent"


@pytest.mark.parametrize("text", ["一百点二十", "三点五十年期", "二十五点三十"])
def test_a_clock_time_needs_an_hour_and_minutes_and_no_unit(text: str) -> None:
    (quantity,) = quantities(text)
    assert "点" in quantity.token.surface
    assert quantity.token.value_itn == quantity.token.surface
    assert quantity.value is None


@pytest.mark.parametrize("text", ["一亿五千零三", "三亿二千零五十"])
def test_zero_inside_the_group_after_yi_keeps_the_surface(text: str) -> None:
    (token,) = quantity_tokens(text)
    assert token.value_itn == text


def test_tokens_from_one_run_tile_it_in_order() -> None:
    text = "收入三万一个月，三点四十分开会"
    for token in quantity_tokens(text):
        assert text[token.char_start : token.char_end] == token.surface
    assert surfaces(text) == ["三万", "一个月", "三点", "四十"]


@pytest.mark.parametrize("text", ["从一到三月份", "一至三月份", "一～三月份", "一-三月份"])
def test_one_opens_a_range_before_a_date_after_cong_or_with_another_link(text: str) -> None:
    assert [(t.surface, t.value_itn) for t in quantity_tokens(text)] == [("一", "1"), ("三月", "3月")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("两点零五分", [("两点", "2点"), ("零五", "05")]),
        ("下午三点零五", [("三点", "3点"), ("零五", "05")]),
        ("三点零五", [("三点零五", "3.05")]),
    ],
)
def test_zero_minutes_are_a_clock_time_only_with_fen_or_a_time_of_day(
    text: str, expected: list[tuple[str, str]]
) -> None:
    assert [(t.surface, t.value_itn) for t in quantity_tokens(text)] == expected


def test_a_run_never_takes_the_bai_of_a_following_percent() -> None:
    assert [(t.surface, t.value_itn) for t in quantity_tokens("百分之五百分之五")] == [
        ("百分之五", "5%"),
        ("百分之五", "5%"),
    ]


@pytest.mark.parametrize(
    "text", ["五十步笑百步", "三十而立", "十二生肖", "三十六计", "十一黄金周", "一号文件", "八月十五"]
)
def test_set_terms_with_numerals_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []


def test_the_fifteenth_of_august_is_still_a_date() -> None:
    assert surfaces("八月十五日") == ["八月", "十五日"]


def test_two_figures_in_wan_run_together_are_not_one_number() -> None:
    (token,) = [t for t in quantity_tokens("四万三万五") if t.surface == "四万三万五"]
    assert token.value_itn == "四万三万五"
