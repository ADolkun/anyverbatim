import itertools
import random

import pytest

from anyverbatim.text.numbers import NUMERALS, itn, quantity_tokens
from anyverbatim.text.numerals import apply, conversions, display

# Plan section 5.3: input after correction, the display form, and the rule it shows.
CONVERTED = [
    ("六百五十六美元", "656美元"),
    ("三块钱", "3块钱"),
    ("一美元", "1美元"),
    ("一块钱", "1块钱"),
    ("百分之八十", "80%"),
    ("百分之零点五", "0.5%"),
    ("百分之一百二十", "120%"),
    ("三个百分点", "3个百分点"),
    ("一个百分点", "1个百分点"),
    ("五十个基点", "50个基点"),
    ("涨了一倍", "涨了1倍"),
    ("三成", "3成"),
    ("三点五", "3.5"),
    ("零点五", "0.5"),
    ("涨了二十", "涨了20"),
    ("一千五", "1500"),
    ("一百二", "120"),
    ("三万", "3万"),
    ("三万五", "3.5万"),
    ("三万五千", "3.5万"),
    ("三十五万", "35万"),
    ("三千五百万", "3500万"),
    ("一亿五", "1.5亿"),
    ("一亿五千万", "1.5亿"),
    ("一亿五千三", "1.53亿"),
    ("十一亿三", "11.3亿"),
    ("一点五万亿", "1.5万亿"),
    ("两万亿美元", "2万亿美元"),
    ("三万五千二百", "3.52万"),
    ("一万零五", "10005"),
    ("十亿零五千", "1000005000"),
    ("4千", "4000"),
    ("3千万", "3000万"),
    ("三个月", "3个月"),
    ("两年", "2年"),
    ("十二天", "12天"),
    ("三家公司", "3家公司"),
    ("两只股票", "2只股票"),
    ("五分钟", "5分钟"),
    ("两个小时", "2个小时"),
    ("七十六岁", "76岁"),
    ("一百二十周年", "120周年"),
    ("共二百三十五位同学", "共235位同学"),
    ("约三百人", "约300人"),
    ("一百多件", "100多件"),
    ("二十余次", "20余次"),
    ("四十左右", "40左右"),
    ("三到五个", "3到5个"),
    ("三万到五万", "3万到5万"),
    # A range start shares its end's 万 or 亿 and keeps its own digit, never written as 2亿 or 200000000.
    ("两到三亿", "2到3亿"),
    ("两到3亿美元", "2到3亿美元"),
    ("三到五万", "3到5万"),
    ("一到三万", "1到3万"),
    ("二十到三十万", "20到30万"),
    ("二零二六年", "2026年"),
    ("二〇二六年", "2026年"),
    ("一九九七年七月一日", "1997年7月1日"),
    ("十二月二十五日", "12月25日"),
    ("三月五号", "3月5号"),
    ("一月份", "1月份"),
    ("二十世纪八十年代", "20世纪80年代"),
    ("下午三点", "下午3点"),
    ("下午四点四十分", "下午4点40分"),
    ("一点半", "1点半"),
    ("三点十分", "3点10分"),
    ("三点半", "3点半"),
    ("三点四十分", "3点40分"),
    ("十二点三十五", "12点35"),
    ("凌晨两点", "凌晨2点"),
    ("明天上午十点开会", "明天上午10点开会"),
    ("三点钟", "3点钟"),
    ("涨了三点", "涨了3点"),
    ("下跌二十四点", "下跌24点"),
    ("收涨五点", "收涨5点"),
    ("大跌了八点", "大跌了8点"),
    ("指数两千八百点", "指数2800点"),
    ("涨了二十四点", "涨了24点"),
    ("十二点", "12点"),
    ("零零后", "00后"),
    ("九零后", "90后"),
    ("零零七", "007"),
    ("沪深三百ETF", "沪深300ETF"),
    ("三萬五", "3.5萬"),
    ("二百五", "250"),
    ("三万一个月", "3万一个月"),
    ("五百亿一个", "500亿一个"),
    ("指数十月份", "指数10月份"),
    ("三、四、五个", "3、4、5个"),
    ("一到二到三个", "1到2到3个"),
    ("从一到三月份", "从1到3月份"),
    ("一至三月", "1至3月"),
    ("一～三月", "1～3月"),
    ("一-三月", "1-3月"),
    ("下午两点到四点", "下午2点到4点"),
    ("晚上八点到十点", "晚上8点到10点"),
    ("两点到三点半", "2点到3点半"),
    ("涨了三点到五点", "涨了3点到5点"),
    ("两点零五分", "2点05分"),
    ("下午三点零五分", "下午3点05分"),
    ("百分之五百分之五", "5%5%"),
    ("上升两点", "上升2点"),
]

UNCHANGED = [
    "三块五",
    "35000",
    "3.5万",
    "高三三个班",
    "三四个月",
    "一二十个",
    "四十五六岁",
    "五六万套",
    "五六十年前",
    "几千",
    "二十几岁",
    "一百几十",
    "几十个",
    "数十亿",
    "数十年",
    "上千个",
    "十几个",
    "十多个",
    "第一",
    "第五个工作日",
    "第二十五页",
    "第二季度",
    "二季度",
    "前三季度",
    "一个",
    "一次",
    "一年",
    "一家",
    "一个月",
    "万一",
    "一律",
    "一旦",
    "一方面",
    "一心一意",
    "星期五",
    "周一",
    "千万不要",
    "不管三七二十一",
    "八九不离十",
    "半斤八两",
    "五四运动",
    "二八年华",
    "相差十万八千里",
    "十一届全国人大一次会议",
    "二十大",
    "六方会谈",
    "腊月二十三",
    "正月十五",
    "农历八月十五",
    "大年三十",
    "三分之一",
    "十分之一",
    "二十分之一",
    "千分之五",
    "几万分之一",
    "九七年",
    "零八年",
    "二零年",
    "一米八",
    "两米五",
    "三大银行",
    "四大行",
    "一万五点五",
    "一千五点五",
    "公元二五三年",
    "我们一块去",
    "两点最重要的内容",
    "有三点需要注意",
    "好一点",
    "一点点",
    "这十点很关键",
    "A三百",
    "百分之三到五",
    "三万到五",
    "三、五万",
    # A start that could take the end's magnitude or its leading place stays unscaled, so the range stays Chinese.
    "两到三十万",
    "两到三千五百万",
    "两到三百五十万",
    "三到二十5",
    "５三百",
    "５三个",
    "5.三百",
    "Ａ三百",
    "三百.5",
    "一到三月份就开始涨",
    "一到十二点就睡",
    "第三到五个",
    "一个到两个",
    "提升两点",
    "数十日",
    "几十点",
    "跌了几十点",
    "八月十五",
    "五十步笑百步",
    "三十而立",
    "十二生肖",
    "三十六计",
    "十一黄金周",
    "一号文件",
    "三万五到四万三万五到四万",
    "三百5",
]


@pytest.mark.parametrize(("text", "expected"), CONVERTED)
def test_amounts_are_written_as_digits(text: str, expected: str) -> None:
    assert display(text) == expected


@pytest.mark.parametrize("text", UNCHANGED)
def test_set_words_approximations_ordinals_and_partial_figures_stay_chinese(text: str) -> None:
    assert display(text) == text


def test_conversions_are_sorted_spans_of_the_input() -> None:
    text = "下午三点十分涨了百分之二十，三万到五万"
    changes = conversions(text)
    assert [text[start:end] for start, end, _ in changes] == ["三", "十", "百分之二十", "三万", "五万"]
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(changes))
    assert apply(text, changes) == "下午3点10分涨了20%，3万到5万"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("两到三亿", "2到3亿"),
        ("二十到三十万", "20到30万"),
        # A start that takes the end's 百 or 千 is written in the end's 万, so its lower bound never reads as 2.
        ("两到三百万", "200到300万"),
        ("两到三千万", "2000到3000万"),
        ("五到两百万", "5到200万"),
        ("三点五到四万", "3.5到4万"),
        ("三到五到七万", "3到5到7万"),
        ("3到五到七万", "3到5到7万"),
        ("五到十二万", "5到12万"),
        ("两到十五万", "2到15万"),
        ("两到十万", "2到10万"),
        ("两到3.5亿", "2到3.5亿"),
        ("两到三亿美元", "2到3亿美元"),
    ],
)
def test_a_range_start_is_written_in_its_ends_magnitude_and_reads_back_the_same(text: str, expected: str) -> None:
    assert display(text) == expected
    assert [t.value_itn for t in quantity_tokens(expected)] == [t.value_itn for t in quantity_tokens(text)]


def test_a_rewrite_that_the_tokenizer_would_read_otherwise_is_dropped() -> None:
    # 数一千万 is an exact figure, while 数1000万 reads as a rough one, so the digits would change its value.
    assert display("人数一千万") == "人数一千万"
    assert itn("人数一千万") != itn("人数1000万")


_ALPHABET = "〇零一二两三四五六七八九十百千万亿点"
_PIECES = (
    *"〇零一二两三四五六七八九十百千万亿",
    *("点", "个", "年", "月", "号", "块", "钱", "美元", "几", "数", "到", "、", "第", "百分之", "分之", "岁", "家"),
    *("倍", "届", "农历", "千万", "万一", "多", "3", "5", "0", "A", "E", "T", "F", " ", "，"),
)


def _samples() -> list[str]:
    short = ["".join(chars) for n in range(1, 4) for chars in itertools.product(_ALPHABET, repeat=n)]
    rng = random.Random(15835)
    mixed = ["".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 8))) for _ in range(5000)]
    table = [text for text, _ in CONVERTED] + UNCHANGED
    return short + mixed + table


SAMPLES = _samples()


def test_p1_display_is_idempotent() -> None:
    assert [text for text in SAMPLES if display(display(text)) != display(text)] == []


def readings(text: str) -> list[tuple[str, str]]:
    # A rewritten Chinese numeral reads as a digit number; every other kind and every value must stay.
    return [("number" if t.kind == "cjk_quantity" else t.kind, t.value_itn) for t in quantity_tokens(text)]


def test_p2_a_rewrite_never_changes_a_numbers_value() -> None:
    assert [text for text in SAMPLES if readings(display(text)) != readings(text)] == []


@pytest.mark.parametrize(("text", "change"), [("５三百", (1, 3, "300")), ("5.三百", (2, 4, "300"))])
def test_p2_sees_figures_that_a_rewrite_would_merge(text: str, change: tuple[int, int, str]) -> None:
    assert readings(apply(text, [change])) != readings(text)
    assert itn(apply(text, [change])) == itn(text), "the joined itn text alone would not tell"


# GB/T 5.3 allows 万 and 亿 beside Arabic digits (3.5万), never 十, 百, 千 or a Chinese digit (1百零8).
_NEVER_BESIDE_DIGITS = frozenset(NUMERALS) - set("万萬亿億")


def test_p3_no_digit_written_next_to_a_chinese_numeral() -> None:
    def mixes(text: str) -> bool:
        out, cursor, edges = [], 0, []
        for start, end, replacement in conversions(text):
            out.append(text[cursor:start])
            edges.append(len("".join(out)))
            out.append(replacement)
            edges.append(len("".join(out)))
            cursor = end
        out.append(text[cursor:])
        written = "".join(out)
        for edge in edges:
            pair = written[max(edge - 1, 0) : edge + 1]
            if len(pair) == 2 and any(a.isdigit() and b in _NEVER_BESIDE_DIGITS for a, b in (pair, pair[::-1])):
                return True
        return False

    assert [text for text in SAMPLES if mixes(text)] == []


def test_p4_display_never_raises() -> None:
    for text in SAMPLES:
        display(text)


@pytest.mark.parametrize("text", ["", "大家好，今天聊聊估值。", "NVDA rose 3.5% to $150", "1,500手和2026年"])
def test_p5_text_without_chinese_numerals_comes_back_unchanged(text: str) -> None:
    assert conversions(text) == []
    assert display(text) == text
