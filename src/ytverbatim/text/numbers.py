"""Quantity tokens: numbers, currency, percentages, tickers, Chinese numerals and spelled-out English numbers.

The patterns descend from the guards in ytscript's ``analyze/correct.py``. The Chinese rules are
stricter than a plain numeral run, because many words contain numeral characters without being
quantities: idioms (``一样``, ``千万不要``, ``万一``), weekdays (``周一``) and approximations
(``两三个`` is "two or three", not 23).
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Literal

import cn2an

# Part of the stage key of every stage that reads quantities: bump it on any change to what is a token or its value.
VERSION = "numbers-v2"

QKind = Literal["number", "currency", "percent", "ticker", "cjk_quantity"]

_NUMERALS = "〇零一二两兩三四五六七八九十百千万萬亿億廿卅"

# Longest alternatives first so 美元 wins over 元 and 个月 over 个.
_CURRENCY_WORDS = (
    "美元",
    "美金",
    "美分",
    "欧元",
    "歐元",
    "港元",
    "港币",
    "港幣",
    "人民币",
    "人民幣",
    "日元",
    "英镑",
    "英鎊",
    "新台币",
    "新臺幣",
    "台币",
    "臺幣",
    "澳元",
    "加元",
    "韩元",
    "韓元",
    "元",
    "块",
    "塊",
)
# Units that state a value rather than count things: 三个百分点 is a change of three points, not three objects.
_VALUE_UNITS = ("个百分点", "个基点", "基点", "倍", "成")
_CALENDAR_UNITS = ("月", "日", "号", "点钟", "点")
_QUARTER_NAMES = ("季度", "季")
# 分 (十分 "very", scores), 部 (三部曲), 手 (二手), 线 (二线城市), 方 (六方会谈), 届, 大 and 国 are left out on purpose:
# after a numeral they are more often part of a word or a name than a count.
_COUNT_CLASSIFIERS = (
    "个月",
    "个",
    "股",
    "年",
    "期",
    "次",
    "天",
    "周",
    "家",
    "种",
    "类",
    "条",
    "只",
    "支",
    "位",
    "名",
    "人",
    "本",
    "项",
    "台",
    "张",
    "份",
    "句",
    "件",
    "套",
    "篇",
    "颗",
    "笔",
    "单",
    "批",
    "轮",
    "场",
    "遍",
    "页",
    "集",
    "步",
    "小时",
    "分钟",
    "秒钟",
    "秒",
    "周岁",
    "岁",
    "周年",
    "世纪",
    "平方米",
    "米",
    "公里",
    "吨",
    "公斤",
    "斤",
    "克",
    "升",
    "度",
)
_MEASURE_WORDS = (*_CURRENCY_WORDS, *_VALUE_UNITS, *_CALENDAR_UNITS, *_QUARTER_NAMES, *_COUNT_CLASSIFIERS)


# Words whose numeral characters do not state a quantity. They are matched before numeral runs, so a
# run that starts earlier (三万一千) still claims its characters, while 万一 on its own is left alone.
_IDIOMS = (
    "一些",
    "一点儿",
    "一点点",
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
    "一一",
    "万一",
    "万万",
    "千千万万",
    "一切",
    "一共",
    "一度",
    "一律",
    "一再",
    "一向",
    "一贯",
    "一味",
    "一举",
    "一时",
    "一同",
    "一并",
    "一路",
    "一流",
    "一线",
    "一方面",
    "一口气",
    "一会儿",
    "一次性",
    "一成不变",
    "一股脑",
    "一股劲",
    "一个劲",
    "一天到晚",
    "一年到头",
    "一年四季",
    "一年半载",
    "三天两头",
    "一五一十",
    "一模一样",
    "一心一意",
    "三三两两",
    "七七八八",
    "十万八千里",
    "百度",
    "进一步",
    "五十步笑百步",
    "三十而立",
    "十二生肖",
    "三十六计",
    "十一黄金周",
    "一号文件",
    "一步一步",
)
_IDIOM_PATTERNS = (
    # "By all means": 千万不要, 千万别, 千万记住. 千万 before a measure word (千万元) stays a number.
    r"千万(?=[不别要记得莫]|小心|注意)",
    # "A little" (好一点, 一点也不); 一点五 is a decimal and 一点钟 or 一点半 a time.
    r"一点(?![〇零一二两兩三四五六七八九十钟半])",
    # Weekdays, not 周1; 每周一次 is "once a week".
    r"(?:周|星期|礼拜)[一二三四五六日天](?![次回遍趟])",
    # 中秋, which GB/T 15835 4.2.3 names; 八月十五日 and 八月十五号 are dates.
    r"八月十五(?![日号])",
)


def _alternation(words: tuple[str, ...]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


# The same dollar amount is written $15, 15 dollars or 15美元; all of them normalise to $15.
_USD_WORDS = frozenset({"美元", "美金"})
# MM is millions in finance usage; T stands for trillions only in upper case, and mm (millimetres) is left out.
_EN_MAGNITUDE = {
    "thousand": 3,
    "million": 6,
    "billion": 9,
    "trillion": 12,
    "k": 3,
    "m": 6,
    "mm": 6,
    "mn": 6,
    "b": 9,
    "bn": 9,
    "t": 12,
    "tn": 12,
}
_EN_UNITS = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ],
        range(20),
        strict=True,
    )
)
_EN_TENS = dict(
    zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10), strict=True)
)
_EN_SCALES = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_EN_NUMBER_WORD = _alternation((*_EN_UNITS, *_EN_TENS, "hundred", *_EN_SCALES))
_NO_SCALE = 99
_TEN, _HUNDRED = 10, 100
_EN_PIECE = re.compile(rf"(?i:\b(?:{_EN_NUMBER_WORD}|and)\b)")

_DIGIT = "[0-9０-９]"
# A thousands group is exactly three digits, so 2023,2024 is two numbers and not 2023,202 followed by 4.
_NUMBER = rf"(?:{_DIGIT}+(?:,{_DIGIT}{{3}}(?!{_DIGIT}))*(?:\.{_DIGIT}+)?|\.{_DIGIT}+)"
_NO_ALNUM_BEFORE = r"(?<![A-Za-z0-9])"
_NO_ALNUM_AFTER = r"(?![A-Za-z0-9])"

_RUN = rf"(?:(?!百分之)[{_NUMERALS}])"

_TOKEN = re.compile(
    "|".join(
        (
            rf"(?P<idiom>{'|'.join(_IDIOM_PATTERNS)}|{_alternation(_IDIOMS)})",
            # 百分之 on its own is a percent; 三分之一 and 千分之五 are fractions.
            rf"(?P<fraction>(?!百分之)(?P<denominator>[{_NUMERALS}]+)分之(?P<numerator>[{_NUMERALS}]+))",
            # A percent takes no measure word: the 支 of 百分之六十支持 starts the next word.
            # A run never takes the 百 of a following 百分之: 百分之五百分之五 is two percents.
            rf"(?P<cjk>(?P<pct>百分之)?(?P<ord>第)?(?P<run>{_RUN}+(?:点{_RUN}+)?)"
            rf"(?(pct)|(?P<unit>{_alternation((*_MEASURE_WORDS, '%'))})?))",
            rf"(?P<cashtag>\$[A-Z]{{1,6}}(?:\.[A-Z]{{1,2}})?{_NO_ALNUM_AFTER})",
            rf"(?P<exchange>(?i:\b(?:NYSE|NASDAQ|AMEX|NYSEARCA|NYSEAMERICAN|BATS|OTC|OTCQX|OTCQB))\s*:\s*"
            rf"[A-Z]{{1,5}}(?:\.[A-Z])?{_NO_ALNUM_AFTER})",
            r"(?P<share_class>(?<![A-Za-z0-9])[A-Z](?=股))",
            rf"(?P<numeric_ticker>{_NO_ALNUM_BEFORE}\d{{2,}}\.[A-Z]{{2,}}{_NO_ALNUM_AFTER})",
            rf"(?P<symbol>{_NO_ALNUM_BEFORE}[A-Z][A-Z0-9]+(?:\.[A-Z]+)?{_NO_ALNUM_AFTER})",
            # A date (2024/10/01) is not a fraction, so neither side of the slash may continue it.
            rf"(?P<digit_fraction>(?<![A-Za-z0-9０-９/])(?P<over>{_DIGIT}+)/(?P<under>{_DIGIT}+)(?![A-Za-z0-9０-９/]))",
            rf"(?P<number>{_NO_ALNUM_BEFORE}(?P<sign>[-+−])?(?P<currency>[$¥€£])?(?P<sign2>[-+−])?"
            rf"(?P<digits>{_NUMBER})"
            r"(?P<en_magnitude>\s?(?i:thousand|million|billion|trillion)\b|(?:BN|MN|TN|MM|bn|mn|tn|[Kk]|M|B|T)(?![A-Za-z0-9]))?"
            r"(?:(?P<percent> ?[%％]|\s?(?i:percent)\b)|(?P<dollars>\s?(?i:dollars?)\b)"
            rf"|(?P<magnitude>[百千][万亿萬億]*|[万亿萬億]+)?(?P<measure>{_alternation(_MEASURE_WORDS)})?))",
            rf"(?P<spelled>(?<![A-Za-z0-9-])(?i:{_EN_NUMBER_WORD})(?:(?:[ -]|\s+and\s+)(?i:{_EN_NUMBER_WORD}))*"
            r"(?![A-Za-z0-9-]))(?P<spelled_unit>\s+(?i:percent|dollars?)\b)?",
        )
    )
)

# A symbol written against a figure (3.35SMH) is still a ticker, unless the run is a figure suffix. The number
# pattern already takes the magnitudes (10BN, $5MM), and lowercase suffixes (bps, x) and single letters never match a
# symbol, so only the uppercase units it does not read are listed.
_FUSED_SYMBOL = re.compile(rf"[A-Z][A-Z0-9]+(?:\.[A-Z]+)?{_NO_ALNUM_AFTER}")
_FIGURE_SUFFIXES = frozenset({"BPS"})

_MIN_BARE_RUN = 2
_YEAR_DIGITS = 4
_VARIANT_NUMERALS = str.maketrans({"兩": "两", "萬": "万", "億": "亿"})
# Stacked magnitudes multiply: 万亿 is 10^12, so exponents add.
_MAGNITUDE = {"百": 2, "千": 3, "万": 4, "萬": 4, "亿": 8, "億": 8}
_DIGIT_VALUE = {ch: str(i) for i, ch in enumerate("零一二三四五六七八九")} | {"〇": "0"}
# 两 counts things (两个) but is never read out as a digit of a code or decimal.
_COUNTING_DIGITS = _DIGIT_VALUE | {"两": "2"}
_PLACE_WORDS = frozenset("十百千万亿")
_ZEROS = frozenset("〇零")
# Two bare digits side by side (四五十, 三四百) are a range, never a positional numeral.
_ADJACENT_DIGITS = re.compile("[一二两三四五六七八九]{2}")
# One digit after 亿 or 万亿 is the next lower unit, as 五 in 一万五: 一亿五 is 1.5亿.
_YI_SHORTHAND = re.compile("(?P<head>.*?(?P<unit>万?亿))(?P<digit>[一二三四五六七八九])")
# Speech marks a skipped place with 零 (一亿零五千三百 is 100,005,300), so a 千 or 百 group straight after 亿 is
# the 万 group: 一亿五千三 is 1.53亿. After 万亿 it is the 亿 group.
_YI_GROUP = re.compile("(?P<head>.*?(?P<unit>万?亿))(?P<group>(?![零〇])[^万亿]*[百千][^万亿]*)")
# A shorthand digit (一万五, 一千五) already stands for the next lower place, so a decimal after it is not one number.
_SHORTHAND_END = re.compile("[百千万亿][一二两三四五六七八九]$")
_TENS = frozenset("十百千廿卅")
_LAST_HOUR, _LAST_MINUTE = 24, 59
# The digit forms of a run of place words alone: 数十 is 数10, and 数百万 or 数千亿 is 数100万 or 数1000亿.
_ROUND_DIGITS = "10"
_ROUND_DIGITS_BEFORE_MAGNITUDE = frozenset({"10", "100", "1000"})
# Marks the value of a rough figure, so 二十几 and 20几 agree with each other but not with 二十.
_ROUGH = "~"
_SHORTHAND_AFTER_UNIT = re.compile("[万亿][一二三四五六七八九]$")
_ZERO_MINUTE = re.compile("[零〇][一二三四五六七八九]")
# Words that make X点 a clock time.
TIMES_OF_DAY = ("上午", "下午", "晚上", "早上", "凌晨", "中午", "傍晚", "夜里")
# A lone numeral joined to a figure by one of these is that range's other end: the 三 of 三到五个.
_RANGE_LINKS = frozenset("到至～~-—、")
_LINKABLE = frozenset("一二两三四五六七八九十")
_FRACTION = re.compile(rf"([{''.join(_DIGIT_VALUE)}]+)([万亿]*)")


def _magnitude_exponent(magnitudes: str) -> int:
    return sum(_MAGNITUDE[ch] for ch in magnitudes)


@dataclass(frozen=True, slots=True)
class QToken:
    """One quantity in a text: ``char_start`` and ``char_end`` are half-open offsets and ``surface`` the text between.

    ``value_itn`` is the inverse text normalization that two spellings of one quantity share; a ticker's is its
    surface.
    """

    char_start: int
    char_end: int
    surface: str
    value_itn: str
    kind: QKind


def _plain(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _ascii_digits(digits: str) -> str:
    return digits.translate({0xFF10 + i: ord("0") + i for i in range(10)}).replace(",", "")


def _is_quantity(integer: str, unit: str) -> bool:
    """Tell a numeral from bare digits that only look like one.

    Without a place word, digits read one by one form a number only as a code or year, which speech
    marks with 〇 or 零 (二〇一八年) unless it is a four-digit year (一九九七年). Anything else is a
    range or an idiom: 两三个, 一一对应, 五五开.
    """
    if any(ch in _PLACE_WORDS for ch in integer):
        return not _ADJACENT_DIGITS.search(integer)
    if len(integer) == 1:
        return True
    return "两" not in integer and (bool(_ZEROS & set(integer)) or (len(integer) == _YEAR_DIGITS and unit == "年"))


def _cn2an_value(integer: str) -> Decimal | None:
    # ValueError is cn2an's "not one number". Its speaking mode (一万五 is 15000) raises KeyError when the lower
    # unit it wants is missing from its table, as after 十 (七十六十五, 一万十五).
    try:
        return Decimal(str(cn2an.cn2an(integer, "smart")))
    except (ValueError, KeyError, InvalidOperation):
        return None


def _integer_value(integer: str) -> str | None:
    if not any(ch in _PLACE_WORDS for ch in integer):
        # Digit by digit, keeping leading zeros as the digit path keeps 0700: 零零后 is 00后.
        return "".join(_COUNTING_DIGITS[ch] for ch in integer)
    value: Decimal | None
    # 万 as a unit of its own appears once in a number: 四万三万五 is two figures run together, not 400035000.
    if integer.replace("万亿", "").count("万") > 1:
        return None
    if shorthand := _YI_SHORTHAND.fullmatch(integer):
        # cn2an's speaking mode has no unit below 亿, so the trailing digit of 一亿五 is placed here: 千万, or 千亿
        # after 万亿.
        head = _cn2an_value(shorthand["head"])
        lower_unit = Decimal(10) ** (_magnitude_exponent(shorthand["unit"]) - 1)
        value = None if head is None else head + int(_DIGIT_VALUE[shorthand["digit"]]) * lower_unit
    elif group := _YI_GROUP.fullmatch(integer):
        # 零 inside the group (一亿五千零三) leaves its place uncertain.
        if _ZEROS & set(group["group"]):
            return None
        head, part = _cn2an_value(group["head"]), _integer_value(group["group"])
        place = Decimal(10) ** (_magnitude_exponent(group["unit"]) - _MAGNITUDE["万"])
        value = None if head is None or part is None or Decimal(part) >= place else head + Decimal(part) * place
    else:
        value = _cn2an_value(integer)
    return None if value is None else _plain(value)


def _cjk_value(integer: str, fraction: str) -> str | None:
    whole = _integer_value(integer)
    if whole is None or not fraction:
        return whole
    if _SHORTHAND_END.search(integer) or _YI_GROUP.fullmatch(integer):
        return None
    # The magnitude after a decimal scales the whole number: 一点五万亿 is 1.5 × 10^12.
    parts = _FRACTION.fullmatch(fraction)
    if parts is None:
        return None
    decimals = "".join(_DIGIT_VALUE[ch] for ch in parts.group(1))
    scale = _magnitude_exponent(parts.group(2))
    return _plain(Decimal(f"{whole}.{decimals}") * Decimal(10) ** scale)


def _top_magnitude(numerals: str) -> str:
    return next((unit for unit in ("万亿", "亿", "万") if unit in numerals), "")


@dataclass(frozen=True, slots=True)
class _Quantity:
    """A token with the parts of it that a numeral rewrite needs.

    ``prefix`` is 百分之 or 第, ``run`` the span of the numeral or digit run (a digit token's magnitude included),
    ``unit`` the measure word or %, ``value`` the run's inverse text normalization without prefix or unit (None for a
    ticker, a spelled-out number, a fraction, a rough figure or a run that is not one number; a code read digit by
    digit keeps its leading zeros) and ``magnitude`` the largest of 万亿, 亿 and 万 the run is spoken or written in.
    """

    token: QToken
    prefix: str
    run: tuple[int, int]
    unit: str
    value: str | None
    magnitude: str


def _normalized(numerals: str) -> str:
    return numerals.translate(_VARIANT_NUMERALS).replace("廿", "二十").replace("卅", "三十")


def _cjk_quantity(
    text: str, start: int, prefix: str, run: tuple[int, int], unit: str, *, lone: bool = False
) -> _Quantity | None:
    """Read one Chinese numeral token; `lone` admits a single bare numeral (a clock's minutes, a range's end)."""
    surface_run = text[run[0] : run[1]]
    numeral_count = sum(1 for ch in surface_run if ch != "点")
    if not (prefix or unit or lone or numeral_count >= _MIN_BARE_RUN):
        return None
    numerals = _normalized(surface_run)
    integer, _, fraction = numerals.partition("点")
    if not _is_quantity(integer, unit):
        return None
    end = run[1] + len(unit)
    surface = text[start:end]
    rough = _is_approximation(text, prefix, run, numerals, unit)
    value = _cjk_value(_with_leading_one(integer) if rough else integer, fraction)
    ordinal = "第" if prefix == "第" else ""
    if value is None:
        itn = surface
    elif prefix == "百分之":
        itn = f"{value}%{unit}"
    elif unit in _USD_WORDS:
        itn = f"{ordinal}${value}"
    else:
        itn = f"{ordinal}{value}{unit}"
    if rough and value is not None:
        itn = _ROUGH + itn
    kind: QKind = "percent" if prefix == "百分之" else "currency" if unit in _CURRENCY_WORDS else "cjk_quantity"
    token = QToken(start, end, surface, itn, kind)
    return _Quantity(token, prefix, run, unit, None if rough else value, _top_magnitude(numerals))


def _is_approximation(text: str, prefix: str, run: tuple[int, int], numerals: str, unit: str) -> bool:
    """几 or 数 next to a run makes it a rough figure: 二十几岁, 几十个, 数十亿, 一百几十.

    A run of place words alone after 数 may be rough (数十亿) or exact after a noun ending in 数 (人数十万), so it
    is read as rough either way, as the digit form 人数10万 is; a digit in the run makes it exact (人数十五万). A date
    is never rough after 数: the 十月 of 指数十月份 is October, while 几十号 and 数十点 stay rough.
    """
    if (
        not prefix
        and _ji_or_shu_before(text, run[0])
        and set(numerals) <= _PLACE_WORDS
        and not _month_after_shu(text, run[0], unit)
    ):
        return True
    return _ji_after(text, run[1])


def _with_leading_one(numerals: str) -> str:
    """Read a run that starts with a bare 百, 千, 万 or 亿 with its 一: 数百万 shares its value with 数100万."""
    return f"一{numerals}" if numerals[:1] in ("百", "千", "万", "亿") else numerals


def _ji_or_shu_before(text: str, start: int) -> bool:
    return text[start - 1 : start] in ("几", "数")


def _month_after_shu(text: str, start: int, unit: str) -> bool:
    return unit == "月" and text[start - 1 : start] == "数"


def _ji_after(text: str, end: int) -> bool:
    return text[end : end + 1] == "几" and text[end : end + 2] != "几乎"


def _fraction(match: re.Match[str]) -> _Quantity:
    """Read a fraction as over/under in digits, so 三分之一 and 1/3 agree; it never gets a run value."""
    if match.group("fraction") is not None:
        parts = [_with_leading_one(_normalized(match.group(name))) for name in ("numerator", "denominator")]
        values = [_integer_value(part) if _is_quantity(part, "") else None for part in parts]
        kind: QKind = "cjk_quantity"
    else:
        values = [_ascii_digits(match.group(name)) for name in ("over", "under")]
        kind = "number"
    over, under = values
    itn = match.group(0) if over is None or under is None else f"{over}/{under}"
    return _bare(QToken(match.start(), match.end(), match.group(0), itn, kind))


def _per_unit_split(text: str, match: re.Match[str]) -> int | None:
    """Where 一 starts a "per unit" phrase after a figure in 万 or 亿: 三万一个月 is 30000 a month, not 31000 months.

    The 一 is split off only when the rest ends in 万 or 亿, or in the shorthand digit after one (三万一一个月).
    After 百 or 千 the shorthand reading stays: 一百一 is 110.
    """
    unit = match.group("unit") or ""
    if match.group("pct") or match.group("ord") or unit not in _COUNT_CLASSIFIERS:
        return None
    start, end = match.span("run")
    rest = text[start : end - 1].translate(_VARIANT_NUMERALS)
    if text[end - 1] != "一" or not (rest.endswith(("万", "亿")) or _SHORTHAND_AFTER_UNIT.search(rest)):
        return None
    return end - 1


def _clock_split(text: str, match: re.Match[str]) -> int | None:
    """Find the 点 of a clock time, where tens follow it: 三点四十分 is 3点 and 40, not a decimal.

    Only an hour up to 24 and minutes up to 59 with no measure word after them read this way, so 一百点二十 and
    三点五十年 are not times.
    """
    if match.group("pct") or match.group("ord") or match.group("unit"):
        return None
    start, end = match.span("run")
    dot = text.find("点", start, end)
    if dot < 0 or not (_TENS & set(text[dot + 1 : end]) or _zero_minutes(text, start, dot, end)):
        return None
    hour, minute = _normalized(text[start:dot]), _normalized(text[dot + 1 : end])
    hour_value = _integer_value(hour) if _is_quantity(hour, "点") else None
    minute_value = _integer_value(minute)
    if hour_value is None or minute_value is None:
        return None
    return dot if Decimal(hour_value) <= _LAST_HOUR and Decimal(minute_value) <= _LAST_MINUTE else None


def _zero_minutes(text: str, start: int, dot: int, end: int) -> bool:
    """两点零五 is 2.05 unless it is plainly a time: 分 after it (两点零五分) or a time of day before (下午三点零五)."""
    if not _ZERO_MINUTE.fullmatch(text, dot + 1, end):
        return False
    return text[end : end + 1] == "分" or text[:start].endswith(TIMES_OF_DAY)


def _cjk_quantities(text: str, match: re.Match[str]) -> list[_Quantity]:
    prefix = match.group("pct") or match.group("ord") or ""
    unit = match.group("unit") or ""
    run = match.span("run")
    split = _per_unit_split(text, match)
    if split is not None:
        head = _cjk_quantity(text, run[0], "", (run[0], split), "")
        if head is not None and head.value is not None:
            tail = _cjk_quantity(text, split, "", (split, run[1]), unit)
            return [head] if tail is None else [head, tail]
    dot = _clock_split(text, match)
    if dot is not None:
        pieces = (
            _cjk_quantity(text, run[0], "", (run[0], dot), "点"),
            _cjk_quantity(text, dot + 1, "", (dot + 1, run[1]), "", lone=True),
        )
        return [piece for piece in pieces if piece is not None]
    quantity = _cjk_quantity(text, match.start(), prefix, run, unit)
    return [] if quantity is None else [quantity]


def _number_quantity(match: re.Match[str]) -> _Quantity:
    digits = _ascii_digits(match.group("digits"))
    magnitude, percent = match.group("magnitude") or "", match.group("percent")
    en_magnitude = (match.group("en_magnitude") or "").strip().casefold()
    exponent = _magnitude_exponent(magnitude) + _EN_MAGNITUDE.get(en_magnitude, 0)
    if exponent:
        digits = _plain(Decimal(digits) * Decimal(10) ** exponent)
    sign = (match.group("sign") or match.group("sign2") or "").replace("−", "-")
    symbol = match.group("currency") or ""
    measure = match.group("measure") or ""
    shown = measure
    if match.group("dollars") or measure in _USD_WORDS:
        symbol = symbol or "$"
        shown = "" if measure in _USD_WORDS else measure
    itn = f"{sign}{symbol}{digits}{'%' if percent else ''}{shown}"
    text = match.string
    written = _ascii_digits(match.group("digits"))
    round_figure = written == _ROUND_DIGITS or (bool(magnitude) and written in _ROUND_DIGITS_BEFORE_MAGNITUDE)
    rough = (
        _ji_or_shu_before(text, match.start())
        and not sign
        and round_figure
        and not _month_after_shu(text, match.start(), measure)
    ) or _ji_after(text, match.end())
    if rough:
        itn = _ROUGH + itn
    kind: QKind = "percent" if percent else "currency" if symbol or shown in _CURRENCY_WORDS else "number"
    run_end = match.end("magnitude") if magnitude else match.end("digits")
    return _Quantity(
        QToken(match.start(), match.end(), match.group(0), itn, kind),
        "",
        (match.start("digits"), run_end),
        match.group("measure") or ("%" if percent else ""),
        None if rough else f"{sign}{digits}",
        _top_magnitude(magnitude.translate(_VARIANT_NUMERALS)),
    )


def _spelled_tokens(match: re.Match[str]) -> list[QToken]:
    """Split a run of English number words into numbers and convert each: fifteen, twenty-five, two hundred and ten.

    Words that cannot continue the current number start the next one, so nineteen ninety is two numbers. A lone
    "one" is a pronoun or determiner far more often than a quantity (no one, one of them), so it counts only with a
    unit after it.
    """
    groups = _spelled_groups(
        [(piece[0].casefold(), piece.start(), piece.end()) for piece in _EN_PIECE.finditer(match.group("spelled"))]
    )
    offset = match.start()
    unit = match.group("spelled_unit")
    tokens: list[QToken] = []
    for number, group in enumerate(groups):
        # The unit belongs to the last number only, and only when no word was dropped between them.
        last = number == len(groups) - 1 and group.end == len(match.group("spelled"))
        suffix = (unit or "").strip().casefold() if last else ""
        if group.words == ["one"] and not suffix:
            continue
        stop = match.end() if suffix else offset + group.end
        kind: QKind
        if suffix == "percent":
            itn, kind = f"{group.value}%", "percent"
        elif suffix:
            itn, kind = f"${group.value}", "currency"
        else:
            itn, kind = str(group.value), "number"
        tokens.append(QToken(offset + group.start, stop, match.string[offset + group.start : stop], itn, kind))
    return tokens


@dataclass
class _SpelledNumber:
    start: int
    end: int = 0
    total: int = 0
    current: int = 0
    scale: int = _NO_SCALE
    last: str = ""
    words: list[str] = field(default_factory=list[str])

    def accepts(self, word: str) -> bool:
        if word in _EN_UNITS or word in _EN_TENS:
            if self.last == "tens":
                return 0 < _EN_UNITS.get(word, 0) < _TEN
            return self.last in ("hundred", "scale", "and")
        if word == "hundred":
            return self.last in ("unit", "tens") and self.current < _HUNDRED
        return self.last in ("unit", "tens", "hundred") and _EN_SCALES[word] < self.scale

    def add(self, word: str, end: int) -> None:
        self.words.append(word)
        self.end = end
        if word == "hundred":
            self.current *= _HUNDRED
            self.last = "hundred"
        elif word in _EN_SCALES:
            self.scale = _EN_SCALES[word]
            self.total += self.current * 10**self.scale
            self.current = 0
            self.last = "scale"
        else:
            self.current += _EN_UNITS.get(word, 0) + _EN_TENS.get(word, 0)
            self.last = "tens" if word in _EN_TENS else "unit"

    @property
    def value(self) -> int:
        return self.total + self.current


def _spelled_groups(pieces: list[tuple[str, int, int]]) -> list[_SpelledNumber]:
    """Group number words into numbers; ``and`` only joins a hundred or a scale to the words after it."""
    groups: list[_SpelledNumber] = []
    group: _SpelledNumber | None = None
    for word, start, end in pieces:
        if word == "and":
            if group is not None and group.last in ("hundred", "scale"):
                group.last = "and"
            else:
                group = None
            continue
        if group is not None and group.accepts(word):
            group.add(word, end)
            continue
        if word in _EN_UNITS or word in _EN_TENS:
            group = _SpelledNumber(start)
            group.add(word, end)
            groups.append(group)
        else:
            group = None
    return groups


def quantity_tokens(text: str) -> list[QToken]:
    """Return the quantity tokens of ``text`` in order, with offsets into ``text`` itself.

    A Chinese numeral run is a quantity only when it has two or more numeral characters, is
    attached to a measure word, or follows ``第``, and is neither an idiom nor a range of bare
    digits. A fraction is written over/under (``三分之一`` and ``1/3`` are ``1/3``) and a rough
    figure with ``几`` or ``数`` carries ``~`` (``二十几岁`` and ``20几岁`` are ``~20``), so each
    agrees with its other spelling but never with an exact value. ``value_itn`` is the inverse
    text normalization of the token: ``一百五十期`` becomes ``150期``, ``3.5亿`` becomes
    ``350000000`` and ``2万亿`` becomes ``2000000000000``. English magnitudes scale the same way
    (``$1.5 billion`` and ``$1.5B`` become ``$1500000000``), a dollar amount is ``$`` however it
    is written (``15 dollars``, ``15美元``), ``15 percent`` is ``15%``, and spelled-out numbers
    convert (``twenty-five`` is ``25``).
    """
    return [quantity.token for quantity in _quantities(text)]


def _quantities(text: str) -> list[_Quantity]:
    found: list[_Quantity] = []
    lone: list[tuple[int, int]] = []
    for match in _TOKEN.finditer(text):
        if match.group("idiom") is not None:
            continue
        if match.group("fraction") is not None or match.group("digit_fraction") is not None:
            found.append(_fraction(match))
        elif match.group("cjk") is not None:
            quantities = _cjk_quantities(text, match)
            found.extend(quantities)
            if not quantities and match.group(0) in _LINKABLE:
                lone.append(match.span())
        elif match.group("number") is not None:
            found.append(_number_quantity(match))
            fused = _fused_symbol(text, match.end())
            if fused is not None:
                found.append(_bare(fused))
        elif match.group("spelled") is not None:
            found.extend(_bare(token) for token in _spelled_tokens(match))
        else:
            surface = match.group(0)
            found.append(_bare(QToken(match.start(), match.end(), surface, surface, "ticker")))
    return _with_range_ends(text, found, lone) if lone else found


def _with_range_ends(text: str, found: list[_Quantity], lone: list[tuple[int, int]]) -> list[_Quantity]:
    """Add each lone numeral that a range link joins to a figure: 三到五个 is 3 to 5个, and 三到五到七个 chains.

    Only a figure with a value anchors a range, so 第三到五 (an ordinal) and 几到五 stay as they are. Neither does a
    percent or a figure in 万 or 亿, whose unit the lone end shares (百分之三到五, 三到五万): read alone it would be a
    bare 5, a value nobody said, so it stays no quantity at all. 一到 before a date or a time is "as soon as"
    (一到三月份就涨), unless 从 opens the range; 一至三月 is a range.
    """
    anchors: dict[tuple[str, int], str] = {
        edge: quantity.unit
        for quantity in found
        if quantity.value is not None and quantity.prefix not in ("第", "百分之") and not quantity.magnitude
        for edge in (("end", quantity.token.char_end), ("start", quantity.token.char_start))
    }
    added = True
    while added:
        added = False
        for start, end in list(lone):
            right = anchors.get(("start", end + 1)) if text[end : end + 1] in _RANGE_LINKS else None
            as_soon_as = text[start] == "一" and text[end : end + 1] == "到" and text[start - 1 : start] != "从"
            if right in _CALENDAR_UNITS and as_soon_as:
                right = None
            left = text[start - 1 : start] in _RANGE_LINKS and ("end", start - 1) in anchors
            quantity = (
                _cjk_quantity(text, start, "", (start, end), "", lone=True) if left or right is not None else None
            )
            if quantity is None or quantity.value is None:
                continue
            found.append(quantity)
            anchors |= {("end", end): "", ("start", start): ""}
            lone.remove((start, end))
            added = True
    return sorted(found, key=lambda quantity: quantity.token.char_start)


def _bare(token: QToken) -> _Quantity:
    return _Quantity(token, "", (token.char_start, token.char_end), "", None, "")


def _fused_symbol(text: str, end: int) -> QToken | None:
    """Read the ticker written directly after a number, where the symbol pattern's boundary stops it matching.

    That boundary refuses only an ASCII letter or digit before the symbol, and a number that ends in a letter (5K,
    $1.2M) never has one after it, so only an ASCII digit needs this.
    """
    if text[end - 1] not in "0123456789":
        return None
    match = _FUSED_SYMBOL.match(text, end)
    if match is None or match[0] in _FIGURE_SUFFIXES:
        return None
    return QToken(match.start(), match.end(), match[0], match[0], "ticker")


def itn(text: str) -> str:
    """Replace every quantity token of ``text`` by its ``value_itn`` and leave the rest untouched."""
    parts: list[str] = []
    cursor = 0
    for token in quantity_tokens(text):
        parts.extend((text[cursor : token.char_start], token.value_itn))
        cursor = token.char_end
    parts.append(text[cursor:])
    return "".join(parts)
