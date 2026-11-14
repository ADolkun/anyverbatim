"""Mandarin amounts written as digits, following GB/T 15835-2011 (出版物上数字用法).

Only quantities the tokenizer (`numbers`) recognizes are rewritten, so every figure written as digits is
one that numeric verification read.

Converts: percentages (百分之八十 → 80%), currency (六百五十六美元 → 656美元), values in 万 or 亿 in the
speaker's unit with an Arabic coefficient (三万五 → 3.5万, 一亿五千三 → 1.53亿), Gregorian years read digit
by digit (二零二六年 → 2026年), counts of two and up (三个, 两年 → 3个, 2年), value units (一倍 → 1倍), dates
(一月一日 → 1月1日), clock times (下午三点, 四点四十分 → 下午3点, 4点40分), bare numbers of two or more numerals,
decimals, and the ends of a range (三到五个 → 3到5个).

Stays Chinese: approximations (三四个, 几十, 二十几), idioms and set words (一样, 万一, 三大), ordinals
(第二十五期, 二季度), a bare 一 before a count word (一个, 一次), lunar dates (腊月二十三), fractions
(三分之一), X点 that is not plainly a clock time (两点最重要, 有三点), spoken two-digit years (九七年), a
figure that a following numeral continues (三块五, 一米八: 3块五 mixes forms, which GB/T 5.3 forbids), set
names (十一届, 二十大) and anything whose digits would run into adjacent digits or letters. Digits are never
written back as Chinese.

A rewrite never changes a number's value: the tokenizer reads the display form to the same `value_itn`,
and a conversion that would not is dropped.
"""

from decimal import Decimal

from ytverbatim.text import numbers

# Part of the numerals stage key: bump it on any rule change so stored results are redone.
VERSION = "numerals-v1"

Conversion = tuple[int, int, str]

_PLACES = frozenset("十百千万亿萬億廿卅")
_TRADITIONAL_UNITS = str.maketrans({"万": "萬", "亿": "億"})
_MAX_COEFFICIENT_DIGITS = 4
_YEAR_DIGITS = 4
_DIGITS = frozenset("0123456789０１２３４５６７８９")
_ASCII_ALNUM = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_NUMERALS = frozenset(numbers._NUMERALS)
# GB/T 4.2.1: dates of the lunar calendar keep Chinese numerals. 初 is left out: 初一 to 初十 are single numerals, which
# are no quantity anyway, while 年初三个月 is "the first three months of the year".
_LUNAR_MARKERS = ("农历", "阴历", "農曆", "陰曆", "腊月", "臘月", "正月", "大年")
# Bare numerals before these name a set rather than count: 十一届, 二十大, 二十国集团, 六方会谈.
_SET_NAME_WORDS = frozenset("届屆大国國方")
_COUNT_CLASSIFIERS = frozenset(numbers._COUNT_CLASSIFIERS)
# 一块 is "together" or "a piece" (一块去, 一块蛋糕) far more often than one yuan, unless 钱 follows.
_PIECE = frozenset("块塊")
_MONEY_AFTER_PIECE = frozenset("钱錢")
# A bare 2点 reads as two o'clock, so X点 is written in digits only where it is plainly a time: minutes or 半 after
# it, or a time of day before it (今天下午 ends in 下午). 两点最重要 ("two points") and 有三点 stay Chinese.
_TIMES_OF_DAY = ("上午", "下午", "晚上", "早上", "凌晨", "中午", "傍晚", "夜里")
_HALF_HOUR = frozenset("半")


def display(text: str) -> str:
    """`text` with its Chinese amounts written as digits; anything else is returned unchanged."""
    return apply(text, conversions(text))


def apply(text: str, changes: list[Conversion]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end, replacement in changes:
        parts.extend((text[cursor:start], replacement))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def conversions(text: str) -> list[Conversion]:
    """Return the rewrites of `text` as sorted, non-overlapping `(start, end, replacement)` spans."""
    quantities = numbers._quantities(text)
    starts = {quantity.token.char_start for quantity in quantities}
    lunar = _lunar(text, quantities)
    planned: dict[int, Conversion] = {}
    range_ends: list[int] = []
    for index, quantity in enumerate(quantities):
        if _is_range_end(text, quantity):
            range_ends.append(index)
        elif not lunar[index] and (change := _conversion(text, quantity, starts)) is not None:
            planned[index] = change
    for index in range_ends:
        if any(partner in planned for partner in _partners(text, quantities, index)):
            start, end = quantities[index].run
            planned[index] = (start, end, str(quantities[index].value))
    return _value_preserving(text, _unfused(text, [change for _, change in sorted(planned.items())]))


def _conversion(text: str, quantity: numbers._Quantity, starts: set[int]) -> Conversion | None:
    token, unit, value = quantity.token, quantity.unit, quantity.value
    if value is None or quantity.prefix == "第" or unit in numbers._QUARTER_NAMES:
        return None
    start, end = quantity.run
    run = text[start:end]
    after = text[token.char_end : token.char_end + 1]
    if _DIGITS & set(run):
        # An engine's 4千 or 3千万: GB/T 5.3 keeps only 万 and 亿 as units beside Arabic digits.
        if not set(run) & {"百", "千"} or any(ch.isascii() and ch.isalpha() for ch in run):
            return None
        return (start, end, _figure(value.lstrip("+-−"), quantity.magnitude, run))
    if _stays_chinese(text, quantity, run, after, starts):
        return None
    if quantity.prefix == "百分之":
        return (token.char_start, token.char_end, f"{value}%")
    return (start, end, _figure(value, quantity.magnitude, run))


def _stays_chinese(text: str, quantity: numbers._Quantity, run: str, after: str, starts: set[int]) -> bool:
    """Whether a Chinese numeral token with a value is still kept as written (plan rules K4, K7, K8, K9, clocks)."""
    token, unit = quantity.token, quantity.unit
    continued = after in _NUMERALS and token.char_end not in starts
    spoken_year = unit == "年" and 1 < len(run) != _YEAR_DIGITS and not (_PLACES & set(run)) and "点" not in run
    single_one = run == "一" and (unit in _COUNT_CLASSIFIERS or (unit in _PIECE and after not in _MONEY_AFTER_PIECE))
    not_a_time = unit == "点" and not _is_clock_time(text, token, after, starts)
    set_name = not unit and not quantity.prefix and after in _SET_NAME_WORDS
    return continued or spoken_year or single_one or not_a_time or set_name


def _is_clock_time(text: str, token: numbers.QToken, after: str, starts: set[int]) -> bool:
    minutes = token.char_end in starts and after in _NUMERALS
    return minutes or after in _HALF_HOUR or text[: token.char_start].endswith(_TIMES_OF_DAY)


def _figure(value: str, magnitude: str, run: str) -> str:
    """Write the value in the speaker's largest of 万亿, 亿 and 万 if that takes four digits at most, else plainly."""
    if not magnitude:
        return value
    coefficient = format((Decimal(value) / Decimal(10) ** numbers._magnitude_exponent(magnitude)).normalize(), "f")
    if len(coefficient.replace(".", "").lstrip("0")) > _MAX_COEFFICIENT_DIGITS:
        return value
    if {"萬", "億"} & set(run):
        magnitude = magnitude.translate(_TRADITIONAL_UNITS)
    return coefficient + magnitude


def _is_range_end(text: str, quantity: numbers._Quantity) -> bool:
    start, end = quantity.run
    return not quantity.prefix and not quantity.unit and end - start == 1 and text[start] in _NUMERALS


def _partners(text: str, quantities: list[numbers._Quantity], index: int) -> list[int]:
    """Find the figures a lone numeral belongs with: the other ends of its range, or the hour before its minutes."""
    token = quantities[index].token
    partners: list[int] = []
    if index > 0:
        before = quantities[index - 1]
        gap = text[before.token.char_end : token.char_start]
        if (gap == "" and before.unit == "点") or (len(gap) == 1 and gap in numbers._RANGE_LINKS):
            partners.append(index - 1)
    if index + 1 < len(quantities):
        gap = text[token.char_end : quantities[index + 1].token.char_start]
        if len(gap) == 1 and gap in numbers._RANGE_LINKS:
            partners.append(index + 1)
    return partners


def _lunar(text: str, quantities: list[numbers._Quantity]) -> list[bool]:
    """Whether each quantity is part of a lunar date: after a marker, directly or through the date tokens before it."""
    lunar: list[bool] = []
    for index, quantity in enumerate(quantities):
        start = quantity.token.char_start
        joined = index > 0 and lunar[-1] and quantities[index - 1].token.char_end == start
        lunar.append(joined or text[:start].endswith(_LUNAR_MARKERS))
    return lunar


def _unfused(text: str, changes: list[Conversion]) -> list[Conversion]:
    """Drop every rewrite whose digits would run into the characters written around them, until none does.

    An ASCII letter or digit before (A三百 is not A300) or an ASCII digit after would make one longer figure, and a
    Chinese numeral on either side would mix forms (GB/T 5.3). An ASCII letter after is fine: the tokenizer splits a
    ticker written against a figure (沪深300ETF). A neighbour that is itself rewritten counts as written.
    """
    kept = list(changes)
    while True:
        fused = [index for index in range(len(kept)) if _fuses(text, kept, index)]
        if not fused:
            return kept
        kept = [change for index, change in enumerate(kept) if index not in fused]


def _fuses(text: str, changes: list[Conversion], index: int) -> bool:
    start, end, replacement = changes[index]
    before, after = text[start - 1 : start], text[end : end + 1]
    if index > 0 and changes[index - 1][1] == start:
        before = changes[index - 1][2][-1:]
    if index + 1 < len(changes) and changes[index + 1][0] == end:
        after = changes[index + 1][2][:1]
    if replacement[:1] in _DIGITS and before and (before in _ASCII_ALNUM or before in _NUMERALS):
        return True
    return replacement[-1:] in _DIGITS and bool(after) and (after in _DIGITS or after in _NUMERALS)


def _value_preserving(text: str, changes: list[Conversion]) -> list[Conversion]:
    """Drop any rewrite that the tokenizer would read to another value (数一千万 is exact, 数1000万 rough)."""
    if not changes:
        return changes
    reference = numbers.itn(text)
    if numbers.itn(apply(text, changes)) == reference:
        return changes
    kept = [change for change in changes if numbers.itn(apply(text, [change])) == reference]
    return kept if numbers.itn(apply(text, kept)) == reference else []
