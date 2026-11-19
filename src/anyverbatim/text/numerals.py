"""Mandarin amounts written as digits, following GB/T 15835-2011 (出版物上数字用法).

Only quantities the tokenizer (`numbers`) recognizes are rewritten, so every figure written as digits is
one that numeric verification read.

Converts: percentages (百分之八十 → 80%), currency (六百五十六美元 → 656美元), values in 万 or 亿 in the
speaker's unit with an Arabic coefficient (三万五 → 3.5万, 一亿五千三 → 1.53亿), Gregorian years read digit
by digit (二零二六年 → 2026年), counts of two and up (三个, 两年 → 3个, 2年), value units (一倍 → 1倍), dates
(一月一日 → 1月1日), clock times (下午三点, 两点零五分 → 下午3点, 2点05分), index points (涨了三点, 指数两千八百点
→ 涨了3点, 指数2800点), bare numbers of two or more numerals, decimals, and ranges, whole or not at all (三到五个
→ 3到5个, 下午两点到四点 → 下午2点到4点; a start that shares its end's 万 or 亿 is written in that unit: 两到三亿
→ 2到3亿, 两到三百万 → 200到300万).

Stays Chinese: approximations (三四个, 几十, 二十几), idioms and set words (一样, 万一, 三大), ordinals
(第二十五期, 二季度), a bare 一 before a count word (一个, 一次), lunar dates (腊月二十三), fractions
(三分之一), X点 up to 10 that is neither a clock time nor a move (两点最重要, 有三点), spoken two-digit
years (九七年), a figure that a following numeral continues (三块五, 一米八: 3块五 mixes forms, which GB/T
5.3 forbids), set names (十一届, 二十大), ranges whose ends share a percent (百分之三到五) or a magnitude on the
left (三万到五) or that hold an ordinal (第三到五个), 一到 meaning "as soon as" (一到三月份就涨) and anything whose
digits would run into adjacent digits or letters. Digits are never written back as Chinese.

A rewrite never changes a number's value: the tokenizer reads the display form as the same quantities with the
same `value_itn`, and a conversion that would not is dropped.
"""

from collections.abc import Callable
from decimal import Decimal

from anyverbatim.text import numbers

# Part of the numerals stage key: bump it on any rule change so stored results are redone.
VERSION = "numerals-v1"

Conversion = tuple[int, int, str]

_PLACES = frozenset("十百千万亿萬億廿卅")
_TRADITIONAL_UNITS = str.maketrans({"万": "萬", "亿": "億"})
_MAX_COEFFICIENT_DIGITS = 4
_YEAR_DIGITS = 4
_DIGITS = frozenset("0123456789０１２３４５６７８９")
_LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Letters and digits, half and full width, that digits written next to them would run into.
_ALNUM = frozenset(_LATIN) | frozenset(chr(ord(ch) + 0xFEE0) for ch in _LATIN) | _DIGITS
_NUMERALS = frozenset(numbers.NUMERALS)
# GB/T 4.2.1: dates of the lunar calendar keep Chinese numerals. 初 is left out: 初一 to 初十 are single numerals, which
# are no quantity anyway, while 年初三个月 is "the first three months of the year".
_LUNAR_MARKERS = ("农历", "阴历", "農曆", "陰曆", "腊月", "臘月", "正月", "大年")
# Bare numerals before these name a set rather than count: 十一届, 二十大, 二十国集团, 六方会谈.
_SET_NAME_WORDS = frozenset("届屆大国國方")
_COUNT_CLASSIFIERS = frozenset(numbers.COUNT_CLASSIFIERS)
# 一块 is "together" or "a piece" (一块去, 一块蛋糕) far more often than one yuan, unless 钱 follows.
_PIECE = frozenset("块塊")
_MONEY_AFTER_PIECE = frozenset("钱錢")
# A bare 2点 reads as two o'clock or "two points" (两点最重要, 有三点), so a small X点 is written in digits only as a
# clock time (minutes or 半 after it, a time of day before it; 今天下午 ends in 下午) or an index move (涨了三点).
# From 11 up it reads as an amount or level (指数两千八百点, 涨了二十四点) and always converts. Bare 升 and 降 end
# ordinary words (提升两点), so only their two-character forms count as moves.
_HALF_HOUR = frozenset("半")
_MOVES = (
    "涨",
    "跌",
    "上涨",
    "下跌",
    "上升",
    "下降",
    "回升",
    "反弹",
    "回落",
    "大涨",
    "大跌",
    "暴涨",
    "暴跌",
    "收涨",
    "收跌",
)
_MOVES_DONE = tuple(f"{move}了" for move in _MOVES)
_SMALLEST_LEVEL = 11


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


def conversions(text: str, *, allowed: Callable[[Conversion], bool] | None = None) -> list[Conversion]:
    """Return the rewrites of `text` as sorted, non-overlapping `(start, end, replacement)` spans.

    `allowed` refuses rewrites the caller cannot carry (a figure its timed words only partly cover). A range (三到五个,
    下午两点到四点, 三点四十) is rewritten whole or not at all, so a refused or dropped end keeps the others Chinese.
    """
    quantities = numbers.quantities(text)
    starts = {quantity.token.char_start for quantity in quantities}
    covered = {i for quantity in quantities for i in range(quantity.token.char_start, quantity.token.char_end)}
    lunar = _lunar(text, quantities)
    groups = _ranges(text, quantities)
    points = _points(text, quantities, groups, starts)
    candidates: dict[int, Conversion] = {}
    for index, quantity in enumerate(quantities):
        if lunar[index]:
            continue
        if _is_range_end(text, quantity):
            start, end = quantity.run
            change = (start, end, quantity.value) if quantity.value is not None else None
        else:
            change = _conversion(text, quantity, starts, points=index in points)
        if change is not None and (allowed is None or allowed(change)):
            candidates[index] = change
    while True:
        chosen = {
            index: candidates[index]
            for group in groups
            if _whole(text, quantities, group, candidates, covered)
            for index in group
            if index in candidates
        }
        planned = sorted(chosen.values())
        kept = set(_value_preserving(text, _unfused(text, planned)))
        if len(kept) == len(planned):
            return planned
        for index, change in chosen.items():
            if change not in kept:
                del candidates[index]


def _conversion(text: str, quantity: numbers.Quantity, starts: set[int], *, points: bool) -> Conversion | None:
    token, unit, value = quantity.token, quantity.unit, quantity.value
    if value is None or quantity.prefix == "第" or unit in numbers.QUARTER_NAMES:
        return None
    start, end = quantity.run
    run = text[start:end]
    after = text[token.char_end : token.char_end + 1]
    if _DIGITS & set(run):
        # An engine's 4千 or 3千万: GB/T 5.3 keeps only 万 and 亿 as units beside Arabic digits.
        if not set(run) & {"百", "千"} or any(ch.isascii() and ch.isalpha() for ch in run):
            return None
        return (start, end, _figure(value.lstrip("+-−"), quantity.magnitude, run))
    if (unit == "点" and not points) or _stays_chinese(quantity, run, after, starts):
        return None
    if quantity.prefix == "百分之":
        return (token.char_start, token.char_end, f"{value}%")
    return (start, end, _figure(value, quantity.magnitude, run))


def _stays_chinese(quantity: numbers.Quantity, run: str, after: str, starts: set[int]) -> bool:
    """Whether a Chinese numeral token with a value is still kept as written (plan rules K4, K7, K8, K9)."""
    token, unit = quantity.token, quantity.unit
    continued = after in _NUMERALS and token.char_end not in starts
    spoken_year = unit == "年" and 1 < len(run) != _YEAR_DIGITS and not (_PLACES & set(run)) and "点" not in run
    single_one = run == "一" and (unit in _COUNT_CLASSIFIERS or (unit in _PIECE and after not in _MONEY_AFTER_PIECE))
    set_name = not unit and not quantity.prefix and after in _SET_NAME_WORDS
    return continued or spoken_year or single_one or set_name


def _points(text: str, quantities: list[numbers.Quantity], groups: list[list[int]], starts: set[int]) -> set[int]:
    """Find the X点 tokens to write in digits; a time or a move at one end of a range carries the others."""
    qualified: set[int] = set()
    for group in groups:
        dots = [index for index in group if quantities[index].unit == "点"]
        if any(_is_points_or_time(text, quantities[index], starts) for index in dots):
            qualified.update(dots)
    return qualified


def _is_points_or_time(text: str, quantity: numbers.Quantity, starts: set[int]) -> bool:
    token = quantity.token
    before, after = text[: token.char_start], text[token.char_end : token.char_end + 1]
    minutes = token.char_end in starts and after in _NUMERALS
    clock = minutes or after in _HALF_HOUR or before.endswith(numbers.TIMES_OF_DAY)
    move = before.endswith(_MOVES) or before.endswith(_MOVES_DONE)
    level = quantity.value is not None and Decimal(quantity.value) >= _SMALLEST_LEVEL
    return clock or move or level


def _figure(value: str, magnitude: str, run: str) -> str:
    """Write the value in the speaker's largest of 万亿, 亿 and 万 if that takes four digits at most, else plainly."""
    if not magnitude:
        return value
    coefficient = format((Decimal(value) / Decimal(10) ** numbers.magnitude_exponent(magnitude)).normalize(), "f")
    if len(coefficient.replace(".", "").lstrip("0")) > _MAX_COEFFICIENT_DIGITS:
        return value
    if {"萬", "億"} & set(run):
        magnitude = magnitude.translate(_TRADITIONAL_UNITS)
    return coefficient + magnitude


def _is_range_end(text: str, quantity: numbers.Quantity) -> bool:
    start, end = quantity.run
    return not quantity.prefix and not quantity.unit and end - start == 1 and text[start] in _NUMERALS


def _ranges(text: str, quantities: list[numbers.Quantity]) -> list[list[int]]:
    """Group the quantities one range joins: ends linked by 到, 至, ～, -, — or 、, and an hour with its minutes."""
    groups: list[list[int]] = []
    for index, quantity in enumerate(quantities):
        if index > 0:
            before = quantities[index - 1]
            gap = text[before.token.char_end : quantity.token.char_start]
            minutes = gap == "" and before.unit == "点" and not quantity.unit and not quantity.prefix
            if minutes or (len(gap) == 1 and gap in numbers.RANGE_LINKS):
                groups[-1].append(index)
                continue
        groups.append([index])
    return groups


def _whole(
    text: str,
    quantities: list[numbers.Quantity],
    group: list[int],
    candidates: dict[int, Conversion],
    covered: set[int],
) -> bool:
    """Whether every end of a range is rewritten or already in digits, and no end was left out of the tokens.

    An end the tokenizer left out shares a unit with its partner (百分之三到五, 三万到五) or is "as soon as"
    (一到三月份), so the whole range stays as said.
    """
    first, last = quantities[group[0]].token, quantities[group[-1]].token
    for link, numeral in ((first.char_start - 1, first.char_start - 2), (last.char_end, last.char_end + 1)):
        if (
            0 <= numeral < len(text)
            and text[link] in numbers.RANGE_LINKS
            and text[numeral] in _NUMERALS
            and numeral not in covered
        ):
            return False
    return all(index in candidates or _in_digits(text, quantities[index]) for index in group)


def _in_digits(text: str, quantity: numbers.Quantity) -> bool:
    start, end = quantity.run
    return quantity.value is not None and bool(_DIGITS & set(text[start:end]))


def _lunar(text: str, quantities: list[numbers.Quantity]) -> list[bool]:
    """Whether each quantity is part of a lunar date: after a marker, directly or through the date tokens before it."""
    lunar: list[bool] = []
    for index, quantity in enumerate(quantities):
        start = quantity.token.char_start
        joined = index > 0 and lunar[-1] and quantities[index - 1].token.char_end == start
        lunar.append(joined or text[:start].endswith(_LUNAR_MARKERS))
    return lunar


def _unfused(text: str, changes: list[Conversion]) -> list[Conversion]:
    """Drop every rewrite whose digits would run into the characters written around them, until none does.

    A letter or digit before, half or full width, or a digit and its decimal point (A三百, ５三百, 5.三百) would
    make one longer figure, as would a digit or a decimal point and digit after; a Chinese numeral on either side
    would mix forms (GB/T 5.3). A letter after is fine: the tokenizer splits a ticker written against a figure
    (沪深300ETF). A neighbour that is itself rewritten counts as written.
    """
    kept = list(changes)
    while True:
        fused = [index for index in range(len(kept)) if _fuses(text, kept, index)]
        if not fused:
            return kept
        kept = [change for index, change in enumerate(kept) if index not in fused]


def _fuses(text: str, changes: list[Conversion], index: int) -> bool:
    start, end, replacement = changes[index]
    before, after = text[max(start - 2, 0) : start], text[end : end + 2]
    if index > 0 and changes[index - 1][1] == start:
        before = changes[index - 1][2][-2:]
    if index + 1 < len(changes) and changes[index + 1][0] == end:
        after = changes[index + 1][2][:2]
    last_before, first_after = before[-1:], after[:1]
    if replacement[:1] in _DIGITS and last_before:
        decimal = last_before == "." and before[:1] in _DIGITS
        if last_before in _ALNUM or last_before in _NUMERALS or decimal:
            return True
    if replacement[-1:] in _DIGITS and first_after:
        decimal = first_after == "." and after[1:2] in _DIGITS
        return first_after in _DIGITS or first_after in _NUMERALS or decimal
    return False


def _value_preserving(text: str, changes: list[Conversion]) -> list[Conversion]:
    """Drop any rewrite after which the tokenizer reads other quantities: more or fewer, or another value.

    Comparing the token sequence, not the joined text, also catches figures that merge (５三百 is 5 and 300, ５300 one
    figure) and a reading that shifts (数一千万 is exact, 数1000万 rough).
    """
    if not changes:
        return changes
    reference = readings(text)
    if readings(apply(text, changes)) == reference:
        return changes
    kept = [change for change in changes if readings(apply(text, [change])) == reference]
    return kept if readings(apply(text, kept)) == reference else []


def readings(text: str) -> list[tuple[str, str]]:
    """Return the quantities of `text` as (kind, `value_itn`), the way a rewrite must leave them."""
    # A rewritten Chinese numeral is read as a digit number, so those two kinds are one.
    return [
        ("number" if token.kind == "cjk_quantity" else token.kind, token.value_itn)
        for token in numbers.quantity_tokens(text)
    ]
