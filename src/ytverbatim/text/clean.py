"""Deterministic text normalization: Simplified script, optional filler removal and punctuation tidying.

The filler and punctuation rules are ported from ytscript's cleaner. Every rule is conservative: English drops only
pure hesitation sounds, never discourse markers, and Chinese drops a filler only where its boundaries show a
disfluency. A rule that would empty a text keeps the original instead, so no stretch loses its time range.

`normalize` returns each single replacement as a (before, after) step of the whole text, so flag offsets can be
moved through every step exactly rather than across one coarse span.
"""

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import opencc

from ytverbatim.text.segment import is_zh

# Part of the normalize stage key: bump it on any rule change so stored results are redone.
VERSION = "clean-v3"

Script = Literal["zh-Hans", "keep"]
Fillers = Literal["keep", "remove"]

# "er" and its drawn-out forms ("errr") are hesitations; "err", the verb, is a word and stays.
_EN_FILLERS = re.compile(r"\b(?:um+|uh+|hmm+|hm+|er(?:rr+)?)\b", re.IGNORECASE)
_ZH_FILLER_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"就是说就是说"), "就是说"),
    (re.compile(r"然后然后"), "然后"),
    (re.compile(r"(?:^|(?<=[，。！？、\s]))嗯+(?=[，。！？、\s]|$)"), ""),
    (re.compile(r"(?:^|(?<=[，。！？、\s]))啊+(?=[，。！？、\s]|$)"), ""),
)
# Demonstratives (这个/那个) may refer to source material even at a boundary, so they are never fillers.
_PUNCTUATION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # A clause never opens on a comma. Trailing marks stay: Whisper ends each Mandarin clause with its boundary.
    (re.compile(r"^[，,、]+\s*"), ""),
    # Two dots stay as written: "U.S.." and a trailing-off "Well.." cannot be told apart.
    (re.compile(r"\.{4,}"), "..."),
    (re.compile(r",{2,}"), ","),
    (re.compile(r"。{3,}"), "……"),
    (re.compile(r"。{2,}"), "。"),
    (re.compile(r"，{2,}"), "，"),
)
# Chinese text writes its marks full-width (GB/T 15834). A mark converts when a Chinese character is next to it, so
# numbers (3.5, 1,000, 10:30), Latin-only stretches and URLs, which have none, stay as written. Both patterns match
# horizontal whitespace only: a line break is never swallowed.
_HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_FULL_WIDTH = {",": "，", "?": "？", "!": "！", ":": "：", ";": "；", ".": "。"}
_FULL_WIDTH_MARKS = "".join(_FULL_WIDTH.values()) + "、…"
_MARK = re.compile(r"(?P<lead>[^\S\n]*)(?P<mark>[,?!:;.])(?P<trail>[^\S\n]*)")
_ELLIPSIS = re.compile(r"[^\S\n]*(?:\.{3,}|。{3,})[^\S\n]*")
_SPACES = re.compile(r"\s{2,}")
_REPEATED_COMMA = re.compile(r"([,，])(?:\s*[,，])+")


def _is_han(char: str) -> bool:
    return _HAN.fullmatch(char) is not None


def _ellipsis(match: re.Match[str]) -> str:
    text = match.string
    left = text[: match.start()].rstrip()[-1:]
    right = text[match.end() :].lstrip()[:1]
    return "……" if _is_han(left) or _is_han(right) else match[0]


def _left_character(before: str) -> str:
    # Read through full-width marks as well, so "你好，," converts whole but "Well…, ok", with no Chinese, stays.
    before = before.rstrip()
    while before and before[-1] in _FULL_WIDTH_MARKS:
        before = before[:-1].rstrip()
    return before[-1:]


def _full_width(match: re.Match[str]) -> str:
    text = match.string
    before, after = text[: match.start()], text[match.end() :]
    mark = match["mark"]
    left = before.rstrip()[-1:]
    # Marks that follow each other ("?!", ",,") are read through, so a run converts whole.
    right = after.lstrip(" \t,?!:;")[:1]
    if mark == ".":
        # A period ends a clause only before a break; "..." and "3.5" keep theirs because no Chinese character precedes.
        sentence_end = not after or bool(match["trail"]) or after[0] == "\n" or _is_han(after[0])
        convert = _is_han(left) and sentence_end
    else:
        convert = _is_han(_left_character(before)) or _is_han(right)
    if not convert:
        return match[0]
    return _FULL_WIDTH[mark]


@dataclass(frozen=True, slots=True)
class Normalized:
    text: str
    # Successive whole-text states, one per replacement, in order: steps[0][0] is the input.
    steps: tuple[tuple[str, str], ...]


@functools.cache
def _t2s() -> Callable[[str], str]:
    converter = opencc.OpenCC("t2s")

    def convert(text: str) -> str:
        return str(converter.convert(text))

    return convert


def to_simplified(text: str) -> str:
    return _t2s()(text)


def normalize(text: str, language: str, *, script: Script, fillers: Fillers) -> Normalized:
    steps: list[tuple[str, str]] = []
    current = text

    def step(after: str) -> None:
        nonlocal current
        if after != current:
            steps.append((current, after))
            current = after

    def apply(pattern: re.Pattern[str], replace: str | Callable[[re.Match[str]], str]) -> None:
        # One match at a time, so each step is a single contiguous replacement.
        position = 0
        while (match := pattern.search(current, position)) is not None:
            after = replace if isinstance(replace, str) else replace(match)
            step(current[: match.start()] + after + current[match.end() :])
            position = match.start() + len(after)
            if match.end() == match.start() and not after:
                position += 1

    if script == "zh-Hans":
        step(to_simplified(current))
        apply(_ELLIPSIS, _ellipsis)
        apply(_MARK, _full_width)
    if fillers == "remove":
        if is_zh(language):
            for pattern, replacement in _ZH_FILLER_RULES:
                apply(pattern, replacement)
        else:
            # A capitalised hesitation ("Um,") still goes; an all-caps one is more likely an acronym ("UM", "ER").
            apply(_EN_FILLERS, lambda match: match[0] if match[0].isupper() else "")
        # "I think, um, that" would otherwise keep both commas around the removed filler.
        apply(_REPEATED_COMMA, lambda match: match[1])
        apply(_SPACES, " ")
    for pattern, replacement in _PUNCTUATION_RULES:
        apply(pattern, replacement)
    step(current.strip())
    if not current:
        return Normalized(text, ())
    return Normalized(current, tuple(steps))
