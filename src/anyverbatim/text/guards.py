"""Deterministic guards on an LLM correction of one transcript segment.

A correction may fix homophones and garbled terms; it must never change what the speaker asserted. Every guard
compares the original with the proposed text and rejects recognized data, negation or direction changes and
destructive rewrites. They are conservative checks, not a proof of semantic equivalence. `check_numeric_fix` is the
inverse guard for numeric verification, where the number is the one thing allowed to change. `name_changes` finds the
Latin-script names a fix changed, for the caller to flag.

Ported from ytscript's ``analyze/correct.py`` with one guard loosened: an uppercase acronym or ticker may take one
inserted or deleted letter or one adjacent transposition (GAP→GAAP, SMCI→MSCI), never a substituted letter.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import takewhile
from typing import Literal

from rapidfuzz.distance import Levenshtein, Opcode, Opcodes

from anyverbatim.text.numbers import quantity_tokens

# Part of the stage key of every stage that applies these guards: bump it on any change to what they accept.
VERSION = "guards-v2"

GuardName = Literal[
    "empty",
    "shrink",
    "balloon",
    "digits",
    "numeric_spans",
    "quantities",
    "cjk_numerals",
    "negation",
    "direction",
    "symbols",
    "cjk_retention",
    "cjk_similarity",
]

MIN_KEEP_RATIO = 0.5
# A real homophone fix keeps roughly the original length, so a fix longer than max(floor, original × ratio) is a
# hallucination.
_BALLOON_RATIO = 1.6
_BALLOON_FLOOR = 20
# A predominantly-CJK segment's fix must keep this share of its CJK characters, or it was translated or
# transliterated.
_CJK_RETENTION_FLOOR = 0.5
# A wholesale rewrite (or a same-length mass swap) changes many characters. The ratio is noisy when one legitimate
# fix dominates a very short segment, so it applies only from a minimum length.
_CJK_MIN_SIMILARITY = 0.65
_CJK_RATIO_MIN_CHARS = 8
_CJK = re.compile(r"[\u4e00-\u9fff]")
_CJK_NUMERAL_CHARS = "〇零一二两兩三四五六七八九十百千万萬亿億廿卅"
_CJK_NUMERAL = re.compile(f"[{_CJK_NUMERAL_CHARS}]")
# `\d` is Unicode-aware and already matches full-width ０-９; spelling them out keeps a later narrowing to ASCII from
# dropping protection for the full-width digits common on CJK finance slides.
_DIGIT_RUN = re.compile(r"[\d０-９]+(?:\.[\d０-９]+)?")
_SIGN_OR_CURRENCY = r"(?:[-+−负負正](?:\s*[$¥€£])?\s*|[$¥€£](?:\s*[-+−负負正])?\s*)"
# Digit quantities keep their sign, currency and unit. Chinese numerals are compared as quantity tokens instead, so
# only a sign or currency symbol in front of them is tracked here (¥一百元 → 一百元 drops the currency).
_QUANTITY = (
    rf"{_SIGN_OR_CURRENCY}?(?:\d+(?:,\d{{3}}(?!\d))*(?:\.\d+)?|\.\d+)(?:\s*[%$¥€£元块塊分])?"
    rf"|{_SIGN_OR_CURRENCY}(?=[{_CJK_NUMERAL_CHARS}])"
)
# A numeric fix may touch only data characters; any other character in a changed run means a reword rode along.
_NON_DATA_CHAR = re.compile(rf"[^\s0-9０-９.,%％$¥€£A-Z{_CJK_NUMERAL_CHARS}]")
# Bare uppercase runs (NVDA, GAAP, SPX): acronyms and tickers, the one symbol class a fix may respell.
_BARE_SYMBOL = re.compile(r"[A-Z]{2,}")
_ASCII_ALNUM = re.compile(r"[A-Za-z0-9]")
# A Latin word for name flags: Latin letters, accented ones too (Zörbix), and digits, joined by ' & . - (O'Brien,
# AT&T, U.S., e-mail). Bare numbers are quantities, which have their own guards. × and ÷ sit among the Latin-1 letters
# but are not letters.
_LATIN_ALNUM = "A-Za-z0-9\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff"
_LATIN_WORD = re.compile(rf"[{_LATIN_ALNUM}]+(?:['&.\-][{_LATIN_ALNUM}]+)*")
_PRONOUN_I = re.compile(r"I(?:'(?:m|d|ll|ve))?")
_SENTENCE_END = ".!?…。！？"
# A period after a title does not end the sentence: the word after it is a name (Dr. Zorbix).
_TITLE_END = re.compile(r"(?<![A-Za-z])(?:Mr|Mrs|Ms|Dr|Prof|St|vs)\.$")
# Besides case, a format-only change alters only spaces and the joiners of a Latin word (Zorb-ix→Zorbix).
_FORMAT_ONLY = re.compile(r"[\s'’&.\-]")
# Dotted, exchange-prefixed and share-class tickers stay exact: BRK.B→BRK.A or $F→$T is a different security even
# though it is one letter away. Bare single letters (F, C, T) collide with English words, so only their explicit
# financial forms are protected.
_DOTTED_TICKER = re.compile(r"[A-Z]{2,}\.[A-Z]+|\d{2,}\.[A-Z]{2,}")
_SINGLE_FINANCIAL_TICKER = re.compile(
    r"\$\s*[A-Z](?:\.[A-Z])?(?![A-Za-z0-9])"
    r"|(?i:\b(?:NYSE|NASDAQ|AMEX|NYSEARCA|NYSEAMERICAN|BATS|OTC|OTCQX|OTCQB))\s*:\s*[A-Z](?:\.[A-Z])?(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[A-Z](?=股)"
)
# The major non-$ symbols too, so a ¥100→€100 swap with unchanged digits is still a changed amount.
_CURRENCY_AMOUNT = re.compile(r"[$¥€£][\d０-９][\d０-９.,]*")
# Spoken units carry value even when the recognizer writes the number as digits. A finite lexicon is deliberately
# conservative; it is not a semantic proofreader.
_NUMBER_WORDS = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundreds?|thousands?|millions?|billions?|trillions?|half|halves|quarters?"
)
_QUANTITY_WORDS = (
    r"negative|positive|minus|plus|dollars?|cents?|euros?|pounds?|pence|pennies|yen|yuan|renminbi|"
    r"sterling|rupees?|won|basis\s+points?|percentage\s+points?|percent(?:age)?|bps"
)
_CJK_CURRENCY_WORDS = (
    r"美元|美金|美分|欧元|歐元|欧分|歐分|港元|港币|港幣|人民币|人民幣|日元|日圆|日圓|"
    r"英镑|英鎊|新台币|新臺幣|台币|臺幣|澳元|澳币|澳幣|加元|新西兰元|紐西蘭元|瑞士法郎|韩元|韓元|韩币|韓幣"
)
_QUANTITY_TOKEN = re.compile(
    rf"{_QUANTITY}|(?i:\b(?:{_NUMBER_WORDS}|{_QUANTITY_WORDS})\b)|"
    rf"{_CJK_CURRENCY_WORDS}|{_SINGLE_FINANCIAL_TICKER.pattern}|{_DOTTED_TICKER.pattern}"
)
# Ordered numeric surface tokens: alphanumeric and dotted tickers (Q2, BRK.B, 0700.HK) and numbers with their sign,
# currency and percent. Letter-only symbols and Chinese numerals have their own guards. A ticker may follow a digit
# directly, as quantity_tokens splits it (1H25 is 1 and H25), so a fused one with a single letter is compared too.
_NUMERIC_SPAN = re.compile(
    r"(?<![A-Za-z])(?:[A-Z][A-Z0-9]+\.[A-Z]+|[A-Z][A-Z0-9]*[0-9][A-Z0-9]*|\d{2,}\.[A-Z]{2,})(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[-+−]?(?:[$¥€£][-+−]?)?(?:\d+(?:,\d{3}(?!\d))*(?:\.\d+)?|\.\d+)%?"
)
_NEGATION = re.compile(
    r"(?i:\b(?:not|no|never|neither|nor|without|cannot|"
    r"(?:ca|could|do|does|did|has|have|had|is|are|was|were|should|would|must|need|dare|wo|sha|ai)n'?t)\b)"
    r"|[不没沒无無未非勿别別莫甭否负負]"
)
_LEXICAL_UNIT = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]")
_ACTION_DIRECTION = re.compile(
    r"[买買卖賣]|(?<![A-Za-z0-9_])(?:buying|bought|buys|buy|selling|sells|sold|sell)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_BUY_FORMS = frozenset({"买", "買", "buy", "buys", "buying", "bought"})
# Opposite pairs grouped by what they describe, so 涨→降 counts as a flip but 多 next to 涨 does not. Each group holds
# its two sides; the longest word matches first, so 做空 is never read as 空.
_DIRECTION_GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "move": (
        ["上涨", "涨", "漲", "升", "up", "rise", "rises", "rising", "rose", "risen", "higher"],
        ["下跌", "跌", "降", "down", "fall", "falls", "falling", "fell", "fallen", "lower"],
    ),
    "amount": (
        ["增加", "增", "increase", "increases", "increased", "increasing"],
        ["减少", "減少", "减", "減", "decrease", "decreases", "decreased", "decreasing"],
    ),
    "stance": (["做多", "利好", "多", "牛", "long", "bullish"], ["做空", "利空", "空", "熊", "short", "bearish"]),
    "result": (["盈", "gain", "gains", "gained", "gaining"], ["亏", "虧", "loss", "losses"]),
}
_DIRECTION_SIDE = {
    word: (group, side)
    for group, sides in _DIRECTION_GROUPS.items()
    for side, words in enumerate(sides)
    for word in words
}
_DIRECTION_WORD = re.compile(
    "|".join(
        re.escape(word) if _CJK.match(word) else rf"(?<![a-z0-9_]){word}(?![a-z0-9_])"
        for word in sorted(_DIRECTION_SIDE, key=len, reverse=True)
    )
)


@dataclass(frozen=True, slots=True)
class EntityChange:
    """One respelled acronym, ticker or name; the offsets are half-open into the corrected text."""

    before: str
    after: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    rejected_by: GuardName | None
    entity_changes: tuple[EntityChange, ...] = ()

    @property
    def ok(self) -> bool:
        return self.rejected_by is None


def check_fix(original: str, fixed: str, *, min_keep_ratio: float = MIN_KEEP_RATIO) -> GuardVerdict:
    """Accept ``fixed`` as a correction of ``original``, or name the first guard it breaks.

    An accepted fix lists every acronym or ticker it respelled, so the caller can flag each one.
    """
    checks: tuple[tuple[GuardName, Callable[[], bool]], ...] = (
        ("empty", lambda: not fixed),
        ("shrink", lambda: len(fixed) < min_keep_ratio * len(original)),
        ("balloon", lambda: len(fixed) > max(_BALLOON_FLOOR, int(len(original) * _BALLOON_RATIO))),
        ("digits", lambda: _DIGIT_RUN.findall(fixed) != _DIGIT_RUN.findall(original)),
        ("numeric_spans", lambda: _numeric_spans(fixed) != _numeric_spans(original)),
        ("quantities", lambda: _quantity_runs(fixed) != _quantity_runs(original)),
        ("cjk_numerals", lambda: _cjk_numerals_changed(original, fixed)),
        ("negation", lambda: _negation_signature(fixed) != _negation_signature(original)),
        ("direction", lambda: _direction_flipped(original, fixed)),
        ("symbols", lambda: _exact_symbols(fixed) != _exact_symbols(original)),
        ("cjk_retention", lambda: _cjk_retention_broken(original, fixed)),
        ("cjk_similarity", lambda: _cjk_similarity_broken(original, fixed)),
    )
    for name, broken in checks:
        if broken():
            return GuardVerdict(name)
    before, after = _BARE_SYMBOL.findall(original), _BARE_SYMBOL.findall(fixed)
    if len(before) != len(after):
        return GuardVerdict("symbols")
    if any(old != new and not _is_near_symbol(old, new) for old, new in zip(before, after, strict=True)):
        return GuardVerdict("symbols")
    return GuardVerdict(None, symbol_changes(original, fixed))


def check_numeric_fix(original: str, fixed: str, suspects: list[str]) -> bool:
    """Accept a numeric fix only if it touches the flagged ``suspects`` and nothing else.

    The inverse of `check_fix`: that one forbids editing numbers, this one requires the edit be confined to the
    flagged tokens. It rejects an empty or echoed result, a truncation or balloon, a CJK-to-non-CJK rewrite, a changed
    run that touches a prose character, and a fix that also rewrites an unflagged data token.
    """
    if not fixed or fixed == original:
        return False
    if len(fixed) < MIN_KEEP_RATIO * len(original):
        return False
    if len(fixed) > max(_BALLOON_FLOOR, int(len(original) * _BALLOON_RATIO)):
        return False
    if _is_predominantly_cjk(original) and not _CJK.search(fixed):
        return False
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, original, fixed, autojunk=False).get_opcodes():
        if tag != "equal" and (_NON_DATA_CHAR.search(original[i1:i2]) or _NON_DATA_CHAR.search(fixed[j1:j2])):
            return False
    unflagged = Counter(_data_tokens(original)) - Counter(_data_tokens(" ".join(suspects)))
    return not unflagged - Counter(_data_tokens(fixed))


def symbol_changes(original: str, fixed: str) -> tuple[EntityChange, ...]:
    """Acronyms and tickers that differ position by position; both texts must hold the same number of them.

    Each uppercase run is named by the quantity ticker that holds it, or else by its whole word. A run can survive a
    merge unchanged while its ticker vanishes into a name (open AI→OpenAI), and that is recorded as ``AI → OpenAI``
    on the merged word; a ticker split out of a word is recorded the other way round.
    """
    before = list(_BARE_SYMBOL.finditer(original))
    after = list(_BARE_SYMBOL.finditer(fixed))
    if len(before) != len(after):
        raise ValueError(f"symbol count changed from {len(before)} to {len(after)}")
    changes: list[EntityChange] = []
    for (old, _), (new, (start, end)) in zip(_symbol_names(original, before), _symbol_names(fixed, after), strict=True):
        change = EntityChange(old, new, start, end)
        # A ticker or word holding two runs (NYSE: AI) changes once.
        if old != new and change not in changes:
            changes.append(change)
    return tuple(changes)


def name_changes(
    original: str, final: str, *, zh: bool, symbols: Sequence[EntityChange] = ()
) -> tuple[EntityChange, ...]:
    """Latin-script names that differ between ``original`` and ``final``, with offsets into ``final``.

    Words are diffed casefolded, and a stretch of changed words is one change. A word that a ticker change in
    ``symbols`` (`symbol_changes`) overlaps is left to the ticker flag and splits its stretch. Format-only changes
    (case, spaces and joiners: Open AI→OpenAI, Zorb-ix→Zorbix) and deletions, which leave nothing to point at, are not
    changes. A change is a name when any of its final words is name-like: in a Chinese segment (``zh``) any Latin
    word, otherwise a word with an uppercase letter after its first character or a capitalized one that does not start
    a sentence, never the pronoun I or its contractions.

    ``before`` is the original text in the change's place, found by aligning the two texts character by character:
    a name that replaced Chinese characters names them (佐比→Zorbix), and one inserted into unchanged text names
    nothing.
    """
    old, new = _name_tokens(original), _name_tokens(final)
    ops = Levenshtein.opcodes([_token_key(t) for t in old], [_token_key(t) for t in new])
    characters: Opcodes | None = None
    changes: list[EntityChange] = []
    for words in _changed_runs(ops, new, symbols):
        if not any(_name_like(w, final, zh) for w in words):
            continue
        start, end = words[0].start(), words[-1].end()
        if characters is None:
            characters = Levenshtein.opcodes(original, final)
        source_start, source_end = _source_range(characters, original, start, end)
        before = original[source_start:source_end].strip()
        if _squash(before) != _squash(final[start:end]):
            changes.append(EntityChange(before, final[start:end], start, end))
    return tuple(changes)


def _name_tokens(text: str) -> list[re.Match[str] | None]:
    """Latin words in order, with None between two words that other text than spaces separates (zorbix 和 xorbyte)."""
    tokens: list[re.Match[str] | None] = []
    previous: re.Match[str] | None = None
    for word in _LATIN_WORD.finditer(text):
        if not any(c.isalpha() for c in word[0]):
            continue
        if previous is not None and text[previous.end() : word.start()].strip():
            tokens.append(None)
        tokens.append(word)
        previous = word
    return tokens


def _token_key(token: re.Match[str] | None) -> str:
    return "" if token is None else token[0].casefold()


def _squash(text: str) -> str:
    return _FORMAT_ONLY.sub("", text).casefold()


def _changed_runs(
    ops: Opcodes, new: Sequence[re.Match[str] | None], symbols: Sequence[EntityChange]
) -> list[list[re.Match[str]]]:
    """Return the final words of each stretch of differing words, split by other text and by ticker changes."""
    runs: list[list[re.Match[str]]] = []
    k = 0
    while k < len(ops):
        if ops[k].tag == "equal":
            k += 1
            continue
        first = k
        while k < len(ops) and ops[k].tag != "equal":
            k += 1
        tokens = new[ops[first].dest_start : ops[k - 1].dest_end]
        runs += _word_runs([None if t is not None and _overlaps(t, symbols) else t for t in tokens])
    return runs


def _overlaps(word: re.Match[str], symbols: Sequence[EntityChange]) -> bool:
    return any(s.char_start < word.end() and word.start() < s.char_end for s in symbols)


def _word_runs(tokens: Sequence[re.Match[str] | None]) -> list[list[re.Match[str]]]:
    runs: list[list[re.Match[str]]] = [[]]
    for token in tokens:
        if token is None:
            runs.append([])
        else:
            runs[-1].append(token)
    return [run for run in runs if run]


def _source_range(ops: Opcodes, original: str, start: int, end: int) -> tuple[int, int]:
    """Return the original range in the place of the final range [start, end).

    Equal and replaced characters map one to one (a Levenshtein replace has as many characters on each side), and a
    position inside inserted text maps to where it was inserted. A deletion right at an edge adds the letters it
    removed there, so a dropped last letter belongs to the name (Zorbyxe→Zorbix) and a dropped word before it, which
    ends in a space, does not (um Zorbyx→Zorbix).
    """
    first = next(k for k, op in enumerate(ops) if op.dest_start <= start < op.dest_end)
    last = next(k for k, op in enumerate(ops) if op.dest_start < end <= op.dest_end)
    source_start, source_end = _source_position(ops[first], start), _source_position(ops[last], end)
    if start == ops[first].dest_start and first > 0 and ops[first - 1].tag == "delete":
        source_start -= _edge_letters(original[ops[first - 1].src_start : ops[first - 1].src_end], at_end=True)
    if end == ops[last].dest_end and last + 1 < len(ops) and ops[last + 1].tag == "delete":
        source_end += _edge_letters(original[ops[last + 1].src_start : ops[last + 1].src_end], at_end=False)
    return source_start, source_end


def _source_position(op: Opcode, position: int) -> int:
    return op.src_start if op.tag == "insert" else op.src_start + position - op.dest_start


def _edge_letters(text: str, *, at_end: bool) -> int:
    """Count the letters, digits and CJK characters at one end of ``text``."""
    return sum(1 for _ in takewhile(str.isalnum, reversed(text) if at_end else text))


def _name_like(word: re.Match[str], text: str, zh: bool) -> bool:
    if zh:
        return True
    surface = word[0]
    if _PRONOUN_I.fullmatch(surface):
        return False
    if any(c.isupper() for c in surface[1:]):
        return True
    return surface[0].isupper() and not _starts_sentence(text, word.start())


def _starts_sentence(text: str, position: int) -> bool:
    before = text[:position].rstrip(" \t\n\"'“‘([«")
    return not before or (before[-1] in _SENTENCE_END and _TITLE_END.search(before) is None)


def _symbol_names(text: str, runs: list[re.Match[str]]) -> list[tuple[str, tuple[int, int]]]:
    tickers = [token for token in quantity_tokens(text) if token.kind == "ticker"]
    names: list[tuple[str, tuple[int, int]]] = []
    for run in runs:
        ticker = next((t for t in tickers if t.char_start <= run.start() and run.end() <= t.char_end), None)
        if ticker is not None:
            names.append((ticker.surface, (ticker.char_start, ticker.char_end)))
            continue
        start, end = run.start(), run.end()
        while start > 0 and _ASCII_ALNUM.match(text[start - 1]):
            start -= 1
        while end < len(text) and _ASCII_ALNUM.match(text[end]):
            end += 1
        names.append((text[start:end], (start, end)))
    return names


def _is_near_symbol(before: str, after: str) -> bool:
    """One inserted or deleted letter (GAP→GAAP) or one adjacent transposition (SMCI→MSCI).

    A substituted letter, at any length, already names a different security or indicator (MS→GS, CPI→PPI).
    """
    if abs(len(before) - len(after)) == 1:
        shorter, longer = sorted((before, after), key=len)
        return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))
    return (
        before != after
        and len(before) == len(after)
        and any(before[:i] + before[i + 1] + before[i] + before[i + 2 :] == after for i in range(len(before) - 1))
    )


def _data_tokens(text: str) -> list[str]:
    return _DIGIT_RUN.findall(text) + _BARE_SYMBOL.findall(text) + _CJK_NUMERAL.findall(text)


def _cjk_numerals_changed(original: str, fixed: str) -> bool:
    """Quantities must keep their values in order, and the ordered numeral characters must not change at all.

    Tickers are left to the symbol guard. The character rule is deliberately blunt: it also refuses a fix that adds
    a numeral (依样→一样), because telling that apart from a dropped or swapped one (买三送一→买送一) is guesswork.
    """

    def values(text: str) -> list[tuple[str, str]]:
        return [(token.value_itn, token.kind) for token in quantity_tokens(text) if token.kind != "ticker"]

    return values(original) != values(fixed) or _CJK_NUMERAL.findall(original) != _CJK_NUMERAL.findall(fixed)


def _numeric_spans(text: str) -> list[str]:
    return _NUMERIC_SPAN.findall(unicodedata.normalize("NFKC", text))


def _quantity_runs(text: str) -> list[str]:
    return [token.casefold() for token in _QUANTITY_TOKEN.findall(unicodedata.normalize("NFKC", text))]


def _negation_signature(text: str) -> list[tuple[int, str]]:
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'").casefold()
    # Positions count lexical units, so punctuation fixes pass while moving 不 between otherwise equal spans does not.
    return [
        (len(_LEXICAL_UNIT.findall(normalized[: match.start()])), match[0]) for match in _NEGATION.finditer(normalized)
    ]


def _direction_signature(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return ["buy" if match[0] in _BUY_FORMS else "sell" for match in _ACTION_DIRECTION.finditer(normalized)]


def _direction_flipped(original: str, fixed: str) -> bool:
    """Buy and sell must match exactly; any other opposite pair may gain or lose a word but never turn over.

    Inserting or dropping 多 or up is common in a homophone fix (很都→很多), while a turn-over or reorder within a group
    (上涨→下跌, up→down) leaves neither side's sequence contained in the other.
    """
    if _direction_signature(fixed) != _direction_signature(original):
        return True
    before, after = _direction_sides(original), _direction_sides(fixed)
    return any(
        not (_is_subsequence(before[group], after[group]) or _is_subsequence(after[group], before[group]))
        for group in _DIRECTION_GROUPS
    )


def _direction_sides(text: str) -> dict[str, list[int]]:
    sides: dict[str, list[int]] = {group: [] for group in _DIRECTION_GROUPS}
    for match in _DIRECTION_WORD.finditer(unicodedata.normalize("NFKC", text).casefold()):
        group, side = _DIRECTION_SIDE[match[0]]
        sides[group].append(side)
    return sides


def _is_subsequence(short: list[int], long: list[int]) -> bool:
    remaining = iter(long)
    return all(item in remaining for item in short)


def _exact_symbols(text: str) -> Counter[str]:
    return Counter(
        [
            *_DOTTED_TICKER.findall(text),
            *_SINGLE_FINANCIAL_TICKER.findall(text),
            *_CURRENCY_AMOUNT.findall(text),
        ]
    )


def _is_predominantly_cjk(text: str) -> bool:
    non_space = len(re.sub(r"\s+", "", text))
    return non_space > 0 and len(_CJK.findall(text)) >= 0.5 * non_space


def _cjk_retention_broken(original: str, fixed: str) -> bool:
    if not _is_predominantly_cjk(original):
        return False
    return len(_CJK.findall(fixed)) < _CJK_RETENTION_FLOOR * len(_CJK.findall(original))


def _cjk_similarity_broken(original: str, fixed: str) -> bool:
    if not _is_predominantly_cjk(original) or len(_CJK.findall(original)) < _CJK_RATIO_MIN_CHARS:
        return False
    return SequenceMatcher(None, original, fixed).ratio() < _CJK_MIN_SIMILARITY
