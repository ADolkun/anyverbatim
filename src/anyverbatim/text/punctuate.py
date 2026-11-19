"""LLM pass that restores punctuation where an ASR stretch has none.

Whisper large-v3 often returns Mandarin with no punctuation at all (sometimes with a space at each pause), and English
with an unpunctuated, lowercase first stretch. Stretches that already carry enough sentence-final marks and, in Latin
script, some capitals pass through untouched. For the rest, the LLM may add or replace punctuation, turn pause spaces
into punctuation and capitalise English sentence starts; it may not add, drop or change a letter, CJK character
or digit. `guard_violation` enforces that on every reply: a violating item is retried once with the diff, and when it
fails again the original text is kept and its index is reported in `PunctuationResult.failed`. A reply that is not the
expected JSON raises `LLMOutputError` rather than shipping unpunctuated text as if it had been handled.

Segments are cut from continuous speech at pauses and a size limit, so a cut can fall mid-sentence or even inside a
word. Each segment therefore travels with the end of the segment before it and the start of the one after it, as
read-only context the reply must not contain, so the model can tell a sentence that ends at the cut from one that
goes on.

In a diarized transcript each segment also carries its speaker's id ("S2", never a name), and the prompt says the
segments are turns of a conversation. Each segment is punctuated by the rules of its own language, which in a
recording can differ from the document's.

Each batch that passes the guard is checkpointed, so a resume never pays twice for a finished batch.
"""

import hashlib
import json
import logging
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from anyverbatim import atomic
from anyverbatim.contract.models import Edit, LLMUsage
from anyverbatim.errors import LLMOutputError
from anyverbatim.llm.jsonio import complete_json
from anyverbatim.llm.protocol import LLMClient, LLMRequest
from anyverbatim.llm.usage import UsageMeter
from anyverbatim.text.segment import is_punctuated, uses_zh_rules

_log = logging.getLogger(__name__)

# Part of every checkpoint and stage key: bump it on any prompt, guard or logic change so stored results are redone.
PROMPT_VERSION = "punctuation-v4"
BATCH_CHARS = 1500
# A Latin stretch this long without one capital is Whisper's lowercase decoding, not a chat fragment or a lone
# lowercase brand: real prose of this length carries at least a sentence start or an "I".
_UNCASED_MIN_WORDS = 8
STAGE = "punctuation"

# Replies echo their input, and the claude CLI also counts thinking tokens here.
_MAX_OUTPUT_TOKENS = 8192
# Marks that are data in a transcript (S&P, 3.5%, #1) and so count as content for the guard.
_DATA_MARKS = frozenset("%&#@")
_MAX_DIFF_PARTS = 6
# How much of the neighbouring segment a segment is shown: enough to see that a word or sentence runs on across the cut.
_CONTEXT_CHARS = 16
_CONTEXT_WORDS = 4
_DIFF_CONTEXT = 12

Style = Literal["zh", "en"]

_RULES = (
    "Keep the SAME language and script (never translate, never convert between Traditional and Simplified). "
    "Keep every letter, CJK character and digit exactly as given, in the same order: never add, drop, repeat, reorder "
    "or change one, and do not fix mis-recognised words. Keep tickers and numbers intact (S&P 500, 3.5%, NVDA). "
    "Never add, remove or change punctuation inside or next to a number (1,000, 3.5, -5%, 1/2, 3:30, 10-K stay "
    "exactly as given). Never lowercase a letter. "
    "Never put quotation marks around arbitrary text. Never merge or split segments; every segment keeps its index. "
    "The segments are consecutive pieces of one continuous speech, cut at pauses or by length, not at sentence "
    "boundaries: a cut can fall mid-sentence or even inside a word. Each segment may come with `before` and `after`, "
    "the end of the previous segment and the start of the next one. They are for reading only: use them to judge "
    "whether the sentence is finished where the segment ends, and never include them in your reply. When the speech "
    "runs on across the cut (the next words complete the word or the sentence), end the segment with no "
    "sentence-final mark, or with a comma at most, and never put a mark inside a word. "
)
_STYLE_RULES: dict[Style, str] = {
    "zh": (
        "Use full-width Chinese punctuation: ，。？！；：、 . A space inside the text marks a pause: replace it with "
        "the punctuation that fits (usually ， or 。) and leave no pause spaces. A space between Latin words or "
        "numbers stays. Leave the case of Latin letters as given. "
    ),
    "en": (
        "Use ASCII punctuation. Capitalise the first letter of each sentence, even when the text already ends in a "
        "period, and capitalise proper nouns and acronyms that are unmistakable (the I pronoun, NVDA, Fed, Q4). Only "
        "ever turn a lowercase letter into its capital; never add, drop, respell or change a word. When `before` shows "
        "that the segment continues its sentence, the segment does not begin one: do not capitalise its first word. "
    ),
}
_SYSTEM = (
    "You restore punctuation in speech-to-text segments that have little or none. You are given numbered segments "
    "and return each one with sentence punctuation added so it reads naturally: periods or the Chinese equivalent at "
    "sentence ends, commas at clause breaks, question and exclamation marks where the speech asks for them. " + _RULES
)
_CONVERSATION = (
    "The segments are turns of a conversation: each carries `speaker`, the id of the person saying it, and speech "
    "never runs on from one speaker's segment into another speaker's. "
)
_SCHEMA_HINT = 'Return JSON {"items": [{"i": <index>, "text": "<punctuated text>"}]} with EVERY segment given.'


class _Item(BaseModel):
    model_config = ConfigDict(strict=True)
    i: int
    text: str


class _Items(BaseModel):
    model_config = ConfigDict(strict=True)
    items: list[_Item]


class _Checkpoint(BaseModel):
    """A batch's guarded reply and what it cost, so a reused batch still counts toward the document's LLM usage."""

    items: list[_Item]
    usage: tuple[LLMUsage, ...]


@dataclass(frozen=True, slots=True)
class PunctuationResult:
    """Results aligned with the input; ``failed`` lists the indices whose original text was kept unpunctuated."""

    texts: tuple[str, ...]
    edits: tuple[tuple[Edit, ...], ...]
    failed: tuple[int, ...]
    usage: tuple[LLMUsage, ...]


@dataclass(frozen=True, slots=True)
class _Piece:
    """A segment to punctuate with the edges of its neighbours as transcribed, which are only there to be read."""

    text: str
    before: str
    after: str
    speaker: str | None = None


@dataclass(frozen=True, slots=True)
class _Rejected:
    reply: str
    problem: str


def needs_punctuation(text: str, language: str) -> bool:
    return bool(text.strip()) and (not is_punctuated(text, language) or _is_uncased(text, language))


def _is_uncased(text: str, language: str) -> bool:
    """Latin-script text long enough to have capitals that has none, even when it already ends in periods."""
    if uses_zh_rules(text, language) or len(text.split()) < _UNCASED_MIN_WORDS:
        return False
    letters = [char for char in text if char.isalpha()]
    return any(char.isascii() for char in letters) and not any(char.isupper() for char in letters)


@dataclass(frozen=True, slots=True)
class _Reduced:
    skeleton: str
    # The skeleton before lowercasing, so a letter the reply lowercased can be told from one it capitalised.
    cased: str
    # The text, and for each skeleton character where it came from in it, so a diff can quote readable text.
    source: str
    positions: tuple[int, ...]

    def quote(self, start: int, end: int) -> str:
        if end <= start:
            return ""
        first, last = self.positions[start], self.positions[end - 1] + 1
        # Whole Latin words read better than the fragment a character diff lands on; CJK is quoted per character.
        while first > 0 and _is_latin_word_char(self.source[first - 1]) and _is_latin_word_char(self.source[first]):
            first -= 1
        while (
            last < len(self.source)
            and _is_latin_word_char(self.source[last])
            and _is_latin_word_char(self.source[last - 1])
        ):
            last += 1
        return self.source[first:last]

    def lead(self, start: int) -> str:
        stop = self.positions[start - 1] + 1 if start > 0 else 0
        return self.source[max(0, stop - _DIFF_CONTEXT) : stop].strip()


def _is_latin_word_char(char: str) -> bool:
    return char.isalnum() and not "\u3400" <= char <= "\u9fff"


def skeleton(text: str) -> str:
    """Reduce `text` to its lowercased letters, CJK characters, digits and data marks, dropping punctuation and spaces.

    Punctuation that belongs to a number stays, exactly as written: any mark before a digit (so between two digits too,
    and a sign), and a mark that joins a digit to a following Latin letter or digit (10-K). 1,000 → 1，000 or 3.5 → 35
    therefore changes the skeleton, while a period or comma after a number at a clause end does not.
    """
    return _reduce(text).skeleton


def _numeric_mark(text: str, position: int) -> bool:
    after = text[position + 1] if position + 1 < len(text) else ""
    before = text[position - 1] if position > 0 else ""
    return after.isdigit() or (before.isdigit() and after.isascii() and after.isalnum())


def _reduce(text: str) -> _Reduced:
    kept: list[str] = []
    cased: list[str] = []
    positions: list[int] = []
    for position, raw in enumerate(text):
        if raw.isspace():
            continue
        if not raw.isalnum() and _numeric_mark(text, position):
            # Raw, not NFKC-folded: a full-width ，between digits is not the , it would fold to.
            kept.append(raw)
            cased.append(raw)
            positions.append(position)
            continue
        char = unicodedata.normalize("NFKC", raw)
        if all(unicodedata.category(c).startswith("P") and c not in _DATA_MARKS for c in char):
            continue
        for c in char:
            if c.isspace() or (unicodedata.category(c).startswith("P") and c not in _DATA_MARKS):
                continue
            lowered = c.lower()
            kept.append(lowered)
            cased.append(c if len(lowered) == 1 else lowered)
            positions.extend([position] * len(lowered))
            if len(lowered) > 1:
                cased.extend(lowered[1:])
    return _Reduced("".join(kept), "".join(cased), text, tuple(positions))


def guard_violation(before: str, after: str) -> str | None:
    """Describe how `after` changes more than punctuation, spacing and capitalisation, or None when it does not.

    Letters may only be capitalised, never lowercased, and punctuation that belongs to a number must stay as written.
    """
    old, new = _reduce(before), _reduce(after)
    if old.skeleton == new.skeleton:
        lowered = [i for i, (a, b) in enumerate(zip(old.cased, new.cased, strict=True)) if a != b and a.isupper()]
        if not lowered:
            return None
        runs: list[list[int]] = []
        for i in lowered:
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        return "lowercased " + ", ".join(f'"{old.quote(a, b)}"' for a, b in runs[:_MAX_DIFF_PARTS])
    parts: list[str] = []
    for tag, a1, a2, b1, b2 in SequenceMatcher(None, old.skeleton, new.skeleton, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        lead = old.lead(a1)
        place = f' after "{lead}"' if lead else " at the start"
        match tag:
            case "delete":
                parts.append(f'dropped "{old.quote(a1, a2)}"{place}')
            case "insert":
                parts.append(f'added "{new.quote(b1, b2)}"{place}')
            case _:
                parts.append(f'changed "{old.quote(a1, a2)}" to "{new.quote(b1, b2)}"{place}')
    if len(parts) > _MAX_DIFF_PARTS:
        parts = [*parts[:_MAX_DIFF_PARTS], f"and {len(parts) - _MAX_DIFF_PARTS} more differences"]
    return "; ".join(parts)


def punctuate(
    texts: Sequence[str],
    *,
    language: str,
    client: LLMClient,
    checkpoint_dir: Path,
    languages: Sequence[str | None] = (),
    speakers: Sequence[str | None] = (),
) -> PunctuationResult:
    """Restore punctuation in the stretches of ``texts`` that lack it; the rest pass through unchanged.

    ``languages`` and ``speakers`` are per text when given: the language a text was spoken in (None for ``language``)
    and, for a diarized transcript, its speaker's id.
    """
    for name, given in (("languages", languages), ("speakers", speakers)):
        if given and len(given) != len(texts):
            raise ValueError(f"{name} covers {len(given)} texts, but there are {len(texts)}")
    run = _Run(
        texts, language=language, client=client, checkpoint_dir=checkpoint_dir, languages=languages, speakers=speakers
    )
    pending: dict[Style, list[int]] = {"zh": [], "en": []}
    for i, text in enumerate(texts):
        spoken = run.language_of(i)
        if needs_punctuation(text, spoken):
            pending["zh" if uses_zh_rules(text, spoken) else "en"].append(i)
    for style, indices in pending.items():
        for batch in _batches(texts, indices, BATCH_CHARS):
            run.punctuate_batch(style, batch)
    result = PunctuationResult(
        texts=tuple(run.texts),
        edits=tuple(tuple(edits) for edits in run.edits),
        failed=tuple(sorted(run.failed)),
        usage=run.meter.summary(),
    )
    _log.info(
        "punctuation: %d of %d stretches needed it, %d edited, %d kept unpunctuated",
        sum(len(batch) for batch in pending.values()),
        len(texts),
        sum(1 for edits in result.edits if edits),
        len(result.failed),
    )
    return result


class _Run:
    def __init__(
        self,
        texts: Sequence[str],
        *,
        language: str,
        client: LLMClient,
        checkpoint_dir: Path,
        languages: Sequence[str | None] = (),
        speakers: Sequence[str | None] = (),
    ) -> None:
        self.texts = list(texts)
        self._source = tuple(texts)
        self._languages = tuple(code or language for code in languages) or (language,) * len(texts)
        self._speakers = tuple(speakers)
        self.edits: list[list[Edit]] = [[] for _ in texts]
        self.failed: list[int] = []
        self.meter = UsageMeter()
        self._client = client
        self._checkpoint_dir = checkpoint_dir

    def punctuate_batch(self, style: Style, indices: list[int]) -> None:
        # Batch-local indices keep the checkpoint key independent of where the batch sits in the document.
        pieces = [self._piece(i) for i in indices]
        originals = [piece.text for piece in pieces]
        path = self._checkpoint_path(style, pieces)
        loaded = self._load(path, originals)
        if loaded is not None:
            outputs, usage = loaded
        else:
            batch = UsageMeter()
            outputs = self._ask_with_retry(style, pieces, batch)
            usage = batch.summary()
            # A batch with a kept-original item is not cached, so a resume gets another try at it.
            if all(text is not None for text in outputs):
                self._save(path, [text or "" for text in outputs], usage)
        self.meter.merge(usage)
        for index, original, output in zip(indices, originals, outputs, strict=True):
            if output is None:
                self.failed.append(index)
            elif output != original:
                self.texts[index] = output
                self.edits[index].append(Edit(stage=STAGE, before=original, after=output, llm=self._client.id))

    def language_of(self, index: int) -> str:
        return self._languages[index]

    def _piece(self, index: int) -> _Piece:
        # The neighbours as transcribed, not as an earlier batch may have punctuated them, so a batch's key and request
        # do not depend on which batches ran before it. A neighbour is cut by its own script, not this segment's, and
        # a blank one (filler removal can empty a stretch) is skipped, since the run-on signal lies beyond it.
        before = next((i for i in range(index - 1, -1, -1) if self._source[i].strip()), None)
        after = next((i for i in range(index + 1, len(self._source)) if self._source[i].strip()), None)
        return _Piece(
            self._source[index],
            self._tail(before) if before is not None else "",
            self._head(after) if after is not None else "",
            self._speakers[index] if self._speakers else None,
        )

    def _tail(self, index: int) -> str:
        text = self._source[index]
        if uses_zh_rules(text, self.language_of(index)):
            return text.strip()[-_CONTEXT_CHARS:]
        return " ".join(text.split()[-_CONTEXT_WORDS:])

    def _head(self, index: int) -> str:
        text = self._source[index]
        if uses_zh_rules(text, self.language_of(index)):
            return text.strip()[:_CONTEXT_CHARS]
        return " ".join(text.split()[:_CONTEXT_WORDS])

    def _ask_with_retry(self, style: Style, pieces: list[_Piece], meter: UsageMeter) -> list[str | None]:
        outputs: list[str | None] = [None] * len(pieces)
        rejected = self._guarded(style, pieces, list(range(len(pieces))), None, outputs, meter)
        if rejected:
            rejected = self._guarded(style, pieces, sorted(rejected), rejected, outputs, meter)
        return outputs

    def _guarded(
        self,
        style: Style,
        pieces: list[_Piece],
        asked: list[int],
        previous: dict[int, _Rejected] | None,
        outputs: list[str | None],
        meter: UsageMeter,
    ) -> dict[int, _Rejected]:
        """Ask for ``asked``, store the replies that pass the guard in ``outputs`` and return the ones that do not."""
        rejected: dict[int, _Rejected] = {}
        for index, text in self._ask(style, pieces, asked, previous, meter).items():
            problem = guard_violation(pieces[index].text, text)
            if problem is None:
                outputs[index] = text
            else:
                rejected[index] = _Rejected(text, problem)
        return rejected

    def _ask(
        self,
        style: Style,
        pieces: list[_Piece],
        asked: list[int],
        previous: dict[int, _Rejected] | None,
        meter: UsageMeter,
    ) -> dict[int, str]:
        client = meter.metered(STAGE, self._client)
        expected = sorted(asked)
        complaint = ""
        for _ in range(2):
            conversation = _CONVERSATION if any(piece.speaker is not None for piece in pieces) else ""
            request = LLMRequest(
                system=_SYSTEM + _STYLE_RULES[style] + conversation,
                user=_prompt(pieces, asked, previous) + complaint,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.0,
            )
            reply = complete_json(client, request, _Items)
            got = sorted(item.i for item in reply.items)
            if got == expected:
                return {item.i: item.text.strip() for item in reply.items}
            complaint = f"\n\nYour previous reply had the indices {got}; it must contain exactly {expected}, once each."
        raise LLMOutputError(f"{self._client.id} did not return exactly the requested indices {expected} twice")

    def _checkpoint_path(self, style: Style, pieces: list[_Piece]) -> Path:
        batch_text = json.dumps(_batch_items(pieces, range(len(pieces))), ensure_ascii=False)
        key = json.dumps([PROMPT_VERSION, self._client.id, style, batch_text], ensure_ascii=False)
        return self._checkpoint_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def _load(self, path: Path, originals: list[str]) -> tuple[list[str | None], tuple[LLMUsage, ...]] | None:
        if not path.exists():
            return None
        try:
            checkpoint = _Checkpoint.model_validate_json(path.read_bytes())
        except ValidationError:
            # Only this module writes these files, atomically; an unreadable one is stale, and redoing it is safe.
            _log.warning("punctuation: ignoring an unreadable checkpoint %s", path.name)
            return None
        items = checkpoint.items
        by_index = {item.i: item.text for item in items}
        if sorted(by_index) != list(range(len(originals))) or len(items) != len(originals):
            return None
        # The guard is re-applied so a stricter guard never trusts a reply an older one let through.
        if any(guard_violation(original, by_index[i]) is not None for i, original in enumerate(originals)):
            return None
        return [by_index[i] for i in range(len(originals))], checkpoint.usage

    def _save(self, path: Path, outputs: list[str], usage: tuple[LLMUsage, ...]) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        items = [_Item(i=i, text=text) for i, text in enumerate(outputs)]
        atomic.write_json(path, _Checkpoint(items=items, usage=usage).model_dump(mode="json"))


def _item(pieces: Sequence[_Piece], i: int) -> dict[str, object]:
    piece = pieces[i]
    item: dict[str, object] = {"i": i}
    if piece.speaker is not None:
        item["speaker"] = piece.speaker
    if piece.before:
        item["before"] = piece.before
    item["text"] = piece.text
    if piece.after:
        item["after"] = piece.after
    return item


def _batch_items(pieces: Sequence[_Piece], indices: Sequence[int]) -> list[dict[str, object]]:
    return [_item(pieces, i) for i in indices]


def _prompt(pieces: list[_Piece], asked: list[int], previous: dict[int, _Rejected] | None) -> str:
    if previous is None:
        return (
            "Restore the punctuation in these segments. "
            + _SCHEMA_HINT
            + "\n\nSEGMENTS:\n"
            + json.dumps(_batch_items(pieces, asked), ensure_ascii=False)
        )
    items: list[dict[str, object]] = [
        {**_item(pieces, i), "rejected_reply": previous[i].reply, "problem": previous[i].problem} for i in asked
    ]
    return (
        "Your punctuation for these segments was rejected because it changed letters, characters or digits, not only "
        "punctuation. Redo each one from its original `text`: touch only punctuation, spacing and sentence-initial "
        "capitals. " + _SCHEMA_HINT + "\n\nSEGMENTS:\n" + json.dumps(items, ensure_ascii=False)
    )


def _batches(texts: Sequence[str], indices: list[int], batch_chars: int) -> list[list[int]]:
    """Group indices into batches of about ``batch_chars`` characters; one oversized text is a batch on its own."""
    batches: list[list[int]] = []
    current: list[int] = []
    running = 0
    for i in indices:
        current.append(i)
        running += len(texts[i])
        if running >= batch_chars:
            batches.append(current)
            current = []
            running = 0
    if current:
        batches.append(current)
    return batches
