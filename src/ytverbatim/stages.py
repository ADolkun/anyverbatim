"""The pure stages of `ytverbatim run`: source transcript → stretches → text stages → contract document.

A *stretch* is the unit every text stage works on: one sentence-like piece of the source transcript, with its time
range, the engine's timed words inside it, its speaker when the source is diarized, and the flags and edits raised on
it so far. The source is cut into stretches once (at its own punctuation, or at pauses and a size limit where it has
none, and always at a change of speaker). Text stages only replace a stretch's text; flags raised earlier are moved
through each replacement by a diff, and every edit is kept.

`build_document` then re-times each stretch's final text against its words, re-cuts the transcript into sentences
within each run of same-speaker stretches (punctuation restored by an LLM can merge or split stretches) and maps every
flag and edit onto the sentence that holds it. Nothing here performs I/O.
"""

import collections
import difflib
import itertools
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from typing import Annotated, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from ytverbatim import __version__, ids
from ytverbatim.acquire.captions import CaptionTrack
from ytverbatim.acquire.creator_note import CreatorNote
from ytverbatim.acquire.metadata import Chapter as SourceChapter
from ytverbatim.acquire.metadata import VideoMetadata
from ytverbatim.asr import chunks
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.contract import (
    Access,
    Chapter,
    ContextDocument,
    DiarizationRun,
    Edit,
    Flag,
    LLMUsage,
    Paragraph,
    Quality,
    Segment,
    Source,
    SourceRef,
    Speaker,
    Transcript,
    TranscriptRun,
    Word,
    YouTubeSource,
)
from ytverbatim.contract import CreatorNote as ContractCreatorNote
from ytverbatim.contract import Generator as GeneratorInfo
from ytverbatim.contract.models import SCHEMA_VERSION
from ytverbatim.diarize import attribution
from ytverbatim.diarize.protocol import Diarization, DiarizerDef
from ytverbatim.errors import ContractError
from ytverbatim.store import digest
from ytverbatim.text import clean, numbers, offsets
from ytverbatim.text import numerals as numerals_module
from ytverbatim.text import segment as seg

# Part of every stage key that runs this module's code, and of the context record: bump it on any logic change.
VERSION = "stages-v1"
UNDETERMINED_LANGUAGE = "und"
CAPTION_MANUAL = "caption_manual"

# The chunk flag that marks a decode the pipeline cannot vouch for; the others only describe how the audio was cut.
_ANOMALY_FLAG = "decoder_anomaly"

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stretch(_Model):
    text: str
    start: Seconds
    end: Seconds
    # The engine's timed words in this stretch, absolute times; the text is re-timed against them after every edit.
    words: tuple[Word, ...] = ()
    # Whether the source had word timings here at all; untimed text in a word-timed stretch is interpolated.
    word_timed: bool = False
    # Offsets into `text`.
    flags: tuple[Flag, ...] = ()
    # Whole-stretch before/after texts, in the order the stages made them.
    edits: tuple[Edit, ...] = ()
    # Left out of every dump while None, so the stage keys of undiarized documents stay what they were.
    speaker: str | None = Field(default=None, exclude_if=lambda value: value is None)
    language: str | None = Field(default=None, exclude_if=lambda value: value is None)


class Transcribed(_Model):
    """The source transcript, cut into stretches, and what the source itself says about its quality."""

    run: TranscriptRun
    language: str
    stretches: tuple[Stretch, ...]
    # None when the source has no speech estimate of its own (whole-file engines, captions).
    speech_seconds: Seconds | None
    decoder_anomalies: int
    chunk_flags: dict[str, int]
    warnings: tuple[str, ...]
    # Both set only for a diarized transcript, and left out of every dump otherwise, as for `Stretch.speaker`.
    diarization: DiarizationRun | None = Field(default=None, exclude_if=lambda value: value is None)
    # The digest of the speaker-attributed stretches (start, end, speaker), which speaker names are bound to: a rerun
    # of a text stage keeps it, and anything that could move a speaker changes it.
    assignment: str | None = Field(default=None, exclude_if=lambda value: value is None)


@dataclass(frozen=True)
class SourceInfo:
    """A source as `build_document` reads it; the adapter that fetched or opened the source builds it.

    `describe` gets the source transcript's stretches, so a source that knows no duration of its own can take the
    transcript's end.
    """

    id: str
    chapters: tuple[SourceChapter, ...]
    describe: Callable[[Sequence[Stretch]], Source]


class Diarized(NamedTuple):
    """A stored diarization of the source's audio and the diarizer that made it."""

    definition: DiarizerDef
    diarization: Diarization


def run_id(definition: EngineDef) -> str:
    return f"asr:{definition.spec.id}"


def diarization_run(definition: DiarizerDef) -> DiarizationRun:
    """Describe the diarizer for the document; `params` include the attribution rules, so a rule change shows."""
    rules: dict[str, JsonValue] = {
        "version": attribution.VERSION,
        "gap_s": attribution.GAP_S,
        "overlap_share": attribution.OVERLAP_SHARE,
        "smoothing": attribution.SMOOTHING,
        "smooth_s": attribution.SMOOTH_S,
        "smooth_words": attribution.SMOOTH_WORDS,
    }
    spec = definition.spec
    return DiarizationRun(
        engine=spec.id,
        model_repo=spec.repo,
        model_revision=spec.revision,
        params={**definition.variant, "attribution": rules},
    )


def from_asr_rows(
    rows: Sequence[ChunkHypothesis],
    definition: EngineDef,
    language_hint: str | None,
    diarized: Diarized | None = None,
) -> Transcribed:
    """Turn one engine's chunk rows (sorted by chunk) into stretches; `language_hint` is the source's declared one.

    With `diarized`, every token gets the speaker attribution gives it, a stretch never crosses a change of speaker,
    and the words a second speaker talks over are flagged `overlapping_speech`.
    """
    spec = definition.spec
    language = _language(rows, language_hint)
    tokens, warnings = _tokens(rows, language)
    run: DiarizationRun | None = None
    if diarized is not None:
        tokens = _attributed(tokens, diarized.diarization)
        if any(token.speaker is not None for token in tokens):
            run = diarization_run(diarized.definition)
        elif tokens:
            warnings.append("the diarizer heard no speaker anywhere, so the transcript has no speaker labels")
    sentences = seg.segment_tokens(tokens, language)
    overlap_flags = [_overlap_flags(sentence) for sentence in sentences]
    worded = {row.chunk for row in rows if any(word.text.strip() for word in row.words)}
    anomalies = {row.chunk: row for row in rows if _ANOMALY_FLAG in row.flags}
    stretches = tuple(
        Stretch(
            text=sentence.text,
            start=sentence.start,
            end=sentence.end,
            words=tuple(
                Word(text=t.text.strip(), start=t.start, end=t.end, conf=t.conf)
                for t in sentence.tokens
                if t.timed and t.text.strip()
            ),
            word_timed=any(t.origin in worded for t in sentence.tokens),
            flags=_anomaly_flags(sentence, anomalies) + overlap_flags[number],
            speaker=sentence.speaker,
        )
        for number, sentence in enumerate(sentences)
    )
    transcript_run = TranscriptRun(
        id=run_id(definition),
        kind="asr",
        engine=spec.id,
        model_repo=spec.repo,
        model_revision=spec.revision,
        params={**definition.variant, "language_hint": language_hint, "sample_rate": spec.sample_rate},
        vad=_vad_info() if spec.max_chunk_s is not None else None,
        chunking=_chunking_info(spec.max_chunk_s, rows),
        aligner=None,
    )
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        counts.update(row.flags)
        if not row.text.strip() and "empty_output" not in row.flags:
            counts["empty_output"] += 1
    return Transcribed(
        run=transcript_run,
        language=language,
        stretches=stretches,
        # Without a VAD (whole-file engines) there is no separate speech estimate; the transcribed span stands in.
        speech_seconds=sum(row.end - row.start for row in rows) if spec.max_chunk_s is not None else None,
        decoder_anomalies=counts[_ANOMALY_FLAG],
        chunk_flags=dict(counts),
        warnings=tuple(warnings),
        diarization=run,
        assignment=digest([[s.start, s.end, s.speaker] for s in stretches]) if run is not None else None,
    )


def _attributed(tokens: Sequence[seg.Token], diarization: Diarization) -> list[seg.Token]:
    """Each token with its speaker, and marked when another speaker talks over it.

    Tokens are attributed rather than the engine's words: they are the timed units sentences are cut from, and for an
    engine without word timings they are the only timed units there are.
    """
    words = [Word(text=token.text.strip(), start=token.start, end=token.end) for token in tokens]
    result = attribution.assign(words, diarization, smoothing=attribution.SMOOTHING)
    return [
        replace(token, speaker=speaker, overlapped=flagged)
        for token, speaker, flagged in zip(tokens, result.speakers, result.overlapped, strict=True)
    ]


def _overlap_flags(sentence: seg.Sentence) -> tuple[Flag, ...]:
    """Flag `overlapping_speech` over each run of the sentence's tokens another speaker talks over."""
    flags: list[Flag] = []
    hits = [i for i, token in enumerate(sentence.tokens) if token.overlapped]
    for _, group in itertools.groupby(enumerate(hits), key=lambda pair: pair[1] - pair[0]):
        run = [i for _, i in group]
        span = sentence.span(run[0], run[-1])
        if span is not None:
            note = "another speaker talks at the same time here"
            flags.append(Flag(kind="overlapping_speech", char_start=span[0], char_end=span[1], note=note))
    return tuple(flags)


def caption_run_id(track: CaptionTrack) -> str:
    return f"{CAPTION_MANUAL}:{track.language}"


def from_captions(track: CaptionTrack, language_hint: str | None) -> Transcribed:
    """Turn a manual caption track into stretches; cue times are spread over each cue's characters."""
    language = language_hint or track.language.split("-")[0]
    zh = seg.uses_zh_rules(" ".join(cue.text for cue in track.segments), language)
    tokens: list[seg.Token] = []
    for index, cue in enumerate(track.segments):
        piece = seg.pseudo_tokens(cue.text, cue.start, max(cue.end, cue.start), language=language, origin=index)
        tokens += _joined(tokens, piece, zh=zh)
    stretches = tuple(
        Stretch(text=sentence.text, start=sentence.start, end=sentence.end)
        for sentence in seg.segment_tokens(_monotonic(tokens), language)
    )
    run = TranscriptRun(
        id=caption_run_id(track),
        kind=CAPTION_MANUAL,
        engine=None,
        model_repo=None,
        model_revision=None,
        params={"language": track.language, "format": track.source_format},
        vad=None,
        chunking=None,
        aligner=None,
    )
    return Transcribed(
        run=run,
        language=language,
        stretches=stretches,
        speech_seconds=None,
        decoder_anomalies=0,
        chunk_flags={},
        warnings=(),
    )


def normalize(
    stretches: Sequence[Stretch], language: str, *, script: clean.Script, fillers: clean.Fillers
) -> tuple[Stretch, ...]:
    """Convert script and remove fillers; flags move with the text and no edit is recorded (nothing was misheard)."""
    out: list[Stretch] = []
    for stretch in stretches:
        result = clean.normalize(stretch.text, language, script=script, fillers=fillers)
        words = stretch.words
        if seg.is_zh(language) and script == "zh-Hans":
            # The words are matched against the text by characters, so they must be in the same script.
            words = tuple(word.model_copy(update={"text": clean.to_simplified(word.text)}) for word in words)
        out.append(
            stretch.model_copy(
                update={
                    "text": result.text,
                    "words": words,
                    "flags": offsets.remap_flags(stretch.flags, result.steps),
                }
            )
        )
    return tuple(out)


def numerals(stretches: Sequence[Stretch], language: str) -> tuple[Stretch, ...]:
    """Write Mandarin amounts as digits (`text.numerals`); flags, words and edits follow, and no edit is recorded.

    Words covering a rewritten figure become one word with the display text, so `build_document` still times the
    figure by the engine. A figure its words only partly match stays as said, so no word loses its time. Edits carry
    the same rewrites, so they read in the form of the text they lead to.
    """
    if not seg.is_zh(language):
        return tuple(stretches)
    out: list[Stretch] = []
    for stretch in stretches:
        changes = _timeable_conversions(stretch)
        out.append(
            stretch.model_copy(
                update={
                    "text": numerals_module.apply(stretch.text, changes),
                    "words": _merged_words(stretch.text, stretch.words, changes) if changes else stretch.words,
                    "flags": offsets.replace_under_flags(stretch.flags, changes),
                    "edits": _carried_edits(stretch.edits, changes),
                }
            )
        )
    return tuple(out)


def _timeable_conversions(stretch: Stretch) -> list[numerals_module.Conversion]:
    changes = numerals_module.conversions(stretch.text)
    if not changes or not stretch.words:
        return changes
    covered = {i for a, b in seg.word_spans(stretch.text, stretch.words).values() for i in range(a, b)}

    def whole_or_untimed(change: numerals_module.Conversion) -> bool:
        figure = [i for i in range(change[0], change[1]) if stretch.text[i].isalnum()]
        timed = sum(i in covered for i in figure)
        return timed in (0, len(figure))

    if all(whole_or_untimed(change) for change in changes):
        return changes
    return numerals_module.conversions(stretch.text, allowed=whole_or_untimed)


def _carried_edits(edits: Sequence[Edit], changes: Sequence[numerals_module.Conversion]) -> tuple[Edit, ...]:
    """Rewrite each edit with the rewrites of the text it leads to, carried back along the chain of edits.

    Each edit's `after` is the next one's `before`, and the last one's is the stretch text. A rewrite reaches an
    earlier text only with a whole quantity the edit left alone, so 张了两点 → 涨了两点 reads 张了2点 → 涨了2点, as the
    move word in the final text decided. A figure the edit changed takes its own text's rewrites, so a numeric fix
    reads 35 → 50, and an earlier text whose figures would read otherwise takes its own rewrites throughout.
    """
    carried = list(changes)
    out: list[Edit] = []
    for edit in reversed(edits):
        after = numerals_module.apply(edit.after, carried)
        back = _carried_back(edit.before, edit.after, carried)
        if numerals_module.readings(numerals_module.apply(edit.before, back)) != numerals_module.readings(edit.before):
            back = numerals_module.conversions(edit.before)
        out.append(edit.model_copy(update={"before": numerals_module.apply(edit.before, back), "after": after}))
        carried = back
    return tuple(reversed(out))


def _carried_back(
    before: str, after: str, changes: Sequence[numerals_module.Conversion]
) -> list[numerals_module.Conversion]:
    """Map rewrites of `after` onto `before` where their whole quantity token is unchanged; `before`'s own elsewhere.

    A token is unchanged when it sits inside one block the diff found equal, with the same boundaries in both texts.
    There the final text decides, rewritten or not; 三万 in 收入三万五美元 → 收入三万美元 was part of 三万五, so
    `before` takes its own rewrite of that figure instead.
    """
    opcodes = difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
    equal = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in opcodes if tag == "equal"]
    before_tokens = [(t.char_start, t.char_end) for t in numbers.quantity_tokens(before)]
    after_tokens = {(t.char_start, t.char_end) for t in numbers.quantity_tokens(after)}

    def shift(token: tuple[int, int]) -> int | None:
        # How far an unchanged token moves from `before` to `after`; None when the edit touched it.
        for i1, i2, j1, _ in equal:
            if i1 <= token[0] and token[1] <= i2 and (token[0] - i1 + j1, token[1] - i1 + j1) in after_tokens:
                return j1 - i1
        return None

    def token_of(start: int, end: int) -> tuple[int, int]:
        return next(((a, b) for a, b in before_tokens if a <= start and end <= b), (start, end))

    kept = {token: moved for token in before_tokens if (moved := shift(token)) is not None}
    back: list[numerals_module.Conversion] = []
    for start, end, replacement in changes:
        for (a, b), moved in kept.items():
            if a + moved <= start and end <= b + moved:
                back.append((start - moved, end - moved, replacement))
                break
    own = [change for change in numerals_module.conversions(before) if token_of(change[0], change[1]) not in kept]
    return sorted(back + own)


def _merged_words(text: str, words: Sequence[Word], changes: Sequence[numerals_module.Conversion]) -> tuple[Word, ...]:
    """Replace the words over each rewritten figure by one word that spells the figure as displayed.

    Changes and the words over them are grouped while their ranges overlap (三点四十 heard as one word holds two
    figures). A group is merged only when its words cover every letter and digit of its figures.
    """
    spans = seg.word_spans(text, words)
    groups: list[tuple[int, int, list[numerals_module.Conversion], list[int]]] = []
    for change in changes:
        start, end, _ = change
        over = [index for index, (a, b) in spans.items() if a < end and start < b]
        lo = min([start, *(spans[index][0] for index in over)])
        hi = max([end, *(spans[index][1] for index in over)])
        if groups and groups[-1][1] > lo:
            g_lo, g_hi, g_changes, g_words = groups[-1]
            groups[-1] = (g_lo, max(g_hi, hi), [*g_changes, change], sorted({*g_words, *over}))
        else:
            groups.append((lo, hi, [change], over))
    covered = {i for a, b in spans.values() for i in range(a, b)}
    merged: dict[int, Word] = {}
    dropped: set[int] = set()
    for lo, hi, group_changes, group_words in groups:
        figure_chars = (i for start, end, _ in group_changes for i in range(start, end) if text[i].isalnum())
        if not group_words or not all(i in covered for i in figure_chars):
            continue
        shifted = [(start - lo, end - lo, replacement) for start, end, replacement in group_changes]
        conf = [c for index in group_words if (c := words[index].conf) is not None]
        merged[group_words[0]] = Word(
            text=numerals_module.apply(text[lo:hi], shifted),
            start=min(words[index].start for index in group_words),
            end=max(words[index].end for index in group_words),
            conf=min(conf) if conf else None,
        )
        dropped.update(group_words[1:])
    return tuple(merged.get(index, word) for index, word in enumerate(words) if index not in dropped)


def with_results(
    stretches: Sequence[Stretch],
    texts: Sequence[str],
    edits: Sequence[Sequence[Edit]],
    flags: Sequence[Sequence[Flag]] = (),
    *,
    record: bool = True,
) -> tuple[Stretch, ...]:
    """Apply one text stage's per-stretch results: earlier flags follow the edits, the stage's own flags are added.

    With `record=False` the edits only move the flags and are not kept: punctuation changes no word, the guard proves
    it, and hundreds of inserted commas per video would bury the edits a reader needs.
    """
    out: list[Stretch] = []
    for index, stretch in enumerate(stretches):
        new_edits = tuple(edits[index])
        moved = offsets.remap_flags(stretch.flags, ((edit.before, edit.after) for edit in new_edits))
        added = tuple(flags[index]) if flags else ()
        kept = stretch.edits + new_edits if record else stretch.edits
        out.append(stretch.model_copy(update={"text": texts[index], "flags": moved + added, "edits": kept}))
    return tuple(out)


def word_confidence(stretches: Sequence[Stretch]) -> list[list[float] | None] | None:
    """Per-stretch word probabilities, or None when no word carries one (an engine that reports none)."""
    confidence = [[word.conf for word in stretch.words if word.conf is not None] or None for stretch in stretches]
    return confidence if any(confidence) else None


def stretch_flag(stretch: Stretch, kind: str, note: str) -> Flag:
    return Flag.model_validate({"kind": kind, "char_start": 0, "char_end": len(stretch.text), "note": note})


def build_document(
    *,
    info: SourceInfo,
    source: Transcribed,
    stretches: Sequence[Stretch],
    chapters: bool,
    created_at: datetime,
    correction_llm: str | None = None,
    llm_punctuated: bool = False,
    warnings: Sequence[str] = (),
    llm_usage: Sequence[LLMUsage] = (),
    speaker_names: Mapping[str, str] | None = None,
) -> ContextDocument:
    """Assemble and validate the document from the final stretches; `llm_punctuated` says an LLM added punctuation.

    `speaker_names` maps speaker ids to the names the user gave them; an id the final segments no longer have is unused.

    Raises `ContractError` when the inputs cannot make a valid document, such as a chapter that ends before it starts.
    """
    try:
        return _build(
            info,
            source,
            stretches,
            chapters=chapters,
            created_at=created_at,
            correction_llm=correction_llm,
            llm_punctuated=llm_punctuated,
            warnings=warnings,
            llm_usage=llm_usage,
            speaker_names=speaker_names or {},
        )
    except ValidationError as exc:
        raise ContractError(f"{info.id}: the transcript does not fit the contract:\n{exc}") from exc


@dataclass(frozen=True, slots=True)
class _Placed:
    """Where one token of the final token stream came from and where it landed."""

    stretch: int
    # The token's first character as an offset into its stretch's text; -1 for a joining space added before it.
    stretch_start: int
    stretch_end: int
    sentence: int
    sentence_start: int


@dataclass(frozen=True, slots=True)
class _Layout:
    sentences: list[seg.Sentence]
    # Per stretch, its tokens in order.
    placed: dict[int, list[_Placed]]
    interpolated: set[int]

    def spans(self, stretch: int, start: int, end: int) -> dict[int, tuple[int, int]]:
        """Find the sentence ranges that hold `[start, end)` of a stretch, by sentence index."""
        hits: dict[int, tuple[int, int]] = {}
        for token in self.placed.get(stretch, ()):
            lo, hi = max(start, token.stretch_start), min(end, token.stretch_end)
            if hi <= lo:
                continue
            length = len(self.sentences[token.sentence].text)
            a = min(max(token.sentence_start + lo - token.stretch_start, 0), length)
            b = min(max(token.sentence_start + hi - token.stretch_start, 0), length)
            if b <= a:
                continue
            known = hits.get(token.sentence)
            hits[token.sentence] = (min(a, known[0]), max(b, known[1])) if known else (a, b)
        return hits

    def sentence_at(self, stretch: int, offset: int) -> int | None:
        tokens = self.placed.get(stretch)
        if not tokens:
            return None
        for token in tokens:
            if offset < token.stretch_end:
                return token.sentence
        return tokens[-1].sentence


def _build(
    info: SourceInfo,
    source: Transcribed,
    stretches: Sequence[Stretch],
    *,
    chapters: bool,
    created_at: datetime,
    correction_llm: str | None,
    llm_punctuated: bool,
    warnings: Sequence[str],
    llm_usage: Sequence[LLMUsage],
    speaker_names: Mapping[str, str],
) -> ContextDocument:
    language = source.language
    layout = _layout(stretches, language)
    sentences = layout.sentences
    notes = [*source.warnings, *warnings]
    text = " ".join(sentence.text for sentence in sentences)
    script = _script(language, text)
    if script == "zh-Hant":
        notes.append("the transcript contains Traditional Chinese characters, which are not converted")
    if not sentences:
        notes.append("no speech was transcribed")
    if layout.interpolated:
        notes.append(
            f"word timings did not cover the text in {len(layout.interpolated)} stretch(es); the uncovered words are "
            "timed between their timed neighbours and their segments are flagged timing_interpolated"
        )

    flags: list[list[Flag]] = [[] for _ in sentences]
    edits: list[list[Edit]] = [[] for _ in sentences]
    for index, stretch in enumerate(stretches):
        for flag in stretch.flags:
            for k, (a, b) in layout.spans(index, flag.char_start, flag.char_end).items():
                flags[k].append(flag.model_copy(update={"char_start": a, "char_end": b}))
        for k, edit in _placed_edits(layout, index, stretch.edits):
            edits[k].append(edit)

    chapter_of, chapter_list = _chapters(info.chapters, sentences, enabled=chapters)
    turns = [sentence.speaker for sentence in sentences] if source.diarization is not None else []
    paragraph_ranges = seg.split_paragraphs(
        [s.start for s in sentences], [s.end for s in sentences], chapter_of, speakers=turns
    )
    paragraph_of = [index for index, (first, last) in enumerate(paragraph_ranges) for _ in range(first, last + 1)]
    run = source.run
    segments = tuple(
        _segment(
            index,
            sentence,
            run,
            paragraph=paragraph_of[index],
            chapter=chapter_of[index],
            flags=flags[index],
            edits=edits[index],
            interpolated=layout.interpolated,
        )
        for index, sentence in enumerate(sentences)
    )
    paragraphs = tuple(
        Paragraph(
            index=index,
            start=sentences[first].start,
            end=max(s.end for s in sentences[first : last + 1]),
            first_segment=first,
            last_segment=last,
            speaker=sentences[first].speaker,
        )
        for index, (first, last) in enumerate(paragraph_ranges)
    )
    return ContextDocument(
        schema_version=SCHEMA_VERSION,
        generator=GeneratorInfo(name="ytverbatim", version=__version__),
        created_at=created_at,
        source=info.describe(source.stretches),
        transcript=Transcript(
            language=language,
            languages=(language,),
            script=script,
            runs=(run,),
            punctuation=_punctuation(text, language, run, llm_punctuated=llm_punctuated),
            correction_llm=correction_llm,
            diarization=source.diarization,
        ),
        speakers=_speakers(segments, speaker_names),
        chapters=chapter_list,
        paragraphs=paragraphs,
        segments=segments,
        quality=_quality(source, segments, notes),
        llm_usage=tuple(llm_usage),
    )


def _speakers(segments: Sequence[Segment], names: Mapping[str, str]) -> tuple[Speaker, ...]:
    """List the final segments' speakers under their ids; one whose text a later stage removed simply drops out."""
    talk = attribution.speech_seconds((s.start, s.end, s.speaker) for s in segments)
    return tuple(Speaker(id=speaker, name=names.get(speaker), speech_s=seconds) for speaker, seconds in talk.items())


def _layout(stretches: Sequence[Stretch], language: str) -> _Layout:
    zh = seg.uses_zh_rules(" ".join(stretch.text for stretch in stretches), language)
    tokens: list[seg.Token] = []
    origins: list[tuple[int, int]] = []
    interpolated: set[int] = set()
    for index, stretch in enumerate(stretches):
        text = stretch.text.strip()
        if not text:
            continue
        piece, complete = _retimed(stretch, text, index, language)
        if not complete:
            interpolated.add(index)
        if stretch.speaker is not None:
            piece = [replace(token, speaker=stretch.speaker) for token in piece]
        joined = _joined(tokens, piece, zh=zh)
        position = len(stretch.text) - len(stretch.text.lstrip()) - (len(joined[0].text) - len(piece[0].text))
        for token in joined:
            origins.append((index, position))
            position += len(token.text)
        tokens += joined
    tokens = _monotonic(tokens)
    kept = [(token, origin) for token, origin in zip(tokens, origins, strict=True) if token.text.strip()]
    sentences = seg.segment_tokens([token for token, _ in kept], language)
    placed: dict[int, list[_Placed]] = {}
    cursor = 0
    for number, sentence in enumerate(sentences):
        group = kept[cursor : cursor + len(sentence.tokens)]
        cursor += len(group)
        raw = "".join(token.text for token, _ in group)
        position = -(len(raw) - len(raw.lstrip()))
        for token, (stretch, start) in group:
            placed.setdefault(stretch, []).append(_Placed(stretch, start, start + len(token.text), number, position))
            position += len(token.text)
    return _Layout(sentences, placed, interpolated)


def _retimed(stretch: Stretch, text: str, index: int, language: str) -> tuple[list[seg.Token], bool]:
    """Tokens of the stretch's current text, timed by its words; False when some of it had to be interpolated."""
    if stretch.words:
        piece, complete = seg.aligned_tokens(
            text, stretch.words, stretch.start, stretch.end, language=language, origin=index
        )
        if "".join(token.text for token in piece) == text:
            return piece, complete
    piece = seg.pseudo_tokens(text, stretch.start, stretch.end, language=language, origin=index)
    return piece, not stretch.word_timed


def _joined(tokens: Sequence[seg.Token], piece: list[seg.Token], *, zh: bool) -> list[seg.Token]:
    # Stretch texts are stripped, so "end." and "Next" would otherwise fuse and hide the sentence break.
    if tokens and piece and not zh and not tokens[-1].text[-1:].isspace() and not piece[0].text[:1].isspace():
        return [replace(piece[0], text=f" {piece[0].text}"), *piece[1:]]
    return piece


def _placed_edits(layout: _Layout, stretch: int, edits: Sequence[Edit]) -> list[tuple[int, Edit]]:
    """Each edit as its changed word runs, positioned in the final text and attached to the sentence holding them."""
    placed: list[tuple[int, Edit]] = []
    for number, edit in enumerate(edits):
        later = [offsets.OffsetMap.between(e.before, e.after) for e in edits[number + 1 :]]
        for run in offsets.changed_runs(edit.before, edit.after):
            if not (run.before.strip() or run.after.strip()):
                continue
            position = run.start
            for mapping in later:
                position = mapping.start(position)
            sentence = layout.sentence_at(stretch, position)
            if sentence is not None:
                placed.append((sentence, Edit(stage=edit.stage, before=run.before, after=run.after, llm=edit.llm)))
    return placed


def _punctuation(
    text: str, language: str, run: TranscriptRun, *, llm_punctuated: bool
) -> Literal["engine", "llm", "caption", "none"]:
    if not seg.is_punctuated(text, language):
        return "none"
    if llm_punctuated:
        return "llm"
    return "caption" if run.kind != "asr" else "engine"


def _anomaly_flags(sentence: seg.Sentence, anomalies: Mapping[int, ChunkHypothesis]) -> tuple[Flag, ...]:
    flags: list[Flag] = []
    for chunk, row in anomalies.items():
        span = sentence.span_of(chunk)
        if span is not None:
            flags.append(
                Flag(
                    kind="decoder_anomaly",
                    char_start=span[0],
                    char_end=span[1],
                    note=f"the decoder looped or stalled in the chunk at {_clock(row.start)}-{_clock(row.end)}",
                )
            )
    return tuple(flags)


def _language(rows: Sequence[ChunkHypothesis], declared: str | None) -> str:
    """Return the language the engine reports for most of the text, else the one YouTube declares."""
    weight: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row.language and row.text.strip():
            weight[row.language] += len(row.text)
    if weight:
        return weight.most_common(1)[0][0]
    return declared or UNDETERMINED_LANGUAGE


def _tokens(rows: Sequence[ChunkHypothesis], language: str) -> tuple[list[seg.Token], list[str]]:
    """Return the tokens of every chunk, timed by its words where they line up with its text, and timing warnings.

    A chunk without any word timings is spread over the chunk, which the warning covers rather than a flag on every
    segment.
    """
    zh = seg.uses_zh_rules(" ".join(row.text for row in rows), language)
    tokens: list[seg.Token] = []
    chunk_level = 0
    for row in rows:
        text = row.text.strip()
        if not text:
            continue
        words = [word for word in row.words if word.text.strip()]
        if words:
            piece, _ = seg.aligned_tokens(text, words, row.start, row.end, language=language, origin=row.chunk)
        else:
            chunk_level += 1
            piece = seg.pseudo_tokens(text, row.start, row.end, language=language, origin=row.chunk)
        tokens += _joined(tokens, piece, zh=zh)
    warnings: list[str] = []
    if chunk_level:
        warnings.append(
            f"timing is chunk-level for {chunk_level} chunk(s): segment times are spread over each chunk "
            "in proportion to characters"
        )
    return _monotonic(tokens), warnings


def _monotonic(tokens: Sequence[seg.Token]) -> list[seg.Token]:
    """Segment starts must not go backwards, and engines occasionally emit a word that does.

    Word times are already clamped to their chunk, so a bogus late start can hold later ones back only to its
    chunk's end.
    """
    out: list[seg.Token] = []
    floor = 0.0
    for token in tokens:
        start = max(token.start, floor)
        out.append(replace(token, start=start, end=max(token.end, start)))
        floor = start
    return out


def _script(language: str, text: str) -> Literal["zh-Hans", "zh-Hant", "n/a"]:
    if not seg.is_zh(language):
        return "n/a"
    return "zh-Hans" if clean.to_simplified(text) == text else "zh-Hant"


def _chapters(
    source_chapters: Sequence[SourceChapter], sentences: Sequence[seg.Sentence], *, enabled: bool
) -> tuple[list[int | None], tuple[Chapter, ...]]:
    """Map sentences to YouTube chapters by start time, dropping chapters that no sentence starts in."""
    if not enabled or not source_chapters or not sentences:
        return [None] * len(sentences), ()
    youtube = sorted(source_chapters, key=lambda c: c.start)
    raw = seg.chapter_indexes([s.start for s in sentences], [c.start for c in youtube])
    order = list(dict.fromkeys(raw))
    compact = {old: new for new, old in enumerate(order)}
    chapters = tuple(
        Chapter(
            index=compact[old],
            title=youtube[old].title,
            start=youtube[old].start,
            end=youtube[old].end,
            source="youtube",
            first_segment=raw.index(old),
            last_segment=len(raw) - 1 - raw[::-1].index(old),
        )
        for old in order
    )
    return [compact[old] for old in raw], chapters


def _segment(
    index: int,
    sentence: seg.Sentence,
    run: TranscriptRun,
    *,
    paragraph: int,
    chapter: int | None,
    flags: Sequence[Flag],
    edits: Sequence[Edit],
    interpolated: Collection[int],
) -> Segment:
    words = tuple(
        Word(text=t.text.strip(), start=t.start, end=t.end, conf=t.conf)
        for t in sentence.tokens
        if t.timed and t.text.strip()
    )
    all_flags = list(flags)
    if any(not t.timed and t.origin in interpolated for t in sentence.tokens):
        all_flags.append(
            Flag(
                kind="timing_interpolated",
                char_start=0,
                char_end=len(sentence.text),
                note="some words here had no engine timing; their times are interpolated from the words around them",
            )
        )
    return Segment(
        id=index,
        speaker=sentence.speaker,
        start=sentence.start,
        end=sentence.end,
        text=sentence.text,
        words=words or None,
        source=SourceRef(kind=run.kind, run=run.id),
        edits=tuple(edits),
        flags=tuple(sorted(all_flags, key=lambda f: f.char_start)),
        paragraph=paragraph,
        chapter=chapter,
    )


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def youtube_source(
    metadata: VideoMetadata, tags: Sequence[str], access: Access, stretches: Sequence[Stretch]
) -> YouTubeSource:
    duration = metadata.duration if metadata.duration is not None else max((s.end for s in stretches), default=0.0)
    return YouTubeSource(
        kind="youtube",
        id=metadata.video_id,
        url=ids.video_url(metadata.video_id),
        title=metadata.title,
        channel=metadata.channel,
        channel_id=metadata.channel_id or "",
        upload_date=_upload_date(metadata.upload_date),
        duration_s=duration,
        description=metadata.description,
        language=metadata.language,
        tags=tuple(tags),
        creator_note=_creator_note(metadata.creator_note),
        access=access,
    )


def _creator_note(note: CreatorNote | None) -> ContractCreatorNote | None:
    return ContractCreatorNote.model_validate(asdict(note)) if note is not None else None


def _upload_date(raw: str | None) -> date | None:
    try:
        return datetime.strptime(raw, "%Y%m%d").date() if raw else None
    except ValueError:
        return None


def _vad_info() -> dict[str, JsonValue]:
    return {"repo": chunks.SILERO_REPO, "revision": chunks.SILERO_REVISION, **asdict(chunks.DEFAULT_VAD_PARAMS)}


def _chunking_info(max_chunk_s: float | None, rows: Sequence[ChunkHypothesis]) -> dict[str, JsonValue] | None:
    if max_chunk_s is None:
        return None
    return {
        "max_chunk_s": max_chunk_s,
        "tail_fraction": chunks.DEFAULT_TAIL_FRACTION,
        "min_pause_s": chunks.DEFAULT_MIN_PAUSE_S,
        "chunks": len(rows),
    }


def _quality(source: Transcribed, segments: Sequence[Segment], warnings: Sequence[str]) -> Quality:
    """Flag counts are the document's own flags by kind, plus the source's chunk diagnostics (forced cuts, ...)."""
    counts: collections.Counter[str] = collections.Counter(flag.kind for s in segments for flag in s.flags)
    for kind, count in source.chunk_flags.items():
        if kind != _ANOMALY_FLAG:
            counts[kind] += count
    transcribed = _covered_seconds([(s.start, s.end) for s in segments])
    return Quality(
        speech_seconds=source.speech_seconds if source.speech_seconds is not None else transcribed,
        transcribed_seconds=transcribed,
        decoder_anomalies=source.decoder_anomalies,
        flags_by_kind=dict(counts),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _covered_seconds(spans: Collection[tuple[float, float]]) -> float:
    total = 0.0
    reach = 0.0
    for start, end in sorted(spans):
        start = max(start, reach)
        if end > start:
            total += end - start
            reach = end
    return total
