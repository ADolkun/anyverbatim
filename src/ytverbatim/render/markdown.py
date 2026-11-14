import math
import re
from collections.abc import Sequence
from datetime import datetime
from itertools import groupby

from ytverbatim.contract import ContextDocument, Flag, RecordingSource, Segment, YouTubeSource
from ytverbatim.text import segment as seg

_HOUR = 3600
_WATCH_URL = "https://youtu.be/"

# `_` only emphasises at a word edge; escaping it inside identifiers would just add noise. `&` is escaped only
# where it would start an HTML entity, so "AT&T" and "R&D" stay untouched. Every `~` is escaped, because GFM
# strikes through text between single tildes as well as double ones.
_ESCAPE = re.compile(
    r"[\\`*\[\]<>~]"
    r"|(?<![^\W_])_"
    r"|_(?![^\W_])"
    r"|&(?=#\d+;|#[xX][0-9a-fA-F]+;|[A-Za-z][A-Za-z0-9]*;)"
)

# The edit stage that records the replaced text for flags whose footnote shows a "was" value.
_BEFORE_STAGE = {"entity_corrected": "correction", "numeric_fixed": "numeric"}
# Timing and overlap provenance for programs; they say nothing a reader of the text can act on.
_JSON_ONLY = frozenset({"timing_interpolated", "overlapping_speech"})
_ORIGIN = {"voice_memos": "Voice Memos"}
# Ideographs, CJK symbols and punctuation, and fullwidth forms: a boundary next to one takes no space.
_CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
# What a segment may end with after its last word (closers, and General Punctuation such as … — ” ’) or start with
# before its first (openers); the boundary is judged by the characters beyond them.
_TRAILING = re.compile(r"[\u2000-\u206f\"')）】」』》〕\]]+$")
_LEADING = re.compile(r"^[\u2000-\u206f\"'(（「『《【〔\[]+")


def render_markdown(doc: ContextDocument) -> str:
    return _Renderer(doc).render()


class _Renderer:
    def __init__(self, doc: ContextDocument) -> None:
        self._doc = doc
        self._hours = doc.source.duration_s >= _HOUR
        self._footnotes: list[str] = []
        self._names = {speaker.id: speaker.label for speaker in doc.speakers}
        # Ids that share a name are one person, so a document whose speakers all share one name needs no labels.
        self._labelled = len(set(self._names.values())) > 1

    def render(self) -> str:
        # The body is rendered first: the provenance line reports how many flags it actually marked.
        body = self._body()
        blocks = [*self._header(), *body]
        if self._footnotes:
            blocks.append("\n".join(f"[^{number}]: {text}" for number, text in enumerate(self._footnotes, start=1)))
        return "\n\n".join(blocks) + "\n"

    def _header(self) -> list[str]:
        source = self._doc.source
        if isinstance(source, YouTubeSource):
            facts = [_escape(source.channel)]
            if source.upload_date is not None:
                facts.append(source.upload_date.isoformat())
            facts += [self._clock(source.duration_s), f"{_WATCH_URL}{source.id}"]
        else:
            facts = self._recording_facts(source)
        title = source.title if isinstance(source, YouTubeSource) else source.title.strip() or source.file_name
        return [f"# {_escape(title)}", " · ".join(facts), self._provenance()]

    def _recording_facts(self, source: RecordingSource) -> list[str]:
        facts: list[str] = []
        if source.recorded_at is not None:
            facts.append(f"Recorded {_recorded(source.recorded_at)}")
        facts += [self._clock(source.duration_s), _escape(_ORIGIN.get(source.origin, source.file_name))]
        if self._labelled:
            facts.append(f"Speakers: {self._shares()}")
        return facts

    def _shares(self) -> str:
        talk: dict[str, float] = {}
        for speaker in self._doc.speakers:
            name = self._names[speaker.id]
            talk[name] = talk.get(name, 0.0) + speaker.speech_s
        total = sum(talk.values())
        return " · ".join(
            f"{_escape(name)} ({round(100 * seconds / total) if total else 0}%)" for name, seconds in talk.items()
        )

    def _provenance(self) -> str:
        transcript = self._doc.transcript
        flags = len(self._footnotes)
        parts = [f"Transcript: {_escape(self._source_label())}"]
        if isinstance(self._doc.source, RecordingSource) and transcript.languages:
            parts.append(f"languages: {', '.join(transcript.languages)}")
        parts += [
            f"punctuation: {transcript.punctuation}",
            f"corrected by: {_escape(transcript.correction_llm or 'none')}",
            f"{flags} flag" if flags == 1 else f"{flags} flags",
        ]
        return " · ".join(parts)

    def _source_label(self) -> str:
        runs = {run.id: run for run in self._doc.transcript.runs}
        used = {segment.source.run for segment in self._doc.segments}
        labels: list[str] = []
        for run in runs.values():
            if used and run.id not in used:
                continue
            label = {"caption_manual": "manual captions", "caption_auto": "auto captions"}.get(
                run.kind, run.engine or run.id
            )
            if label not in labels:
                labels.append(label)
        return " + ".join(labels)

    def _body(self) -> list[str]:
        doc = self._doc
        blocks: list[str] = []
        if not doc.chapters:
            blocks.append("## Transcript")
        current_chapter: int | None = None
        for chapter_index, group in groupby(self._blocks(), key=lambda segments: segments[0].chapter):
            if chapter_index is not None and chapter_index != current_chapter:
                chapter = doc.chapters[chapter_index]
                blocks.append(f"## {self._link(chapter.start)} {_escape(chapter.title)}")
                current_chapter = chapter_index
            blocks += [self._paragraph(segments) for segments in group]
        return blocks

    def _blocks(self) -> list[list[Segment]]:
        """Group the segments into rendered paragraphs: one per paragraph and chapter, one person's turns joined.

        Paragraphs of two ids that share a name are one person's, so they join where the paragraph splitter itself
        would not have broken them: no chapter start, a short pause, and the joined block within its size limits.
        """
        blocks: list[list[Segment]] = []
        for _, group in groupby(self._doc.segments, key=lambda s: (s.paragraph, s.chapter)):
            segments = list(group)
            if blocks and self._same_person(blocks[-1], segments):
                blocks[-1] += segments
            else:
                blocks.append(segments)
        return blocks

    def _same_person(self, block: list[Segment], following: list[Segment]) -> bool:
        last, first = block[-1], following[0]
        before, after = last.speaker, first.speaker
        if before is None or after is None or before == after or self._names[before] != self._names[after]:
            return False
        return (
            last.chapter == first.chapter
            and first.start - last.end < seg.PARAGRAPH_PAUSE_S
            and len(block) + len(following) <= seg.PARAGRAPH_MAX_SENTENCES
            and following[-1].end - block[0].start < seg.PARAGRAPH_MAX_S
        )

    def _paragraph(self, segments: list[Segment]) -> str:
        first = segments[0]
        paragraph = self._doc.paragraphs[first.paragraph]
        start = paragraph.start if first.id == paragraph.first_segment else first.start
        head = self._link(start)
        if self._labelled and paragraph.speaker is not None:
            head += f" **{_escape(self._names[paragraph.speaker])}:**"
        return f"{head} {_joined(segments, [self._segment(segment) for segment in segments])}"

    def _segment(self, segment: Segment) -> str:
        text = segment.text.replace("\r", " ").replace("\n", " ")
        escaped = _escape_positions(text)
        markers: dict[int, list[str]] = {}
        for flag in sorted(_visible_flags(segment), key=lambda f: (f.char_end, f.char_start, f.kind, f.note)):
            self._footnotes.append(self._footnote(segment, flag))
            markers.setdefault(flag.char_end, []).append(f"[^{len(self._footnotes)}]")
        out: list[str] = []
        for offset, char in enumerate(text):
            out += markers.get(offset, [])
            out.append(f"\\{char}" if offset in escaped else char)
        out += markers.get(len(text), [])
        return "".join(out).strip()

    def _footnote(self, segment: Segment, flag: Flag) -> str:
        head = f"{self._clock(segment.start)} — {flag.kind}"
        parts = [f"{head}: {_escape(flag.note)}" if flag.note else head]
        before = _before_text(segment, flag)
        if before is not None:
            parts.append(f"was {_escape(before)}")
        if flag.readings:
            heard = " / ".join(f"{_escape(r.text)} ({_escape(r.source)})" for r in flag.readings)
            parts.append(f"heard {heard}")
        return "; ".join(parts)

    def _link(self, seconds: float) -> str:
        source = self._doc.source
        if isinstance(source, YouTubeSource):
            return f"[{self._clock(seconds)}]({_WATCH_URL}{source.id}?t={_whole(seconds)})"
        return f"[{self._clock(seconds)}]"

    def _clock(self, seconds: float) -> str:
        total = _whole(seconds)
        minutes, secs = divmod(total, 60)
        if self._hours:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def _whole(seconds: float) -> int:
    return math.floor(seconds)


def _recorded(moment: datetime) -> str:
    offset = moment.strftime("%z")
    return f"{moment:%Y-%m-%d %H:%M} {offset[:3]}:{offset[3:5]}"


def _joined(segments: Sequence[Segment], rendered: Sequence[str]) -> str:
    """Join rendered segments with one space where Latin script meets Latin script, and with none at CJK."""
    out = ""
    left = ""
    for segment, text in zip(segments, rendered, strict=True):
        if not text:
            continue
        stripped = segment.text.strip()
        if out:
            right = _LEADING.sub("", stripped)[:1]
            out += "" if _CJK.match(left) or _CJK.match(right) else " "
        out += text
        left = _TRAILING.sub("", stripped)[-1:]
    return out


def _escape_positions(text: str) -> set[int]:
    return {match.start() for match in _ESCAPE.finditer(text)}


def _escape(text: str) -> str:
    flat = text.replace("\r", " ").replace("\n", " ")
    escaped = _escape_positions(flat)
    return "".join(f"\\{char}" if offset in escaped else char for offset, char in enumerate(flat))


def _visible_flags(segment: Segment) -> list[Flag]:
    low_confidence = [flag for flag in segment.flags if flag.kind == "low_confidence"]
    return [
        flag
        for flag in segment.flags
        if flag.kind not in _JSON_ONLY
        and (
            flag.kind != "numeric_single_source"
            or any(flag.char_start < low.char_end and low.char_start < flag.char_end for low in low_confidence)
        )
    ]


def _before_text(segment: Segment, flag: Flag) -> str | None:
    stage = _BEFORE_STAGE.get(flag.kind)
    if stage is None:
        return None
    span = segment.text[flag.char_start : flag.char_end]
    # Edits carry no offsets, and a changed run can be narrower than the flagged span (十五→五十 inside 五十个百分点),
    # so an edit whose text lies inside the span, or contains it, counts; an exact match wins over a partial one.
    edits = [edit for edit in segment.edits if edit.stage == stage and edit.before]
    exact = [edit.before for edit in edits if edit.after == span]
    overlapping = [edit.before for edit in edits if edit.after and (edit.after in span or span in edit.after)]
    matches = exact or overlapping
    return matches[-1] if matches else None
