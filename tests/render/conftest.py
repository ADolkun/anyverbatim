from collections.abc import Callable, Mapping, Sequence
from datetime import date

import pytest

from anyverbatim.contract import (
    Access,
    Chapter,
    ContextDocument,
    DiarizationRun,
    Edit,
    Flag,
    Generator,
    Paragraph,
    Quality,
    RecordingSource,
    Segment,
    SourceRef,
    Speaker,
    Transcript,
    TranscriptRun,
    YouTubeSource,
)

RUN = TranscriptRun(
    id="asr-1",
    kind="asr",
    engine="example-engine",
    model_repo=None,
    model_revision=None,
    params={},
    vad=None,
    chunking=None,
    aligner=None,
)
ACCESS = Access(
    auth_source="none",
    cookies_used=False,
    logged_in=None,
    premium=None,
    member=None,
    availability="public",
    ytdlp_version="2026.10.26",
    warnings=(),
)
DIARIZATION = DiarizationRun(engine="d-example", model_repo="example/diarizer", model_revision="0" * 40, params={})
AUDIO_SHA256 = "0123456789abcdef" + "0" * 48
YOUTUBE: dict[str, object] = {
    "kind": "youtube",
    "id": "AAAAAAAAAAA",
    "url": "https://www.youtube.com/watch?v=AAAAAAAAAAA",
    "title": "Example Title",
    "channel": "Example Channel",
    "channel_id": "UCexample",
    "upload_date": date(2026, 1, 1),
    "description": "",
    "tags": (),
    "access": ACCESS,
}
RECORDING: dict[str, object] = {
    "kind": "recording",
    "id": f"rec-{AUDIO_SHA256[:16]}",
    "audio_sha256": AUDIO_SHA256,
    "title": "Example Title",
    "file_name": "Example.m4a",
    "origin": "voice_memos",
    "recorded_at": None,
    "language": None,
}


def _seg(
    text: str,
    start: float,
    end: float,
    *,
    paragraph: int = 0,
    chapter: int | None = None,
    flags: Sequence[Flag] = (),
    edits: Sequence[Edit] = (),
    speaker: str | None = None,
) -> Segment:
    return Segment(
        id=0,
        start=start,
        end=end,
        text=text,
        source=SourceRef(kind="asr", run=RUN.id),
        edits=tuple(edits),
        flags=tuple(flags),
        paragraph=paragraph,
        chapter=chapter,
        speaker=speaker,
    )


def _flag(
    kind: str,
    text: str,
    span: str,
    *,
    note: str = "",
    occurrence: int = 0,
    readings: Sequence[tuple[str, str]] = (),
) -> Flag:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(span, start + 1)
    return Flag.model_validate(
        {
            "kind": kind,
            "char_start": start,
            "char_end": start + len(span),
            "note": note,
            "readings": [{"source": source, "text": reading} for source, reading in readings],
        }
    )


@pytest.fixture
def seg() -> Callable[..., Segment]:
    return _seg


@pytest.fixture
def flag() -> Callable[..., Flag]:
    return _flag


@pytest.fixture
def make_doc() -> Callable[..., ContextDocument]:
    def build(
        segments: Sequence[Segment],
        *,
        language: str = "en",
        duration_s: float = 600.0,
        chapters: Mapping[int, str] | None = None,
        run: TranscriptRun = RUN,
        punctuation: str = "engine",
        correction_llm: str | None = None,
        source: Mapping[str, object] | None = None,
        languages: Sequence[str] | None = None,
        speakers: Mapping[str, tuple[str | None, float]] | None = None,
    ) -> ContextDocument:
        """`source` overrides source fields (`kind="recording"` for a recording); `speakers` maps id → (name, talk)."""
        numbered = [s.model_copy(update={"id": index}) for index, s in enumerate(segments)]
        paragraphs = _ranges(numbered, lambda s: s.paragraph)
        chapter_ranges = _ranges(numbered, lambda s: s.chapter) if chapters else {}
        overrides = {"duration_s": duration_s, **(source or {})}
        described = (
            RecordingSource.model_validate({**RECORDING, **overrides})
            if overrides.get("kind") == "recording"
            else YouTubeSource.model_validate({**YOUTUBE, "language": language, **overrides})
        )
        return ContextDocument.model_validate(
            {
                "schema_version": "2.0",
                "generator": Generator(name="anyverbatim", version="0.0.1"),
                "created_at": "2026-01-02T03:04:05+00:00",
                "source": described,
                "transcript": Transcript.model_validate(
                    {
                        "language": language,
                        "languages": tuple(languages) if languages is not None else (language,),
                        "script": "zh-Hans" if language.startswith("zh") else "n/a",
                        "runs": (run,),
                        "punctuation": punctuation,
                        "correction_llm": correction_llm,
                        "diarization": DIARIZATION if speakers else None,
                    }
                ),
                "speakers": tuple(
                    Speaker(id=speaker, name=name, speech_s=seconds)
                    for speaker, (name, seconds) in (speakers or {}).items()
                ),
                "chapters": tuple(
                    Chapter(
                        index=index,
                        title=chapters[index] if chapters else "",
                        start=numbered[first].start,
                        end=numbered[last].end,
                        source="youtube",
                        first_segment=first,
                        last_segment=last,
                    )
                    for index, (first, last) in chapter_ranges.items()
                ),
                "paragraphs": tuple(
                    Paragraph(
                        index=index,
                        start=numbered[first].start,
                        end=numbered[last].end,
                        first_segment=first,
                        last_segment=last,
                        speaker=numbered[first].speaker,
                    )
                    for index, (first, last) in paragraphs.items()
                ),
                "segments": tuple(
                    s.model_copy(update={"source": SourceRef(kind=run.kind, run=run.id)}) for s in numbered
                ),
                "quality": Quality(
                    speech_seconds=0, transcribed_seconds=0, decoder_anomalies=0, flags_by_kind={}, warnings=()
                ),
                "llm_usage": (),
            }
        )

    return build


def _ranges(segments: Sequence[Segment], key: Callable[[Segment], int | None]) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    for segment in segments:
        index = key(segment)
        if index is None:
            continue
        first, _ = ranges.get(index, (segment.id, segment.id))
        ranges[index] = (first, segment.id)
    return ranges
