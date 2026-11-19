import struct
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from anyverbatim.acquire.creator_note import CreatorNote
from anyverbatim.acquire.metadata import Chapter, VideoMetadata
from anyverbatim.asr import registry
from anyverbatim.asr.models import AnomalySpan, ChunkHypothesis
from anyverbatim.asr.protocol import ChunkFlag
from anyverbatim.contract import Access, ContextDocument, Word, YouTubeSource, dump_document, load_document
from anyverbatim.diarize import registry as diarize_registry
from anyverbatim.diarize.protocol import Diarization
from anyverbatim.errors import ContractError
from anyverbatim.render import render_markdown
from anyverbatim.stages import Diarized, SourceInfo, Stretch, Transcribed, build_document, from_asr_rows, youtube_source

ACCESS = Access(
    auth_source="none",
    cookies_used=False,
    logged_in=None,
    premium=None,
    member=None,
    availability="public",
    ytdlp_version="2099.01.01",
    warnings=(),
)
NOW = datetime(2030, 1, 1, tzinfo=UTC)


def metadata(*, language: str | None = "en", chapters: Sequence[Chapter] = (), duration: float | None = 100.0):
    return VideoMetadata(
        video_id="synthVid001",
        title="Synthetic title",
        channel="Synthetic channel",
        channel_id="UCsynthetic",
        upload_date="20300101",
        duration=duration,
        description="",
        language=language,
        availability="public",
        live_status=None,
        chapters=tuple(chapters),
        subtitle_langs=(),
        auto_caption_lang=None,
    )


def words(pieces: Sequence[str], start: float = 0.0, step: float = 0.4, gaps: dict[int, float] | None = None):
    out: list[Word] = []
    clock = start
    for index, piece in enumerate(pieces):
        out.append(Word(text=piece, start=clock, end=clock + step))
        clock += step + (gaps or {}).get(index, 0.0)
    return tuple(out)


def row(
    chunk: int,
    text: str,
    start: float,
    end: float,
    *,
    hypothesis: str = "w-prod",
    language: str | None = "en",
    chunk_words: tuple[Word, ...] = (),
    flags: tuple[ChunkFlag, ...] = (),
    anomaly_spans: tuple[AnomalySpan, ...] = (),
) -> ChunkHypothesis:
    return ChunkHypothesis(
        hypothesis=hypothesis,
        file="synthVid001",
        chunk=chunk,
        start=start,
        end=end,
        text=text,
        language=language,
        words=chunk_words,
        flags=flags,
        anomaly_spans=anomaly_spans,
    )


def build(
    rows: Sequence[ChunkHypothesis],
    *,
    engine: str = "w-prod",
    chapters: bool = True,
    meta: VideoMetadata | None = None,
    diarized: Diarized | None = None,
) -> ContextDocument:
    meta = meta or metadata()
    return assemble(from_asr_rows(rows, registry.get(engine), meta.language, diarized), meta=meta, chapters=chapters)


def assemble(
    source: Transcribed,
    stretches: Sequence[Stretch] | None = None,
    *,
    meta: VideoMetadata | None = None,
    chapters: bool = True,
) -> ContextDocument:
    meta = meta or metadata()

    def describe(stretches: Sequence[Stretch]) -> YouTubeSource:
        return youtube_source(meta, ("synthetic",), ACCESS, stretches)

    return build_document(
        info=SourceInfo(meta.video_id, meta.chapters, describe),
        source=source,
        stretches=source.stretches if stretches is None else stretches,
        chapters=chapters,
        created_at=NOW,
    )


def diarized(turns: Sequence[tuple[float, float, int]], *, seconds: float = 20.0, channels: int = 8) -> Diarized:
    """A stored diarization in which each (start, end, channel) turn is active at probability 0.9."""
    frame_s = 0.01
    frames = round(seconds / frame_s)
    probs = [0.0] * (frames * channels)
    for start, end, channel in turns:
        for frame in range(round(start / frame_s), round(end / frame_s)):
            probs[frame * channels + channel] = 0.9
    data = b"".join(struct.pack("<e", p) for p in probs)
    diarization = Diarization(turns=(), threshold=0.5, frame_s=frame_s, frames=frames, channels=channels, probs=data)
    return Diarized(diarize_registry.get("d-nemotron-8bit"), diarization)


def test_whisper_words_give_word_level_segments_and_a_valid_document() -> None:
    pieces = [" The", " market", " rose", " today.", " Then", " it", " fell", " sharply."]
    doc = build([row(0, "".join(pieces), 0.0, 5.0, chunk_words=words(pieces, 0.5, 0.5))])

    assert [s.text for s in doc.segments] == ["The market rose today.", "Then it fell sharply."]
    first, second = doc.segments
    assert (first.start, first.end, second.start, second.end) == (0.5, 2.5, 2.5, 4.5)
    assert first.words is not None
    assert [w.text for w in first.words] == ["The", "market", "rose", "today."]
    assert first.words[0].start == 0.5
    assert first.words[-1].end == pytest.approx(2.5)
    assert doc.transcript.punctuation == "engine"
    assert doc.transcript.language == "en"
    assert doc.transcript.script == "n/a"
    assert doc.transcript.correction_llm is None
    assert doc.quality.warnings == ()
    assert doc.chapters == ()
    assert [s.chapter for s in doc.segments] == [None, None]
    again = load_document(dump_document(doc))
    assert again == doc
    assert "The market rose today." in render_markdown(doc)


def test_run_records_the_engine_and_a_whole_file_run_has_no_vad_or_chunking() -> None:
    doc = build([row(0, " Hi.", 0, 1, chunk_words=words([" Hi."]))])
    (run,) = doc.transcript.runs
    assert run.engine == "w-prod"
    assert run.kind == "asr"
    assert run.model_repo
    assert run.model_revision
    assert run.vad is None
    assert run.chunking is None
    assert run.params["mode"] == "whole_file"
    assert run.params["language_hint"] == "en"
    assert {s.source.run for s in doc.segments} == {run.id}


def test_a_chunked_engine_records_vad_and_chunking() -> None:
    rows = [row(0, "Hello.", 0, 4, hypothesis="q-base"), row(1, "World.", 5, 9, hypothesis="q-base")]
    doc = build(rows, engine="q-base")
    (run,) = doc.transcript.runs
    assert run.vad is not None
    assert run.vad["threshold"] == 0.5
    assert "revision" in run.vad
    assert run.chunking is not None
    assert run.chunking["max_chunk_s"] == 30.0
    assert run.chunking["chunks"] == 2
    assert doc.quality.speech_seconds == pytest.approx(8.0)


def test_engines_without_word_times_get_chunk_level_timing_and_a_warning() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta eta."
    doc = build([row(0, text, 10.0, 20.0, hypothesis="q-base")], engine="q-base")
    first, second = doc.segments
    assert first.words is None
    assert second.words is None
    assert first.start == 10.0
    assert second.end == pytest.approx(20.0)
    # 15 of the 35 non-space characters belong to the first sentence.
    assert first.end == pytest.approx(10.0 + 10.0 * 15 / 35, abs=1e-3)
    assert second.start == pytest.approx(first.end)
    assert len(doc.quality.warnings) == 1
    assert "chunk-level" in doc.quality.warnings[0]
    assert doc.segments[0].flags == ()


def test_words_missing_from_the_end_are_interpolated_up_to_the_chunk_end() -> None:
    doc = build([row(0, "Alpha beta gamma.", 0, 3, chunk_words=words([" Alpha", " beta"]))])
    (segment,) = doc.segments
    assert segment.text == "Alpha beta gamma."
    assert (segment.start, segment.end) == (0.0, 3.0)
    assert segment.words is not None
    assert [(w.text, w.start, w.end) for w in segment.words] == [
        ("Alpha", 0.0, 0.4),
        ("beta", 0.4, 0.8),
    ]
    (flag,) = segment.flags
    assert (flag.kind, flag.char_start, flag.char_end) == ("timing_interpolated", 0, len(segment.text))
    assert any("did not cover" in warning for warning in doc.quality.warnings)


def test_one_dropped_word_mid_file_keeps_every_other_time_exact() -> None:
    pieces = [" The", " market", " rose", " today.", " Then", " it", " fell", " sharply.", " Prices", " recovered."]
    ws = words(pieces, 1.0, 0.5)
    kept = ws[:5] + ws[6:]  # the engine's words lack " it"
    doc = build([row(0, "".join(pieces), 0.0, 100.0, chunk_words=kept)])

    assert [s.text for s in doc.segments] == ["The market rose today.", "Then it fell sharply.", "Prices recovered."]
    timed = {w.text: (w.start, w.end) for s in doc.segments for w in s.words or ()}
    assert timed == {w.text.strip(): (w.start, w.end) for w in kept}
    first, second, third = doc.segments
    assert first.flags == ()
    assert third.flags == ()
    assert [f.kind for f in second.flags] == ["timing_interpolated"]
    assert second.words is not None
    assert [w.text for w in second.words] == ["Then", "fell", "sharply."]
    assert (second.start, second.end) == (ws[4].start, ws[7].end)


def test_words_that_match_nothing_still_fall_back_to_chunk_level_timing() -> None:
    text = "Alpha beta gamma. Delta epsilon."
    doc = build([row(0, text, 10.0, 20.0, chunk_words=words([" Something", " else"], 10.0))])
    first, second = doc.segments
    assert first.words is None
    assert second.words is None
    assert (first.start, second.end) == (10.0, 20.0)
    assert first.end == pytest.approx(10.0 + 10.0 * 15 / 28, abs=1e-3)
    assert all([f.kind for f in s.flags] == ["timing_interpolated"] for s in doc.segments)


def test_word_times_are_clamped_to_their_chunk() -> None:
    first = (Word(text=" One", start=0.5, end=1.0), Word(text=" two.", start=999.0, end=999.5))
    second = words([" Three", " four."], 11.0, 0.5)
    doc = build(
        [
            row(0, " One two.", 0, 10, hypothesis="w-vad", chunk_words=first),
            row(1, " Three four.", 10, 20, hypothesis="w-vad", chunk_words=second),
        ],
        engine="w-vad",
    )
    one, three = doc.segments
    assert one.words is not None
    assert one.words[1].start == 10.0
    assert one.end == 10.0
    assert three.words is not None
    assert [(w.start, w.end) for w in three.words] == [(11.0, 11.5), (11.5, 12.0)]


def test_a_chapter_that_ends_before_it_starts_is_a_contract_error() -> None:
    pieces = [" One.", " Two."]
    with pytest.raises(ContractError, match="synthVid001"):
        build(
            [row(0, "".join(pieces), 0, 3, chunk_words=words(pieces))],
            meta=metadata(chapters=[Chapter("Backwards", 5.0, 1.0)]),
        )


def test_zh_tagged_english_joins_chunks_with_a_space_and_uses_english_rules() -> None:
    rows = [
        row(0, "The market rose today.", 0, 4, hypothesis="q-base", language="zh"),
        row(1, "Then it fell.", 4.2, 9, hypothesis="q-base", language="zh"),
    ]
    doc = build(rows, engine="q-base", meta=metadata(language="zh"))
    assert [s.text for s in doc.segments] == ["The market rose today.", "Then it fell."]
    assert doc.transcript.punctuation == "engine"


def test_mandarin_without_punctuation_is_cut_at_pauses_and_flagged_unpunctuated() -> None:
    pieces = list("今天我们聊聊财报") + list("营收增长很快") + list("估值也不低")
    text = "".join(pieces)
    doc = build(
        [row(0, text, 0, 12, language="zh", chunk_words=words(pieces, 0, 0.25, {7: 0.8, 13: 1.0}))],
        meta=metadata(language="zh"),
    )
    assert [s.text for s in doc.segments] == ["今天我们聊聊财报", "营收增长很快", "估值也不低"]
    assert doc.transcript.punctuation == "none"
    assert doc.transcript.language == "zh"
    assert doc.transcript.script == "zh-Hans"
    assert all(s.words is not None for s in doc.segments)


def test_traditional_characters_are_recorded_not_converted() -> None:
    doc = build([row(0, "這個軟體很好。", 0, 3, language="zh", hypothesis="q-base")], engine="q-base")
    assert doc.transcript.script == "zh-Hant"
    assert doc.segments[0].text == "這個軟體很好。"
    assert any("Traditional" in warning for warning in doc.quality.warnings)


def test_chunks_join_into_one_sentence_when_the_cut_fell_mid_sentence() -> None:
    rows = [
        row(0, "The market rose", 0, 4, hypothesis="q-base"),
        row(1, "sharply today. Then it fell.", 4.2, 9, hypothesis="q-base"),
    ]
    doc = build(rows, engine="q-base")
    assert [s.text for s in doc.segments] == ["The market rose sharply today.", "Then it fell."]


def test_decoder_anomaly_flags_the_segments_of_that_chunk_only() -> None:
    rows = [
        row(0, "First chunk here. It is fine.", 0, 6, hypothesis="q-base"),
        row(1, "Second chunk loops. Loops again.", 7, 13, hypothesis="q-base", flags=("decoder_anomaly", "forced_cut")),
        row(2, "Third chunk is fine.", 14, 18, hypothesis="q-base", flags=("token_cap_hit",)),
    ]
    doc = build(rows, engine="q-base")
    flagged = {s.text: [f.kind for f in s.flags] for s in doc.segments}
    assert flagged == {
        "First chunk here.": [],
        "It is fine.": [],
        "Second chunk loops.": ["decoder_anomaly"],
        "Loops again.": ["decoder_anomaly"],
        "Third chunk is fine.": [],
    }
    loops = doc.segments[2]
    assert (loops.flags[0].char_start, loops.flags[0].char_end) == (0, len(loops.text))
    assert "0:07-0:13" in loops.flags[0].note
    assert doc.quality.decoder_anomalies == 1
    # The document's flags by kind (two segments carry the anomaly), plus the chunk diagnostics.
    assert doc.quality.flags_by_kind == {"decoder_anomaly": 2, "forced_cut": 1, "token_cap_hit": 1}
    assert "decoder_anomaly" in render_markdown(doc)


FOUR_SENTENCES = [
    " Alpha",
    " beta",
    " gamma.",
    " Delta",
    " epsilon",
    " zeta.",
    " Eta",
    " theta",
    " iota.",
    " Kappa",
    " lambda",
    " mu.",
]


def looped_row(*spans: tuple[float, float], flags: tuple[ChunkFlag, ...] = ("redecoded", "decoder_anomaly")):
    """One whole-file row of four three-word sentences, one second a word, with the given unrepaired spans."""
    return row(
        0,
        "".join(FOUR_SENTENCES),
        0,
        12,
        chunk_words=words(FOUR_SENTENCES, 0.0, 1.0),
        flags=flags,
        anomaly_spans=tuple(AnomalySpan(start=a, end=b) for a, b in spans),
    )


def test_a_span_flags_only_the_sentences_it_covers() -> None:
    doc = build([looped_row((3.0, 6.0))])
    flagged = {s.text: [(s.text[f.char_start : f.char_end], f.note) for f in s.flags] for s in doc.segments}
    note = "the decoder looped or stalled at 0:03-0:06 and it could not be repaired"
    assert flagged == {
        "Alpha beta gamma.": [],
        "Delta epsilon zeta.": [("Delta epsilon zeta.", note)],
        "Eta theta iota.": [],
        "Kappa lambda mu.": [],
    }


def test_a_sentence_straddling_a_span_edge_is_flagged_on_its_in_span_words_only() -> None:
    doc = build([looped_row((4.0, 7.5))])
    flagged = {s.text: [s.text[f.char_start : f.char_end] for f in s.flags] for s in doc.segments}
    assert flagged == {
        "Alpha beta gamma.": [],
        "Delta epsilon zeta.": ["epsilon zeta."],
        "Eta theta iota.": ["Eta theta"],
        "Kappa lambda mu.": [],
    }


def test_every_span_of_a_row_is_flagged_with_its_own_note() -> None:
    doc = build([looped_row((0.0, 2.0), (9.0, 12.0))])
    tail = " and it could not be repaired"
    notes = [(s.text, f.note) for s in doc.segments for f in s.flags]
    assert notes == [
        ("Alpha beta gamma.", f"the decoder looped or stalled at 0:00-0:02{tail}"),
        ("Kappa lambda mu.", f"the decoder looped or stalled at 0:09-0:12{tail}"),
    ]


def test_touching_spans_flag_a_token_across_their_boundary_once_and_name_the_union() -> None:
    pieces = [" Alpha", " beta", " gamma.", " Delta", " epsilon", " zeta."]
    loop = row(
        0,
        "".join(pieces),
        0,
        12,
        chunk_words=words(pieces, 0.0, 2.0),
        flags=("decoder_anomaly",),
        anomaly_spans=(AnomalySpan(start=5.0, end=9.0), AnomalySpan(start=3.0, end=5.0)),
    )
    note = "the decoder looped or stalled at 0:03-0:09 and it could not be repaired"
    doc = build([loop])
    flagged = {s.text: [(s.text[f.char_start : f.char_end], f.note) for f in s.flags] for s in doc.segments}
    assert flagged == {
        "Alpha beta gamma.": [("beta gamma.", note)],
        "Delta epsilon zeta.": [("Delta epsilon", note)],
    }
    assert loop.anomaly_spans == (AnomalySpan(start=5.0, end=9.0), AnomalySpan(start=3.0, end=5.0))


def collapsed_row(pieces: Sequence[str], times: Sequence[tuple[float, float]], span: tuple[float, float]):
    ws = tuple(Word(text=p, start=a, end=b) for p, (a, b) in zip(pieces, times, strict=True))
    return row(
        0,
        "".join(pieces),
        0,
        40,
        chunk_words=ws,
        flags=("redecoded", "decoder_anomaly"),
        anomaly_spans=(AnomalySpan(start=span[0], end=span[1]),),
    )


def test_zero_length_words_on_either_edge_of_a_span_are_flagged() -> None:
    pieces = [" Alpha.", " again", " again", " Gamma", " delta.", " again", " again", " Epsilon."]
    times = [(1, 2), (10, 10), (10, 10), (12, 13), (13, 14), (20, 20), (20, 20), (25, 26)]
    doc = build([collapsed_row(pieces, times, (10.0, 20.0))])
    flagged = {s.text: [s.text[f.char_start : f.char_end] for f in s.flags] for s in doc.segments}
    assert flagged == {
        "Alpha.": [],
        "again again Gamma delta.": ["again again Gamma delta."],
        "again again Epsilon.": ["again again"],
    }


def test_a_loop_collapsed_onto_the_end_of_a_window_is_flagged_even_with_no_other_word_in_it() -> None:
    pieces = [" Alpha.", " again", " again", " again", " Epsilon."]
    times = [(1, 2), (20, 20), (20, 20), (20, 20), (25, 26)]
    doc = build([collapsed_row(pieces, times, (10.0, 20.0))])
    assert [(s.text, [s.text[f.char_start : f.char_end] for f in s.flags]) for s in doc.segments] == [
        ("Alpha.", []),
        ("again again again Epsilon.", ["again again again"]),
    ]
    assert doc.quality.flags_by_kind == {"decoder_anomaly": 1, "redecoded": 1}


def test_a_span_no_word_falls_in_flags_the_whole_row_so_the_anomaly_is_not_invisible() -> None:
    pieces = [" Alpha.", " Beta.", " Gamma."]
    doc = build([collapsed_row(pieces, [(1, 2), (2, 3), (25, 26)], (10.0, 20.0))])
    assert doc.quality.decoder_anomalies == 1
    assert [len(s.flags) for s in doc.segments] == [1, 1, 1]
    assert doc.segments[0].flags[0].note == "the decoder looped or stalled in the chunk at 0:00-0:40"
    assert doc.quality.flags_by_kind == {"decoder_anomaly": 3, "redecoded": 1}


def test_the_flag_counts_are_the_localized_flags_plus_the_rows_own_redecoded() -> None:
    doc = build([looped_row((3.0, 9.0))])
    assert doc.quality.decoder_anomalies == 1
    assert doc.quality.flags_by_kind == {"decoder_anomaly": 2, "redecoded": 1}


def test_a_row_without_spans_is_flagged_whole_with_the_chunk_note() -> None:
    doc = build([looped_row()])
    assert [len(s.flags) for s in doc.segments] == [1, 1, 1, 1]
    assert doc.segments[0].flags[0].note == "the decoder looped or stalled in the chunk at 0:00-0:12"
    assert doc.quality.flags_by_kind == {"decoder_anomaly": 4, "redecoded": 1}


def test_a_span_footnotes_only_the_words_inside_it() -> None:
    body = render_markdown(build([looped_row((3.0, 6.0))]))
    assert body.count("[^") == 2
    assert "Delta epsilon zeta.[^1]" in body
    assert "gamma.[^" not in body
    assert "iota.[^" not in body


def test_a_flag_covers_only_the_anomalous_chunks_part_of_a_shared_sentence() -> None:
    rows = [
        row(0, "The market rose", 0, 4, hypothesis="q-base"),
        row(1, "sharply today.", 4.2, 8, hypothesis="q-base", flags=("decoder_anomaly",)),
    ]
    (segment,) = build(rows, engine="q-base").segments
    (flag,) = segment.flags
    assert segment.text[flag.char_start : flag.char_end] == "sharply today."


def test_empty_chunks_are_counted_but_leave_no_segment() -> None:
    rows = [row(0, "", 0, 3), row(1, "Hello there.", 3, 6, hypothesis="q-base")]
    doc = build(rows, engine="q-base")
    assert [s.text for s in doc.segments] == ["Hello there."]
    assert doc.quality.flags_by_kind["empty_output"] == 1


def test_a_silent_video_still_yields_a_valid_empty_document() -> None:
    doc = build([row(0, "", 0, 60)], meta=metadata(chapters=[Chapter("One", 0, 60)]))
    assert doc.segments == ()
    assert doc.paragraphs == ()
    assert doc.chapters == ()
    assert "no speech" in doc.quality.warnings[0]
    assert load_document(dump_document(doc)) == doc
    assert render_markdown(doc).startswith("# Synthetic title")


def test_paragraphs_break_at_pauses_and_chapters_map_by_start_time() -> None:
    pieces = [" One.", " Two.", " Three.", " Four.", " Five."]
    ws = words(pieces, 0.0, 0.5, gaps={0: 0.1, 1: 2.5, 2: 0.1, 3: 0.1})
    chapters = [Chapter("Intro", 0.0, 3.0), Chapter("Silent", 3.0, 3.5), Chapter("Main", 3.5, 20.0)]
    doc = build([row(0, "".join(pieces), 0, 10, chunk_words=ws)], meta=metadata(chapters=chapters))

    assert [s.paragraph for s in doc.segments] == [0, 0, 1, 1, 1]
    assert [(c.title, c.first_segment, c.last_segment, c.source) for c in doc.chapters] == [
        ("Intro", 0, 1, "youtube"),
        ("Main", 2, 4, "youtube"),
    ]
    assert [c.index for c in doc.chapters] == [0, 1]
    assert [s.chapter for s in doc.segments] == [0, 0, 1, 1, 1]
    assert "## [00:00](https://youtu.be/synthVid001?t=0) Intro" in render_markdown(doc)


def test_a_paragraph_never_crosses_a_chapter_boundary() -> None:
    pieces = [" One.", " Two.", " Three.", " Four."]
    ws = words(pieces, 0.0, 0.5, gaps={0: 0.1, 1: 0.1, 2: 0.1})
    doc = build(
        [row(0, "".join(pieces), 0, 10, chunk_words=ws)],
        meta=metadata(chapters=[Chapter("A", 0.0, 1.0), Chapter("B", 1.0, 10.0)]),
    )
    assert [s.chapter for s in doc.segments] == [0, 0, 1, 1]
    assert [s.paragraph for s in doc.segments] == [0, 0, 1, 1]


def test_chapters_can_be_turned_off() -> None:
    pieces = [" One.", " Two."]
    doc = build(
        [row(0, "".join(pieces), 0, 3, chunk_words=words(pieces))],
        meta=metadata(chapters=[Chapter("A", 0.0, 3.0)]),
        chapters=False,
    )
    assert doc.chapters == ()
    assert [s.chapter for s in doc.segments] == [None, None]


def test_non_monotonic_word_starts_are_clamped_not_rejected() -> None:
    ws = (
        Word(text=" One", start=1.0, end=1.5),
        Word(text=" two.", start=0.9, end=1.4),
        Word(text=" Three.", start=2.0, end=2.4),
    )
    doc = build([row(0, " One two. Three.", 0, 3, chunk_words=ws)])
    assert [s.start for s in doc.segments] == [1.0, 2.0]
    assert doc.segments[0].words is not None
    assert doc.segments[0].words[1].start == 1.0


def test_language_comes_from_the_engine_else_the_declared_language_else_und() -> None:
    assert build([row(0, "Hi.", 0, 1, language="de", hypothesis="q-base")], engine="q-base").transcript.language == "de"
    declared = build([row(0, "Hi.", 0, 1, language=None, hypothesis="q-base")], engine="q-base", meta=metadata())
    assert declared.transcript.language == "en"
    unknown = build(
        [row(0, "Hi.", 0, 1, language=None, hypothesis="q-base")], engine="q-base", meta=metadata(language=None)
    )
    assert unknown.transcript.language == "und"


def test_video_and_quality_fields() -> None:
    pieces = [" Hello", " there."]
    doc = build([row(0, "".join(pieces), 0, 4, chunk_words=words(pieces, 1.0, 0.5))])
    video = doc.source
    assert isinstance(video, YouTubeSource)
    assert video.access == ACCESS
    assert (video.id, video.url, video.title, video.channel) == (
        "synthVid001",
        "https://www.youtube.com/watch?v=synthVid001",
        "Synthetic title",
        "Synthetic channel",
    )
    assert video.upload_date is not None
    assert video.upload_date.isoformat() == "2030-01-01"
    assert video.tags == ("synthetic",)
    assert video.duration_s == 100.0
    assert doc.quality.transcribed_seconds == pytest.approx(1.0)
    assert doc.quality.speech_seconds == doc.quality.transcribed_seconds
    assert doc.generator.name == "anyverbatim"
    assert doc.created_at == NOW


def test_an_unparseable_upload_date_becomes_none() -> None:
    meta = metadata()
    bad = VideoMetadata(**{**meta.__dict__, "upload_date": "soon", "duration": None})
    doc = build([row(0, "Hi.", 0, 7, hypothesis="q-base")], engine="q-base", meta=bad)
    assert isinstance(doc.source, YouTubeSource)
    assert doc.source.upload_date is None
    assert doc.source.duration_s == 7.0


def test_the_library_surface_is_exported_lazily() -> None:
    import anyverbatim

    assert anyverbatim.process.__module__ == "anyverbatim.pipeline"
    assert anyverbatim.ContextDocument is ContextDocument
    assert anyverbatim.Options.__module__ == "anyverbatim.config"
    assert anyverbatim.LLMClient.__module__ == "anyverbatim.llm.protocol"
    with pytest.raises(AttributeError):
        _ = anyverbatim.nonexistent  # type: ignore[attr-defined]
    assert {"process", "Options", "ContextDocument", "LLMClient", "__version__"} <= set(dir(anyverbatim))
    assert "process" in vars(anyverbatim), "a resolved name is cached in the module"


def test_importing_the_package_loads_neither_the_pipeline_nor_its_heavy_dependencies() -> None:
    probe = "import sys, anyverbatim; print(sorted({'anyverbatim.pipeline', 'yt_dlp', 'pydantic'} & set(sys.modules)))"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_correction_context_caps_the_glossary_at_twenty_terms_and_carries_the_creator_note() -> None:
    from anyverbatim.pipeline import correction_context

    note = CreatorNote(text="市盈率 not 适应率", author="Example Channel", like_count=3, published="20300101")
    tags = tuple(f"term{i}" for i in range(25))
    context = correction_context(replace(metadata(), creator_note=note), tags)
    assert context.glossary == tags[:20]
    assert context.creator_note == "市盈率 not 适应率"
    bare = correction_context(metadata(), ())
    assert (bare.glossary, bare.creator_note) == ((), None)


def test_configured_glossary_terms_come_before_the_tags_within_the_cap() -> None:
    from anyverbatim.pipeline import correction_context

    tags = tuple(f"tag{n}" for n in range(25))
    context = correction_context(metadata(), ("tag3", *tags), ("Acme", "tag3"))
    assert context.glossary[:4] == ("Acme", "tag3", "tag0", "tag1")
    assert len(context.glossary) == 20
    assert correction_context(metadata(), tags, ()).glossary == tags[:20]


def test_info_tags_keeps_only_string_tags_of_a_raw_extraction() -> None:
    from anyverbatim.pipeline import info_tags

    assert info_tags({"tags": ["GAAP", 3, "市盈率", None]}) == ("GAAP", "市盈率")
    assert info_tags({"tags": "GAAP"}) == ()
    assert info_tags({}) == ()
    assert info_tags(["tags"]) == ()


THREE_VOICES = (" we", " met", " today", " so", " and", " we", " left")


def three_voices() -> tuple[list[ChunkHypothesis], Diarized]:
    # Words 0-2 (0-1.2 s), 3 (1.2-1.6 s) and 4-6 (1.6-2.8 s) are three voices, with no pause and no punctuation.
    spoken = words(THREE_VOICES)
    rows = [row(0, "".join(w.text for w in spoken), 0.0, 3.0, chunk_words=spoken)]
    return rows, diarized([(0.0, 1.2, 0), (1.2, 1.6, 5), (1.6, 2.8, 2)])


def test_a_diarized_transcript_breaks_sentences_and_paragraphs_at_every_speaker_change() -> None:
    rows, voices = three_voices()
    doc = build(rows, diarized=voices)
    assert [(s.text, s.speaker) for s in doc.segments] == [("we met today", "S1"), ("so", "S2"), ("and we left", "S3")]
    assert [(p.first_segment, p.speaker) for p in doc.paragraphs] == [(0, "S1"), (1, "S2"), (2, "S3")]
    assert [s.id for s in doc.speakers] == ["S1", "S2", "S3"]
    run = doc.transcript.diarization
    assert run is not None
    assert run.engine == "d-nemotron-8bit"
    assert run.params["threshold"] == 0.5


def test_a_text_stage_emptying_a_speaker_drops_it_without_renumbering_the_others() -> None:
    rows, voices = three_voices()
    source = from_asr_rows(rows, registry.get("w-prod"), "en", voices)
    emptied = [s.model_copy(update={"text": ""}) if s.speaker == "S2" else s for s in source.stretches]
    doc = assemble(source, emptied)
    assert [s.id for s in doc.speakers] == ["S1", "S3"]
    assert [(s.text, s.speaker) for s in doc.segments] == [("we met today", "S1"), ("and we left", "S3")]
    assert [p.speaker for p in doc.paragraphs] == ["S1", "S3"]
    assert "**Speaker 3:** and we left" in render_markdown(doc)


def test_sentences_are_re_cut_only_within_one_speakers_stretches() -> None:
    rows, voices = three_voices()
    source = from_asr_rows(rows, registry.get("w-prod"), "en", voices)
    # Punctuation that would join the first two stretches into one sentence if they were one speaker's.
    texts = {"S1": "We met today,", "S2": "so.", "S3": "And we left."}
    stretches = [s.model_copy(update={"text": texts[s.speaker or ""]}) for s in source.stretches]
    doc = assemble(source, stretches)
    assert [(s.text, s.speaker) for s in doc.segments] == [
        ("We met today,", "S1"),
        ("so.", "S2"),
        ("And we left.", "S3"),
    ]


def test_one_voice_is_one_listed_speaker_and_the_markdown_has_no_labels() -> None:
    spoken = words(THREE_VOICES)
    rows = [row(0, "".join(w.text for w in spoken), 0.0, 3.0, chunk_words=spoken)]
    doc = build(rows, diarized=diarized([(0.0, 2.8, 4)]))
    assert [s.id for s in doc.speakers] == ["S1"]
    assert {s.speaker for s in doc.segments} == {"S1"}
    assert "Speaker" not in render_markdown(doc)


def test_a_diarizer_that_hears_nobody_leaves_the_transcript_unlabelled_with_a_warning() -> None:
    spoken = words(THREE_VOICES)
    rows = [row(0, "".join(w.text for w in spoken), 0.0, 3.0, chunk_words=spoken)]
    source = from_asr_rows(rows, registry.get("w-prod"), "en", diarized([]))
    assert source.diarization is None
    assert source.assignment is None
    doc = assemble(source)
    assert doc.speakers == ()
    assert doc.transcript.diarization is None
    assert any("heard no speaker" in warning for warning in doc.quality.warnings)


def test_undiarized_stretches_and_transcripts_dump_exactly_as_before() -> None:
    spoken = words(THREE_VOICES)
    source = from_asr_rows(
        [row(0, "".join(w.text for w in spoken), 0.0, 3.0, chunk_words=spoken)], registry.get("w-prod"), "en"
    )
    dumped = source.model_dump(mode="json")
    assert "diarization" not in dumped
    assert "assignment" not in dumped
    assert all("speaker" not in s and "language" not in s for s in dumped["stretches"])

    rows, voices = three_voices()
    attributed = from_asr_rows(rows, registry.get("w-prod"), "en", voices).model_dump(mode="json")
    assert [s["speaker"] for s in attributed["stretches"]] == ["S1", "S2", "S3"]
    assert isinstance(attributed["assignment"], str)
    assert attributed["diarization"]["engine"] == "d-nemotron-8bit"


def test_a_long_turn_repeats_its_label_on_each_paragraph_and_a_backchannel_is_its_own_line() -> None:
    first = [f" {n}." for n in ("one", "two", "three", "four", "five", "six", "seven", "eight")]
    pieces = [*first, " yeah", " sure.", " nine."]
    spoken = words(pieces)
    rows = [row(0, "".join(w.text for w in spoken), 0.0, 5.0, chunk_words=spoken)]
    # Eight sentences of one voice, a two-word backchannel from another, then the first voice again.
    doc = build(rows, diarized=diarized([(0.0, 3.2, 0), (3.2, 4.0, 1), (4.0, 4.4, 0)]), chapters=False)
    assert [(p.speaker, p.first_segment, p.last_segment) for p in doc.paragraphs] == [
        ("S1", 0, 5),
        ("S1", 6, 7),
        ("S2", 8, 8),
        ("S1", 9, 9),
    ]
    lines = [line for line in render_markdown(doc).splitlines() if line.startswith("[")]
    assert [line.split("** ", 1)[0].split(" ", 1)[1] + "**" for line in lines] == [
        "**Speaker 1:**",
        "**Speaker 1:**",
        "**Speaker 2:**",
        "**Speaker 1:**",
    ]
    assert lines[2].endswith("**Speaker 2:** yeah sure.")
