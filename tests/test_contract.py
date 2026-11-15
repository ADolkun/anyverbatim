import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from ytverbatim.auth import chromium
from ytverbatim.contract import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    ContextDocument,
    CreatorNote,
    dump_document,
    export_schema,
    load_document,
)
from ytverbatim.contract.models import Access, RecordingSource, YouTubeSource
from ytverbatim.errors import ContractError

Raw = dict[str, Any]

FIXTURES = Path(__file__).parent / "fixtures"
V1_0_FIXTURE = FIXTURES / "contract" / "context_v1_0.json"
# The pipeline fixture's document as 1.1 wrote it; the committed 2.0 one is the same document upgraded.
V1_1_FIXTURE = FIXTURES / "contract" / "context_v1_1.json"
V2_OF_V1_1 = FIXTURES / "pipeline" / "fxMixZhEn01" / "context.json"
AUDIO_SHA256 = "0123456789abcdef" + "0" * 48
NOTE = {"text": "Correction: the figure is 12%.", "author": "@example", "like_count": None, "published": "1 day ago"}

RUN = {
    "id": "asr-1",
    "kind": "asr",
    "engine": "example-engine",
    "model_repo": "example/model",
    "model_revision": None,
    "params": {"temperature": [0.0, 0.2], "beam": 1, "alpha": {"z": 1, "a": 2}},
    "vad": None,
    "chunking": {"max_s": 30},
    "aligner": None,
}


def _segment(
    index: int, text: str, start: float, end: float, *, paragraph: int, chapter: int | None, speaker: str | None = None
) -> Raw:
    return {
        "id": index,
        "start": start,
        "end": end,
        "text": text,
        "words": None,
        "source": {"kind": "asr", "run": "asr-1"},
        "edits": [],
        "flags": [],
        "paragraph": paragraph,
        "chapter": chapter,
        "speaker": speaker,
    }


def make_raw() -> Raw:
    segments = [
        _segment(0, "大家好，欢迎收看。", 0.0, 2.5, paragraph=0, chapter=0),
        _segment(1, "Today we cover NVDA at 150 dollars.", 2.5, 6.0, paragraph=0, chapter=0),
        _segment(2, "好的，我们开始。", 6.0, 8.2, paragraph=1, chapter=1),
    ]
    segments[1]["words"] = [
        {"text": "Today", "start": 2.5, "end": 2.9, "conf": 0.98},
        {"text": "we", "start": 2.9, "end": 3.0, "conf": None},
    ]
    segments[1]["edits"] = [{"stage": "correction", "before": "Envidia", "after": "NVDA", "llm": "example-llm"}]
    segments[1]["flags"] = [
        {
            "kind": "numeric_disputed",
            "char_start": 21,
            "char_end": 24,
            "note": "engines disagree",
            "readings": [
                {"source": "asr:example-engine", "text": "150"},
                {"source": "caption_auto", "text": "115"},
            ],
        }
    ]
    return {
        "schema_version": "2.0",
        "generator": {"name": "ytverbatim", "version": "0.0.1"},
        "created_at": "2026-01-02T03:04:05+00:00",
        "source": {
            "kind": "youtube",
            "id": "AAAAAAAAAAA",
            "url": "https://www.youtube.com/watch?v=AAAAAAAAAAA",
            "title": "Example Title",
            "channel": "Example Channel",
            "channel_id": "UCexample",
            "upload_date": "2026-01-01",
            "duration_s": 8.2,
            "description": "An example video.",
            "language": "zh",
            "tags": ["example"],
            "creator_note": None,
            "access": {
                "auth_source": "none",
                "cookies_used": False,
                "logged_in": None,
                "premium": None,
                "member": None,
                "availability": "public",
                "ytdlp_version": "2026.10.26",
                "warnings": ["no JS runtime found"],
            },
        },
        "transcript": {
            "language": "zh",
            "languages": ["zh", "en"],
            "script": "zh-Hans",
            "runs": [RUN],
            "punctuation": "engine",
            "correction_llm": None,
            "diarization": None,
        },
        "speakers": [],
        "chapters": [
            {
                "index": 0,
                "title": "Intro",
                "start": 0.0,
                "end": 6.0,
                "source": "youtube",
                "first_segment": 0,
                "last_segment": 1,
            },
            {
                "index": 1,
                "title": "Main",
                "start": 6.0,
                "end": 8.2,
                "source": "youtube",
                "first_segment": 2,
                "last_segment": 2,
            },
        ],
        "paragraphs": [
            {"index": 0, "start": 0.0, "end": 6.0, "first_segment": 0, "last_segment": 1},
            {"index": 1, "start": 6.0, "end": 8.2, "first_segment": 2, "last_segment": 2},
        ],
        "segments": segments,
        "quality": {
            "speech_seconds": 7.5,
            "transcribed_seconds": 8.2,
            "decoder_anomalies": 0,
            "flags_by_kind": {"numeric_disputed": 1},
            "warnings": [],
        },
        "llm_usage": [
            {"stage": "correction", "client": "example", "calls": 1, "input_tokens": 100, "output_tokens": None}
        ],
    }


def make_recording() -> Raw:
    """A two-speaker recording: S1 speaks segments 0 and 2, S2 segment 1, one paragraph each."""
    raw = make_raw()
    raw["source"] = {
        "kind": "recording",
        "id": f"rec-{AUDIO_SHA256[:16]}",
        "audio_sha256": AUDIO_SHA256,
        "title": "Weekly sync",
        "file_name": "Weekly sync.m4a",
        "origin": "voice_memos",
        "recorded_at": "2026-11-03T14:05:00+08:00",
        "duration_s": 8.2,
        "language": None,
    }
    raw["chapters"] = []
    raw["transcript"]["diarization"] = {
        "engine": "d-example",
        "model_repo": "example/diarizer",
        "model_revision": "0" * 40,
        "params": {"smoothing": {"min_words": 2, "min_s": 0.5}},
    }
    raw["speakers"] = [
        {"id": "S1", "name": None, "speech_s": 4.7},
        {"id": "S2", "name": "Bob", "speech_s": 3.5},
    ]
    raw["paragraphs"] = [
        {"index": 0, "start": 0.0, "end": 2.5, "first_segment": 0, "last_segment": 0, "speaker": "S1"},
        {"index": 1, "start": 2.5, "end": 6.0, "first_segment": 1, "last_segment": 1, "speaker": "S2"},
        {"index": 2, "start": 6.0, "end": 8.2, "first_segment": 2, "last_segment": 2, "speaker": "S1"},
    ]
    for index, (segment, speaker) in enumerate(zip(raw["segments"], ("S1", "S2", "S1"), strict=True)):
        segment.update(paragraph=index, chapter=None, speaker=speaker)
    return raw


def test_checked_in_schema_matches_export() -> None:
    checked_in = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert checked_in == export_schema(), (
        "context.v2.schema.json is stale; regenerate it with: uv run python -c "
        "'import json; from ytverbatim.contract import SCHEMA_PATH, export_schema; "
        'SCHEMA_PATH.write_text(json.dumps(export_schema(), indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")\''
    )


def test_round_trip_is_stable() -> None:
    doc = ContextDocument.model_validate(make_raw())
    text = dump_document(doc)
    assert text.endswith("}\n")
    assert "大家好，欢迎收看。" in text
    reloaded = load_document(text)
    assert reloaded == doc
    assert dump_document(reloaded) == text


def test_a_recording_round_trips() -> None:
    doc = ContextDocument.model_validate(make_recording())
    assert isinstance(doc.source, RecordingSource) and doc.source.recorded_at is not None
    assert [s.speaker for s in doc.segments] == ["S1", "S2", "S1"]
    text = dump_document(doc)
    assert load_document(text) == doc
    assert dump_document(load_document(text)) == text


def test_round_trip_through_a_file(tmp_path: Path) -> None:
    doc = ContextDocument.model_validate(make_raw())
    path = tmp_path / "context.json"
    path.write_text(dump_document(doc), encoding="utf-8")
    assert load_document(path) == doc


def test_dump_key_order_is_stable() -> None:
    raw = make_raw()
    doc = ContextDocument.model_validate(raw)
    text = dump_document(doc)
    assert list(json.loads(text))[:3] == ["schema_version", "generator", "created_at"]
    run = json.loads(text)["transcript"]["runs"][0]
    assert list(run["params"]) == ["alpha", "beam", "temperature"]
    assert list(run["params"]["alpha"]) == ["a", "z"]

    shuffled = deepcopy(raw)
    shuffled["transcript"]["runs"][0]["params"] = dict(reversed(list(RUN["params"].items())))
    assert dump_document(ContextDocument.model_validate(shuffled)) == text


def test_times_round_to_milliseconds() -> None:
    raw = make_raw()
    raw["segments"][0]["end"] = 2.50049
    raw["segments"][1]["start"] = 2.5004
    doc = ContextDocument.model_validate(raw)
    assert doc.segments[0].end == 2.5
    assert doc.segments[1].start == 2.5


def test_words_may_be_absent_without_a_flag() -> None:
    doc = ContextDocument.model_validate(make_raw())
    assert doc.segments[0].words is None
    assert doc.segments[0].flags == ()


def test_no_chapters_is_valid() -> None:
    raw = make_raw()
    raw["chapters"] = []
    for segment in raw["segments"]:
        segment["chapter"] = None
    assert ContextDocument.model_validate(raw).chapters == ()


def test_empty_document_is_valid() -> None:
    raw = make_raw()
    raw.update(chapters=[], paragraphs=[], segments=[])
    assert ContextDocument.model_validate(raw).segments == ()


@pytest.mark.parametrize(
    ("start", "end"),
    [(3.0, 4.0), (3.0, 7.0), (2.5, 6.0)],
    ids=["nested", "overlapping", "identical"],
)
def test_overlapping_and_nested_cues_are_valid(start: float, end: float) -> None:
    raw = make_raw()
    raw["segments"][2]["start"] = start
    raw["segments"][2]["end"] = end
    doc = ContextDocument.model_validate(raw)
    assert (doc.segments[2].start, doc.segments[2].end) == (start, end)


def test_every_supported_auth_source_is_recordable() -> None:
    recordable = set(get_args(Access.model_fields["auth_source"].annotation))
    assert recordable == {"none", "cookie-file", *chromium.SUPPORTED_BROWSERS}


def test_documents_are_frozen() -> None:
    doc = ContextDocument.model_validate(make_raw())
    with pytest.raises(ValidationError):
        doc.source.__setattr__("title", "other")


def _bad_id(raw: Raw) -> None:
    raw["segments"][1]["id"] = 5


def _bad_first_id(raw: Raw) -> None:
    for segment in raw["segments"]:
        segment["id"] += 1


def _segment_ends_before_start(raw: Raw) -> None:
    raw["segments"][1]["end"] = 1.0


def _segments_go_backwards(raw: Raw) -> None:
    raw["segments"][2]["start"] = 2.0
    raw["segments"][2]["end"] = 4.0


def _word_goes_backwards(raw: Raw) -> None:
    raw["segments"][1]["words"][1]["start"] = 2.0
    raw["segments"][1]["words"][1]["end"] = 2.1


def _word_ends_before_start(raw: Raw) -> None:
    raw["segments"][1]["words"][0]["end"] = 1.0


def _flag_past_text(raw: Raw) -> None:
    raw["segments"][1]["flags"][0]["char_end"] = 999


def _flag_empty_span(raw: Raw) -> None:
    raw["segments"][1]["flags"][0]["char_end"] = 21


def _paragraph_gap(raw: Raw) -> None:
    raw["paragraphs"][1]["first_segment"] = 2
    raw["paragraphs"][0]["last_segment"] = 0
    raw["segments"][1]["paragraph"] = 0


def _paragraph_overlap(raw: Raw) -> None:
    raw["paragraphs"][1]["first_segment"] = 1


def _paragraph_short_of_the_end(raw: Raw) -> None:
    raw["paragraphs"].pop()


def _paragraph_index_skips(raw: Raw) -> None:
    raw["paragraphs"][1]["index"] = 2


def _segment_in_wrong_paragraph(raw: Raw) -> None:
    raw["segments"][2]["paragraph"] = 0


def _paragraph_reference_out_of_range(raw: Raw) -> None:
    raw["segments"][2]["paragraph"] = 9


def _chapter_overlap(raw: Raw) -> None:
    raw["chapters"][1]["first_segment"] = 1


def _chapter_gap(raw: Raw) -> None:
    raw["chapters"][0]["last_segment"] = 0


def _chapters_without_chapter_refs(raw: Raw) -> None:
    for segment in raw["segments"]:
        segment["chapter"] = None


def _chapter_refs_without_chapters(raw: Raw) -> None:
    raw["chapters"] = []


def _segment_in_wrong_chapter(raw: Raw) -> None:
    raw["segments"][1]["chapter"] = 1


def _paragraphs_without_segments(raw: Raw) -> None:
    raw.update(chapters=[], segments=[])


def _unknown_run(raw: Raw) -> None:
    raw["segments"][0]["source"]["run"] = "missing"


def _run_kind_mismatch(raw: Raw) -> None:
    raw["segments"][0]["source"]["kind"] = "caption_auto"


def _duplicate_run(raw: Raw) -> None:
    raw["transcript"]["runs"].append(deepcopy(RUN))


def _macos_home_path(raw: Raw) -> None:
    raw["source"]["access"]["warnings"] = ["cookie db at /Users/example/Library/Cookies"]


def _linux_home_path(raw: Raw) -> None:
    raw["source"]["access"]["warnings"] = ["failed reading /home/example/.config/chromium"]


def _bare_home_path(raw: Raw) -> None:
    raw["source"]["access"]["warnings"] = ["cannot open /home/example"]


def _unknown_field(raw: Raw) -> None:
    raw["source"]["surprise"] = 1


def _unknown_source_kind(raw: Raw) -> None:
    raw["source"]["kind"] = "podcast"


def _youtube_without_access(raw: Raw) -> None:
    del raw["source"]["access"]


def _access_at_the_root(raw: Raw) -> None:
    raw["access"] = raw["source"].pop("access")


def _language_listed_twice(raw: Raw) -> None:
    raw["transcript"]["languages"] = ["zh", "zh"]


def _speaker_without_a_list(raw: Raw) -> None:
    raw["segments"][0]["speaker"] = "S1"


def _paragraph_speaker_without_a_list(raw: Raw) -> None:
    raw["paragraphs"][0]["speaker"] = "S1"


def _language_not_listed(raw: Raw) -> None:
    raw["transcript"]["languages"] = ["en"]


def _no_languages_with_segments(raw: Raw) -> None:
    raw["transcript"]["languages"] = []


def _naive_timestamp(raw: Raw) -> None:
    raw["created_at"] = "2026-01-02T03:04:05"


def _negative_time(raw: Raw) -> None:
    raw["segments"][0]["start"] = -1.0


def _non_finite_time(raw: Raw) -> None:
    raw["source"]["duration_s"] = float("inf")


def _unknown_speaker(raw: Raw) -> None:
    raw["segments"][2]["speaker"] = raw["paragraphs"][2]["speaker"] = "S9"


def _listed_speaker_without_segments(raw: Raw) -> None:
    raw["speakers"].append({"id": "S3", "name": None, "speech_s": 0.0})


def _paragraph_spans_speakers(raw: Raw) -> None:
    raw["paragraphs"][1:] = [
        {"index": 1, "start": 2.5, "end": 8.2, "first_segment": 1, "last_segment": 2, "speaker": "S2"}
    ]
    raw["segments"][2]["paragraph"] = 1


def _slash_in_file_name(raw: Raw) -> None:
    raw["source"]["file_name"] = "Recordings/Weekly sync.m4a"


def _empty_file_name(raw: Raw) -> None:
    raw["source"]["file_name"] = ""


def _segment_without_speaker(raw: Raw) -> None:
    raw["segments"][1]["speaker"] = None


def _paragraph_without_speaker(raw: Raw) -> None:
    raw["paragraphs"][1]["speaker"] = None


def _speakers_out_of_number_order(raw: Raw) -> None:
    raw["speakers"].reverse()


def _speaker_listed_twice(raw: Raw) -> None:
    raw["speakers"].append(dict(raw["speakers"][0]))


def _renumber(raw: Raw, old: str, new: str) -> None:
    for item in (*raw["speakers"], *raw["segments"], *raw["paragraphs"]):
        for key in ("id", "speaker"):
            if item.get(key) == old:
                item[key] = new


def _speaker_id_zero(raw: Raw) -> None:
    _renumber(raw, "S1", "S0")


def _speaker_id_shape(raw: Raw) -> None:
    _renumber(raw, "S2", "B2")


def _blank_speaker_name(raw: Raw) -> None:
    raw["speakers"][0]["name"] = "  "


def _name_of_another_default(raw: Raw) -> None:
    raw["speakers"][0]["name"] = "Speaker 2"


def _diarized_without_speakers(raw: Raw) -> None:
    raw["speakers"] = []
    for item in (*raw["segments"], *raw["paragraphs"]):
        item["speaker"] = None


def _speakers_without_diarization(raw: Raw) -> None:
    raw["transcript"]["diarization"] = None


def _id_not_from_the_hash(raw: Raw) -> None:
    raw["source"]["id"] = "rec-ffffffffffffffff"


def _uppercase_hash(raw: Raw) -> None:
    raw["source"]["audio_sha256"] = AUDIO_SHA256.upper()
    raw["source"]["id"] = f"rec-{AUDIO_SHA256[:16].upper()}"


def _naive_recorded_at(raw: Raw) -> None:
    raw["source"]["recorded_at"] = "2026-11-03T14:05:00"


def _recording_with_access(raw: Raw) -> None:
    raw["source"]["access"] = make_raw()["source"]["access"]


REJECTED: dict[str, tuple[Callable[[Raw], None], str]] = {
    "dense ids": (_bad_id, "dense"),
    "ids start at zero": (_bad_first_id, "dense"),
    "segment reversed": (_segment_ends_before_start, "before it starts"),
    "segment starts ordered": (_segments_go_backwards, "before segment 1"),
    "words monotonic": (_word_goes_backwards, "starts before the word"),
    "word reversed": (_word_ends_before_start, "before it starts"),
    "flag past text": (_flag_past_text, "past the"),
    "flag empty span": (_flag_empty_span, "empty or reversed"),
    "paragraph gap": (_paragraph_gap, "uncovered"),
    "paragraph overlap": (_paragraph_overlap, "overlaps"),
    "paragraph short": (_paragraph_short_of_the_end, "ranges"),
    "paragraph index skips": (_paragraph_index_skips, "indexes"),
    "segment paragraph mismatch": (_segment_in_wrong_paragraph, "does not contain"),
    "paragraph ref out of range": (_paragraph_reference_out_of_range, "does not contain"),
    "chapter overlap": (_chapter_overlap, "overlaps"),
    "chapter gap": (_chapter_gap, "uncovered"),
    "chapter ref missing": (_chapters_without_chapter_refs, "does not contain"),
    "chapter ref without chapters": (_chapter_refs_without_chapters, "has none"),
    "segment chapter mismatch": (_segment_in_wrong_chapter, "does not contain"),
    "paragraphs without segments": (_paragraphs_without_segments, "no segments"),
    "unknown run": (_unknown_run, "absent from transcript.runs"),
    "run kind mismatch": (_run_kind_mismatch, "is caption_auto but run"),
    "duplicate run": (_duplicate_run, "duplicate transcript run"),
    "macOS home path": (_macos_home_path, "home path"),
    "linux home path": (_linux_home_path, "home path"),
    "bare home path": (_bare_home_path, "home path"),
    "unknown field": (_unknown_field, "surprise"),
    "naive timestamp": (_naive_timestamp, "created_at"),
    "negative time": (_negative_time, "start"),
    "non-finite time": (_non_finite_time, "duration_s"),
    "unknown source kind": (_unknown_source_kind, "podcast"),
    "youtube without access": (_youtube_without_access, "access"),
    "access at the root": (_access_at_the_root, "access"),
    "language listed twice": (_language_listed_twice, "twice"),
    "speaker without a list": (_speaker_without_a_list, "unknown speaker 'S1'"),
    "paragraph speaker without a list": (
        _paragraph_speaker_without_a_list,
        "names a speaker but the document lists no",
    ),
    "language not listed": (_language_not_listed, "not in transcript.languages"),
    "no languages with segments": (_no_languages_with_segments, "languages is empty"),
}

REJECTED_RECORDINGS: dict[str, tuple[Callable[[Raw], None], str]] = {
    "unknown speaker": (_unknown_speaker, "unknown speaker 'S9'"),
    "listed speaker without segments": (_listed_speaker_without_segments, "'S3' is listed but has no segment"),
    "paragraph spans speakers": (_paragraph_spans_speakers, "never spans speakers"),
    "slash in file name": (_slash_in_file_name, "file_name\n  String should match pattern"),
    "empty file name": (_empty_file_name, "file_name\n  String should match pattern"),
    "segment without speaker": (_segment_without_speaker, "segment 1 has no speaker"),
    "paragraph without speaker": (_paragraph_without_speaker, "paragraph 1 has no speaker"),
    "speakers out of number order": (_speakers_out_of_number_order, "in order of their id number"),
    "speaker listed twice": (_speaker_listed_twice, "lists an id twice"),
    "speaker id zero": (_speaker_id_zero, "id\n  String should match pattern"),
    "speaker id shape": (_speaker_id_shape, "id\n  String should match pattern"),
    "blank speaker name": (_blank_speaker_name, "blank name"),
    "name of another default label": (_name_of_another_default, "the default label of S2"),
    "diarized without speakers": (_diarized_without_speakers, "lists no speakers"),
    "speakers without diarization": (_speakers_without_diarization, "diarization is None"),
    "id not from the hash": (_id_not_from_the_hash, "first 16 hex digits"),
    "uppercase hash": (_uppercase_hash, "audio_sha256\n  String should match pattern"),
    "naive recorded_at": (_naive_recorded_at, "recorded_at"),
    "recording with access": (_recording_with_access, "access"),
}


@pytest.mark.parametrize("name", REJECTED)
def test_validator_rejects(name: str) -> None:
    mutate, expected = REJECTED[name]
    raw = make_raw()
    mutate(raw)
    with pytest.raises(ValidationError, match=expected):
        ContextDocument.model_validate(raw)


@pytest.mark.parametrize("name", REJECTED_RECORDINGS)
def test_recording_validator_rejects(name: str) -> None:
    mutate, expected = REJECTED_RECORDINGS[name]
    raw = make_recording()
    mutate(raw)
    with pytest.raises(ValidationError, match=expected):
        ContextDocument.model_validate(raw)


def test_a_later_speaker_id_may_speak_first() -> None:
    raw = make_recording()
    _renumber(raw, "S1", "S5")
    raw["speakers"].reverse()
    doc = ContextDocument.model_validate(raw)
    assert [s.id for s in doc.speakers] == ["S2", "S5"] and doc.segments[0].speaker == "S5"


def test_speaker_ids_may_skip_numbers() -> None:
    raw = make_recording()
    _renumber(raw, "S2", "S3")
    doc = ContextDocument.model_validate(raw)
    assert [s.id for s in doc.speakers] == ["S1", "S3"]


def test_a_speaker_may_carry_its_own_default_label() -> None:
    raw = make_recording()
    raw["speakers"][0]["name"] = "Speaker 1"
    assert ContextDocument.model_validate(raw).speakers[0].label == "Speaker 1"


def test_two_ids_may_share_a_name() -> None:
    raw = make_recording()
    for speaker in raw["speakers"]:
        speaker["name"] = "Alice"
    assert {s.name for s in ContextDocument.model_validate(raw).speakers} == {"Alice"}


def test_a_diarized_recording_with_no_speech_lists_no_speakers() -> None:
    raw = make_recording()
    raw.update(speakers=[], paragraphs=[], segments=[])
    assert ContextDocument.model_validate(raw).transcript.diarization is not None


def test_overlapping_speech_is_a_flag_kind() -> None:
    raw = make_recording()
    raw["segments"][1]["flags"] = [{"kind": "overlapping_speech", "char_start": 0, "char_end": 5, "note": "S1 too"}]
    assert ContextDocument.model_validate(raw).segments[1].flags[0].kind == "overlapping_speech"


def test_load_wraps_validation_errors() -> None:
    raw = make_raw()
    raw["segments"][1]["id"] = 7
    with pytest.raises(ContractError, match="dense"):
        load_document(json.dumps(raw))


def test_url_paths_are_not_mistaken_for_home_paths() -> None:
    raw = make_raw()
    raw["source"]["access"]["warnings"] = ["see https://example.com/home/user/ and ~/notes and /Users"]
    assert youtube(ContextDocument.model_validate(raw)).access.warnings


@pytest.mark.parametrize("version", ["3.0", "0.9", "10.0", "1.2"])
def test_unknown_major_is_refused(version: str) -> None:
    raw = make_raw()
    raw["schema_version"] = version
    with pytest.raises(ContractError, match="unsupported") as caught:
        load_document(json.dumps(raw))
    expected = "upgrade ytverbatim" if version in ("3.0", "10.0") else "produce the document again"
    assert expected in caught.value.fix


@pytest.mark.parametrize("version", [None, 1, "1", "1.x", "v1.0", "1.0.1", ""])
def test_malformed_version_is_refused(version: object) -> None:
    raw = make_raw()
    raw["schema_version"] = version
    with pytest.raises(ContractError, match="schema_version"):
        load_document(json.dumps(raw))


def test_minor_version_is_accepted_and_unknown_fields_dropped() -> None:
    raw = make_raw()
    raw["schema_version"] = "2.7"
    raw["new_top_level"] = {"anything": True}
    raw["source"]["new_field"] = 1
    raw["source"]["access"]["new_access_field"] = 1
    raw["segments"][1]["flags"][0]["new_flag_field"] = "x"
    raw["segments"][1]["words"][0]["new_word_field"] = 1
    raw["transcript"]["runs"][0]["vad"] = {"kept": "free-form blobs are untouched"}
    doc = load_document(json.dumps(raw))
    assert doc.schema_version == SCHEMA_VERSION
    assert doc.transcript.runs[0].vad == {"kept": "free-form blobs are untouched"}
    assert "new_" not in dump_document(doc)


def test_a_newer_minor_version_prunes_each_source_kind_by_its_own_fields() -> None:
    raw = make_recording()
    raw["schema_version"] = "2.7"
    raw["source"]["new_field"] = 1
    raw["speakers"][0]["new_speaker_field"] = 1
    raw["transcript"]["diarization"]["new_run_field"] = 1
    doc = load_document(json.dumps(raw))
    assert doc == ContextDocument.model_validate(make_recording())


def test_a_newer_minor_version_drops_unknown_fields_inside_the_creator_note() -> None:
    raw = make_raw()
    raw["schema_version"] = "2.7"
    raw["source"]["creator_note"] = {**NOTE, "new_note_field": 1}
    doc = load_document(json.dumps(raw))
    assert youtube(doc).creator_note == CreatorNote.model_validate(NOTE)


def test_a_1_0_document_is_upgraded() -> None:
    doc = load_document(V1_0_FIXTURE)
    assert doc.schema_version == SCHEMA_VERSION == "2.0"
    source = youtube(doc)
    assert source.creator_note is None and source.id == "AAAAAAAAAAA"
    assert source.access.availability == "public"
    assert doc.speakers == () and doc.transcript.diarization is None
    assert doc.transcript.languages == (doc.transcript.language,)
    assert all(s.speaker is None for s in doc.segments) and all(p.speaker is None for p in doc.paragraphs)
    dumped = json.loads(dump_document(doc))
    assert "video" not in dumped and "access" not in dumped
    assert dumped["source"]["creator_note"] is None
    assert doc.segments[1].text == "Today we cover NVDA at 150 dollars."


def test_a_1_1_document_upgrades_to_exactly_what_2_0_writes() -> None:
    upgraded = load_document(V1_1_FIXTURE)
    assert youtube(upgraded).creator_note is not None
    assert upgraded == load_document(V2_OF_V1_1)
    assert dump_document(upgraded) == V2_OF_V1_1.read_text(encoding="utf-8")


@pytest.mark.parametrize("fixture", [V1_0_FIXTURE, V1_1_FIXTURE], ids=["1.0", "1.1"])
def test_a_1_x_document_is_still_validated_strictly(fixture: Path) -> None:
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    raw["video"]["new_field"] = 1
    with pytest.raises(ContractError, match="new_field"):
        load_document(json.dumps(raw))
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    raw["access"]["new_access_field"] = 1
    with pytest.raises(ContractError, match="new_access_field"):
        load_document(json.dumps(raw))
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    raw["segments"][1]["id"] = 7
    with pytest.raises(ContractError, match="dense"):
        load_document(json.dumps(raw))


def test_a_1_x_document_cannot_carry_2_0_fields() -> None:
    raw = json.loads(V1_1_FIXTURE.read_text(encoding="utf-8"))
    raw["segments"][0]["speaker"] = "S1"
    with pytest.raises(ContractError, match="unknown speaker"):
        load_document(json.dumps(raw))


@pytest.mark.parametrize(
    ("place", "field"),
    [(None, "speakers"), (None, "source"), ("transcript", "languages"), ("transcript", "diarization")],
)
def test_a_1_x_document_carrying_a_2_0_field_is_refused_not_overwritten(place: str | None, field: str) -> None:
    raw = json.loads(V1_1_FIXTURE.read_text(encoding="utf-8"))
    (raw[place] if place else raw)[field] = []
    name = f"{place}.{field}" if place else field
    with pytest.raises(ContractError, match=f"cannot carry the 2.0 fields {name}"):
        load_document(json.dumps(raw))


def test_the_creator_note_round_trips() -> None:
    raw = make_raw()
    raw["source"]["creator_note"] = NOTE
    doc = ContextDocument.model_validate(raw)
    assert youtube(doc).creator_note == CreatorNote.model_validate(NOTE)
    assert load_document(dump_document(doc)) == doc


def test_a_creator_note_needs_all_its_fields() -> None:
    raw = make_raw()
    raw["source"]["creator_note"] = {"text": "x", "author": "y"}
    with pytest.raises(ValidationError, match="published"):
        ContextDocument.model_validate(raw)


def test_same_version_still_rejects_unknown_fields() -> None:
    raw = make_raw()
    raw["source"]["new_field"] = 1
    with pytest.raises(ContractError, match="new_field"):
        load_document(json.dumps(raw))


def test_minor_version_still_validates_contract_rules() -> None:
    raw = make_raw()
    raw["schema_version"] = "2.1"
    raw["segments"][1]["id"] = 7
    with pytest.raises(ContractError, match="dense"):
        load_document(json.dumps(raw))


@pytest.mark.parametrize("text", ["not json", "[]", '"1.0"'])
def test_non_document_input_is_refused(text: str) -> None:
    with pytest.raises(ContractError):
        load_document(text)


def youtube(doc: ContextDocument) -> YouTubeSource:
    assert isinstance(doc.source, YouTubeSource)
    return doc.source
