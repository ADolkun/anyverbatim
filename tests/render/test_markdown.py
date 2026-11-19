from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from anyverbatim.cli import main
from anyverbatim.contract import ContextDocument, Edit, Flag, Segment, TranscriptRun
from anyverbatim.render import render_markdown

MakeDoc = Callable[..., ContextDocument]
MakeSeg = Callable[..., Segment]
MakeFlag = Callable[..., Flag]

FIXTURES = Path(__file__).parent.parent / "fixtures"
GOLDEN = FIXTURES / "render"


def zh_doc(make_doc: MakeDoc, seg: MakeSeg, flag: MakeFlag) -> ContextDocument:
    price = "今天我们聊聊英伟达NVDA，股价涨到了150美元。"
    return make_doc(
        [
            seg("大家好，欢迎收看本期节目。", 0.0, 3.4, paragraph=0, chapter=0),
            seg(
                price,
                3.4,
                9.9,
                paragraph=0,
                chapter=0,
                flags=[
                    flag(
                        "numeric_disputed",
                        price,
                        "150",
                        note="engines disagree",
                        readings=[("asr:example-engine", "150"), ("caption_auto", "115")],
                    ),
                    flag("entity_corrected", price, "英伟达", note="company name"),
                ],
                edits=[Edit(stage="correction", before="英伟大", after="英伟达", llm="example-llm")],
            ),
            seg("去年的价格也是150美元。", 65.2, 70.0, paragraph=1, chapter=1),
        ],
        language="zh",
        duration_s=125.5,
        chapters={0: "开场", 1: "财报"},
        punctuation="engine",
        correction_llm="example-llm",
    )


def en_doc(make_doc: MakeDoc, seg: MakeSeg, flag: MakeFlag) -> ContextDocument:
    first = "Revenue grew 12% to $4.2 billion in Q3."
    second = "Guidance for next_quarter stays at *flat* [sic] and A&T."
    return make_doc(
        [
            seg(
                first,
                3.0,
                8.0,
                flags=[
                    flag("numeric_single_source", first, "12%", note="one engine only"),
                    flag("numeric_single_source", first, "$4.2 billion", note="one engine only"),
                    flag("low_confidence", first, "$4.2 billion", note="low decoder confidence"),
                ],
            ),
            seg(second, 8.0, 14.5),
            seg("Thanks for watching.", 3700.0, 3703.4, paragraph=1),
        ],
        language="en",
        duration_s=3725.0,
        punctuation="llm",
    )


def recording_doc(make_doc: MakeDoc, seg: MakeSeg, flag: MakeFlag) -> ContextDocument:
    budget = "我这边预算还是120万，没有变。"
    return make_doc(
        [
            seg("好，我们开始吧。", 0.0, 2.0, paragraph=0, speaker="S1"),
            seg("First item is the Q3 budget.", 2.0, 8.5, paragraph=0, speaker="S1"),
            seg(
                budget,
                9.0,
                14.0,
                paragraph=1,
                speaker="S2",
                flags=[
                    flag("low_confidence", budget, "120万", note="engine confidence 0.31"),
                    flag("overlapping_speech", budget, "我这边", note="a second speaker is active"),
                ],
            ),
            seg("OK, so we keep it flat.", 15.0, 17.0, paragraph=2, speaker="S1"),
            seg("Next is hiring.", 17.0, 18.5, paragraph=2, speaker="S1"),
        ],
        language="zh",
        languages=("zh", "en"),
        duration_s=2832.0,
        punctuation="llm",
        correction_llm="example-llm",
        source={"kind": "recording", "recorded_at": datetime(2026, 11, 3, 14, 5, tzinfo=timezone(timedelta(hours=8)))},
        speakers={"S1": (None, 12.0), "S2": (None, 6.0)},
    )


@pytest.mark.parametrize(("name", "build"), [("zh", zh_doc), ("en", en_doc), ("recording", recording_doc)])
def test_golden(name: str, build: Callable[[MakeDoc, MakeSeg, MakeFlag], ContextDocument], make_doc, seg, flag) -> None:
    rendered = render_markdown(build(make_doc, seg, flag))
    assert rendered == (GOLDEN / f"{name}.md").read_text(encoding="utf-8")


def test_footnote_sits_at_the_flagged_occurrence_and_other_occurrences_stay_plain(make_doc, seg, flag) -> None:
    text = "Buy 150 now, 150 later, 150 always."
    doc = make_doc([seg(text, 0.0, 5.0, flags=[flag("numeric_disputed", text, "150", occurrence=1, note="n")])])
    body = render_markdown(doc)
    assert "Buy 150 now, 150[^1] later, 150 always." in body
    assert body.count("[^1]") == 2


def test_numbering_follows_document_order_across_segments(make_doc, seg, flag) -> None:
    first, second = "alpha beta", "gamma delta"
    doc = make_doc(
        [
            seg(first, 0.0, 1.0, flags=[flag("low_confidence", first, "beta")]),
            seg(second, 1.0, 2.0, flags=[flag("low_confidence", second, "gamma")]),
        ]
    )
    body = render_markdown(doc)
    assert "alpha beta[^1] gamma[^2] delta" in body
    assert body.index("[^1]: 00:00") < body.index("[^2]: 00:01")


def test_links_use_floored_whole_seconds(make_doc, seg) -> None:
    doc = make_doc([seg("hello", 61.999, 63.0)], duration_s=100.0)
    assert "[01:01](https://youtu.be/AAAAAAAAAAA?t=61) hello" in render_markdown(doc)


def test_no_chapters_gets_a_single_transcript_heading(make_doc, seg) -> None:
    doc = make_doc([seg("one", 0.0, 1.0), seg("two", 1.0, 2.0, paragraph=1)])
    body = render_markdown(doc)
    assert body.count("## Transcript") == 1
    assert body.count("\n## ") == 1


def test_chapter_heading_links_to_chapter_start(make_doc, seg) -> None:
    doc = make_doc([seg("one", 5.5, 9.0, chapter=0)], chapters={0: "First"})
    assert "## [00:05](https://youtu.be/AAAAAAAAAAA?t=5) First" in render_markdown(doc)


def test_hour_long_video_uses_h_mm_ss_everywhere(make_doc, seg) -> None:
    doc = make_doc([seg("late", 3725.9, 3730.0)], duration_s=3600.0)
    body = render_markdown(doc)
    assert "[1:02:05](https://youtu.be/AAAAAAAAAAA?t=3725) late" in body
    assert "· 1:00:00 ·" in body


def test_short_video_uses_mm_ss(make_doc, seg) -> None:
    doc = make_doc([seg("early", 5.0, 6.0)], duration_s=3599.9)
    body = render_markdown(doc)
    assert "[00:05]" in body
    assert "· 59:59 ·" in body


def test_zh_segments_join_without_separator_and_en_with_a_space(make_doc, seg) -> None:
    zh = render_markdown(make_doc([seg("你好。", 0.0, 1.0), seg("再见。", 1.0, 2.0)], language="zh"))
    en = render_markdown(make_doc([seg("Hello.", 0.0, 1.0), seg("Bye.", 1.0, 2.0)], language="en"))
    assert "你好。再见。" in zh
    assert "Hello. Bye." in en


@pytest.mark.parametrize(
    ("left", "right", "joined"),
    [
        ("Really strong quarter.", "Next we look at margins.", "Really strong quarter. Next we look at margins."),
        ("He said “stop.”", "Then he left.", "He said “stop.” Then he left."),
        ("It fell (sharply).", '"Why?" she asked.', 'It fell (sharply). "Why?" she asked.'),
        ("收入3.5万美元。", "Really strong quarter.", "收入3.5万美元。Really strong quarter."),
        ("Really strong quarter.", "大家好。", "Really strong quarter.大家好。"),
        ("大家好欢迎回来", "今天聊聊ETF", "大家好欢迎回来今天聊聊ETF"),
        ("今天聊聊利润", "really strong", "今天聊聊利润really strong"),
        ("引用", "「原话」", "引用「原话」"),
        ("“走吧。”", "然后", "“走吧。”然后"),
        ("我想想……", "好吧。", "我想想……好吧。"),
        ("好的!", "我们", "好的!我们"),
        ("好的？", "我们", "好的？我们"),
        ("就这样——", "下一个", "就这样——下一个"),
        ("增长了5%", "然后", "增长了5%然后"),
        ("今年", "2025年", "今年2025年"),
        ("到了2025", "年底", "到了2025年底"),
        ("他说‘好’", "我同意", "他说‘好’我同意"),
        ("Revenue grew 5%", "then fell.", "Revenue grew 5% then fell."),
    ],
    ids=[
        "end.Next",
        "latin-closer-latin",
        "latin-quote-opener",
        "cjk-stop-latin",
        "latin-stop-cjk",
        "cjk-cjk",
        "cjk-latin",
        "cjk-bracket",
        "cjk-quote-closer",
        "cjk-ellipsis",
        "cjk-ascii-bang",
        "cjk-fullwidth-question",
        "cjk-dash",
        "percent-cjk",
        "cjk-digit",
        "digit-cjk",
        "cjk-single-quote",
        "percent-latin",
    ],
)
def test_segments_join_by_the_script_at_each_boundary(make_doc, seg, left: str, right: str, joined: str) -> None:
    doc = make_doc([seg(left, 0.0, 1.0), seg(right, 1.0, 2.0)], language="zh")
    assert f"[00:00](https://youtu.be/AAAAAAAAAAA?t=0) {joined}\n" in render_markdown(doc)


def test_the_joiner_reads_the_text_not_the_footnote_marker(make_doc, seg, flag) -> None:
    text = "价格涨到12"
    doc = make_doc([seg(text, 0.0, 1.0, flags=[flag("low_confidence", text, "12")]), seg("today.", 1.0, 2.0)])
    assert "价格涨到12[^1] today." in render_markdown(doc)


def recording(**fields: object) -> dict[str, object]:
    return {"kind": "recording", **fields}


def test_a_recording_has_plain_timestamps_and_no_links(make_doc, seg) -> None:
    doc = make_doc([seg("one", 5.0, 6.0, chapter=0)], chapters={0: "First"}, source=recording())
    body = render_markdown(doc)
    assert "## [00:05] First" in body
    assert "[00:05] one" in body
    assert "youtu" not in body


def test_a_recording_header_omits_an_unknown_date_and_names_a_plain_file(make_doc, seg) -> None:
    doc = make_doc([seg("x", 0.0, 1.0)], duration_s=3725.0, source=recording(origin="file", file_name="a_b [1].m4a"))
    lines = render_markdown(doc).splitlines()
    assert lines[2] == "1:02:05 · a_b \\[1\\].m4a"
    assert lines[4] == "Transcript: example-engine · languages: en · punctuation: engine · corrected by: none · 0 flags"


def test_a_recording_time_is_shown_in_its_own_offset(make_doc, seg) -> None:
    moment = datetime(2026, 11, 3, 6, 5, tzinfo=UTC)
    doc = make_doc([seg("x", 0.0, 1.0)], source=recording(recorded_at=moment))
    assert "Recorded 2026-11-03 06:05 +00:00 · 10:00 · Voice Memos\n" in render_markdown(doc)


def test_a_single_speaker_gets_no_labels(make_doc, seg) -> None:
    doc = make_doc(
        [seg("one.", 0.0, 1.0, speaker="S1"), seg("two.", 5.0, 6.0, paragraph=1, speaker="S1")],
        source=recording(),
        speakers={"S1": (None, 2.0)},
    )
    body = render_markdown(doc)
    assert "Speaker" not in body
    assert "[00:00] one.\n\n[00:05] two." in body


def test_ids_sharing_a_name_are_one_person(make_doc, seg) -> None:
    doc = make_doc(
        [
            seg("Hi.", 0.0, 1.0, paragraph=0, speaker="S1"),
            seg("Hello.", 1.0, 2.0, paragraph=1, speaker="S2"),
            seg("And me.", 2.0, 3.0, paragraph=2, speaker="S3"),
            seg("Still me.", 9.0, 10.0, paragraph=3, speaker="S3"),
            seg("Bye.", 10.0, 11.0, paragraph=4, speaker="S1"),
        ],
        source=recording(),
        speakers={"S1": ("Alice", 2.0), "S2": ("Bob", 1.0), "S3": ("Bob", 2.0)},
    )
    body = render_markdown(doc)
    assert "Speakers: Alice (40%) · Bob (60%)" in body
    paragraphs = [line for line in body.splitlines() if line.startswith("[")]
    assert paragraphs == [
        "[00:00] **Alice:** Hi.",
        "[00:01] **Bob:** Hello. And me.",
        "[00:09] **Bob:** Still me.",
        "[00:10] **Alice:** Bye.",
    ]


def test_ids_sharing_a_name_stay_apart_across_a_long_pause(make_doc, seg) -> None:
    doc = make_doc(
        [
            seg("Hi.", 0.0, 1.0, paragraph=0, speaker="S1"),
            seg("Hello.", 1.0, 2.0, paragraph=1, speaker="S2"),
            seg("A minute later.", 62.0, 63.0, paragraph=2, speaker="S3"),
        ],
        source=recording(),
        speakers={"S1": ("Alice", 1.0), "S2": ("Bob", 1.0), "S3": ("Bob", 1.0)},
    )
    paragraphs = [line for line in render_markdown(doc).splitlines() if line.startswith("[")]
    assert paragraphs == ["[00:00] **Alice:** Hi.", "[00:01] **Bob:** Hello.", "[01:02] **Bob:** A minute later."]


def test_ids_sharing_a_name_join_only_within_the_paragraph_size_limit(make_doc, seg) -> None:
    # Four sentences and three make seven, past the splitter's six.
    first = [seg(f"One {n}.", float(n), n + 1.0, paragraph=0, speaker="S1") for n in range(4)]
    second = [seg(f"Two {n}.", 4.0 + n, 5.0 + n, paragraph=1, speaker="S2") for n in range(3)]
    doc = make_doc(
        [*first, *second, seg("Hi.", 8.0, 9.0, paragraph=2, speaker="S3")],
        source=recording(),
        speakers={"S1": ("Bob", 4.0), "S2": ("Bob", 3.0), "S3": ("Alice", 1.0)},
    )
    paragraphs = [line for line in render_markdown(doc).splitlines() if line.startswith("[")]
    assert [line.split("** ")[0] for line in paragraphs] == ["[00:00] **Bob:", "[00:04] **Bob:", "[00:08] **Alice:"]


def test_an_untitled_recording_is_headed_by_its_file_name(make_doc, seg) -> None:
    doc = make_doc([seg("x", 0.0, 1.0)], source=recording(title="  ", file_name="Memo 3.m4a"))
    assert render_markdown(doc).startswith("# Memo 3.m4a\n")


def test_ids_that_all_share_one_name_get_no_labels(make_doc, seg) -> None:
    doc = make_doc(
        [seg("Hi.", 0.0, 1.0, speaker="S1"), seg("Hello.", 1.0, 2.0, paragraph=1, speaker="S2")],
        source=recording(),
        speakers={"S1": ("Alice", 1.0), "S2": ("Alice", 1.0)},
    )
    body = render_markdown(doc)
    assert "Alice" not in body
    assert "Speakers" not in body
    assert "[00:00] Hi. Hello." in body


def test_names_are_escaped(make_doc, seg) -> None:
    doc = make_doc(
        [seg("Hi.", 0.0, 1.0, speaker="S1"), seg("Yo.", 1.0, 2.0, paragraph=1, speaker="S2")],
        source=recording(),
        speakers={"S1": ("*Al*", 1.0), "S2": (None, 1.0)},
    )
    body = render_markdown(doc)
    assert "[00:00] **\\*Al\\*:** Hi." in body
    assert "Speakers: \\*Al\\* (50%) · Speaker 2 (50%)" in body


def test_overlapping_speech_stays_in_the_json(make_doc, seg, flag) -> None:
    text = "we both talk"
    overlap = flag("overlapping_speech", text, "both", note="a second speaker is active")
    doc = make_doc(
        [seg(text, 0.0, 2.0, flags=[overlap], speaker="S1")], source=recording(), speakers={"S1": (None, 2.0)}
    )
    body = render_markdown(doc)
    assert "overlapping" not in body
    assert "[^" not in body
    assert "0 flags" in body
    assert doc.segments[0].flags == (overlap,)


def test_numeric_single_source_needs_overlapping_low_confidence(make_doc, seg, flag) -> None:
    text = "up 12 and 34 today"
    lone = flag("numeric_single_source", text, "12", note="alone")
    covered = flag("numeric_single_source", text, "34", note="covered")
    low = flag("low_confidence", text, "34")
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=[lone, covered, low])]))
    assert "up 12 and 34[^1][^2] today" in body
    assert "alone" not in body
    assert "numeric_single_source: covered" in body


def test_timing_interpolated_stays_in_the_json(make_doc, seg, flag) -> None:
    text = "Prices rose 12 today"
    timing = flag("timing_interpolated", text, text, note="interpolated")
    disputed = flag("numeric_disputed", text, "12", note="heard 20")
    doc = make_doc([seg(text, 0.0, 2.0, flags=[timing, disputed])])
    body = render_markdown(doc)
    assert "timing_interpolated" not in body
    assert "interpolated" not in body
    assert "Prices rose 12[^1] today" in body
    assert "1 flag" in body
    assert doc.segments[0].flags == (timing, disputed)


def test_numeric_single_source_ignores_low_confidence_elsewhere_in_segment(make_doc, seg, flag) -> None:
    text = "up 12 and shaky speech"
    lone = flag("numeric_single_source", text, "12", note="alone")
    low = flag("low_confidence", text, "shaky")
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=[lone, low])]))
    assert "up 12 and shaky[^1] speech" in body
    assert "alone" not in body


def test_entity_corrected_and_numeric_fixed_footnotes_show_the_before_text(make_doc, seg, flag) -> None:
    text = "Nvidia hit 150 today"
    edits = [
        Edit(stage="correction", before="Envidia", after="Nvidia", llm=None),
        Edit(stage="numeric", before="115", after="150", llm=None),
    ]
    flags = [flag("entity_corrected", text, "Nvidia", note="ticker"), flag("numeric_fixed", text, "150", note="n")]
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=flags, edits=edits)]))
    assert "00:00 — entity_corrected: ticker; was Envidia" in body
    assert "00:00 — numeric_fixed: n; was 115" in body


def test_the_before_text_shows_when_the_edit_is_narrower_than_the_flagged_span(make_doc, seg, flag) -> None:
    text = "涨了五十个百分点"
    edits = [Edit(stage="numeric", before="十五", after="五十", llm=None)]
    flags = [flag("numeric_fixed", text, "五十个百分点", note="十五个百分点 → 五十个百分点")]
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=flags, edits=edits)]))
    assert "numeric_fixed: 十五个百分点 → 五十个百分点; was 十五" in body


def test_a_numeric_fix_in_digits_shows_the_figure_it_replaced(make_doc, seg, flag) -> None:
    text = "涨了50个百分点"
    edits = [Edit(stage="numeric", before="35", after="50", llm=None)]
    flags = [flag("numeric_fixed", text, "50个百分点", note="三十五个百分点 → 五十个百分点")]
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=flags, edits=edits)]))
    assert "50个百分点[^1]" in body
    assert "; was 35" in body


def test_an_edit_outside_the_flagged_span_shows_no_before_text(make_doc, seg, flag) -> None:
    text = "Nvidia hit 150 today"
    edits = [Edit(stage="correction", before="Envidia", after="Nvidia", llm=None)]
    body = render_markdown(make_doc([seg(text, 0.0, 2.0, flags=[flag("entity_corrected", text, "150")], edits=edits)]))
    assert "was" not in body


def test_footnote_lists_readings_with_sources(make_doc, seg, flag) -> None:
    text = "price is 150"
    f = flag("numeric_disputed", text, "150", note="n", readings=[("asr:x", "150"), ("caption_auto", "115")])
    body = render_markdown(make_doc([seg(text, 0.0, 1.0, flags=[f])]))
    assert "[^1]: 00:00 — numeric_disputed: n; heard 150 (asr:x) / 115 (caption_auto)" in body


def test_overlapping_flags_keep_text_intact_and_order_deterministically(make_doc, seg, flag) -> None:
    text = "the quick brown fox"
    flags = [
        flag("low_confidence", text, "quick brown", note="b"),
        flag("decoder_anomaly", text, "quick", note="a"),
        flag("entity_corrected", text, "brown fox", note="c"),
        flag("low_confidence", text, "quick brown", note="a"),
    ]
    doc = make_doc([seg(text, 0.0, 2.0, flags=flags)])
    body = render_markdown(doc)
    assert "the quick[^1] brown[^2][^3] fox[^4]" in body
    shuffled = doc.segments[0].model_copy(update={"flags": tuple(reversed(flags))})
    assert render_markdown(doc.model_copy(update={"segments": (shuffled,)})) == body
    footnotes = [line for line in body.splitlines() if line.startswith("[^")]
    assert footnotes == [
        "[^1]: 00:00 — decoder_anomaly: a",
        "[^2]: 00:00 — low_confidence: a",
        "[^3]: 00:00 — low_confidence: b",
        "[^4]: 00:00 — entity_corrected: c",
    ]


def test_a_localized_decoder_anomaly_footnotes_only_the_words_inside_the_span(make_doc, seg, flag) -> None:
    text = "Alpha beta gamma. Delta epsilon zeta."
    note = "the decoder looped or stalled at 0:03-0:06 and it could not be repaired"
    body = render_markdown(
        make_doc([seg(text, 0.0, 6.0, flags=[flag("decoder_anomaly", text, "Delta epsilon zeta.", note=note)])])
    )
    assert "Alpha beta gamma. Delta epsilon zeta.[^1]" in body
    assert body.count("[^") == 2
    assert f"[^1]: 00:00 — decoder_anomaly: {note}" in body


def test_flag_touching_the_segment_end_and_adjacent_segment(make_doc, seg, flag) -> None:
    first, second = "ends here", "starts there"
    doc = make_doc(
        [
            seg(first, 0.0, 1.0, flags=[flag("low_confidence", first, "here")]),
            seg(second, 1.0, 2.0, flags=[flag("low_confidence", second, "starts")]),
        ]
    )
    assert "ends here[^1] starts[^2] there" in render_markdown(doc)


def test_escaping_is_minimal_and_leaves_cjk_tickers_and_numbers_alone(make_doc, seg) -> None:
    text = "英伟达NVDA涨到$1,150.50（+3.2%）, AT&T, snake_case, 2*3 [x] <b> `c` a_ _b \\ &amp;"
    body = render_markdown(make_doc([seg(text, 0.0, 1.0)]))
    expected = (
        "英伟达NVDA涨到$1,150.50（+3.2%）, AT&T, snake_case, 2\\*3 \\[x\\] \\<b\\> \\`c\\` a\\_ \\_b \\\\ \\&amp;"
    )
    assert expected in body


def test_tildes_cannot_strike_text_through(make_doc, seg) -> None:
    body = render_markdown(make_doc([seg("涨幅~~3%~~ and ~5~ points", 0.0, 1.0)]))
    assert "涨幅\\~\\~3%\\~\\~ and \\~5\\~ points" in body


def test_newlines_in_text_do_not_start_block_syntax(make_doc, seg, flag) -> None:
    text = "intro\n# not a heading\n- not a list"
    f = flag("low_confidence", text, "heading")
    body = render_markdown(make_doc([seg(text, 0.0, 1.0, flags=[f])]))
    assert "intro # not a heading[^1] - not a list" in body


def test_title_and_chapter_titles_are_escaped(make_doc, seg) -> None:
    doc = make_doc([seg("x", 0.0, 1.0, chapter=0)], chapters={0: "Q&A [live]"})
    body = render_markdown(doc)
    assert "Q&A \\[live\\]" in body


def test_provenance_line_labels(make_doc, seg, flag) -> None:
    text = "a b"
    manual = TranscriptRun(
        id="cap-1",
        kind="caption_manual",
        engine=None,
        model_repo=None,
        model_revision=None,
        params={},
        vad=None,
        chunking=None,
        aligner=None,
    )
    doc = make_doc([seg(text, 0.0, 1.0, flags=[flag("low_confidence", text, "b")])], run=manual)
    assert "Transcript: manual captions · punctuation: engine · corrected by: none · 1 flag\n" in render_markdown(doc)


def test_header_omits_unknown_upload_date(make_doc, seg) -> None:
    body = render_markdown(make_doc([seg("x", 0.0, 1.0)], source={"upload_date": None}))
    assert "Example Channel · 10:00 · https://youtu.be/AAAAAAAAAAA" in body


def test_document_without_segments_renders_header_only(make_doc) -> None:
    body = render_markdown(make_doc([]))
    assert body.startswith("# Example Title\n")
    assert body.rstrip().endswith("## Transcript")
    assert "0 flags" in body


def test_the_render_command_reads_a_1_1_document_as_its_2_0_upgrade(tmp_path: Path) -> None:
    context = tmp_path / "context.json"
    context.write_bytes((FIXTURES / "contract" / "context_v1_1.json").read_bytes())
    result = CliRunner().invoke(main, ["render", "--stdout", str(context)])
    assert result.exit_code == 0, result.output
    assert result.output == (FIXTURES / "pipeline" / "fxMixZhEn01" / "context.md").read_text(encoding="utf-8")
