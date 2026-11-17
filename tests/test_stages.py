import itertools
import random
from datetime import UTC, datetime

import pytest

from ytverbatim.acquire.captions import CaptionKind, CaptionSegment, CaptionTrack
from ytverbatim.acquire.metadata import VideoMetadata
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.contract import Access, ContextDocument, Edit, Flag, Word, dump_document, load_document
from ytverbatim.render import render_markdown
from ytverbatim.stages import (
    SourceInfo,
    Stretch,
    Transcribed,
    build_document,
    from_asr_rows,
    from_captions,
    normalize,
    numerals,
    with_results,
    word_confidence,
    youtube_source,
)
from ytverbatim.text.numbers import quantity_tokens
from ytverbatim.text.numerals import apply, conversions, readings

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
LLM = "fake:llm"


def metadata(language: str = "zh") -> VideoMetadata:
    return VideoMetadata(
        video_id="synthVid001",
        title="Synthetic title",
        channel="Synthetic channel",
        channel_id="UCsynthetic",
        upload_date="20300101",
        duration=100.0,
        description="",
        language=language,
        availability="public",
        live_status=None,
        chapters=(),
        subtitle_langs=(),
        auto_caption_lang=None,
    )


def flag_on(text: str, part: str, kind: str = "decoder_anomaly") -> Flag:
    start = text.index(part)
    return Flag.model_validate({"kind": kind, "char_start": start, "char_end": start + len(part), "note": part})


def source_of(rows: list[ChunkHypothesis], language: str = "zh") -> Transcribed:
    return from_asr_rows(rows, registry.get("w-prod"), language)


def row(text: str, start: float, end: float, words: tuple[Word, ...] = (), language: str = "zh") -> ChunkHypothesis:
    return ChunkHypothesis(
        hypothesis="w-prod",
        file="synthVid001",
        chunk=0,
        start=start,
        end=end,
        text=text,
        language=language,
        words=words,
    )


def document(
    source: Transcribed, stretches: tuple[Stretch, ...], language: str = "zh", *, llm_punctuated: bool = False
) -> ContextDocument:
    meta = metadata(language)
    doc = build_document(
        info=SourceInfo(meta.video_id, meta.chapters, lambda s: youtube_source(meta, (), ACCESS, s)),
        source=source,
        stretches=stretches,
        chapters=False,
        created_at=NOW,
        llm_punctuated=llm_punctuated,
    )
    assert load_document(dump_document(doc)) == doc
    return doc


def flagged(doc: ContextDocument) -> list[tuple[int, str, str]]:
    return [(s.id, f.kind, s.text[f.char_start : f.char_end]) for s in doc.segments for f in s.flags]


def test_flags_stay_on_their_characters_across_normalize_and_punctuation() -> None:
    source = source_of([row("嗯 這個GAAP數據很好 嗯 然後估值也不低", 0.0, 8.0)])
    (stretch,) = source.stretches
    stretch = stretch.model_copy(
        update={"flags": (flag_on(stretch.text, "GAAP數據"), flag_on(stretch.text, "估值", "low_confidence"))}
    )
    (normal,) = normalize((stretch,), "zh", script="zh-Hans", fillers="remove")
    assert normal.text == "这个GAAP数据很好 然后估值也不低"
    punctuated = "这个GAAP数据很好。然后估值也不低。"
    edit = Edit(stage="punctuation", before=normal.text, after=punctuated, llm=LLM)
    # As the pipeline does: punctuation moves the flags but is not recorded as edits.
    final = with_results((normal,), [punctuated], [[edit]], record=False)

    doc = document(source, final, llm_punctuated=True)
    assert [s.text for s in doc.segments] == ["这个GAAP数据很好。", "然后估值也不低。"]
    assert flagged(doc) == [(0, "decoder_anomaly", "GAAP数据"), (1, "low_confidence", "估值")]
    assert all(s.edits == () for s in doc.segments)
    assert doc.transcript.punctuation == "llm" and doc.transcript.script == "zh-Hans"


def test_flags_stay_on_their_characters_when_marks_become_full_width() -> None:
    source = source_of([row("OK, 这个GAAP数据很好 , 然后估值也不低.", 0.0, 8.0)])
    (stretch,) = source.stretches
    stretch = stretch.model_copy(
        update={"flags": (flag_on(stretch.text, "GAAP数据"), flag_on(stretch.text, "估值", "low_confidence"))}
    )
    (normal,) = normalize((stretch,), "zh", script="zh-Hans", fillers="keep")
    assert normal.text == "OK，这个GAAP数据很好，然后估值也不低。"
    assert [normal.text[f.char_start : f.char_end] for f in normal.flags] == ["GAAP数据", "估值"]


def test_a_corrected_word_is_interpolated_while_the_rest_keeps_engine_times() -> None:
    pieces = [" The", " GAP", " margin", " rose."]
    words = tuple(Word(text=p, start=1.0 + i, end=1.5 + i, conf=0.9) for i, p in enumerate(pieces))
    source = source_of([row("".join(pieces), 0.0, 10.0, words, "en")], "en")
    (stretch,) = source.stretches
    after = "The GAAP margin rose."
    corrected = Edit(stage="correction", before=stretch.text, after=after, llm=LLM)
    final = with_results((stretch,), [after], [[corrected]], [[flag_on(after, "GAAP", "entity_corrected")]])

    doc = document(source, final, "en")
    (segment,) = doc.segments
    assert segment.words is not None
    assert [(w.text, w.start) for w in segment.words] == [("The", 1.0), ("margin", 3.0), ("rose.", 4.0)]
    assert flagged(doc) == [(0, "timing_interpolated", after), (0, "entity_corrected", "GAAP")]
    assert [(e.stage, e.before, e.after) for e in segment.edits] == [("correction", "GAP", "GAAP")]
    markdown = render_markdown(doc)
    assert "GAAP[^1]" in markdown and "was GAP" in markdown and "timing_interpolated" not in markdown
    assert any("did not cover" in w for w in doc.quality.warnings)


def test_latin_and_digit_pieces_the_engine_split_keep_their_times_without_interpolation() -> None:
    pieces = ["好，", "看", "X", "LP", "的", "位", "置。", "成", "交", "17", "1.", "6", "万"]
    words = tuple(Word(text=p, start=1.0 + i, end=1.5 + i) for i, p in enumerate(pieces))
    source = source_of([row("好，看XLP的位置。成交171.6万", 0.0, 20.0, words)])
    stored = [word.text for stretch in source.stretches for word in stretch.words]
    assert stored == ["好，", "看", "XLP", "的", "位", "置。", "成", "交", "171.6", "万"]

    doc = document(source, source.stretches)
    assert flagged(doc) == []
    assert not any("did not cover" in w for w in doc.quality.warnings)
    timed = {w.text: (w.start, w.end) for segment in doc.segments for w in segment.words or ()}
    assert timed["XLP"] == (3.0, 4.5)
    assert timed["171.6"] == (10.0, 12.5)


def test_a_correction_that_lengthens_a_ticker_interpolates_the_whole_run() -> None:
    lead, tail = "参考一", "的走势" + "很好" * 20
    chars = [*lead, "QQ", *tail]
    words = tuple(Word(text=c, start=1.0 + i * 0.3, end=1.2 + i * 0.3) for i, c in enumerate(chars))
    source = source_of([row(f"{lead}QQ{tail}", 0.0, 30.0, words)])
    (stretch,) = source.stretches
    after = f"{lead}QQQ{tail}"
    edit = Edit(stage="correction", before=stretch.text, after=after, llm=LLM)
    final = with_results((stretch,), [after], [[edit]], [[]])

    doc = document(source, final)
    assert sum("QQQ" in segment.text for segment in doc.segments) == 1
    assert not any(segment.text.endswith("QQ") or segment.text.startswith("Q") for segment in doc.segments)
    assert not any(w.text.startswith("QQ") for segment in doc.segments for w in segment.words or ())
    assert any(kind == "timing_interpolated" for _, kind, _ in flagged(doc))


def test_text_the_engine_never_timed_is_still_interpolated_and_flagged() -> None:
    words = tuple(Word(text=p, start=1.0 + i, end=1.5 + i) for i, p in enumerate(["好", "看", "的", "位", "置"]))
    source = source_of([row("好看XLP的位置", 0.0, 10.0, words)])
    doc = document(source, source.stretches)
    assert flagged(doc) == [(0, "timing_interpolated", "好看XLP的位置")]
    assert any("did not cover" in w for w in doc.quality.warnings)


def test_numeric_edits_after_punctuation_land_in_the_final_text() -> None:
    source = source_of([row("营收增长了十五个百分点 利润也不错", 0.0, 6.0)])
    (stretch,) = source.stretches
    punctuated = "营收增长了十五个百分点，利润也不错。"
    fixed = "营收增长了五十个百分点，利润也不错。"
    first = with_results(
        (stretch,),
        [punctuated],
        [[Edit(stage="punctuation", before=stretch.text, after=punctuated, llm=LLM)]],
        record=False,
    )
    numeric_flag = flag_on(fixed, "五十个百分点", "numeric_fixed")
    final = with_results(
        first, [fixed], [[Edit(stage="numeric", before=punctuated, after=fixed, llm=LLM)]], [[numeric_flag]]
    )
    (segment,) = document(source, final).segments
    assert [(e.stage, e.before, e.after) for e in segment.edits] == [("numeric", "十五", "五十")]
    assert segment.text[segment.flags[0].char_start : segment.flags[0].char_end] == "五十个百分点"


def test_a_manual_caption_track_is_a_source_of_its_own() -> None:
    track = CaptionTrack(
        language="en",
        kind=CaptionKind.MANUAL,
        source_format="json3",
        segments=(
            CaptionSegment(start=1.0, end=3.0, text="Markets rose"),
            CaptionSegment(start=3.0, end=5.0, text="today. Then they fell."),
        ),
    )
    source = from_captions(track, "en")
    assert source.run.kind == "caption_manual" and source.run.id == "caption_manual:en"
    doc = document(source, source.stretches, "en")
    assert [s.text for s in doc.segments] == ["Markets rose today.", "Then they fell."]
    assert all(s.words is None and s.flags == () for s in doc.segments)
    assert doc.transcript.punctuation == "caption" and doc.segments[0].start == 1.0
    assert "Transcript: manual captions" in render_markdown(doc)


def test_word_confidence_is_none_when_no_word_has_a_probability() -> None:
    plain = (Word(text="估值", start=0.0, end=1.0), Word(text="很高", start=1.0, end=2.0))
    assert word_confidence(source_of([row("估值很高", 0.0, 2.0, plain)]).stretches) is None
    assert word_confidence(source_of([row("估值很高", 0.0, 2.0)]).stretches) is None


def test_word_confidence_lists_the_probabilities_each_stretch_has() -> None:
    scored = (Word(text="估值", start=0.0, end=1.0, conf=0.9), Word(text="很高", start=1.0, end=2.0, conf=0.3))
    unscored = (Word(text="然后", start=3.0, end=4.0),)
    source = source_of([row("估值很高", 0.0, 2.0, scored), row("然后", 3.0, 4.0, unscored)])
    assert len(source.stretches) == 2
    assert word_confidence(source.stretches) == [[0.9, 0.3], None]


def timed_row(pieces: list[str], start: float = 1.0) -> ChunkHypothesis:
    words = tuple(Word(text=p, start=start + i, end=start + i + 0.5, conf=0.9 - i / 100) for i, p in enumerate(pieces))
    return row("".join(pieces), 0.0, start + len(pieces) + 1.0, words)


def test_flags_stay_on_their_quantity_across_the_numerals_stage() -> None:
    source = source_of([row("收入三十五万美元，第一季度涨了百分之八十", 0.0, 6.0)])
    (stretch,) = source.stretches
    flags = (
        flag_on(stretch.text, "三十五万美元", "numeric_single_source"),
        flag_on(stretch.text, "第一季度", "low_confidence"),
        flag_on(stretch.text, "百分之八十", "numeric_disputed"),
    )
    (shown,) = numerals((stretch.model_copy(update={"flags": flags}),), "zh")
    assert shown.text == "收入35万美元，第一季度涨了80%"
    assert [shown.text[f.char_start : f.char_end] for f in shown.flags] == ["35万美元", "第一季度", "80%"]
    assert [f.note for f in shown.flags] == [f.note for f in flags]


@pytest.mark.parametrize(
    ("pieces", "shown", "figure", "first", "last"),
    [
        (["收入", "三", "万", "美元", "左右"], "收入3万美元左右", "3万", 1, 2),
        (["涨到", "一百", "五十", "块"], "涨到150块", "150", 1, 2),
        (["在", "二零", "二六年", "以后"], "在2026年以后", "2026年", 1, 2),
        (["成交", "4", "千", "手"], "成交4000手", "4000", 1, 2),
        (["成交", "1", ".5", "千万", "手"], "成交1500万手", "1500万", 1, 3),
        (["下午三点四十", "分开会"], "下午3点40分开会", "下午3点40", 0, 0),
    ],
)
def test_words_split_over_a_figure_retime_it_without_interpolation(
    pieces: list[str], shown: str, figure: str, first: int, last: int
) -> None:
    source = source_of([timed_row(pieces)])
    assert flagged(document(source, source.stretches)) == []
    (stretch,) = numerals(source.stretches, "zh")
    assert stretch.text == shown
    doc = document(source, (stretch,))
    assert flagged(doc) == [] and not any("did not cover" in w for w in doc.quality.warnings)
    (segment,) = doc.segments
    timed = {w.text: (w.start, w.end, w.conf) for w in segment.words or ()}
    assert timed[figure] == (1.0 + first, 1.5 + last, 0.9 - last / 100)


def test_a_figure_its_words_only_partly_cover_stays_as_said_and_keeps_its_word_times() -> None:
    source = source_of([timed_row(["收入", "三", "美元", "左右"])])
    (stretch,) = source.stretches
    with_missing_word = stretch.model_copy(update={"text": "收入三万美元左右"})
    before = document(source, (with_missing_word,))
    (shown,) = numerals((with_missing_word,), "zh")
    assert shown == with_missing_word
    after = document(source, (shown,))
    assert flagged(after) == flagged(before) == [(0, "timing_interpolated", "收入三万美元左右")]
    assert after.segments[0].words == before.segments[0].words


def test_a_range_with_an_end_its_words_only_partly_cover_stays_whole() -> None:
    source = source_of([timed_row(["要", "三到十五", "个月"])])
    (stretch,) = source.stretches
    assert numerals((stretch,), "zh")[0].text == "要3到15个月"
    words = (*stretch.words[:1], stretch.words[1].model_copy(update={"text": "三到十"}), *stretch.words[2:])
    (shown,) = numerals((stretch.model_copy(update={"words": words}),), "zh")
    assert shown.text == "要三到十五个月" and shown.words == words


@pytest.mark.parametrize(
    "text",
    [
        "目标价三千目标价三千",
        "是八十是八十",
        "涨到四千涨到四千",
        "三万五到四万三万五到四万",
        "百分之五十分",
        "收入三万，收入三万",
    ],
)
def test_every_flag_stays_on_its_own_figure_when_figures_repeat(text: str) -> None:
    tokens = quantity_tokens(text)
    flags = tuple(
        Flag.model_validate(
            {"kind": "numeric_single_source", "char_start": t.char_start, "char_end": t.char_end, "note": t.surface}
        )
        for t in tokens
    )
    (shown,) = numerals((Stretch(text=text, start=0.0, end=1.0, flags=flags),), "zh")
    changes = conversions(text)
    expected = [
        apply(
            t.surface,
            [(a - t.char_start, b - t.char_start, r) for a, b, r in changes if t.char_start <= a < t.char_end],
        )
        for t in tokens
    ]
    assert [shown.text[f.char_start : f.char_end] for f in shown.flags] == expected
    assert [f.note for f in shown.flags] == [t.surface for t in tokens]


def test_an_edit_reads_with_the_rewrites_of_the_text_it_led_to() -> None:
    stretch = Stretch(
        text="指数涨了两点",
        start=0.0,
        end=2.0,
        edits=(Edit(stage="correction", before="指数张了两点", after="指数涨了两点", llm=LLM),),
    )
    (shown,) = numerals((stretch,), "zh")
    assert shown.text == "指数涨了2点"
    assert [(e.before, e.after) for e in shown.edits] == [("指数张了2点", "指数涨了2点")]


def test_edits_are_rewritten_and_land_on_their_sentence_with_the_was_text() -> None:
    source = source_of([row("开盘平稳 营收增长了三十五个百分点 利润也不错", 0.0, 9.0)])
    (stretch,) = source.stretches
    punctuated = "开盘平稳。营收增长了三十五个百分点，利润也不错。"
    fixed = "开盘平稳。营收增长了五十个百分点，利润也不错。"
    first = with_results(
        (stretch,),
        [punctuated],
        [[Edit(stage="punctuation", before=stretch.text, after=punctuated, llm=LLM)]],
        record=False,
    )
    flag = flag_on(fixed, "五十个百分点", "numeric_fixed")
    verified = with_results(
        first, [fixed], [[Edit(stage="numeric", before=punctuated, after=fixed, llm=LLM)]], [[flag]]
    )
    (shown,) = numerals(verified, "zh")
    assert [(e.before, e.after) for e in shown.edits] == [
        ("开盘平稳。营收增长了35个百分点，利润也不错。", "开盘平稳。营收增长了50个百分点，利润也不错。")
    ]
    doc = document(source, (shown,), llm_punctuated=True)
    assert [s.text for s in doc.segments] == ["开盘平稳。", "营收增长了50个百分点，利润也不错。"]
    assert doc.segments[0].edits == ()
    assert [(e.stage, e.before, e.after) for e in doc.segments[1].edits] == [("numeric", "35", "50")]
    assert flagged(doc) == [(1, "numeric_fixed", "50个百分点")]
    assert "numeric_fixed: 五十个百分点; was 35" in render_markdown(doc)


def test_an_edit_is_rewritten_even_where_the_final_text_has_no_numeral() -> None:
    stretch = Stretch(
        text="涨了50%",
        start=0.0,
        end=2.0,
        edits=(Edit(stage="numeric", before="涨了百分之五", after="涨了50%", llm=LLM),),
    )
    (shown,) = numerals((stretch,), "zh")
    assert shown.text == "涨了50%" and shown.edits[0].before == "涨了5%"


def test_the_numerals_stage_leaves_other_languages_alone() -> None:
    stretch = Stretch(text="三万美元 is 30,000 dollars", start=0.0, end=2.0)
    assert numerals((stretch,), "en") == (stretch,)
    assert numerals((stretch,), "zh")[0].text == "3万美元 is 30,000 dollars"


_PHRASES = (
    *("收入三十五万美元", "第一季度", "涨了百分之八十", "三到五个", "下午三点四十分", "二〇二六年三月五日"),
    *("一亿五千三", "三块五", "涨了二十四点", "两千八百点", "一个", "三家公司", "四千", "4千", "1.5千万", "的"),
    *("我们", "看", "ETF", "三百ETF", "两点半", "百分之三到五", "三、四、五个", "一到三月份", "十二点", "零点五"),
    *("两块钱", "3", "5", "八点零八分", "三万五到四万"),
)


def test_the_stage_never_adds_interpolation_or_loses_a_flag_however_words_split() -> None:
    rng = random.Random(15835)
    for _ in range(400):
        text = "".join(rng.choice(_PHRASES) for _ in range(rng.randint(2, 4)))
        pieces: list[str] = []
        while sum(map(len, pieces)) < len(text):
            start = sum(map(len, pieces))
            pieces.append(text[start : start + rng.randint(1, 3)])
        kept = [piece for piece in pieces if rng.random() >= 0.1]
        words = tuple(Word(text=p, start=1.0 + k * 0.5, end=1.4 + k * 0.5, conf=0.9) for k, p in enumerate(kept))
        source = source_of([row(text, 0.0, 2.0 + len(kept) * 0.5, words)])
        first, *rest = source.stretches
        flags = tuple(
            Flag.model_validate(
                {"kind": "numeric_single_source", "char_start": t.char_start, "char_end": t.char_end, "note": t.surface}
            )
            for t in quantity_tokens(first.text)
        )
        stretches = (first.model_copy(update={"flags": flags}), *rest)
        shown = numerals(stretches, "zh")
        interpolated = [
            sum(k == "timing_interpolated" for _, k, _ in flagged(document(source, s))) for s in (stretches, shown)
        ]
        assert interpolated[1] <= interpolated[0], text
        assert len(shown[0].flags) == len(flags), text
        assert all(a.start <= b.start for s in shown for a, b in itertools.pairwise(s.words)), text


@pytest.mark.parametrize(
    ("before", "after", "shown_before", "shown_after"),
    [
        ("收入三万五美元", "收入三万美元", "收入3.5万美元", "收入3万美元"),
        ("涨了三十五个百分点", "涨了三十个百分点", "涨了35个百分点", "涨了30个百分点"),
        ("一百五十块", "一百块", "150块", "100块"),
        ("跌了十五点", "跌了五点", "跌了15点", "跌了5点"),
        ("百分之三十五", "百分之三十", "35%", "30%"),
        ("两点十五分", "两点十分", "2点15分", "2点10分"),
        ("三万五千", "三万", "3.5万", "3万"),
        # The final text's 300 would fuse with the 5 the edit removed, so the earlier text keeps its own form.
        ("5三百块", "，三百块", "5三百块", "，300块"),
    ],
)
def test_a_figure_the_edit_changed_reads_its_own_value_in_the_rewritten_edit(
    before: str, after: str, shown_before: str, shown_after: str
) -> None:
    stretch = Stretch(
        text=after, start=0.0, end=1.0, edits=(Edit(stage="numeric", before=before, after=after, llm=LLM),)
    )
    (shown,) = numerals((stretch,), "zh")
    (edit,) = shown.edits
    assert (edit.before, edit.after) == (shown_before, shown_after)
    assert readings(edit.before) == readings(before) and edit.after == shown.text


def test_rewrites_carried_along_a_chain_of_edits_keep_every_text_reading_its_values() -> None:
    chain = ["指数张了两点，收入三万五", "指数涨了两点，收入三万五", "指数涨了两点，收入三万"]
    edits = tuple(
        Edit(stage=stage, before=chain[i], after=chain[i + 1], llm=LLM)
        for i, stage in enumerate(("correction", "numeric"))
    )
    (shown,) = numerals((Stretch(text=chain[-1], start=0.0, end=1.0, edits=edits),), "zh")
    assert [(e.before, e.after) for e in shown.edits] == [
        ("指数张了2点，收入3.5万", "指数涨了2点，收入3.5万"),
        ("指数涨了2点，收入3.5万", "指数涨了2点，收入3万"),
    ]
    assert [readings(e.before) for e in shown.edits] == [readings(text) for text in chain[:-1]]


@pytest.mark.parametrize(
    ("spoken", "fixed", "figure", "footnote"),
    [
        ("营收增长了三十五个百分点。", "营收增长了三十个百分点。", "三十个百分点", "; was 35"),
        ("收入三万五美元。", "收入三万美元。", "三万美元", "; was 3.5"),
        ("指数跌了十五点。", "指数跌了五点。", "五点", "; was 15"),
    ],
)
def test_a_numeric_fix_that_shortens_a_figure_shows_the_figure_it_replaced(
    spoken: str, fixed: str, figure: str, footnote: str
) -> None:
    source = source_of([row(spoken, 0.0, 5.0)])
    (stretch,) = source.stretches
    verified = with_results(
        (stretch,),
        [fixed],
        [[Edit(stage="numeric", before=stretch.text, after=fixed, llm=LLM)]],
        [[flag_on(fixed, figure, "numeric_fixed")]],
    )
    doc = document(source, numerals(verified, "zh"))
    assert footnote in render_markdown(doc)


def lid_row(chunk: int, text: str, start: float, end: float, language: str) -> ChunkHypothesis:
    return ChunkHypothesis(
        hypothesis="w-runs-lid", file="synthRec001", chunk=chunk, start=start, end=end, text=text, language=language
    )


EN_DOMINANT = [
    lid_row(0, "so we met today and talked about the budget for next year", 0.0, 6.0, "en"),
    lid_row(1, "我們開會了 嗯 三個人都在", 6.0, 9.0, "zh"),
    lid_row(2, "and then everyone went back to work after lunch", 9.0, 14.0, "en"),
]
ZH_DOMINANT = [
    lid_row(0, "今天我们开会讨论明年的预算大家都来了", 0.0, 6.0, "zh"),
    lid_row(1, "um I think so", 6.0, 8.0, "en"),
]


def lid_source(rows: list[ChunkHypothesis], *, stretch_languages: bool = True) -> Transcribed:
    return from_asr_rows(rows, registry.get("w-runs-lid"), None, stretch_languages=stretch_languages)


def test_a_stretch_decoded_in_another_language_carries_it_and_is_cut_at_the_change() -> None:
    source = lid_source(EN_DOMINANT)
    assert source.language == "en"
    assert [(s.text, s.language) for s in source.stretches] == [
        ("so we met today and talked about the budget for next year", None),
        ("我們開會了 嗯 三個人都在", "zh"),
        ("and then everyone went back to work after lunch", None),
    ]
    assert all(s.language is None for s in lid_source(EN_DOMINANT, stretch_languages=False).stretches)


def test_a_mandarin_stretch_in_an_english_recording_gets_mandarin_normalize_and_numerals() -> None:
    source = lid_source(EN_DOMINANT)
    normalized = normalize(source.stretches, source.language, script="zh-Hans", fillers="remove")
    assert normalized[1].text == "我们开会了 三个人都在", "Simplified, and the 嗯 at a pause goes"
    assert numerals(normalized, source.language)[1].text == "我们开会了 3个人都在"
    assert numerals(normalized, source.language)[0] == normalized[0]
    doc = document(source, numerals(normalized, source.language), "en")
    assert [s.text for s in doc.segments][1] == "我们开会了 3个人都在"
    assert doc.transcript.script == "zh-Hans"


def test_traditional_words_in_an_english_stretch_are_converted_with_their_text() -> None:
    pieces = [" they", " said", " 這個", " 軟體", " ok"]
    words = tuple(Word(text=p, start=1.0 + i, end=1.5 + i, conf=0.9) for i, p in enumerate(pieces))
    source = source_of([row("".join(pieces), 0.0, 10.0, words, "en")], "en")
    (stretch,) = source.stretches
    (normal,) = normalize((stretch,), "en", script="zh-Hans", fillers="keep")
    assert normal.text == "they said 这个 软体 ok"
    assert [w.text for w in normal.words] == ["they", "said", "这个", "软体", "ok"]
    assert [(w.start, w.end, w.conf) for w in normal.words] == [(w.start, w.end, w.conf) for w in stretch.words]
    assert len(normal.words) == len(pieces)
    (kept,) = normalize((stretch,), "en", script="keep", fillers="keep")
    assert kept.words == stretch.words


def test_an_english_stretch_in_a_mandarin_recording_keeps_english_rules() -> None:
    source = lid_source(ZH_DOMINANT)
    assert source.language == "zh" and [s.language for s in source.stretches] == [None, "en"]
    normalized = normalize(source.stretches, source.language, script="zh-Hans", fillers="remove")
    assert normalized[1].text == "I think so", "an English hesitation goes by the English rule"
    assert numerals((Stretch(text="三个 one", start=0.0, end=1.0, language="en"),), "zh")[0].text == "三个 one"


def test_a_stretch_without_a_language_or_speaker_dumps_as_before() -> None:
    plain = Stretch(text="x", start=0.0, end=1.0)
    assert "language" not in plain.model_dump() and "speaker" not in plain.model_dump()
    assert Stretch(text="x", start=0.0, end=1.0, language="zh").model_dump()["language"] == "zh"


SPEECH_HEAVY_ZH = [
    lid_row(0, "我们开会", 0.0, 6.0, "zh"),
    lid_row(1, "讨论预算", 6.0, 12.0, "zh"),
    lid_row(2, "and that was the whole meeting really", 12.0, 15.0, "en"),
]


def test_a_recordings_document_language_is_the_one_spoken_longest_not_the_one_with_most_characters() -> None:
    source = lid_source(SPEECH_HEAVY_ZH)
    assert source.language == "zh"
    assert [s.language for s in source.stretches] == [None, "en"]
    assert lid_source(SPEECH_HEAVY_ZH, stretch_languages=False).language == "en", "YouTube still counts characters"


def test_a_mixed_document_lists_its_languages_by_speech_and_reads_script_and_punctuation_per_language() -> None:
    source = lid_source(EN_DOMINANT)
    stretches = normalize(source.stretches, source.language, script="zh-Hans", fillers="keep")
    punctuated = [s.model_copy(update={"text": s.text.capitalize() + "."}) for s in stretches]
    punctuated[1] = punctuated[1].model_copy(update={"text": "我们开会了，嗯，三个人都在。"})
    doc = document(source, tuple(punctuated), "en", llm_punctuated=True)
    assert doc.transcript.language == "en" and doc.transcript.languages == ("en", "zh")
    assert (doc.transcript.script, doc.transcript.punctuation) == ("zh-Hans", "llm")
    unpunctuated = document(source, stretches, "en")
    assert unpunctuated.transcript.punctuation == "none"
    single = lid_source(EN_DOMINANT[:1])
    assert document(single, single.stretches, "en").transcript.languages == ("en",)


def test_a_row_with_nothing_to_read_takes_no_language_of_its_own() -> None:
    rows = [lid_row(0, "我们今天讨论明年的预算。", 0.0, 6.0, "zh"), lid_row(1, "。", 6.0, 6.5, "en")]
    source = lid_source(rows)
    assert [(s.text, s.language) for s in source.stretches] == [("我们今天讨论明年的预算。。", None)]
