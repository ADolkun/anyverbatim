import json
import zlib
from pathlib import Path

import numpy as np
import pytest
from fake_engine import RATE, write_silence
from numpy.typing import NDArray
from pydantic import JsonValue
from scripted_engine import LOOP_TEXT, ScriptedEngine, WindowedEngine, looping, said, sound, spaced, timed, windowed

from anyverbatim.asr import worker
from anyverbatim.asr.chunks import Chunk
from anyverbatim.asr.guard import MIN_REDECODE_S, QualityGuard, Trigger, triggers
from anyverbatim.asr.models import HYPOTHESES_FILE, AnomalySpan, ChunkHypothesis, FileJob, WorkerRequest
from anyverbatim.asr.protocol import AsrContext, ChunkInput, EngineDef, EngineSpec, GuardedChunk, RawChunk, Window
from anyverbatim.asr.quality import COMPRESSION_RATIO_THRESHOLD, compression_ratio
from anyverbatim.contract.models import Word
from anyverbatim.errors import EngineError

DURATION = 10.0
# Compresses past Whisper's 2.4 when repeated four times, yet is too few repeats for the repetition detectors.
PHRASE = "the market rallied on strong earnings, "


def tone(seconds: float, *, quiet: tuple[float, float] | None = None) -> NDArray[np.float32]:
    t = np.arange(round(seconds * RATE)) / RATE
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if quiet is not None:
        audio[round(quiet[0] * RATE) : round(quiet[1] * RATE)] = 0.0
    return audio


def chunk_input(
    audio: NDArray[np.float32], *, language: str | None = "en", context: AsrContext | None = None
) -> ChunkInput:
    return ChunkInput(audio, 100.0, 100.0 + len(audio) / RATE, language, context)


def collapsed(count: int = 12) -> tuple[Word, ...]:
    return tuple(Word(text=f"unique{i}", start=1.0, end=1.0) for i in range(count))


@pytest.mark.parametrize(
    ("raw", "duration", "expected"),
    [
        (sound(), DURATION, ()),
        (RawChunk("fine", generated_tokens=64, max_tokens=64), DURATION, ("token_cap",)),
        (RawChunk("fine", generated_tokens=64), DURATION, ()),
        (RawChunk("fine", max_tokens=64), DURATION, ()),
        (RawChunk(LOOP_TEXT), DURATION, ("repetition_loop", "compression_ratio")),
        (RawChunk("风险限制" * 8), DURATION, ("repetition_loop", "compression_ratio")),
        (RawChunk(PHRASE * 4), DURATION, ("compression_ratio",)),
        (RawChunk("收益率曲线倒挂意味着衰退风险" * 4), DURATION, ("compression_ratio",)),
        (RawChunk("。。。"), DURATION, ("empty_output",)),
        (RawChunk(""), DURATION, ("empty_output",)),
        (RawChunk("。。。"), MIN_REDECODE_S - 0.1, ()),
        (RawChunk(""), MIN_REDECODE_S - 0.1, ()),
        (RawChunk("fine", words=collapsed()), DURATION, ("word_timing_collapse",)),
        (RawChunk("fine", words=collapsed(9)), DURATION, ()),
        (RawChunk("damaged � text"), DURATION, ("replacement_character",)),
        (
            RawChunk(LOOP_TEXT, generated_tokens=64, max_tokens=64, words=collapsed()),
            DURATION,
            ("token_cap", "repetition_loop", "compression_ratio", "word_timing_collapse"),
        ),
    ],
)
def test_trigger_matrix(raw: RawChunk, duration: float, expected: tuple[Trigger, ...]) -> None:
    assert triggers(raw, duration=duration) == expected


def test_compression_ratio_matches_whisper() -> None:
    text = "风险限制" * 8
    data = text.encode("utf-8")
    assert compression_ratio(text) == len(data) / len(zlib.compress(data))


def spoken(texts: list[str], *, seconds_each: float) -> tuple[Word, ...]:
    return tuple(Word(text=text, start=i * seconds_each, end=(i + 1) * seconds_each) for i, text in enumerate(texts))


def test_a_long_decode_is_measured_per_thirty_second_window_of_its_words() -> None:
    # Ten distinct 12 s sentences: each window compresses poorly, though the whole text repeats enough to cross 2.4.
    sentences = [f"{PHRASE.strip()} number {i} " for i in range(10)]
    whole = RawChunk("".join(sentences) * 3, words=spoken(sentences * 3, seconds_each=12.0))
    assert compression_ratio(whole.text) > COMPRESSION_RATIO_THRESHOLD
    assert "compression_ratio" not in triggers(whole, duration=360.0)
    looped = (*sentences, *[PHRASE] * 8, *sentences)
    assert "compression_ratio" in triggers(
        RawChunk("".join(looped), words=spoken(list(looped), seconds_each=3.0)), duration=3.0 * len(looped)
    )


def test_a_long_decode_without_word_times_is_not_measured_whole() -> None:
    assert "compression_ratio" not in triggers(RawChunk(PHRASE * 40), duration=360.0)


def test_dense_punctuation_words_trigger_even_when_the_text_has_letters() -> None:
    words = tuple(Word(text="!", start=1 + i * 0.5, end=1 + i * 0.5 + 0.01) for i in range(10))
    words = (*words[:-1], Word(text="!", start=5.5, end=6.0))
    assert triggers(RawChunk("real text !!!!!!!!!!", words=words), duration=DURATION) == ("nonlexical_output",)


def test_a_sound_decode_is_returned_untouched_without_decoding_again() -> None:
    engine = ScriptedEngine([sound("never asked")])
    raw = sound()
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), raw)
    assert guarded.raw is raw
    assert guarded.flags == ()
    assert engine.heard == []


def test_a_passing_redecode_replaces_the_whole_chunk() -> None:
    first = RawChunk(
        "你好",
        language="zh",
        generated_tokens=4,
        max_tokens=64,
        confidence=0.9,
        words=(Word(text="你好", start=0, end=1),),
    )
    second = RawChunk(
        "世界", generated_tokens=6, max_tokens=64, confidence=0.5, words=(Word(text="世界", start=0.2, end=0.8),)
    )
    audio = tone(DURATION, quiet=(4.9, 5.1))
    engine = ScriptedEngine([first, second])
    guarded = QualityGuard()(engine, chunk_input(audio, language="zh"), looping())
    assert guarded.flags == ("redecoded",)
    merged = guarded.raw
    assert merged.text == "你好世界"
    assert merged.language == "zh"
    assert (merged.generated_tokens, merged.max_tokens) == (10, 128)
    cut = len(engine.heard[0]) / RATE
    assert merged.confidence == pytest.approx((0.9 * cut + 0.5 * (DURATION - cut)) / DURATION)
    assert [(w.text, w.start, w.end) for w in merged.words] == [
        ("你好", 0, 1),
        ("世界", pytest.approx(cut + 0.2), pytest.approx(cut + 0.8)),
    ]


def test_english_halves_are_joined_with_a_space() -> None:
    engine = ScriptedEngine([sound("first half."), sound(" second half")])
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), looping())
    assert guarded.raw.text == "first half. second half"


def test_halves_come_from_the_chunks_own_audio_cut_at_the_quietest_point_near_the_middle() -> None:
    audio = tone(DURATION, quiet=(4.2, 4.5))
    engine = ScriptedEngine([sound()])
    QualityGuard()(engine, chunk_input(audio), looping())
    first, second = engine.heard
    assert np.array_equal(np.concatenate([first, second]), audio)
    assert np.shares_memory(first, audio)
    assert np.shares_memory(second, audio)
    assert 4.2 <= len(first) / RATE <= 4.5


def test_energy_ties_cut_nearest_the_middle_and_the_cut_stays_in_the_middle_window() -> None:
    engine = ScriptedEngine([sound()])
    QualityGuard()(engine, chunk_input(np.zeros(RATE * 10, dtype=np.float32)), looping())
    assert len(engine.heard[0]) == pytest.approx(5 * RATE, abs=RATE // 10)
    loud_middle = tone(DURATION, quiet=(0.0, 1.0))
    engine = ScriptedEngine([sound()])
    QualityGuard()(engine, chunk_input(loud_middle), looping())
    assert 3.0 <= len(engine.heard[0]) / RATE <= 7.0


def test_the_redecode_reuses_the_language_and_context() -> None:
    seen: list[tuple[str | None, AsrContext | None]] = []

    class Spy(ScriptedEngine):
        def transcribe(
            self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None
        ) -> RawChunk:
            seen.append((language, context))
            return super().transcribe(audio, language=language, context=context)

    context = AsrContext(system_prompt="finance")
    QualityGuard()(Spy([sound()]), chunk_input(tone(DURATION), language="zh", context=context), looping())
    assert seen == [("zh", context)] * 2


def test_a_still_failing_redecode_is_the_only_one_and_the_chunk_is_flagged() -> None:
    engine = ScriptedEngine([looping()])
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), looping())
    assert len(engine.heard) == 2
    assert guarded.flags == ("redecoded", "decoder_anomaly", "token_cap_hit")


def test_the_attempt_with_fewer_failures_is_kept() -> None:
    worse = RawChunk(LOOP_TEXT, generated_tokens=64, max_tokens=64)
    engine = ScriptedEngine([RawChunk(LOOP_TEXT, generated_tokens=5, max_tokens=64), sound()])
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), worse)
    assert guarded.flags == ("redecoded", "decoder_anomaly")
    assert LOOP_TEXT in guarded.raw.text
    assert guarded.raw.text.endswith("a sound decode")


def test_empty_output_is_decoded_again_and_flagged_apart_from_decoder_failures() -> None:
    engine = ScriptedEngine([RawChunk("")])
    guarded = QualityGuard()(engine, chunk_input(tone(DURATION)), RawChunk(""))
    assert len(engine.heard) == 2
    assert guarded.flags == ("redecoded", "empty_output")
    rescued = QualityGuard()(ScriptedEngine([sound("left"), sound("right")]), chunk_input(tone(DURATION)), RawChunk(""))
    assert (rescued.raw.text, rescued.flags) == ("left right", ("redecoded",))


def test_a_short_empty_chunk_is_left_alone() -> None:
    engine = ScriptedEngine([sound()])
    guarded = QualityGuard()(engine, chunk_input(tone(MIN_REDECODE_S - 0.1)), RawChunk(""))
    assert (engine.heard, guarded.flags) == ([], ())


def test_a_whole_call_decode_without_windows_is_only_flagged() -> None:
    engine = ScriptedEngine([sound()])
    raw = looping()
    guarded = QualityGuard(whole_call=True)(engine, chunk_input(tone(DURATION)), raw)
    assert engine.heard == []
    assert guarded.raw is raw
    assert (guarded.flags, guarded.anomaly_spans) == (("decoder_anomaly", "token_cap_hit"), ())
    empty = QualityGuard(whole_call=True)(engine, chunk_input(tone(DURATION)), RawChunk(""))
    assert (engine.heard, empty.flags) == ([], ("empty_output",))
    sound_decode = sound()
    assert QualityGuard(whole_call=True)(engine, chunk_input(tone(DURATION)), sound_decode).raw is sound_decode


def test_text_beats_silence_when_both_attempts_fail() -> None:
    original = looping()
    guarded = QualityGuard()(ScriptedEngine([RawChunk(""), RawChunk("")]), chunk_input(tone(DURATION)), original)
    assert guarded.raw is original
    assert guarded.flags == ("redecoded", "decoder_anomaly", "token_cap_hit")


def test_a_loop_across_the_cut_is_caught_in_the_merged_text() -> None:
    half = " ".join(["repeat risk"] * 5)
    guarded = QualityGuard()(ScriptedEngine([sound(half)]), chunk_input(tone(DURATION)), looping())
    assert guarded.flags == ("redecoded", "decoder_anomaly")


def test_a_chunk_too_short_to_split_is_flagged_without_decoding_again() -> None:
    engine = ScriptedEngine([sound()])
    guarded = QualityGuard()(engine, chunk_input(tone(MIN_REDECODE_S - 0.1)), RawChunk(LOOP_TEXT))
    assert engine.heard == []
    assert guarded.flags == ("decoder_anomaly",)


CALL_S = 32.0
WINDOW_2 = (8.0, 16.0)
# Inside window 2's middle 40%, so its halves are cut here.
QUIET = (11.0, 11.3)


def four_windows(second: list[str] | None = None, fourth: list[str] | None = None) -> RawChunk:
    return windowed(
        (0.0, 8.0, spaced("alpha bravo charlie")),
        (*WINDOW_2, second or spaced(LOOP_TEXT)),
        (16.0, 24.0, spaced("delta echo foxtrot")),
        (24.0, CALL_S, fourth or spaced("golf hotel india")),
    )


def whole_call(engine: ScriptedEngine, raw: RawChunk, *, language: str | None = "en") -> GuardedChunk:
    return QualityGuard(whole_call=True)(engine, chunk_input(tone(CALL_S, quiet=QUIET), language=language), raw)


def test_a_sound_whole_call_decode_is_returned_untouched_without_decoding_again() -> None:
    engine = WindowedEngine([], [said(" never asked")])
    raw = four_windows(second=spaced("kilo lima mike"))
    guarded = whole_call(engine, raw)
    assert guarded.raw is raw
    assert (guarded.flags, guarded.anomaly_spans) == ((), ())
    assert engine.windows_heard == []


def test_a_looped_window_is_decoded_again_as_halves_and_spliced_back() -> None:
    first = RawChunk("left", words=(Word(text=" left", start=0.0, end=1.0, conf=0.5),))
    # No leading space on the second half's first word: the seam still gets exactly one.
    second = RawChunk("right", words=(Word(text="right", start=0.5, end=1.5, conf=1.0),))
    engine = WindowedEngine([], [first, second])
    audio = tone(CALL_S, quiet=QUIET)
    raw = four_windows()
    guarded = QualityGuard(whole_call=True)(engine, chunk_input(audio), raw)

    (left, left_language), (right, right_language) = engine.windows_heard
    window = audio[round(WINDOW_2[0] * RATE) : round(WINDOW_2[1] * RATE)]
    assert np.array_equal(np.concatenate([left, right]), window)
    assert QUIET[0] <= WINDOW_2[0] + len(left) / RATE <= QUIET[1]
    assert (left_language, right_language) == ("en", "en")
    assert engine.heard == []

    cut = WINDOW_2[0] + len(left) / RATE
    repaired = guarded.raw
    assert repaired.words[:3] == raw.words[:3]
    assert repaired.words[5:] == raw.words[23:]
    assert [(w.text, w.start, w.end) for w in repaired.words[3:5]] == [
        (" left", 8.0, 9.0),
        ("right", pytest.approx(cut + 0.5, abs=1e-3), pytest.approx(cut + 1.5, abs=1e-3)),
    ]
    assert repaired.text == "alpha bravo charlie left right delta echo foxtrot golf hotel india"
    assert repaired.confidence == 0.75
    assert (repaired.language, repaired.windows) == ("en", ())
    assert (guarded.flags, guarded.anomaly_spans) == (("redecoded",), ())


def test_halves_that_still_loop_are_kept_on_a_tie_and_their_window_is_the_span() -> None:
    engine = WindowedEngine([], [said(*spaced(LOOP_TEXT))])
    guarded = whole_call(engine, four_windows())
    assert len(engine.windows_heard) == 2
    assert guarded.raw.text.count("repeat risk") == 20
    assert (guarded.flags, guarded.anomaly_spans) == (("redecoded", "decoder_anomaly"), (WINDOW_2,))


def test_the_original_window_is_kept_when_its_halves_are_worse() -> None:
    # The same loop, with its word times collapsed as well.
    worse = RawChunk(LOOP_TEXT, words=tuple(Word(text=text, start=1.0, end=1.0) for text in spaced(LOOP_TEXT)))
    raw = four_windows()
    guarded = whole_call(WindowedEngine([], [worse]), raw)
    assert guarded.raw is raw
    assert (guarded.flags, guarded.anomaly_spans) == (("redecoded", "decoder_anomaly"), (WINDOW_2,))


def test_each_looped_window_is_decoded_again_once() -> None:
    engine = WindowedEngine([], [said(" left"), said(" right"), said(" up"), said(" down")])
    audio = tone(CALL_S, quiet=QUIET)
    guarded = QualityGuard(whole_call=True)(engine, chunk_input(audio), four_windows(fourth=spaced(LOOP_TEXT)))
    heard = [samples for samples, _ in engine.windows_heard]
    assert len(heard) == 4
    assert np.array_equal(np.concatenate(heard[:2]), audio[8 * RATE : 16 * RATE])
    assert np.array_equal(np.concatenate(heard[2:]), audio[24 * RATE : 32 * RATE])
    assert guarded.raw.text == "alpha bravo charlie left right delta echo foxtrot up down"
    assert (guarded.flags, guarded.anomaly_spans) == (("redecoded",), ())


def spaced_words(text: str, start: float) -> tuple[Word, ...]:
    return tuple(Word(text=f" {w}", start=start + i, end=start + i + 0.5) for i, w in enumerate(text.split()))


def test_a_failure_no_single_window_explains_flags_the_whole_decode_without_decoding_again() -> None:
    # Five words piled on the seam from each side: a timing collapse only when both windows are read together.
    piled = [Word(text=f" unique{i}", start=8.0, end=8.0) for i in range(10)]
    words = (*spaced_words("alpha bravo", 0.0), *piled, *spaced_words("delta echo", 10.0))
    raw = RawChunk(
        text="".join(word.text for word in words).strip(),
        language="en",
        words=words,
        windows=(Window(0.0, 8.0, 7), Window(8.0, 16.0, 7)),
    )
    assert triggers(raw, duration=CALL_S) == ("word_timing_collapse",)
    engine = WindowedEngine([], [said(" never asked")])
    guarded = whole_call(engine, raw)
    assert guarded.raw is raw
    assert (guarded.flags, guarded.anomaly_spans) == (("decoder_anomaly",), ())
    assert engine.windows_heard == []


def test_windows_from_an_engine_that_cannot_decode_one_fail_loud() -> None:
    with pytest.raises(EngineError, match="transcribe_window"):
        whole_call(ScriptedEngine([sound()]), four_windows())


def test_window_word_counts_that_miss_words_fail_loud() -> None:
    raw = four_windows()
    short = RawChunk(raw.text, language="en", words=raw.words[:-1], windows=raw.windows)
    with pytest.raises(EngineError, match=f"hold {len(raw.words)} words, but it returned {len(raw.words) - 1}"):
        whole_call(WindowedEngine([], [sound()]), short)


def test_a_decode_made_without_a_hint_is_repaired_in_the_language_it_detected() -> None:
    engine = WindowedEngine([], [said(" left"), said(" right")])
    raw = four_windows()
    whole_call(engine, RawChunk(raw.text, language="nn", words=raw.words, windows=raw.windows), language=None)
    assert [language for _, language in engine.windows_heard] == ["nn", "nn"]


def test_a_window_too_short_to_split_is_only_flagged_with_its_span() -> None:
    raw = windowed((0.0, 8.0, spaced("alpha bravo charlie")), (8.0, 9.5, spaced(LOOP_TEXT)))
    engine = WindowedEngine([], [said(" never asked")])
    guarded = whole_call(engine, raw)
    assert engine.windows_heard == []
    assert guarded.raw is raw
    assert (guarded.flags, guarded.anomaly_spans) == (("decoder_anomaly",), ((8.0, 9.5),))


def test_halves_with_text_but_no_words_are_judged_on_the_words_the_splice_would_write() -> None:
    raw = four_windows()
    engine = WindowedEngine([], [RawChunk(" left side speech"), RawChunk(" right side speech")])
    guarded = whole_call(engine, raw)
    assert len(engine.windows_heard) == 2
    assert guarded.raw is raw
    assert (guarded.flags, guarded.anomaly_spans) == (("redecoded", "decoder_anomaly"), (WINDOW_2,))


def test_a_failure_outside_the_failing_windows_leaves_the_whole_row_flagged() -> None:
    # Window 0 loops and stays looped; five words from each side piled on the 16 s seam collapse only together.
    looped = timed(spaced(LOOP_TEXT), 0.0, 8.0)
    piled = [Word(text=f" pile{i}", start=16.0, end=16.0) for i in range(10)]
    before = (*timed(spaced("alpha bravo"), 8.0, 15.0), *piled[:5])
    after = (*piled[5:], *timed(spaced("delta echo"), 17.0, 24.0))
    words = (*looped, *before, *after)
    raw = RawChunk(
        "".join(word.text for word in words).strip(),
        language="en",
        words=words,
        windows=(Window(0.0, 8.0, len(looped)), Window(8.0, 16.0, len(before)), Window(16.0, 24.0, len(after))),
    )
    guarded = whole_call(WindowedEngine([], [said(*spaced(LOOP_TEXT))]), raw)
    assert "word_timing_collapse" in triggers(guarded.raw, duration=CALL_S)
    assert (guarded.flags, guarded.anomaly_spans) == (("redecoded", "decoder_anomaly"), ())


def test_the_first_window_is_repaired_like_any_other() -> None:
    raw = windowed((0.0, 8.0, spaced(LOOP_TEXT)), (8.0, 16.0, spaced("delta echo foxtrot")))
    engine = WindowedEngine([], [said(" left"), said(" right")])
    audio = tone(16.0, quiet=(3.0, 3.3))
    guarded = QualityGuard(whole_call=True)(engine, chunk_input(audio), raw)
    left, right = (samples for samples, _ in engine.windows_heard)
    assert np.array_equal(np.concatenate([left, right]), audio[: 8 * RATE])
    assert guarded.raw.text == "left right delta echo foxtrot"
    assert (guarded.flags, guarded.anomaly_spans) == (("redecoded",), ())


def test_a_whole_call_repaired_into_silence_is_marked_empty_like_a_chunk() -> None:
    garbage = windowed((0.0, 8.0, [" !!!"] * 40))
    guarded = QualityGuard(whole_call=True)(WindowedEngine([], [RawChunk("")]), chunk_input(tone(8.0)), garbage)
    assert (guarded.raw.text, guarded.flags, guarded.anomaly_spans) == ("", ("redecoded", "empty_output"), ())


@pytest.mark.parametrize(
    ("before", "halves", "after", "text"),
    [
        # A half's first word with no leading space still gets one at an English seam.
        (["Alpha", " bravo"], [" left", "right"], [" delta"], "Alpha bravo left right delta"),
        # Mandarin seams take no space, but a space Whisper gave a word stays, as in Whisper's own text.
        (["你好", "世界"], ["左边", " 右边"], ["再见"], "你好世界左边 右边再见"),
        (["你好", "世界"], [" Apple", " stock"], [" OK"], "你好世界 Apple stock OK"),
        (["你好", "世界"], ["Apple", "stock"], ["再见"], "你好世界Apple stock再见"),
        ([" alpha", " bravo"], ["你好", "世界"], [" delta"], "alpha bravo你好世界 delta"),
    ],
)
def test_seams_add_a_space_only_where_neither_side_has_one_and_neither_is_wide(
    before: list[str], halves: list[str], after: list[str], text: str
) -> None:
    raw = windowed((0.0, 8.0, before), (*WINDOW_2, list("风险限制" * 8)), (16.0, 24.0, after))
    guarded = whole_call(WindowedEngine([], [said(halves[0]), said(halves[1])]), raw)
    assert guarded.raw.text == text
    assert guarded.flags == ("redecoded",)


def request(tmp_path: Path, wav: Path, *, always: bool = False, guard: bool = True) -> WorkerRequest:
    spec = EngineSpec("scripted", "fake", "example/fake", "0" * 40, 30.0, False, False, False, RATE)
    definition = EngineDef(spec, "scripted_engine.create", {"always": always, "language": "pass"})
    job = FileJob(id="f1", wav=str(wav), language="en", chunks=(Chunk(0.0, 6.0), Chunk(7.0, 9.0)))
    return WorkerRequest(engine=definition, out_dir=str(tmp_path / "out"), files=(job,), quality_guard=guard)


def rows(tmp_path: Path) -> list[ChunkHypothesis]:
    lines = (tmp_path / "out" / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    return [ChunkHypothesis.model_validate_json(line) for line in lines]


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return write_silence(tmp_path / "in.wav", 10.0)


def test_run_request_redecodes_a_chunk_whose_first_decode_loops(tmp_path: Path, wav: Path) -> None:
    req = request(tmp_path, wav)
    engine = ScriptedEngine([looping(), sound("left"), sound("right"), sound("untouched")])
    metrics = worker.run_request(req, engine=engine, guard=QualityGuard())
    first, second = rows(tmp_path)
    assert (first.text, first.flags) == ("left right", ("redecoded",))
    assert (first.start, first.end) == (0.0, 6.0)
    assert (second.text, second.flags) == ("untouched", ())
    assert len(engine.heard) == 4
    assert metrics.chunks_decoded == 2


def test_run_request_flags_a_chunk_that_keeps_looping(tmp_path: Path, wav: Path) -> None:
    engine = ScriptedEngine([looping()])
    worker.run_request(request(tmp_path, wav), engine=engine, guard=QualityGuard())
    assert [r.flags for r in rows(tmp_path)] == [("token_cap_hit", "redecoded", "decoder_anomaly")] * 2
    # Per chunk: the first decode plus one re-decode of two halves, however badly the halves fare.
    assert len(engine.heard) == 2 * (1 + 2)


def whole_file_request(tmp_path: Path, wav: Path, **variant: JsonValue) -> WorkerRequest:
    spec = EngineSpec("scripted-whole", "fake", "example/fake", "0" * 40, None, False, False, False, RATE)
    definition = EngineDef(spec, "scripted_engine.create", {"always": False, "language": "pass", **variant})
    req = WorkerRequest(engine=definition, out_dir=str(tmp_path / "out"), files=(FileJob(id="f1", wav=str(wav)),))
    (tmp_path / "request.json").write_text(req.model_dump_json(), encoding="utf-8")
    return req


def test_main_only_flags_a_whole_file_hypothesis_whose_engine_reports_no_windows(tmp_path: Path, wav: Path) -> None:
    req = whole_file_request(tmp_path, wav)
    assert worker.main([str(tmp_path / "request.json")]) == 0
    (only,) = rows(tmp_path)
    assert only.text == LOOP_TEXT
    assert only.flags == ("token_cap_hit", "decoder_anomaly")
    assert only.anomaly_spans == ()
    assert worker.default_guard(req) is not None


@pytest.mark.parametrize(
    ("always", "text", "flags", "spans"),
    [
        (False, "a sound decode left right", ("redecoded",), ()),
        (
            True,
            f"a sound decode {LOOP_TEXT} {LOOP_TEXT}",
            ("redecoded", "decoder_anomaly"),
            (AnomalySpan(start=5.0, end=10.0),),
        ),
    ],
)
def test_main_repairs_the_looped_window_of_a_whole_file_hypothesis(
    tmp_path: Path,
    wav: Path,
    always: bool,
    text: str,
    flags: tuple[str, ...],
    spans: tuple[AnomalySpan, ...],
) -> None:
    whole_file_request(tmp_path, wav, windowed=True, always=always)
    assert worker.main([str(tmp_path / "request.json")]) == 0
    (only,) = rows(tmp_path)
    assert (only.text, only.flags, only.anomaly_spans) == (text, flags, spans)


def test_main_guards_by_default_and_the_request_can_opt_out(tmp_path: Path, wav: Path) -> None:
    for guard in (True, False):
        base = tmp_path / str(guard)
        base.mkdir()
        req = request(base, wav, guard=guard)
        path = base / "request.json"
        path.write_text(req.model_dump_json(), encoding="utf-8")
        assert worker.main([str(path)]) == 0
        first, second = rows(base)
        assert (first.text == LOOP_TEXT) is (not guard)
        assert first.flags == (("redecoded",) if guard else ("token_cap_hit",))
        assert second.flags == ()
    assert json.loads(path.read_text(encoding="utf-8"))["quality_guard"] is False
