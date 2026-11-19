from pathlib import Path
from typing import override

import numpy as np
import pytest
from fake_engine import definition, write_silence
from numpy.typing import NDArray
from pydantic import ValidationError
from scripted_engine import LOOP_TEXT, said, spaced, windowed

from anyverbatim import store
from anyverbatim.asr import registry, worker
from anyverbatim.asr.chunks import Chunk
from anyverbatim.asr.guard import QualityGuard
from anyverbatim.asr.models import HYPOTHESES_FILE, AnomalySpan, ChunkHypothesis, FileJob, WorkerRequest
from anyverbatim.asr.protocol import AsrContext, EngineDef, RawChunk
from anyverbatim.asr.worker import DecodeUnit, pick_language, plan_runs
from anyverbatim.contract.models import Word
from anyverbatim.errors import EngineError
from anyverbatim.mlx_worker import without_wav

ZH = {"zh": 0.6, "en": 0.3, "ja": 0.1}
EN = {"zh": 0.2, "en": 0.7, "ja": 0.1}
CHUNKS = (Chunk(1.0, 4.0), Chunk(5.0, 8.0), Chunk(9.0, 12.0, forced_cut=True), Chunk(12.0, 15.0))


class DetectingEngine:
    """Answers detection from a script, one entry per call, and records every call in order."""

    def __init__(self, detections: list[dict[str, float]] | None = None) -> None:
        self.detections = list(detections or [])
        self.calls: list[tuple[str, int, str | None]] = []

    def load(self) -> None:
        self.calls.append(("load", 0, None))

    def detect_language(self, audio: NDArray[np.float32]) -> dict[str, float]:
        self.calls.append(("detect", len(audio), None))
        return self.detections.pop(0)

    def transcribe(self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None) -> RawChunk:
        samples = len(audio)
        self.calls.append(("transcribe", samples, language))
        return RawChunk(
            text=f"{language}:{samples}",
            language=language,
            words=(Word(text="a", start=0.0, end=0.5), Word(text="b", start=1.5, end=2.0)),
        )

    def decodes(self) -> list[tuple[int, str | None]]:
        return [(samples, language) for kind, samples, language in self.calls if kind == "transcribe"]


def lid(mode: str) -> EngineDef:
    return definition(mode=mode, language="detect")


def job(wav: Path, *, language: str | None = None, languages: tuple[str, ...] = ()) -> FileJob:
    return FileJob(id="f1", wav=str(wav), language=language, chunks=CHUNKS, languages=languages)


def run(tmp_path: Path, engine: DetectingEngine, definition_: EngineDef, *jobs: FileJob) -> list[ChunkHypothesis]:
    request = WorkerRequest(engine=definition_, out_dir=str(tmp_path / "out"), files=jobs)
    worker.run_request(request, engine=engine)
    lines = (tmp_path / "out" / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    return [ChunkHypothesis.model_validate_json(line) for line in lines]


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return write_silence(tmp_path / "speech.wav", 16.0)


def test_the_argmax_is_restricted_to_the_allowed_set_with_the_unrestricted_probability() -> None:
    probabilities = {"ja": 0.5, "zh": 0.3, "en": 0.2}
    assert pick_language(probabilities, ("zh", "en")) == ("zh", 0.3)
    assert pick_language(probabilities, ()) == ("ja", 0.5)
    assert pick_language({"zh": 0.0, "en": 0.0, "ja": 1.0}, ("en", "zh")) == ("en", 0.0)


def test_an_allowed_language_the_detector_does_not_know_fails_loud() -> None:
    with pytest.raises(ValueError, match="'zz'"):
        pick_language({"zh": 1.0}, ("zh", "zz"))


def test_consecutive_same_language_chunks_merge_into_runs() -> None:
    units = [
        DecodeUnit(Chunk(1.0, 4.0), "zh", 0.9),
        DecodeUnit(Chunk(5.0, 8.0), "zh", 0.6),
        DecodeUnit(Chunk(9.0, 12.0, forced_cut=True), "en", 0.8),
        DecodeUnit(Chunk(12.0, 15.0, forced_cut=True), "zh", 0.7),
    ]
    assert plan_runs(units) == [
        DecodeUnit(Chunk(1.0, 8.0), "zh", 0.6),
        DecodeUnit(Chunk(9.0, 12.0, forced_cut=True), "en", 0.8),
        DecodeUnit(Chunk(12.0, 15.0, forced_cut=True), "zh", 0.7),
    ]
    assert plan_runs([DecodeUnit(Chunk(0.0, 1.0), "en"), DecodeUnit(Chunk(2.0, 3.0), "en", 0.5)]) == [
        DecodeUnit(Chunk(0.0, 3.0), "en")
    ]
    assert plan_runs([]) == []


def test_each_run_is_decoded_once_with_its_language_on_its_slice_with_absolute_times(tmp_path: Path, wav: Path) -> None:
    engine = DetectingEngine([ZH, ZH, EN, ZH])
    rows = run(tmp_path, engine, lid("runs"), job(wav, languages=("zh", "en")))
    assert [call[0] for call in engine.calls] == ["load", *["detect"] * 4, *["transcribe"] * 3]
    assert engine.decodes() == [(7 * 16_000, "zh"), (3 * 16_000, "en"), (3 * 16_000, "zh")]
    assert [(r.chunk, r.start, r.end, r.language, r.language_prob) for r in rows] == [
        (0, 1.0, 8.0, "zh", 0.6),
        (1, 9.0, 12.0, "en", 0.7),
        (2, 12.0, 15.0, "zh", 0.6),
    ]
    assert [(w.start, w.end) for w in rows[0].words] == [(1.0, 1.5), (2.5, 3.0)]
    assert [(w.start, w.end) for w in rows[2].words] == [(12.0, 12.5), (13.5, 14.0)]
    assert [r.flags for r in rows] == [(), ("forced_cut",), ()]


def test_detection_is_restricted_to_the_files_allowed_set(tmp_path: Path, wav: Path) -> None:
    japanese = {"zh": 0.3, "en": 0.1, "ja": 0.6}
    rows = run(tmp_path, DetectingEngine([japanese] * 4), lid("runs"), job(wav, languages=("zh", "en")))
    assert [(r.language, r.language_prob) for r in rows] == [("zh", 0.3)]
    unrestricted = run(tmp_path / "any", DetectingEngine([japanese] * 4), lid("runs"), job(wav))
    assert [r.language for r in unrestricted] == ["ja"]


def test_per_chunk_detection_decodes_every_chunk_with_its_own_language(tmp_path: Path, wav: Path) -> None:
    engine = DetectingEngine([ZH, ZH, EN, ZH])
    rows = run(tmp_path, engine, lid("chunk"), job(wav, languages=("zh", "en")))
    assert [kind for kind, _, _ in engine.calls] == ["load", *["detect", "transcribe"] * 4]
    assert engine.decodes() == [(3 * 16_000, language) for language in ("zh", "zh", "en", "zh")]
    assert [(r.start, r.end, r.language, r.language_prob) for r in rows] == [
        (1.0, 4.0, "zh", 0.6),
        (5.0, 8.0, "zh", 0.6),
        (9.0, 12.0, "en", 0.7),
        (12.0, 15.0, "zh", 0.6),
    ]


@pytest.mark.parametrize("mode", ["chunk", "runs"])
def test_a_declared_language_wins_over_detection(tmp_path: Path, wav: Path, mode: str) -> None:
    engine = DetectingEngine()
    rows = run(tmp_path, engine, lid(mode), job(wav, language="en", languages=("zh", "en")))
    assert "detect" not in [kind for kind, _, _ in engine.calls]
    assert {language for _, language in engine.decodes()} == {"en"}
    assert {r.language_prob for r in rows} == {None}
    if mode == "runs":
        assert [(r.start, r.end) for r in rows] == [(1.0, 15.0)]


def test_the_pass_policy_decodes_a_declared_language_without_detection(tmp_path: Path, wav: Path) -> None:
    engine = DetectingEngine()
    rows = run(tmp_path, engine, definition(), job(wav, language="zh"))
    assert "detect" not in [kind for kind, _, _ in engine.calls]
    assert engine.decodes() == [(3 * 16_000, "zh")] * 4
    assert [(r.language, r.language_prob) for r in rows] == [("zh", None)] * 4


def test_a_finished_runs_output_resumes_without_loading_or_detecting(tmp_path: Path, wav: Path) -> None:
    run(tmp_path, DetectingEngine([ZH, ZH, EN, ZH]), lid("runs"), job(wav))
    again = DetectingEngine()
    request = WorkerRequest(engine=lid("runs"), out_dir=str(tmp_path / "out"), files=(job(wav),))
    metrics = worker.run_request(request, engine=again)
    assert again.calls == []
    assert (metrics.chunks_decoded, metrics.chunks_skipped) == (0, 3)


def test_an_interrupted_runs_output_detects_again_and_decodes_only_the_missing_runs(tmp_path: Path, wav: Path) -> None:
    rows = run(tmp_path, DetectingEngine([ZH, ZH, EN, ZH]), lid("runs"), job(wav))
    sink = tmp_path / "out" / HYPOTHESES_FILE
    sink.write_text(rows[0].model_dump_json() + "\n", encoding="utf-8")
    again = DetectingEngine([ZH, ZH, EN, ZH])
    request = WorkerRequest(engine=lid("runs"), out_dir=str(tmp_path / "out"), files=(job(wav),))
    metrics = worker.run_request(request, engine=again)
    assert again.decodes() == [(3 * 16_000, "en"), (3 * 16_000, "zh")]
    assert (metrics.chunks_decoded, metrics.chunks_skipped) == (2, 1)


def test_runs_that_no_longer_match_the_detection_fail_loud(tmp_path: Path, wav: Path) -> None:
    rows = run(tmp_path, DetectingEngine([ZH, ZH, EN, ZH]), lid("runs"), job(wav))
    (tmp_path / "out" / HYPOTHESES_FILE).write_text(rows[0].model_dump_json() + "\n", encoding="utf-8")
    request = WorkerRequest(engine=lid("runs"), out_dir=str(tmp_path / "out"), files=(job(wav),))
    with pytest.raises(EngineError, match="plan now says"):
        worker.run_request(request, engine=DetectingEngine([ZH, EN, EN, ZH]))


def test_an_engine_that_cannot_detect_is_refused(tmp_path: Path, wav: Path) -> None:
    class Deaf:
        def load(self) -> None: ...

        def transcribe(self, audio: object, *, language: str | None, context: AsrContext | None) -> RawChunk:
            return RawChunk(text="")

    request = WorkerRequest(engine=lid("chunk"), out_dir=str(tmp_path / "out"), files=(job(wav),))
    with pytest.raises(EngineError, match="no detect_language"):
        worker.run_request(request, engine=Deaf())


def test_a_detection_failure_names_the_file_and_span(tmp_path: Path, wav: Path) -> None:
    request = WorkerRequest(engine=lid("chunk"), out_dir=str(tmp_path / "out"), files=(job(wav),))
    with pytest.raises(EngineError, match=r"language detection failed on f1 \(1\.0-4\.0s\)"):
        worker.run_request(request, engine=DetectingEngine([]))


def test_the_runs_guard_repairs_windows_like_the_whole_file_control(tmp_path: Path) -> None:
    def whole_call(hypothesis: str) -> object:
        request = WorkerRequest(engine=registry.get(hypothesis), out_dir=str(tmp_path), files=())
        return getattr(worker.default_guard(request), "_whole_call", None)

    assert [whole_call(h) for h in ("w-prod", "w-vad", "w-vad-lid", "w-runs-lid")] == [True, False, False, True]


def test_a_runs_rows_anomaly_span_is_its_window_after_the_run_start(tmp_path: Path, wav: Path) -> None:
    class LoopingRuns(DetectingEngine):
        @override
        def transcribe(
            self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None
        ) -> RawChunk:
            super().transcribe(audio, language=language, context=context)
            return windowed((0.0, 2.0, spaced("alpha bravo")), (2.0, 6.0, spaced(LOOP_TEXT)), language=language)

        def transcribe_window(self, audio: NDArray[np.float32], *, language: str | None) -> RawChunk:
            return said(*spaced(LOOP_TEXT))

    request = WorkerRequest(engine=lid("runs"), out_dir=str(tmp_path / "out"), files=(job(wav),))
    worker.run_request(request, engine=LoopingRuns([ZH, ZH, EN, EN]), guard=worker.default_guard(request))
    rows = [
        ChunkHypothesis.model_validate_json(line)
        for line in (tmp_path / "out" / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert [(row.start, row.flags, row.anomaly_spans) for row in rows] == [
        (1.0, ("redecoded", "decoder_anomaly"), (AnomalySpan(start=3.0, end=7.0),)),
        (9.0, ("redecoded", "decoder_anomaly"), (AnomalySpan(start=11.0, end=15.0),)),
    ]


def test_an_allowed_set_is_refused_by_an_engine_that_does_not_detect(wav: Path) -> None:
    for hypothesis in ("w-prod", "w-vad", "q-base"):
        with pytest.raises(ValidationError, match="does not detect languages"):
            WorkerRequest(
                engine=registry.get(hypothesis),
                out_dir="/out",
                files=(FileJob(id="f1", wav="/a.wav", languages=("zh",)),),
            )


def test_detection_needs_a_chunked_spec() -> None:
    whole = definition(max_chunk_s=None, language="detect")
    with pytest.raises(ValidationError, match="needs a max_chunk_s"):
        WorkerRequest(engine=whole, out_dir="/out", files=())


# Digests of these requests at anyverbatim a3b4b2a, before `languages` existed; a change re-decodes every video. w-vad's
# moved once, deliberately, when the chunked Whisper engines stopped sending the Mandarin prompt.
PINNED = {
    ("w-prod", True): "1221a3727a6ecbac4f22b80d412a787aca952b696b07c90e0b4937cf970ab635",
    ("w-vad", True): "5e2287329479abaf3fbb93b5dcbb1ded37b67ca5209516d0b8c5e115311e192d",
    ("w-prod", False): "53817feee8b08b35f5df4d522bdc4b3b1103022b20a06e238acf7f6d83cfaa99",
    ("w-vad", False): "cd784da7526bd03b403cdd54a36a28a6d97f632dfc9b9ad33981af517ff7f4b8",
}


@pytest.mark.parametrize(("hypothesis", "declared"), list(PINNED))
def test_the_control_fingerprints_are_unchanged(hypothesis: str, declared: bool) -> None:
    if declared:
        chunks = None if hypothesis == "w-prod" else (Chunk(0.0, 4.0), Chunk(5.0, 9.5, True))
        file = FileJob(id="abcdefghijk", wav="/a.wav", language="zh", chunks=chunks)
    else:
        file = FileJob(id="abcdefghijk", wav="/a.wav")
    request = WorkerRequest(engine=registry.get(hypothesis), out_dir="/x", files=(file,))
    fingerprint = request.fingerprint()
    assert store.digest(without_wav(fingerprint)) == PINNED[(hypothesis, declared)]
    assert "languages" not in str(fingerprint)
    assert "languages" not in request.model_dump_json()


def test_an_allowed_set_is_part_of_the_fingerprint() -> None:
    def fingerprint(languages: tuple[str, ...]) -> object:
        file = FileJob(id="f1", wav="/a.wav", languages=languages)
        return WorkerRequest(engine=registry.get("w-runs-lid"), out_dir="/x", files=(file,)).fingerprint()

    assert fingerprint(("zh", "en")) != fingerprint(())
    assert FileJob.model_validate_json(
        FileJob(id="f1", wav="/a.wav", languages=("zh",)).model_dump_json()
    ).languages == ("zh",)


def test_an_interrupted_per_chunk_output_detects_only_the_chunks_it_still_decodes(tmp_path: Path, wav: Path) -> None:
    rows = run(tmp_path, DetectingEngine([ZH, ZH, EN, ZH]), lid("chunk"), job(wav))
    sink = tmp_path / "out" / HYPOTHESES_FILE
    sink.write_text("".join(row.model_dump_json() + "\n" for row in rows[:2]), encoding="utf-8")
    again = DetectingEngine([EN, ZH])
    request = WorkerRequest(engine=lid("chunk"), out_dir=str(tmp_path / "out"), files=(job(wav),))
    metrics = worker.run_request(request, engine=again)
    assert [kind for kind, _, _ in again.calls] == ["load", "detect", "transcribe", "detect", "transcribe"]
    assert again.decodes() == [(3 * 16_000, "en"), (3 * 16_000, "zh")]
    assert (metrics.chunks_decoded, metrics.chunks_skipped) == (2, 2)


def test_the_per_chunk_guard_redecodes_the_halves_in_the_detected_language(tmp_path: Path, wav: Path) -> None:
    class SilentFirst(DetectingEngine):
        @override
        def transcribe(
            self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None
        ) -> RawChunk:
            raw = super().transcribe(audio, language=language, context=context)
            return RawChunk(text="", language=language) if len(self.decodes()) == 1 else raw

    engine = SilentFirst([EN])
    single = FileJob(id="f1", wav=str(wav), chunks=(Chunk(1.0, 4.0),), languages=("zh", "en"))
    request = WorkerRequest(engine=lid("chunk"), out_dir=str(tmp_path / "out"), files=(single,))
    worker.run_request(request, engine=engine, guard=QualityGuard())
    assert [language for _, language in engine.decodes()] == ["en", "en", "en"]
    (row,) = [
        ChunkHypothesis.model_validate_json(line)
        for line in (tmp_path / "out" / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert ("redecoded" in row.flags, row.language, row.language_prob) == (True, "en", 0.7)


def test_allowed_languages_reduce_to_primary_codes_and_unknown_ones_fail_before_any_decode() -> None:
    assert FileJob(id="f1", wav="/a.wav", languages=("zh-Hans", "EN_us", "zh")).languages == ("zh", "en")
    with pytest.raises(ValidationError, match="language codes"):
        FileJob(id="f1", wav="/a.wav", languages=("zh", " "))
    declared = FileJob(id="f1", wav="/a.wav", language="zh", languages=("zh", "xx"))
    with pytest.raises(ValidationError, match=r"cannot detect the allowed languages \['xx'\]"):
        WorkerRequest(engine=registry.get("w-runs-lid"), out_dir="/out", files=(declared,))


def test_the_whisper_language_copy_matches_the_installed_tokenizer() -> None:
    tokenizer = pytest.importorskip("mlx_whisper.tokenizer")
    assert frozenset(tokenizer.LANGUAGES) == registry.WHISPER_LANGUAGES


def test_runs_need_the_detect_policy() -> None:
    for policy in ("auto", "pass"):
        with pytest.raises(ValidationError, match="must be 'detect'"):
            WorkerRequest(engine=definition(mode="runs", language=policy), out_dir="/out", files=())


# A w-prod row as anyverbatim a3b4b2a serialized it, before `language_prob` existed.
MAIN_ROW = (
    '{"hypothesis":"w-prod","file":"abcdefghijk","chunk":0,"start":0.0,"end":12.0,"text":"示例","language":"zh",'
    '"words":[{"text":"示例","start":0.5,"end":1.0,"conf":0.75}],"generated_tokens":null,"max_tokens":null,'
    '"confidence":0.75,"flags":["decoder_anomaly"]}'
)


def test_a_control_row_serializes_as_it_did_before_language_prob() -> None:
    row = ChunkHypothesis.model_validate_json(MAIN_ROW)
    assert row.language_prob is None
    assert row.anomaly_spans == ()
    assert row.model_dump_json() == MAIN_ROW
    detected = row.model_copy(update={"language_prob": 0.5}).model_dump_json()
    assert '"language_prob":0.5' in detected


def test_anomaly_spans_are_written_only_when_there_are_some() -> None:
    row = ChunkHypothesis.model_validate_json(MAIN_ROW)
    spanned = row.model_copy(update={"anomaly_spans": (AnomalySpan(start=3.5, end=7.25),)}).model_dump_json()
    assert spanned.endswith(',"anomaly_spans":[{"start":3.5,"end":7.25}]}')
    assert ChunkHypothesis.model_validate_json(spanned).anomaly_spans == (AnomalySpan(start=3.5, end=7.25),)
