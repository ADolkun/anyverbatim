import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from fake_engine import write_silence
from pydantic import JsonValue

from anyverbatim import pipeline, store
from anyverbatim.asr import registry, worker
from anyverbatim.asr.chunks import Chunk
from anyverbatim.asr.engines import whisper_mlx
from anyverbatim.asr.engines.whisper_mlx import WhisperEngine
from anyverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, FileJob, WorkerRequest
from anyverbatim.asr.protocol import Window
from anyverbatim.errors import ConfigError, DependencyError

REPO_DIR = "models--mlx-community--whisper-large-v3-mlx"
RESULT: dict[str, Any] = {
    "text": " 示例，你好 ",
    "language": "zh",
    "segments": [
        {
            "seek": 0,
            "text": " 示例，你好",
            "words": [
                {"word": "示例，", "start": 0.5, "end": 1.0, "probability": 0.75},
                {"word": "你好", "start": 1.0, "end": 1.5, "probability": 0.25},
                {"word": "漏", "start": None, "end": None, "probability": 0.9},
            ],
        },
        {"seek": 0, "text": "x", "words": [{"word": "x", "start": 2.0, "end": 2.5}]},
    ],
}


class FakeMx:
    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []
        self.clear_error: list[Exception | None] = []
        self.float16 = object()

    def set_cache_limit(self, limit: int) -> int:
        self.events.append(("limit", limit))
        return 123

    def clear_cache(self) -> None:
        self.events.append(("clear", None))
        if self.clear_error and (error := self.clear_error.pop(0)) is not None:
            raise error


class FakeMel:
    """Records how the mel was built, so a test can check it matches mlx-whisper's own detection input."""

    def __init__(self, steps: tuple[Any, ...]) -> None:
        self.steps = steps

    def astype(self, dtype: object) -> FakeMel:
        return FakeMel((*self.steps, ("astype", dtype)))


class FakeModel:
    def __init__(self, mx: FakeMx) -> None:
        self.mx = mx
        self.dims = ModuleType("dims")
        self.dims.n_mels = 128  # type: ignore[attr-defined]
        self.probabilities = {"en": 0.25, "zh": 0.7, "yue": 0.05}
        self.mels: list[FakeMel] = []

    def detect_language(self, mel: FakeMel) -> tuple[None, dict[str, float]]:
        self.mx.events.append(("detect", None))
        self.mels.append(mel)
        return None, self.probabilities


class FakeWhisper:
    def __init__(self, mx: FakeMx) -> None:
        self.mx = mx
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.result: dict[str, Any] = RESULT
        # Returned in turn before `result`, for a test that needs successive decodes to differ.
        self.results: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.loads: list[tuple[str, object]] = []
        self.model = FakeModel(mx)

    def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
        self.mx.events.append(("transcribe", None))
        self.calls.append((audio, kwargs))
        if self.error is not None:
            raise self.error
        return self.results.pop(0) if self.results else self.result


@pytest.fixture
def mx() -> FakeMx:
    return FakeMx()


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "default-hub"
    path = cache / REPO_DIR / "snapshots" / registry.get("w-prod").spec.revision
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "weights.npz").write_bytes(b"")
    constants = ModuleType("huggingface_hub.constants")
    constants.default_cache_path = str(cache)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", ModuleType("huggingface_hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "candidates-hub"))
    return path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, mx: FakeMx, snapshot: Path) -> FakeWhisper:
    whisper = FakeWhisper(mx)
    package, transcribe_module, tokenizer, audio = (
        ModuleType("mlx_whisper"),
        ModuleType("mlx_whisper.transcribe"),
        ModuleType("mlx_whisper.tokenizer"),
        ModuleType("mlx_whisper.audio"),
    )
    package.transcribe = whisper.transcribe  # type: ignore[attr-defined]

    class ModelHolder:
        @classmethod
        def get_model(cls, model_path: str, dtype: object) -> FakeModel:
            whisper.loads.append((model_path, dtype))
            return whisper.model

    transcribe_module.ModelHolder = ModelHolder  # type: ignore[attr-defined]
    audio.SAMPLE_RATE = 16_000  # type: ignore[attr-defined]
    audio.N_SAMPLES = 480_000  # type: ignore[attr-defined]
    audio.N_FRAMES = 3000  # type: ignore[attr-defined]
    audio.FRAMES_PER_SECOND = 100  # type: ignore[attr-defined]
    audio.log_mel_spectrogram = lambda samples, n_mels, padding: FakeMel((samples, n_mels, padding))  # type: ignore[attr-defined]
    audio.pad_or_trim = lambda mel, length, *, axis: FakeMel((*mel.steps, ("trim", length, axis)))  # type: ignore[attr-defined]
    tokenizer.LANGUAGES = {"en": "english", "zh": "chinese", "yue": "cantonese"}  # type: ignore[attr-defined]
    tokenizer.TO_LANGUAGE_CODE = {"chinese": "zh", "mandarin": "zh"}  # type: ignore[attr-defined]
    core = ModuleType("mlx.core")
    core.set_cache_limit = mx.set_cache_limit  # type: ignore[attr-defined]
    core.clear_cache = mx.clear_cache  # type: ignore[attr-defined]
    core.float16 = mx.float16  # type: ignore[attr-defined]
    core.get_peak_memory = lambda: 42  # type: ignore[attr-defined]
    mlx = ModuleType("mlx")
    mlx.core = core  # type: ignore[attr-defined]
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", core),
        ("mlx_whisper", package),
        ("mlx_whisper.transcribe", transcribe_module),
        ("mlx_whisper.tokenizer", tokenizer),
        ("mlx_whisper.audio", audio),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return whisper


def loaded(hypothesis: str, fake: FakeWhisper) -> WhisperEngine:
    engine = registry.build_engine(registry.get(hypothesis))
    assert isinstance(engine, WhisperEngine)
    engine.load()
    return engine


AUDIO = np.zeros(16_000, dtype=np.float32)


@pytest.mark.parametrize(
    ("hypothesis", "mode", "prompt"),
    [
        ("w-prod", "whole_file", whisper_mlx.ZH_INITIAL_PROMPT),
        ("w-vad", "chunk", None),
        ("w-vad-lid", "chunk", None),
        ("w-runs-lid", "runs", None),
    ],
)
def test_every_mode_makes_the_production_call_and_only_whole_file_sends_the_zh_prompt(
    hypothesis: str, mode: str, prompt: str | None, fake: FakeWhisper, snapshot: Path
) -> None:
    engine = loaded(hypothesis, fake)
    assert engine.mode == mode
    engine.transcribe(AUDIO, language="zh", context=None)
    audio, kwargs = fake.calls[0]
    assert audio is AUDIO
    assert kwargs == {
        "path_or_hf_repo": str(snapshot),
        "word_timestamps": True,
        "language": "zh",
        "initial_prompt": prompt,
        "condition_on_previous_text": False,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    }


def windows(*texts: tuple[int, str]) -> dict[str, Any]:
    """A transcribe() result with one segment per `(seek, text)`, as mlx-whisper reports its windows."""
    segments = [
        {"seek": seek, "text": text, "words": [{"word": text, "start": seek / 100, "end": seek / 100 + 1.0}]}
        for seek, text in texts
    ]
    return {"text": "".join(text for _, text in texts), "language": "zh", "segments": segments}


@pytest.mark.parametrize(
    "echo",
    [
        "请使用简体中文并加上标点符号。",
        " 以下是普通话内容，请使用简体中文并加上标点符号。",
        "使用简体中文并加上标点符号。我觉得这个方案可以",
        "因此,您可以使用简体中文并加上标点符号。",
        "请帮助翻译,并且使用简体中文并加上标点符号。",
        "请使用简体中文 并加上标点符号",
        "大家好。请使用简体中文，并加上标点符号！今天我们聊聊利润",
    ],
)
def test_a_whole_file_decode_that_echoes_the_prompt_is_decoded_again_without_it(echo: str, fake: FakeWhisper) -> None:
    fake.results = [windows((0, echo), (3000, "后面是真正的内容")), RESULT]
    raw = loaded("w-prod", fake).transcribe(AUDIO, language="zh", context=None)
    assert [kwargs["initial_prompt"] for _, kwargs in fake.calls] == [whisper_mlx.ZH_INITIAL_PROMPT, None]
    first, second = (kwargs for _, kwargs in fake.calls)
    assert {**first, "initial_prompt": None} == second
    assert all(audio is AUDIO for audio, _ in fake.calls)
    assert (raw.text, raw.flags) == ("示例，你好", ("prompt_echo",))
    assert [w.text for w in raw.words] == ["示例，", "你好", "x"]


def test_the_sentence_said_after_the_first_window_is_not_an_echo(fake: FakeWhisper) -> None:
    fake.result = windows((0, "大家好"), (3000, "请使用简体中文并加上标点符号。"))
    raw = loaded("w-prod", fake).transcribe(AUDIO, language="zh", context=None)
    assert len(fake.calls) == 1
    assert raw.flags == ()


def test_a_result_without_window_offsets_is_checked_whole(fake: FakeWhisper) -> None:
    fake.results = [{"text": "大家好。使用简体中文并加上标点符号。", "language": "zh", "segments": []}, RESULT]
    raw = loaded("w-prod", fake).transcribe(AUDIO, language="zh", context=None)
    assert len(fake.calls) == 2
    assert raw.flags == ("prompt_echo",)


def test_a_decode_that_does_not_echo_the_prompt_is_kept(fake: FakeWhisper) -> None:
    fake.result = windows((0, "请使用这个方法，加上标点"))
    raw = loaded("w-prod", fake).transcribe(AUDIO, language="zh", context=None)
    assert len(fake.calls) == 1
    assert (raw.text, raw.flags) == ("请使用这个方法，加上标点", ())


@pytest.mark.parametrize(("hypothesis", "language"), [("w-vad", "zh"), ("w-runs-lid", "zh"), ("w-prod", "en")])
def test_an_unprompted_decode_is_never_decoded_again(hypothesis: str, language: str, fake: FakeWhisper) -> None:
    fake.result = windows((0, whisper_mlx.ZH_INITIAL_PROMPT))
    raw = loaded(hypothesis, fake).transcribe(AUDIO, language=language, context=None)
    assert len(fake.calls) == 1
    assert raw.flags == ()


@pytest.mark.parametrize(
    ("again", "text", "flags"),
    [(RESULT, "示例，你好", ("prompt_echo",)), (windows(), "", ("prompt_echo", "empty_output"))],
)
def test_the_worker_records_the_prompt_echo_on_the_row(
    again: dict[str, Any], text: str, flags: tuple[str, ...], tmp_path: Path, fake: FakeWhisper
) -> None:
    fake.results = [windows((0, whisper_mlx.ZH_INITIAL_PROMPT)), again]
    wav = write_silence(tmp_path / "short.wav", 3.0)
    out = tmp_path / "out"
    request = WorkerRequest(
        engine=registry.get("w-prod"), out_dir=str(out), files=(FileJob(id="f1", wav=str(wav), language="zh"),)
    )
    worker.run_request(request, guard=worker.default_guard(request))
    (line,) = (out / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    row = ChunkHypothesis.model_validate_json(line)
    assert (row.text, row.flags) == (text, flags)


def test_only_the_chunked_engines_decode_keys_moved() -> None:
    """The digests before the chunked engines stopped sending the prompt: w-prod's must stay, the others must move."""
    before = {
        "w-prod": (
            "3f2bcd6d9c661862434054d6307f45fa3d6b6bc78b4d9fdce1c11c211a0661be",
            "ed67dbc2591a6fe695795fc3bd64325846344b502e23824c3639f93168e3721e",
        ),
        "w-vad": (
            "2d566afc742bf9ef15567688d1b83e64b7517536c21692acf2bc9b7885a4eb67",
            "36fc36b3e760e8724bf24d0290f73cc0954d66a0a9a76590fa78856fe21fa7e9",
        ),
        "w-vad-lid": (
            "5d3dc8b46b992456e72ee80bc4559035acdcc53fd6383132a2e3ee8625490965",
            "56c12dd6fa4241a37ac35a7748574533ad3cc78bc149ad812233fa9793e75901",
        ),
        "w-runs-lid": (
            "a5fd0b576afc7212e45b42249c8a05a647da4217534f7ec539cea6f1ed1b0c3e",
            "167af45c26a58c3c12df4428926e315fc6747ab100c259955c77bae2fbaae1c8",
        ),
    }
    now = {}
    for hypothesis in before:
        definition = registry.get(hypothesis)
        request = WorkerRequest(engine=definition, out_dir="/o", files=(FileJob(id="f", wav="/a.wav", language="zh"),))
        now[hypothesis] = (store.digest(pipeline._engine_key(definition)), store.digest(request.fingerprint()))
    assert now["w-prod"] == before["w-prod"]
    for hypothesis in ("w-vad", "w-vad-lid", "w-runs-lid"):
        assert now[hypothesis][0] != before[hypothesis][0]
        assert now[hypothesis][1] != before[hypothesis][1]


def test_the_decode_reports_its_windows_from_each_segments_seek(fake: FakeWhisper) -> None:
    def segment(seek: int, *times: float) -> dict[str, Any]:
        # mlx-whisper clears the text and words of an instantaneous segment but keeps its seek.
        words = [{"word": " w", "start": t, "end": t + 0.5} for t in times]
        return {"seek": seek, "text": " w" * len(times), "words": words}

    # Window 0 has three segments, one of them cleared; the window after 59.5 s left only a cleared segment, and the
    # one after it was judged silent and left none.
    fake.result = {
        "text": "",
        "language": "en",
        "segments": [
            segment(0, 1.0, 2.0),
            segment(0),
            segment(0, 10.0, 28.0),
            segment(2950, 30.0),
            segment(6000),
            segment(9000, 91.0, 95.0),
        ],
    }
    raw = loaded("w-prod", fake).transcribe(np.zeros(100 * 16_000, dtype=np.float32), language="en", context=None)
    assert raw.windows == (
        Window(0.0, 29.5, 4),
        Window(29.5, 59.5, 1),
        Window(60.0, 90.0, 0),
        Window(90.0, 100.0, 2),
    )
    assert sum(window.words for window in raw.windows) == len(raw.words)


def test_a_result_without_window_offsets_reports_no_windows(fake: FakeWhisper) -> None:
    segment = {"text": " x", "words": [{"word": " x", "start": 0.0, "end": 1.0}]}
    fake.result = {"text": " x", "language": "en", "segments": [segment]}
    assert loaded("w-prod", fake).transcribe(AUDIO, language="en", context=None).windows == ()


@pytest.mark.parametrize("hypothesis", ["w-prod", "w-vad", "w-vad-lid", "w-runs-lid"])
def test_a_single_window_is_decoded_with_the_production_call_and_never_the_prompt(
    hypothesis: str, fake: FakeWhisper, snapshot: Path
) -> None:
    # Even a result that echoes the prompt is kept: none was sent.
    fake.result = windows((0, whisper_mlx.ZH_INITIAL_PROMPT))
    raw = loaded(hypothesis, fake).transcribe_window(AUDIO, language="zh-Hans")
    ((audio, kwargs),) = fake.calls
    assert audio is AUDIO
    assert kwargs == {
        "path_or_hf_repo": str(snapshot),
        "word_timestamps": True,
        "language": "zh",
        "initial_prompt": None,
        "condition_on_previous_text": False,
        "temperature": whisper_mlx.TEMPERATURE_SCHEDULE,
    }
    assert raw.flags == ()
    assert raw.windows == (Window(0.0, 1.0, 1),)


def test_the_worker_repairs_a_looped_window_of_a_whole_file_decode(tmp_path: Path, fake: FakeWhisper) -> None:
    loop = "风险限制" * 8
    looped = [{"word": c, "start": 10.0 + i * 0.25, "end": 10.25 + i * 0.25} for i, c in enumerate(loop)]
    fake.results = [
        {
            "text": f"开头的话。{loop}结尾的话。",
            "language": "zh",
            "segments": [
                {"seek": 0, "text": "开头的话。", "words": [{"word": "开头的话。", "start": 1.0, "end": 3.0}]},
                {"seek": 1000, "text": loop, "words": looped},
                {"seek": 2000, "text": "结尾的话。", "words": [{"word": "结尾的话。", "start": 21.0, "end": 23.0}]},
            ],
        },
        windows((0, "左边")),
        windows((0, "右边")),
    ]
    wav = write_silence(tmp_path / "loop.wav", 30.0)
    out = tmp_path / "out"
    request = WorkerRequest(
        engine=registry.get("w-prod"), out_dir=str(out), files=(FileJob(id="f1", wav=str(wav), language="zh"),)
    )
    worker.run_request(request, guard=worker.default_guard(request))
    whole, left, right = fake.calls
    assert whole[1]["initial_prompt"] == whisper_mlx.ZH_INITIAL_PROMPT
    assert left[1] == right[1] == {**whole[1], "initial_prompt": None}
    assert len(left[0]) + len(right[0]) == 10 * 16_000
    (line,) = (out / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    row = ChunkHypothesis.model_validate_json(line)
    assert (row.text, row.flags, row.anomaly_spans) == ("开头的话。左边右边结尾的话。", ("redecoded",), ())


def test_no_language_means_no_prompt_and_auto_detection(fake: FakeWhisper) -> None:
    loaded("w-prod", fake).transcribe(AUDIO, language=None, context=None)
    assert fake.calls[0][1]["language"] is None
    assert fake.calls[0][1]["initial_prompt"] is None


def test_english_gets_no_prompt(fake: FakeWhisper) -> None:
    loaded("w-prod", fake).transcribe(AUDIO, language="en-US", context=None)
    assert fake.calls[0][1]["language"] == "en"
    assert fake.calls[0][1]["initial_prompt"] is None


def test_the_temperature_schedule_is_the_increasing_fallback() -> None:
    schedule = whisper_mlx.TEMPERATURE_SCHEDULE
    assert len(schedule) > 1
    assert list(schedule) == sorted(schedule)
    assert schedule[0] == 0.0


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("zh-Hans", "zh"), ("ZH_cn", "zh"), ("en", "en"), ("auto", None), (" ", None), ("-x", None), ("mandarin", None)],
)
def test_language_tags_reduce_to_the_primary_code(tag: str, expected: str | None, fake: FakeWhisper) -> None:
    assert whisper_mlx._normalize_language(tag) == expected


def test_an_unknown_language_warns_and_auto_detects(fake: FakeWhisper, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert whisper_mlx._normalize_language("xx-YY") is None
    assert "xx-YY" in caplog.text


def test_words_and_confidence_come_from_the_word_probabilities(fake: FakeWhisper) -> None:
    raw = loaded("w-vad", fake).transcribe(AUDIO, language="zh", context=None)
    assert raw.text == "示例，你好"
    assert raw.language == "zh"
    assert [(w.text, w.start, w.end, w.conf) for w in raw.words] == [
        ("示例，", 0.5, 1.0, 0.75),
        ("你好", 1.0, 1.5, 0.25),
        ("x", 2.0, 2.5, None),
    ]
    assert raw.confidence == 0.5
    assert raw.generated_tokens is None
    assert raw.max_tokens is None


def test_a_decode_without_words_has_no_confidence(fake: FakeWhisper) -> None:
    fake.result = {"text": "", "language": "en", "segments": []}
    raw = loaded("w-prod", fake).transcribe(AUDIO, language="en", context=None)
    assert (raw.text, raw.words, raw.confidence) == ("", (), None)


def test_load_resolves_the_default_cache_snapshot_not_the_worker_cache(
    fake: FakeWhisper, snapshot: Path, mx: FakeMx
) -> None:
    loaded("w-prod", fake)
    assert fake.loads == [(str(snapshot), mx.float16)]
    assert "candidates-hub" not in str(snapshot)


def test_a_missing_snapshot_fails_loud_with_the_download_command(fake: FakeWhisper, snapshot: Path) -> None:
    (snapshot / "config.json").unlink()
    engine = registry.build_engine(registry.get("w-prod"))
    with pytest.raises(
        DependencyError, match=r"hf download mlx-community/whisper-large-v3-mlx --revision [0-9a-f]{40}"
    ):
        engine.load()
    assert fake.loads == []


def test_a_snapshot_without_weights_is_missing(fake: FakeWhisper, snapshot: Path) -> None:
    (snapshot / "weights.npz").unlink()
    with pytest.raises(DependencyError, match="missing"):
        registry.build_engine(registry.get("w-prod")).load()


def test_a_missing_whisper_extra_names_the_install(fake: FakeWhisper, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with pytest.raises(DependencyError, match=r"install anyverbatim\[whisper\]"):
        registry.build_engine(registry.get("w-prod")).load()


def test_transcribing_before_loading_is_a_programming_error(fake: FakeWhisper) -> None:
    engine = registry.build_engine(registry.get("w-prod"))
    with pytest.raises(RuntimeError, match="load"):
        engine.transcribe(AUDIO, language=None, context=None)


def test_decode_runs_inside_a_cache_limit_of_zero_and_restores_it(fake: FakeWhisper, mx: FakeMx) -> None:
    loaded("w-prod", fake).transcribe(AUDIO, language=None, context=None)
    assert mx.events == [("limit", 0), ("clear", None), ("transcribe", None), ("clear", None), ("limit", 123)]


def test_a_decode_error_restores_the_cache_limit_and_propagates(fake: FakeWhisper, mx: FakeMx) -> None:
    error = ValueError("decode failed")
    fake.error = error
    engine = loaded("w-prod", fake)
    with pytest.raises(ValueError, match="decode failed") as raised:
        engine.transcribe(AUDIO, language=None, context=None)
    assert raised.value is error
    assert mx.events[-1] == ("limit", 123)


def test_a_cleanup_failure_does_not_mask_the_decode_error(
    fake: FakeWhisper, mx: FakeMx, caplog: pytest.LogCaptureFixture
) -> None:
    error = ValueError("decode failed")
    fake.error = error
    engine = loaded("w-prod", fake)
    mx.clear_error = [None, RuntimeError("cleanup failed")]
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="decode failed") as raised:
        engine.transcribe(AUDIO, language=None, context=None)
    assert raised.value is error
    assert mx.events[-1] == ("limit", 123)
    assert "cleanup failed" in caplog.text


def test_a_cleanup_failure_after_success_is_raised(fake: FakeWhisper, mx: FakeMx) -> None:
    engine = loaded("w-prod", fake)
    mx.clear_error = [None, RuntimeError("cleanup failed")]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        engine.transcribe(AUDIO, language=None, context=None)
    assert mx.events[-1] == ("limit", 123)


def test_mode_must_be_known_and_agree_with_the_spec() -> None:
    whole, chunked = registry.get("w-prod"), registry.get("w-vad")
    with pytest.raises(ConfigError, match="mode"):
        whisper_mlx.create(whole.spec, {"mode": "other"})
    with pytest.raises(ConfigError, match="contradicts"):
        whisper_mlx.create(whole.spec, {"mode": "chunk", "zh_prompt": False})
    with pytest.raises(ConfigError, match="contradicts"):
        whisper_mlx.create(chunked.spec, {"mode": "whole_file"})


@pytest.mark.parametrize(
    ("hypothesis", "variant"),
    [
        ("w-vad", {"mode": "chunk"}),
        ("w-vad", {"mode": "chunk", "zh_prompt": True}),
        ("w-runs-lid", {"mode": "runs"}),
        ("w-prod", {"mode": "whole_file", "zh_prompt": False}),
    ],
)
def test_only_whole_file_mode_may_send_the_zh_prompt(hypothesis: str, variant: dict[str, JsonValue]) -> None:
    with pytest.raises(ConfigError, match="only whole_file mode sends the Mandarin prompt"):
        whisper_mlx.create(registry.get(hypothesis).spec, variant)


def test_the_registry_pins_the_default_cache_revision() -> None:
    for hypothesis in ("w-prod", "w-vad"):
        spec = registry.get(hypothesis).spec
        assert (spec.repo, spec.revision) == (
            "mlx-community/whisper-large-v3-mlx",
            "49e6aa286ad60c14352c404340ded53710378a11",
        )


def test_the_worker_decodes_a_whole_file_in_one_call_with_absolute_times(tmp_path: Path, fake: FakeWhisper) -> None:
    wav = write_silence(tmp_path / "-dash.wav", 12.0)
    out = tmp_path / "out"
    request = WorkerRequest(
        engine=registry.get("w-prod"), out_dir=str(out), files=(FileJob(id="f1", wav=str(wav), language="zh"),)
    )
    metrics = worker.run_request(request)
    assert metrics.chunks_decoded == 1
    assert len(fake.calls) == 1
    assert len(fake.calls[0][0]) == 12 * 16_000
    (line,) = (out / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    row = ChunkHypothesis.model_validate_json(line)
    assert (row.start, row.end, row.text, row.language, row.confidence) == (0.0, 12.0, "示例，你好", "zh", 0.5)
    assert [w.text for w in row.words] == ["示例，", "你好", "x"]


def test_detection_uses_the_loaded_model_on_mlx_whispers_own_detection_input(fake: FakeWhisper, mx: FakeMx) -> None:
    engine = loaded("w-runs-lid", fake)
    assert engine.detect_language(AUDIO) == {"en": 0.25, "zh": 0.7, "yue": 0.05}
    (mel,) = fake.model.mels
    assert mel.steps == (AUDIO, 128, 480_000, ("trim", 3000, -2), ("astype", mx.float16))
    assert len(fake.loads) == 1
    assert ("detect", None) in mx.events
    assert mx.events[-1] == ("limit", 123)


def test_detecting_before_loading_is_a_programming_error(fake: FakeWhisper) -> None:
    engine = registry.build_engine(registry.get("w-runs-lid"))
    assert isinstance(engine, WhisperEngine)
    with pytest.raises(RuntimeError, match="load"):
        engine.detect_language(AUDIO)


def test_the_worker_decodes_runs_with_their_language_and_no_zh_prompt(
    tmp_path: Path, fake: FakeWhisper, monkeypatch: pytest.MonkeyPatch
) -> None:
    detections = [{"zh": 0.8, "en": 0.2}, {"zh": 0.1, "en": 0.9}, {"zh": 0.7, "en": 0.3}]

    def detect(mel: Any) -> tuple[None, dict[str, float]]:
        return None, detections.pop(0)

    monkeypatch.setattr(fake.model, "detect_language", detect)
    wav = write_silence(tmp_path / "-dash.wav", 30.0)
    chunks = (Chunk(0.5, 9.0), Chunk(10.0, 19.0), Chunk(20.0, 29.0))
    request = WorkerRequest(
        engine=registry.get("w-runs-lid"),
        out_dir=str(tmp_path / "out"),
        files=(FileJob(id="f1", wav=str(wav), chunks=chunks, languages=("zh", "en")),),
    )
    worker.run_request(request)
    assert [(len(audio), kwargs["language"], kwargs["initial_prompt"]) for audio, kwargs in fake.calls] == [
        (136_000, "zh", None),
        (144_000, "en", None),
        (144_000, "zh", None),
    ]
    rows = [
        ChunkHypothesis.model_validate_json(line)
        for line in (tmp_path / "out" / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert [(r.start, r.end, r.language_prob) for r in rows] == [(0.5, 9.0, 0.8), (10.0, 19.0, 0.9), (20.0, 29.0, 0.7)]
    assert [w.start for w in rows[1].words] == [10.5, 11.0, 12.0]
