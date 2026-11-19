import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import JsonValue

from anyverbatim.asr import registry
from anyverbatim.asr.engines import firered
from anyverbatim.errors import ConfigError, EngineError

SPEC = registry.get("f-aed").spec


class FakeMx:
    @staticmethod
    def array(values: NDArray[np.float32]) -> tuple[str, NDArray[np.float32]]:
        return ("mx", values)


class FakeModel:
    def __init__(self, text: str = "nvda rose 3.5 percent", confidence: float = 0.9137) -> None:
        self.text = text
        self.confidence = confidence
        self.calls: list[tuple[object, dict[str, object]]] = []

    def generate(self, audio: object, **kwargs: object) -> SimpleNamespace:
        self.calls.append((audio, kwargs))
        return SimpleNamespace(
            text=self.text, segments=[{"text": self.text, "confidence": self.confidence}], generation_tokens=7
        )


def install(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> list[tuple[str, dict[str, object]]]:
    loads: list[tuple[str, dict[str, object]]] = []
    stt, mlx, core = ModuleType("mlx_audio.stt"), ModuleType("mlx"), ModuleType("mlx.core")

    def load(repo: str, **kwargs: object) -> FakeModel:
        loads.append((repo, kwargs))
        return model

    vars(stt)["load"] = load
    vars(core)["array"] = FakeMx.array
    vars(mlx)["core"] = core
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return loads


def loaded_engine(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> tuple[Any, list[tuple[str, dict[str, object]]]]:
    loads = install(monkeypatch, model)
    engine = firered.create(SPEC, registry.get("f-aed").variant)
    engine.load()
    return engine, loads


def test_loads_the_pinned_repo_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    _, loads = loaded_engine(monkeypatch, FakeModel())

    assert loads == [(SPEC.repo, {"revision": SPEC.revision})]


def test_decodes_with_the_exact_kwargs_and_keeps_the_lowercase_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    engine, _ = loaded_engine(monkeypatch, model)
    audio = np.zeros(16_000 * 20, dtype=np.float32)

    raw = engine.transcribe(audio, language="en", context=None)

    [(passed, kwargs)] = model.calls
    assert passed == ("mx", audio)
    assert kwargs == {
        "beam_size": 3,
        "chunk_duration": 60.0,
        "softmax_smoothing": 1.25,
        "length_penalty": 0.6,
        "eos_penalty": 1.0,
        "max_len": 0,
    }
    assert raw.text == "nvda rose 3.5 percent"


def test_segment_confidence_and_token_count_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = loaded_engine(monkeypatch, FakeModel(confidence=0.9137))

    raw = engine.transcribe(np.zeros(16_000, dtype=np.float32), language=None, context=None)

    assert raw.confidence == 0.9137
    assert raw.generated_tokens == 7
    assert raw.max_tokens is None


def test_an_empty_decode_has_no_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = loaded_engine(monkeypatch, FakeModel(text="", confidence=0.0))

    raw = engine.transcribe(np.zeros(16_000, dtype=np.float32), language=None, context=None)

    assert raw.text == ""
    assert raw.confidence is None


def test_input_over_the_chunk_limit_is_refused_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    engine, _ = loaded_engine(monkeypatch, model)

    with pytest.raises(EngineError, match=r"at most 30s"):
        engine.transcribe(np.zeros(16_000 * 31, dtype=np.float32), language=None, context=None)

    assert model.calls == []


def test_transcribe_before_load_is_an_engine_error() -> None:
    engine = firered.create(SPEC, {"beam_size": 3})

    with pytest.raises(EngineError, match="load"):
        engine.transcribe(np.zeros(16_000, dtype=np.float32), language=None, context=None)


def test_a_failing_load_names_the_repo_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    stt = ModuleType("mlx_audio.stt")

    def load(repo: str, **kwargs: object) -> None:
        raise OSError("no route to host")

    vars(stt)["load"] = load
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", stt)
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hub")

    with pytest.raises(EngineError, match=rf"{SPEC.repo}@{SPEC.revision}.*/tmp/hub.*no route to host"):
        firered.create(SPEC, {"beam_size": 3}).load()


@pytest.mark.parametrize("beam_size", [None, 0, "3", True])
def test_the_variant_needs_a_positive_integer_beam_size(beam_size: JsonValue) -> None:
    with pytest.raises(ConfigError, match="beam_size"):
        firered.create(SPEC, {"beam_size": beam_size})
