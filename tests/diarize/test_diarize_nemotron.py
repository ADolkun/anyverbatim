import re
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from anyverbatim.diarize import nemotron, registry
from anyverbatim.errors import ConfigError, DependencyError, EngineError

RATE = 16_000


class FakeModel:
    def __init__(self, *, sample_rate: int = RATE, speakers: int = 8) -> None:
        self.sample_rate = sample_rate
        self.config = SimpleNamespace(
            num_speakers=speakers, output_subsampling_factor=1, processor_config=SimpleNamespace(hop_length=160)
        )
        self.calls: list[tuple[NDArray[np.float32], dict[str, object]]] = []

    def generate(self, audio: NDArray[np.float32], **kwargs: object) -> SimpleNamespace:
        self.calls.append((audio, kwargs))
        probs = np.zeros((300, 8), dtype=np.float32)
        probs[:150, 2] = 0.9
        probs[120:, 0] = 0.7
        segments = [
            SimpleNamespace(start=1.2, end=3.0, speaker=0),
            SimpleNamespace(start=0.0, end=1.5, speaker=2),
        ]
        return SimpleNamespace(segments=segments, speaker_probs=probs, num_speakers=2)


def install_fake(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> list[tuple[str, dict[str, object]]]:
    loads: list[tuple[str, dict[str, object]]] = []
    package, vad = ModuleType("mlx_audio"), ModuleType("mlx_audio.vad")

    def load(repo: str, **kwargs: object) -> FakeModel:
        loads.append((repo, kwargs))
        return model

    vars(vad)["load"] = load
    monkeypatch.setitem(sys.modules, "mlx_audio", package)
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", vad)
    return loads


def loaded(diarizer_id: str = "d-nemotron-8bit") -> nemotron.NemotronDiarizer:
    definition = registry.get(diarizer_id)
    built = nemotron.create(definition.spec, definition.variant)
    built.load()
    return built


def test_loads_the_pinned_revision_strictly(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = install_fake(monkeypatch, FakeModel())
    loaded()
    assert loads == [
        (
            "mlx-community/Nemotron-3-Diarization-8bit",
            {"revision": "dd8b8ce3d69a80540a7e50150e2beed1c34fee37", "strict": True},
        )
    ]


def test_fake_output_becomes_sorted_turns_and_probabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    install_fake(monkeypatch, model)
    audio = np.zeros(3 * RATE, dtype=np.float32)
    raw = loaded().diarize(audio)
    assert [(t.speaker, t.start, t.end) for t in raw.turns] == [(2, 0.0, 1.5), (0, 1.2, 3.0)]
    assert raw.probs.shape == (300, 8)
    assert raw.probs.dtype == np.float32
    assert raw.frame_s == pytest.approx(0.01)
    samples, kwargs = model.calls[0]
    assert samples is audio
    assert kwargs == {"sample_rate": RATE, "threshold": 0.5, "min_duration": 0.0, "merge_gap": 0.0, "verbose": False}


def test_a_model_that_disagrees_with_the_registry_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, FakeModel(speakers=4))
    with pytest.raises(EngineError, match="4 speakers"):
        loaded()


def test_diarize_before_load_is_a_programming_error() -> None:
    definition = registry.get("d-nemotron-8bit")
    with pytest.raises(RuntimeError, match="load"):
        nemotron.create(definition.spec, definition.variant).diarize(np.zeros(10, dtype=np.float32))


def test_a_missing_mlx_audio_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", None)
    definition = registry.get("d-nemotron-8bit")
    with pytest.raises(DependencyError, match=r"anyverbatim\[asr\]"):
        nemotron.create(definition.spec, definition.variant).load()


def test_both_diarizers_are_registered_with_pinned_revisions() -> None:
    assert set(registry.DIARIZERS) == {"d-nemotron-8bit", "d-nemotron-bf16"}
    for diarizer_id, entry in registry.DIARIZERS.items():
        assert entry.spec.id == diarizer_id
        assert re.fullmatch(r"[0-9a-f]{40}", entry.spec.revision)
        assert entry.threshold() == 0.5
        assert entry.spec.max_speakers == 8
    assert registry.get("d-nemotron-bf16").spec.repo == "mlx-community/Nemotron-3-Diarization"


def test_an_unknown_diarizer_lists_the_known_ones() -> None:
    with pytest.raises(ConfigError, match="d-nemotron-8bit"):
        registry.get("nope")


def test_missing_mlx_audio_is_reported_before_any_work(monkeypatch: pytest.MonkeyPatch) -> None:

    def nothing_installed(name: str) -> None:
        return None

    monkeypatch.setattr(registry, "find_spec", nothing_installed)
    with pytest.raises(DependencyError, match=r"uv sync --extra asr"):
        registry.check_dependencies(registry.get("d-nemotron-8bit"))
