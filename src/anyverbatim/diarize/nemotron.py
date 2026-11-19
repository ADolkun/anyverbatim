"""Nemotron-3-Diarization through mlx-audio, for the `d-nemotron-*` diarizers.

`mlx_audio` is untyped and heavy, so it is imported on `load()` and never leaks past this module.
"""

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import JsonValue

from anyverbatim.diarize.protocol import DiarizerSpec, RawDiarization, Turn
from anyverbatim.errors import DependencyError, EngineError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

SAMPLE_RATE = 16_000
# Every default of `generate` that could change the turns is pinned, so an mlx-audio upgrade cannot shift results.
_MIN_DURATION_S = 0.0
_MERGE_GAP_S = 0.0


class _Segment(Protocol):
    start: float
    end: float
    speaker: int


class _Output(Protocol):
    segments: list[_Segment]
    speaker_probs: Any


class _ProcessorConfig(Protocol):
    hop_length: int


class _Config(Protocol):
    num_speakers: int
    output_subsampling_factor: int
    processor_config: _ProcessorConfig


class _Model(Protocol):
    config: _Config

    @property
    def sample_rate(self) -> int: ...

    def generate(self, audio: object, **kwargs: object) -> _Output: ...


class _Loader(Protocol):
    def load(self, model_path: str, **kwargs: object) -> _Model: ...


class NemotronDiarizer:
    def __init__(self, spec: DiarizerSpec, *, threshold: float) -> None:
        self._spec = spec
        self._threshold = threshold
        self._model: _Model | None = None

    def load(self) -> None:
        try:
            vad = importlib.import_module("mlx_audio.vad")
        except ImportError as exc:
            raise DependencyError("install anyverbatim[asr]") from exc
        # strict: a weight file that does not cover every parameter must fail here, not diarize from random weights.
        model = cast(_Loader, vad).load(self._spec.repo, revision=self._spec.revision, strict=True)
        if model.sample_rate != SAMPLE_RATE or model.config.num_speakers != self._spec.max_speakers:
            raise EngineError(
                f"{self._spec.id} loaded a model for {model.sample_rate} Hz and {model.config.num_speakers} speakers; "
                f"the registry expects {SAMPLE_RATE} Hz and {self._spec.max_speakers}"
            )
        self._model = model

    def diarize(self, audio: NDArray[np.float32]) -> RawDiarization:
        if self._model is None:
            raise RuntimeError("load() must be called before diarize()")
        np = importlib.import_module("numpy")
        model = self._model
        output = model.generate(
            audio,
            sample_rate=SAMPLE_RATE,
            threshold=self._threshold,
            min_duration=_MIN_DURATION_S,
            merge_gap=_MERGE_GAP_S,
            verbose=False,
        )
        probs = np.asarray(output.speaker_probs, dtype=np.float32).reshape(-1, model.config.num_speakers)
        config = model.config
        frame_s = config.processor_config.hop_length * config.output_subsampling_factor / model.sample_rate
        turns = tuple(
            Turn(start=float(s.start), end=float(s.end), speaker=s.speaker)
            for s in sorted(output.segments, key=lambda s: (s.start, s.speaker))
        )
        return RawDiarization(turns=turns, probs=probs, frame_s=frame_s)


def create(spec: DiarizerSpec, variant: Mapping[str, JsonValue]) -> NemotronDiarizer:
    threshold = variant.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise EngineError(f"{spec.id} needs a numeric threshold, got {threshold!r}")
    return NemotronDiarizer(spec, threshold=float(threshold))
