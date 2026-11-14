"""Types shared by the diarization worker, its parent and the attribution step.

Nothing here imports numpy or MLX: the parent reads the stored float16 probabilities with `struct`.
"""

import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ytverbatim.errors import EngineError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

METRICS_FILE = "metrics.json"
ERROR_FILE = "error.json"
REQUEST_FILE = "request.json"
FINGERPRINT_FILE = "fingerprint.json"
TURNS_SUFFIX = ".turns.json"
PROBS_SUFFIX = ".probs.f16"
# Little-endian IEEE half floats, row-major (frames, channels). Half precision rounds a probability just above the
# threshold to it (0.5002 stores as 0.5, so it reads as inactive); the band is under 0.00025 wide around 0.5, far
# below anything the threshold is meant to separate.
_HALF = struct.Struct("<e")

# File ids become file names in the output folder; uniqueness is checked case-insensitively for APFS.
_FILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class DiarizerSpec:
    """What the pipeline needs to know about a diarizer without loading it.

    `max_speakers` is the model's channel count: it never reports more speakers than that.
    """

    id: str
    family: str
    repo: str
    revision: str
    weights_bytes: int
    max_speakers: int
    sample_rate: int


@dataclass(frozen=True, slots=True)
class DiarizerDef:
    """A diarizer as the worker receives it: spec, adapter factory (dotted path) and variant parameters."""

    spec: DiarizerSpec
    factory: str
    variant: dict[str, JsonValue]

    def threshold(self) -> float:
        value = self.variant.get("threshold")
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
            raise ValueError(f"{self.spec.id} needs a variant threshold between 0 and 1, got {value!r}")
        return float(value)


class Turn(_Model):
    """A stretch where one speaker channel is active; channels are the model's own, in arrival order."""

    start: Seconds
    end: Seconds
    speaker: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"turn of speaker {self.speaker} ends ({self.end}) before it starts ({self.start})")
        return self


@dataclass(frozen=True, slots=True, eq=False)
class RawDiarization:
    """One diarizer run over a whole file: its turns and its (frames, channels) activity probabilities."""

    turns: tuple[Turn, ...]
    probs: "NDArray[np.float32]"
    frame_s: float


class Diarizer(Protocol):
    def load(self) -> None: ...

    def diarize(self, audio: "NDArray[np.float32]") -> RawDiarization: ...


DiarizerFactory = Callable[[DiarizerSpec, Mapping[str, JsonValue]], Diarizer]


class FileJob(_Model):
    id: str
    wav: str

    @model_validator(mode="after")
    def _safe_id(self) -> Self:
        if _FILE_ID.fullmatch(self.id) is None:
            raise ValueError(f"file id {self.id!r} must be letters, digits, '-' or '_' (it names output files)")
        return self


class DiarizeRequest(_Model):
    diarizer: DiarizerDef
    out_dir: str
    files: tuple[FileJob, ...]
    hf_hub_cache: str | None = None
    # The parent kills the worker above this footprint; the worker also hands it to MLX as its memory limit.
    memory_cap_bytes: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def _valid(self) -> Self:
        # APFS is case-insensitive by default, so ids differing only in case would share one output file.
        ids = [job.id.casefold() for job in self.files]
        if len(set(ids)) != len(ids):
            raise ValueError("file ids must be unique within a request, ignoring case")
        self.diarizer.threshold()
        return self

    def fingerprint(self) -> dict[str, JsonValue]:
        """Everything that decides the output; where it and the weights live and the memory cap do not."""
        return self.model_dump(mode="json", exclude={"out_dir", "hf_hub_cache", "memory_cap_bytes"})


class FileDiarization(_Model):
    """The `<id>.turns.json` of one file; its probabilities sit beside it in `<id>.probs.f16`.

    `num_speakers` counts the channels with at least one turn.
    """

    diarizer: str
    file: str
    duration_s: Seconds
    threshold: float = Field(gt=0, lt=1)
    frame_s: float = Field(gt=0, allow_inf_nan=False)
    frames: Annotated[int, Field(ge=0)]
    channels: Annotated[int, Field(gt=0)]
    num_speakers: Annotated[int, Field(ge=0)]
    turns: tuple[Turn, ...]


class RunMetrics(_Model):
    """What one worker invocation did. A resumed run counts only the files it diarized itself."""

    diarizer: str
    files: int
    files_diarized: int
    files_skipped: int
    load_s: Seconds
    diarize_s: Seconds
    audio_s: Seconds
    peak_memory_bytes: int | None
    max_rss_bytes: int
    peak_footprint_bytes: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class Diarization:
    """A stored diarization loaded for attribution; `probs` is indexed `[frame * channels + channel]`."""

    turns: tuple[Turn, ...]
    threshold: float
    frame_s: float
    frames: int
    channels: int
    probs: bytes

    def prob(self, frame: int, channel: int) -> float:
        value: float = _HALF.unpack_from(self.probs, _HALF.size * (frame * self.channels + channel))[0]
        return value


def turns_path(out_dir: Path, file_id: str) -> Path:
    return out_dir / f"{file_id}{TURNS_SUFFIX}"


def probs_path(out_dir: Path, file_id: str) -> Path:
    return out_dir / f"{file_id}{PROBS_SUFFIX}"


def read_file_diarization(out_dir: Path, file_id: str, diarizer_id: str) -> FileDiarization:
    """Read one file's turns, refusing a file written for another id or by another diarizer."""
    path = turns_path(out_dir, file_id)
    try:
        stored = FileDiarization.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"{path} is unreadable: {exc}") from exc
    if stored.file != file_id or stored.diarizer != diarizer_id:
        raise EngineError(
            f"{path} holds {stored.diarizer} output for file {stored.file!r}, not {diarizer_id} output for {file_id!r}"
        )
    return stored


def load(out_dir: Path, file_id: str, diarizer_id: str) -> Diarization:
    """Read one file's stored diarization; raise `EngineError` if its two parts are missing or disagree."""
    meta = read_file_diarization(out_dir, file_id, diarizer_id)
    path = probs_path(out_dir, file_id)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise EngineError(f"{path} is unreadable: {exc}") from exc
    expected = meta.frames * meta.channels * _HALF.size
    if len(data) != expected:
        raise EngineError(f"{path} holds {len(data)} bytes, not the {expected} its turns file describes")
    return Diarization(
        turns=meta.turns,
        threshold=meta.threshold,
        frame_s=meta.frame_s,
        frames=meta.frames,
        channels=meta.channels,
        probs=data,
    )
