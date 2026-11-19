import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from anyverbatim import legacy
from anyverbatim.asr import registry
from anyverbatim.asr.chunks import Chunk
from anyverbatim.asr.protocol import AsrContext, ChunkFlag, EngineDef
from anyverbatim.contract.models import Word
from anyverbatim.mlx_worker import ERROR_FILE as ERROR_FILE
from anyverbatim.mlx_worker import FINGERPRINT_FILE as FINGERPRINT_FILE
from anyverbatim.mlx_worker import METRICS_FILE as METRICS_FILE
from anyverbatim.mlx_worker import REQUEST_FILE as REQUEST_FILE

HYPOTHESES_FILE = "hypotheses.jsonl"

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnomalySpan(_Model):
    """A stretch of a row, in absolute seconds, that the guard could not repair."""

    start: Seconds
    end: Seconds


class ChunkHypothesis(_Model):
    """One line of `hypotheses.jsonl`; times are absolute seconds in the source file.

    Under a runs hypothesis a line is one same-language run, and `chunk` is the run's index.
    """

    hypothesis: str
    file: str
    chunk: Annotated[int, Field(ge=0)]
    start: Seconds
    end: Seconds
    text: str
    language: str | None = None
    # The detector's unrestricted probability for the language this row was decoded with; for a run, the lowest of
    # its chunks'. None when no detection chose it.
    # Left out of the line while None, so a control row reads the same to a anyverbatim that predates the field.
    language_prob: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, exclude_if=lambda v: v is None)
    words: tuple[Word, ...] = ()
    generated_tokens: int | None = None
    max_tokens: int | None = None
    confidence: float | None = Field(default=None, allow_inf_nan=False)
    flags: tuple[ChunkFlag, ...] = ()
    # Where a `decoder_anomaly` still lies after the guard's repair; empty means anywhere in the row, as for every
    # chunk row and any decode the guard could not localize. Left out of the line while empty, so every other row
    # serializes byte for byte as it did before the field existed.
    anomaly_spans: tuple[AnomalySpan, ...] = Field(default=(), exclude_if=lambda v: not v)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"chunk {self.chunk} of {self.file} ends ({self.end}) before it starts ({self.start})")
        return self


class RunMetrics(_Model):
    """What one worker invocation did. A resumed run counts only the chunks it decoded itself."""

    hypothesis: str
    files: int
    chunks_decoded: int
    chunks_skipped: int
    load_s: Seconds
    decode_s: Seconds
    audio_s: Seconds
    peak_memory_bytes: int | None
    max_rss_bytes: int
    # The worker's own lifetime peak physical footprint, which the memory watch's cap is measured against.
    peak_footprint_bytes: int | None = None


class FileJob(_Model):
    """One WAV to transcribe. Without `chunks` the worker plans them with the VAD.

    `language` is the metadata language, as production would pass it; the engine's language policy decides whether
    the engine sees it. `languages` is the set a `detect` engine may choose from; empty allows any.
    """

    id: str
    wav: str
    language: str | None = None
    context: AsrContext | None = None
    chunks: tuple[Chunk, ...] | None = None
    # Left out of every dump while empty, so requests made before the field existed keep their fingerprint and with
    # it their worker folder.
    languages: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)

    @field_validator("languages")
    @classmethod
    def _primary_codes(cls, languages: tuple[str, ...]) -> tuple[str, ...]:
        """Reduce tags such as `zh-Hans` to their primary code, as the Whisper adapter does, keeping first order."""
        codes = [re.split(r"[-_]", tag.strip(), maxsplit=1)[0].lower() for tag in languages]
        if not all(codes):
            raise ValueError(f"allowed languages must be language codes, got {list(languages)}")
        return tuple(dict.fromkeys(codes))


class WorkerRequest(_Model):
    engine: EngineDef
    out_dir: str
    files: tuple[FileJob, ...]
    hf_hub_cache: str | None = None
    # The parent kills the worker above this footprint; the worker also hands it to MLX as its memory limit.
    memory_cap_bytes: Annotated[int, Field(gt=0)] | None = None
    # Off only for diagnostics: the guard hides an engine's raw failure rate by re-decoding what failed.
    quality_guard: bool = True

    @model_validator(mode="after")
    def _plans_match_chunking(self) -> Self:
        ids = [job.id for job in self.files]
        if len(set(ids)) != len(ids):
            raise ValueError("file ids must be unique within a request")
        engine = self.engine
        if engine.spec.max_chunk_s is None and any(job.chunks is not None for job in self.files):
            raise ValueError(f"{engine.spec.id} decodes whole files and cannot take a chunk plan")
        detects = engine.language_policy() == "detect"
        if engine.spec.max_chunk_s is None and (detects or engine.decodes_runs()):
            raise ValueError(f"{engine.spec.id} detects languages per chunk, so its spec needs a max_chunk_s")
        if engine.decodes_runs() and not detects:
            raise ValueError(f"{engine.spec.id} decodes same-language runs, so its language policy must be 'detect'")
        if not detects and any(job.languages for job in self.files):
            raise ValueError(f"{engine.spec.id} does not detect languages, so it cannot take an allowed language set")
        known = registry.detection_languages(engine)
        unknown = sorted({code for job in self.files for code in job.languages} - known) if known is not None else []
        if unknown:
            raise ValueError(f"{engine.spec.id} cannot detect the allowed languages {unknown}")
        return self

    def fingerprint(self) -> dict[str, JsonValue]:
        """Everything that decides the decoded text; where the output and the weights live and the memory cap do not.

        Worker folders are keyed by this alone, so a worker code change that alters the decoded text for the same
        request decodes nothing again until a version field is added here (which moves every folder once).
        """
        dumped = self.model_dump(mode="json", exclude={"out_dir", "hf_hub_cache", "memory_cap_bytes"})
        dumped["engine"]["factory"] = legacy.fingerprint_factory(self.engine.factory)
        return dumped
