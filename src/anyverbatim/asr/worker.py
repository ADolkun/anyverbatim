"""Child process that loads one engine and decodes a request's files; run as `python -P -m anyverbatim.asr.worker`.

Everything that needs MLX happens here, never in the parent. `-P` keeps the working directory off `sys.path`, so a
checkout in the current directory cannot shadow the installed package.
"""

import math
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from anyverbatim import atomic, mlx_worker
from anyverbatim.asr.audio import Audio, frame_energy, load_wav
from anyverbatim.asr.chunks import Chunk, plan_chunks
from anyverbatim.asr.guard import QualityGuard
from anyverbatim.asr.models import (
    ERROR_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    AnomalySpan,
    ChunkHypothesis,
    FileJob,
    RunMetrics,
    WorkerRequest,
)
from anyverbatim.asr.protocol import (
    ChunkFlag,
    ChunkInput,
    Engine,
    EngineDef,
    Guard,
    GuardedChunk,
    LanguageDetector,
    RawChunk,
)
from anyverbatim.asr.registry import build_engine
from anyverbatim.asr.vad import speech_regions
from anyverbatim.contract.models import Word
from anyverbatim.errors import EngineError
from anyverbatim.mlx_worker import MlxAllocator, bound_allocator

_PLAN_TOLERANCE_S = 1e-3


@dataclass(frozen=True, slots=True)
class DecodeUnit:
    """One decode: the span of audio handed to the engine and the language hint it is decoded with."""

    chunk: Chunk
    language: str | None = None
    language_prob: float | None = None


def run_request(
    request: WorkerRequest,
    *,
    engine: Engine | None = None,
    guard: Guard | None = None,
    mlx: MlxAllocator | None = None,
) -> RunMetrics:
    """Decode every file, appending one flushed JSONL line per chunk and skipping chunks already on disk.

    `engine` replaces the one built from the request, for in-process tests. `guard` is the hook for the quality
    checks: it sees every decoded chunk and may decode again before the line is written. `mlx` is the allocator to
    bound before anything loads and to empty after every chunk; `main` passes `mlx.core`.
    """
    spec = request.engine.spec
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ERROR_FILE).unlink(missing_ok=True)
    sink_path = out / HYPOTHESES_FILE
    _claim_output(out, request)
    # Written again only once every chunk is on disk, so its presence says this request's output is complete.
    (out / METRICS_FILE).unlink(missing_ok=True)
    done = _read_done(sink_path, spec.id)
    if mlx is not None:
        bound_allocator(mlx, request.memory_cap_bytes)
    session = _Session(request.engine, engine, guard, mlx)
    runs = request.engine.decodes_runs()
    decoded = skipped = 0
    audio_s = 0.0
    with sink_path.open("a", encoding="utf-8") as sink:
        for job in request.files:
            audio = load_wav(Path(job.wav))
            chunks = _plan(job, request.engine, audio)
            if runs:
                finished = [row for (file, _), row in done.items() if file == job.id]
                if _tiles(finished, chunks):
                    skipped += len(finished)
                    continue
                spans = plan_runs([session.label(job, audio, chunk) for chunk in chunks])
            else:
                spans = [DecodeUnit(chunk) for chunk in chunks]
            for index, span in enumerate(spans):
                existing = done.get((job.id, index))
                if existing is not None:
                    _require_same_chunk(existing, span.chunk)
                    skipped += 1
                    continue
                unit = span if runs else session.label(job, audio, span.chunk)
                row = session.decode(spec.id, job, audio, index, unit)
                sink.write(row.model_dump_json() + "\n")
                sink.flush()
                decoded += 1
                audio_s += unit.chunk.duration
    metrics = RunMetrics(
        hypothesis=spec.id,
        files=len(request.files),
        chunks_decoded=decoded,
        chunks_skipped=skipped,
        load_s=session.load_s,
        decode_s=session.decode_s,
        audio_s=audio_s,
        peak_memory_bytes=mlx_worker.peak_memory_bytes(),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_footprint_bytes=mlx_worker.peak_footprint_bytes(),
    )
    atomic.write_json(out / METRICS_FILE, metrics.model_dump(mode="json"))
    return metrics


class _Session:
    """Builds and loads the engine on the first chunk that needs decoding, so a fully resumed run loads nothing."""

    def __init__(
        self, definition: EngineDef, engine: Engine | None, guard: Guard | None, mlx: MlxAllocator | None
    ) -> None:
        self._definition = definition
        self._engine = engine
        self._guard = guard
        self._mlx = mlx
        self._loaded = False
        self.load_s = 0.0
        self.decode_s = 0.0

    def label(self, job: FileJob, audio: Audio, chunk: Chunk) -> DecodeUnit:
        """Choose the language `chunk` is decoded with: the declared one, else under `detect` the detected one."""
        declared = self._definition.engine_language(job.language)
        if declared is not None or self._definition.language_policy() != "detect":
            return DecodeUnit(chunk, declared)
        engine = self._ready()
        hypothesis = self._definition.spec.id
        if not isinstance(engine, LanguageDetector):
            raise EngineError(f"{hypothesis} detects languages, but its engine has no detect_language")
        started = time.perf_counter()
        try:
            language, probability = pick_language(
                engine.detect_language(audio.slice(chunk.start, chunk.end)), job.languages
            )
        except Exception as exc:
            where = f"{job.id} ({chunk.start:.1f}-{chunk.end:.1f}s)"
            raise EngineError(f"{hypothesis} language detection failed on {where}: {exc}") from exc
        finally:
            self.decode_s += time.perf_counter() - started
        if self._mlx is not None:
            self._mlx.clear_cache()
        return DecodeUnit(chunk, language, probability)

    def decode(self, hypothesis: str, job: FileJob, audio: Audio, index: int, unit: DecodeUnit) -> ChunkHypothesis:
        engine = self._ready()
        chunk, language = unit.chunk, unit.language
        samples = audio.slice(chunk.start, chunk.end)
        started = time.perf_counter()
        try:
            raw = engine.transcribe(samples, language=language, context=job.context)
            guarded = (
                GuardedChunk(raw)
                if self._guard is None
                else self._guard(engine, ChunkInput(samples, chunk.start, chunk.end, language, job.context), raw)
            )
        except Exception as exc:
            where = f"{job.id} chunk {index} ({chunk.start:.1f}-{chunk.end:.1f}s)"
            raise EngineError(f"{hypothesis} failed on {where}: {exc}") from exc
        finally:
            self.decode_s += time.perf_counter() - started
        if self._mlx is not None:
            self._mlx.clear_cache()
        return _row(hypothesis, job.id, index, unit, guarded)

    def _ready(self) -> Engine:
        if self._engine is None:
            self._engine = build_engine(self._definition)
        if not self._loaded:
            started = time.perf_counter()
            self._engine.load()
            self.load_s = time.perf_counter() - started
            self._loaded = True
        return self._engine


def _row(hypothesis: str, file: str, index: int, unit: DecodeUnit, guarded: GuardedChunk) -> ChunkHypothesis:
    raw, chunk = guarded.raw, unit.chunk
    flags: list[ChunkFlag] = ["forced_cut"] if chunk.forced_cut else []
    if raw.generated_tokens is not None and raw.max_tokens is not None and raw.generated_tokens >= raw.max_tokens:
        flags.append("token_cap_hit")
    flags.extend(raw.flags)
    flags.extend(guarded.flags)
    return ChunkHypothesis(
        hypothesis=hypothesis,
        file=file,
        chunk=index,
        start=chunk.start,
        end=chunk.end,
        text=raw.text,
        language=raw.language,
        language_prob=unit.language_prob,
        words=_absolute(raw, chunk.start),
        generated_tokens=raw.generated_tokens,
        max_tokens=raw.max_tokens,
        confidence=raw.confidence,
        flags=tuple(dict.fromkeys(flags)),
        anomaly_spans=tuple(
            AnomalySpan(start=chunk.start + start, end=chunk.start + end) for start, end in guarded.anomaly_spans
        ),
    )


def _absolute(raw: RawChunk, offset: float) -> tuple[Word, ...]:
    return tuple(Word(text=w.text, start=w.start + offset, end=w.end + offset, conf=w.conf) for w in raw.words)


def _plan(job: FileJob, definition: EngineDef, audio: Audio) -> list[Chunk]:
    if audio.duration == 0:
        return []
    max_chunk_s = definition.spec.max_chunk_s
    if max_chunk_s is None:
        return [Chunk(0.0, audio.duration)]
    if job.chunks is not None:
        return list(job.chunks)
    return plan_chunks(speech_regions(audio), audio.duration, max_chunk_s=max_chunk_s, frame_energy=frame_energy(audio))


def pick_language(probabilities: Mapping[str, float], allowed: Sequence[str]) -> tuple[str, float]:
    """Return the most probable language within `allowed` (any when empty) and the detector's probability for it.

    The probability is the unrestricted one, so a low value says the audio sounded like none of the allowed
    languages. A tie goes to the earlier entry of `allowed`.
    """
    missing = [code for code in allowed if code not in probabilities]
    if missing:
        raise ValueError(f"the allowed languages {missing} are not ones the detector knows")
    candidates = list(allowed or probabilities)
    if not candidates:
        raise ValueError("the detector returned no languages")
    best = max(candidates, key=lambda code: probabilities[code])
    return best, probabilities[best]


def plan_runs(units: Sequence[DecodeUnit]) -> list[DecodeUnit]:
    """Merge consecutive units with the same language into one run spanning the first's start to the last's end.

    The pauses between a run's chunks are decoded with it, as a whole-file decode would. A run keeps the forced cut
    of its last chunk, since that is where it ends, and the lowest language probability among its chunks.
    """
    runs: list[DecodeUnit] = []
    for unit in units:
        last = runs[-1] if runs else None
        if last is None or last.language != unit.language:
            runs.append(unit)
            continue
        probability = (
            None
            if last.language_prob is None or unit.language_prob is None
            else min(last.language_prob, unit.language_prob)
        )
        span = Chunk(last.chunk.start, unit.chunk.end, forced_cut=unit.chunk.forced_cut)
        runs[-1] = DecodeUnit(span, unit.language, probability)
    return runs


def _tiles(rows: Sequence[ChunkHypothesis], chunks: Sequence[Chunk]) -> bool:
    """Whether finished run rows cover every chunk of the plan, so a resumed run need not detect languages again."""
    ordered = sorted(rows, key=lambda row: row.chunk)
    if [row.chunk for row in ordered] != list(range(len(ordered))):
        return False
    remaining = iter(chunks)
    for row in ordered:
        chunk = next(remaining, None)
        if chunk is None or not math.isclose(chunk.start, row.start, abs_tol=_PLAN_TOLERANCE_S):
            return False
        while not math.isclose(chunk.end, row.end, abs_tol=_PLAN_TOLERANCE_S):
            chunk = next(remaining, None)
            if chunk is None or chunk.end > row.end + _PLAN_TOLERANCE_S:
                return False
    return next(remaining, None) is None


def default_guard(request: WorkerRequest) -> Guard | None:
    """Return the guard `main` runs: none when the request opts out, else one scoped to how the engine decodes.

    A whole-file hypothesis and a runs hypothesis each decode many of the engine's windows in one call, so the guard
    repairs the failed window rather than the whole call; a chunked one decodes at most one window per call.
    """
    if not request.quality_guard:
        return None
    engine = request.engine
    return QualityGuard(whole_call=engine.spec.max_chunk_s is None or engine.decodes_runs())


def _claim_output(out: Path, request: WorkerRequest) -> None:
    hypotheses = out / HYPOTHESES_FILE
    existing = HYPOTHESES_FILE if hypotheses.exists() and hypotheses.stat().st_size > 0 else None
    mlx_worker.claim_output(out, request.fingerprint(), existing=existing)


def _read_done(path: Path, hypothesis: str) -> dict[tuple[str, int], ChunkHypothesis]:
    if not path.exists():
        return {}
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    if len(complete) != len(data):
        # A kill between write and flush leaves a partial last line; that chunk is simply decoded again.
        with path.open("r+b") as handle:
            handle.truncate(len(complete))
    done: dict[tuple[str, int], ChunkHypothesis] = {}
    for number, line in enumerate(complete.decode("utf-8").splitlines(), start=1):
        try:
            row = ChunkHypothesis.model_validate_json(line)
        except ValueError as exc:
            raise EngineError(f"{path.name} line {number} is corrupt: {exc}") from exc
        if row.hypothesis != hypothesis:
            raise EngineError(f"{path.name} holds hypothesis {row.hypothesis!r}, not {hypothesis!r}")
        done[(row.file, row.chunk)] = row
    return done


def _require_same_chunk(existing: ChunkHypothesis, chunk: Chunk) -> None:
    if not (
        math.isclose(existing.start, chunk.start, abs_tol=_PLAN_TOLERANCE_S)
        and math.isclose(existing.end, chunk.end, abs_tol=_PLAN_TOLERANCE_S)
    ):
        raise EngineError(
            f"{existing.file} chunk {existing.chunk} was decoded for {existing.start:.3f}-{existing.end:.3f}s but the "
            f"plan now says {chunk.start:.3f}-{chunk.end:.3f}s; use a fresh output directory for a different plan"
        )


def main(argv: Sequence[str] | None = None) -> int:
    return mlx_worker.run_child(
        argv,
        module="anyverbatim.asr.worker",
        noun="worker",
        parse=WorkerRequest.model_validate_json,
        run_request=lambda request: run_request(request, guard=default_guard(request), mlx=mlx_worker.mlx_allocator()),
    )


if __name__ == "__main__":
    sys.exit(main())
