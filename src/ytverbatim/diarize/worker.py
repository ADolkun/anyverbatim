"""Child process that loads one diarizer and diarizes a request's files: `python -P -m ytverbatim.diarize.worker`.

Everything that needs MLX happens here, never in the parent. Each file's output is written whole, probabilities first
and turns last, so a file with a turns file is done and a rerun skips it without loading the model.
"""

import importlib
import json
import os
import resource
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from ytverbatim import atomic, sysmem
from ytverbatim.asr.audio import Audio, load_wav
from ytverbatim.asr.models import WorkerError
from ytverbatim.asr.worker import MlxAllocator, bound_allocator
from ytverbatim.diarize.protocol import (
    ERROR_FILE,
    FINGERPRINT_FILE,
    METRICS_FILE,
    TURNS_SUFFIX,
    Diarizer,
    DiarizeRequest,
    FileDiarization,
    FileJob,
    RawDiarization,
    RunMetrics,
    load,
    probs_path,
    turns_path,
)
from ytverbatim.diarize.registry import build_diarizer
from ytverbatim.errors import EngineError

EXIT_FAILED = 1
EXIT_BAD_REQUEST = 2
_PROB_DIMS = 2


class _MlxCore(Protocol):
    def get_peak_memory(self) -> int: ...


def run_request(
    request: DiarizeRequest, *, diarizer: Diarizer | None = None, mlx: MlxAllocator | None = None
) -> RunMetrics:
    """Diarize every file not already on disk.

    `diarizer` replaces the one built from the request, for in-process tests. `mlx` is the allocator to bound before
    anything loads and to empty after every file; `main` passes `mlx.core`.
    """
    spec = request.diarizer.spec
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ERROR_FILE).unlink(missing_ok=True)
    _claim_output(out, request)
    # Written again only once every file is on disk, so its presence says this request's output is complete.
    (out / METRICS_FILE).unlink(missing_ok=True)
    if mlx is not None:
        bound_allocator(mlx, request.memory_cap_bytes)
    session = _Session(request, diarizer)
    done = skipped = 0
    audio_s = 0.0
    for job in request.files:
        if turns_path(out, job.id).is_file():
            load(out, job.id, spec.id)
            skipped += 1
            continue
        audio = load_wav(Path(job.wav))
        result = session.diarize(job, audio)
        probs = result.probs.astype("<f2")
        atomic.write_bytes(probs_path(out, job.id), probs.tobytes())
        stored = FileDiarization(
            diarizer=spec.id,
            file=job.id,
            duration_s=audio.duration,
            threshold=request.diarizer.threshold(),
            frame_s=result.frame_s,
            frames=probs.shape[0],
            channels=probs.shape[1],
            num_speakers=len({turn.speaker for turn in result.turns}),
            turns=result.turns,
        )
        atomic.write_json(turns_path(out, job.id), stored.model_dump(mode="json"))
        if mlx is not None:
            mlx.clear_cache()
        done += 1
        audio_s += audio.duration
    metrics = RunMetrics(
        diarizer=spec.id,
        files=len(request.files),
        files_diarized=done,
        files_skipped=skipped,
        load_s=session.load_s,
        diarize_s=session.diarize_s,
        audio_s=audio_s,
        peak_memory_bytes=_peak_memory_bytes(),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_footprint_bytes=_peak_footprint_bytes(),
    )
    atomic.write_json(out / METRICS_FILE, metrics.model_dump(mode="json"))
    return metrics


class _Session:
    """Builds and loads the diarizer on the first file that needs it, so a fully resumed run loads nothing."""

    def __init__(self, request: DiarizeRequest, diarizer: Diarizer | None) -> None:
        self._request = request
        self._diarizer = diarizer
        self._loaded = False
        self.load_s = 0.0
        self.diarize_s = 0.0

    def diarize(self, job: FileJob, audio: Audio) -> RawDiarization:
        diarizer = self._ready()
        started = time.perf_counter()
        try:
            raw = diarizer.diarize(audio.slice())
        except Exception as exc:
            raise EngineError(f"{self._request.diarizer.spec.id} failed on {job.id}: {exc}") from exc
        finally:
            self.diarize_s += time.perf_counter() - started
        channels = self._request.diarizer.spec.max_speakers
        if raw.probs.ndim != _PROB_DIMS or raw.probs.shape[1] != channels:
            raise EngineError(f"{job.id}: expected (frames, {channels}) probabilities, got {raw.probs.shape}")
        # NaN fails both comparisons, so this also refuses non-finite values.
        if not ((raw.probs >= 0) & (raw.probs <= 1)).all():
            raise EngineError(f"{job.id}: the diarizer returned probabilities outside [0, 1] or not finite")
        if any(turn.speaker >= channels for turn in raw.turns):
            raise EngineError(f"{job.id}: a turn names a speaker channel beyond the model's {channels}")
        return raw

    def _ready(self) -> Diarizer:
        if self._diarizer is None:
            self._diarizer = build_diarizer(self._request.diarizer)
        if not self._loaded:
            started = time.perf_counter()
            self._diarizer.load()
            self.load_s = time.perf_counter() - started
            self._loaded = True
        return self._diarizer


def _claim_output(out: Path, request: DiarizeRequest) -> None:
    """Record which request this output directory belongs to, or refuse to resume one that belongs to another."""
    path = out / FINGERPRINT_FILE
    current = request.fingerprint()
    if not path.exists():
        if any(out.glob(f"*{TURNS_SUFFIX}")):
            raise EngineError(
                f"{out} holds diarization output but no {FINGERPRINT_FILE}, so what produced it is unknown; use a "
                "fresh output directory"
            )
        atomic.write_json(path, current)
        return
    try:
        stored: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"{path} is unreadable: {exc}") from exc
    if stored != current:
        known = stored if isinstance(stored, dict) else {}
        changed = sorted(key for key in current.keys() | known.keys() if current.get(key) != known.get(key))
        raise EngineError(
            f"{out} holds output of a different request ({', '.join(changed) or 'format'} changed); resuming would "
            "mix the two, so use a fresh output directory"
        )


def _mlx_allocator() -> MlxAllocator | None:
    # Without MLX installed no diarizer can allocate through it, and the adapter's own load reports the missing extra.
    try:
        return cast(MlxAllocator, importlib.import_module("mlx.core"))
    except ImportError:
        return None


def _peak_footprint_bytes() -> int | None:
    sample = sysmem.footprint(os.getpid())
    return None if sample is None else sample.peak


def _peak_memory_bytes() -> int | None:
    core = sys.modules.get("mlx.core")
    return None if core is None else cast(_MlxCore, core).get_peak_memory()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -P -m ytverbatim.diarize.worker REQUEST.json", file=sys.stderr)
        return EXIT_BAD_REQUEST
    try:
        request = DiarizeRequest.model_validate_json(Path(args[0]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"invalid diarization request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST
    if request.hf_hub_cache is not None:
        # huggingface_hub reads this when it is first imported, which only happens inside an adapter.
        os.environ["HF_HUB_CACHE"] = request.hf_hub_cache
    try:
        run_request(request, mlx=_mlx_allocator())
    except Exception as exc:
        error = WorkerError(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic.write_json(out / ERROR_FILE, error.model_dump(mode="json"))
        print(f"{error.type}: {error.message}", file=sys.stderr)
        return EXIT_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
