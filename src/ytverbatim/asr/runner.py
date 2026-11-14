"""Parent side of the ASR worker: one worker at a time per machine, in its own process, with a typed result.

Nothing here imports MLX or numpy; the engine libraries only ever load inside the child.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from ytverbatim import atomic, mlx_worker, sysmem
from ytverbatim.asr.models import FINGERPRINT_FILE, HYPOTHESES_FILE, METRICS_FILE, RunMetrics, WorkerRequest
from ytverbatim.errors import ConfigError
from ytverbatim.mlx_worker import CAP_SETTING, without_wav

WORKER_MODULE = "ytverbatim.asr.worker"
# With the 10% available-memory floor, a cap above 90% of RAM can never fire before the floor does, so it would be a
# cap in name only.
MAX_MEMORY_CAP_FRACTION = 0.9
_GIB = 1 << 30


@dataclass(frozen=True)
class WorkerResult:
    hypotheses: Path
    metrics: RunMetrics
    # ru_maxrss of the largest child this process has waited for, in bytes on macOS; an upper bound for this worker.
    children_max_rss_bytes: int
    elapsed_s: float


def memory_cap_bytes(gib: float | None) -> int | None:
    """Convert the `asr.memory_cap_gib` setting; None keeps the default cap."""
    if gib is None:
        return None
    if not math.isfinite(gib) or gib <= 0:
        raise ConfigError(f"asr.{CAP_SETTING} must be a positive number of GiB, got {gib}")
    cap = max(1, round(gib * _GIB))
    ceiling = sysmem.physical_memory_bytes() * MAX_MEMORY_CAP_FRACTION
    if cap > ceiling:
        raise ConfigError(
            f"asr.{CAP_SETTING} = {gib:g} is above 90% of this Mac's memory ({ceiling / _GIB:.1f} GiB), where the "
            "watch would only fire once the whole machine is short of memory; set it lower"
        )
    return cap


def run_worker(request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
    """Run the request in a worker subprocess and return where its output is, or raise `EngineError`.

    The worker is killed if its footprint passes `request.memory_cap_bytes` (default: half of physical memory) or the
    machine runs short of memory; the chunks it finished stay on disk for the next run to resume from.
    """
    done = mlx_worker.run(
        request, label=request.engine.spec.id, module=WORKER_MODULE, metrics=RunMetrics, python=python
    )
    return WorkerResult(done.out / HYPOTHESES_FILE, done.metrics, done.children_max_rss_bytes, done.elapsed_s)


def finished(request: WorkerRequest) -> Path | None:
    """Return the hypotheses a completed run of this request left in its `out_dir`, or None.

    Reading them needs neither the WAV nor the worker, so a change to what is made of the rows never decodes again.
    The WAV path is not compared: the same audio read from a moved or differently spelled folder decodes the same.
    """
    out = Path(request.out_dir)
    hypotheses = out / HYPOTHESES_FILE
    if not ((out / METRICS_FILE).is_file() and hypotheses.is_file()):
        return None
    stored = stored_fingerprint(out)
    return hypotheses if stored is not None and without_wav(stored) == without_wav(request.fingerprint()) else None


def follow_moved_audio(request: WorkerRequest) -> None:
    """Re-stamp `request.out_dir` when it holds this same request for a WAV path that has since changed.

    The worker then resumes the folder instead of refusing it as another request's output. Only for a caller that
    owns the folder exclusively, as the pipeline does under its per-video lock.
    """
    request = mlx_worker.anchored(request)
    out = Path(request.out_dir)
    stored = stored_fingerprint(out)
    current = request.fingerprint()
    if stored is not None and stored != current and without_wav(stored) == without_wav(current):
        atomic.write_json(out / FINGERPRINT_FILE, current)


def stored_fingerprint(out: Path) -> JsonValue | None:
    """Return the fingerprint of the request whose output `out` holds, or None if it holds none readable."""
    try:
        stored: JsonValue = json.loads((out / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stored
