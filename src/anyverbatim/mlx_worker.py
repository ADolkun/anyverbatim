"""What the ASR and diarization workers share.

That is the machine-wide MLX lock, the parent's subprocess skeleton, and the child's entry point, output claiming and
memory bounds.

Nothing here imports MLX or numpy at import time; the child reaches `mlx.core` only through `importlib`.
"""

import contextlib
import fcntl
import importlib
import json
import logging
import os
import resource
import sys
import traceback
import wave
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from anyverbatim import atomic, paths, proc, sysmem
from anyverbatim.errors import (
    EngineError,
    EngineMemoryError,
    MediaValidationError,
    ProcMemoryExceeded,
    stderr_preview,
)

log = logging.getLogger(__name__)

LOCK_NAME = "mlx-worker.lock"
METRICS_FILE = "metrics.json"
ERROR_FILE = "error.json"
REQUEST_FILE = "request.json"
FINGERPRINT_FILE = "fingerprint.json"
EXIT_FAILED = 1
EXIT_BAD_REQUEST = 2
TIMEOUT_FACTOR = 3.0
TIMEOUT_FLOOR_S = 30 * 60.0
DEFAULT_MEMORY_CAP_FRACTION = 0.5
CAP_SETTING = "memory_cap_gib"
# Measured on f-aed: a 1 or 2 GiB freed-buffer cache decoded no faster than none (about 9.6-9.8 s per chunk either
# way) and only added its own size to the footprint.
CACHE_LIMIT_BYTES = 0
# The process footprint always exceeds MLX's own active memory (Python, weights being read, Metal bookkeeping), so a
# limit equal to the cap would never come into play before the parent kills the worker.
MLX_LIMIT_FRACTION = 0.85


class _Job(Protocol):
    @property
    def wav(self) -> str: ...

    def model_copy(self, *, update: Mapping[str, Any] | None = None) -> Self: ...


class Request(Protocol):
    """What either worker's request model offers the shared code."""

    @property
    def out_dir(self) -> str: ...
    @property
    def hf_hub_cache(self) -> str | None: ...
    @property
    def memory_cap_bytes(self) -> int | None: ...
    @property
    def files(self) -> Sequence[_Job]: ...

    def model_copy(self, *, update: Mapping[str, Any] | None = None) -> Self: ...
    def model_dump_json(self, *, indent: int | None = None) -> str: ...


class Fingerprinted(Request, Protocol):
    def fingerprint(self) -> dict[str, JsonValue]: ...


class WorkerError(BaseModel):
    """The structured failure the worker leaves in `error.json` for the parent to read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    message: str
    traceback: str


@dataclass(frozen=True)
class Completed[M: BaseModel]:
    out: Path
    metrics: M
    # ru_maxrss of the largest child this process has waited for, in bytes on macOS; an upper bound for this worker.
    children_max_rss_bytes: int
    elapsed_s: float


class _MlxCore(Protocol):
    def get_peak_memory(self) -> int: ...


class MlxAllocator(Protocol):
    def set_memory_limit(self, limit: int) -> int: ...
    def set_cache_limit(self, limit: int) -> int: ...
    def clear_cache(self) -> None: ...


def worker_argv(module: str, request_path: Path, *, python: str | None = None) -> list[str]:
    """Return the child's command line; `-P` keeps the working directory off its `sys.path`."""
    return [python or sys.executable, "-P", "-m", module, str(request_path)]


def timeout_s(audio_seconds: float) -> float:
    return TIMEOUT_FACTOR * audio_seconds + TIMEOUT_FLOOR_S


def default_memory_cap_bytes() -> int:
    return int(sysmem.physical_memory_bytes() * DEFAULT_MEMORY_CAP_FRACTION)


def anchored[R: Request](request: R) -> R:
    """Return the request with every path absolute, since the worker runs in its output folder.

    mlx-whisper and mlx-audio load a directory named like the model repo from the working directory in place of the
    pinned download, so the child's working directory must hold no such directory. Absolute paths are left as they are,
    which keeps fingerprints that include the WAV path unchanged.
    """
    files = tuple(job.model_copy(update={"wav": str(Path(job.wav).absolute())}) for job in request.files)
    cache = None if request.hf_hub_cache is None else str(Path(request.hf_hub_cache).absolute())
    return request.model_copy(
        update={"out_dir": str(Path(request.out_dir).absolute()), "files": files, "hf_hub_cache": cache}
    )


def run[R: Request, M: BaseModel](
    request: R,
    *,
    label: str,
    module: str,
    metrics: type[M],
    python: str | None = None,
    smaller: str = "use a smaller engine",
    kept: str = "finished chunks",
) -> Completed[M]:
    """Run the request in a worker subprocess and return where its output is, or raise `EngineError`.

    The worker is killed if its footprint passes `request.memory_cap_bytes` (default: half of physical memory) or the
    machine runs short of memory; what it finished stays on disk for the next run to resume from.
    """
    cap = request.memory_cap_bytes or default_memory_cap_bytes()
    request = anchored(request).model_copy(update={"memory_cap_bytes": cap})
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    request_path = out / REQUEST_FILE
    limit = timeout_s(sum(_wav_seconds(Path(job.wav)) for job in request.files))
    with mlx_lock():
        # Under the lock, so a run waiting its turn cannot overwrite the request a running worker was given, and a
        # report left by an earlier run cannot be read as this worker's failure.
        (out / ERROR_FILE).unlink(missing_ok=True)
        atomic.write_text(request_path, request.model_dump_json(indent=2) + "\n")
        try:
            result = proc.run(
                worker_argv(module, request_path, python=python),
                timeout_s=limit,
                memory_watch=proc.MemoryWatch(cap),
                cwd=out,
            )
        except ProcMemoryExceeded as exc:
            detail = f"{label} worker stopped to protect the machine: {exc}"
            raise EngineMemoryError(detail, memory_fix(exc, smaller=smaller, kept=kept), exc) from exc
    children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if result.returncode != 0:
        raise EngineError(_failure(label, out, result.returncode, result.stderr))
    try:
        run_metrics = metrics.model_validate_json((out / METRICS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"{label} worker exited cleanly but left no readable {METRICS_FILE}: {exc}") from exc
    return Completed(out, run_metrics, children_rss, result.elapsed_s)


@contextlib.contextmanager
def mlx_lock() -> Generator[int]:
    """Hold the machine-wide MLX lock; two models at once would fight over unified memory.

    Yields the locked descriptor. A child given a copy (`pass_fds`) keeps the lock held until it exits, even if this
    process dies first, because a flock belongs to the open file and not to the process that took it.
    """
    fd = os.open(paths.locks_dir() / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("waiting for another MLX worker (ASR or diarization) to finish (%s)", LOCK_NAME)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        os.close(fd)


def memory_fix(exc: ProcMemoryExceeded, *, smaller: str = "use a smaller engine", kept: str = "finished chunks") -> str:
    """Say how to avoid another kill by the memory watch; `smaller` names the lighter choice, `kept` what survives."""
    resume = f"{kept} are kept and the next run resumes from them"
    if exc.trigger == "available":
        return f"close memory-heavy apps, then run again ({resume})"
    return (
        f"{smaller}, or raise asr.{CAP_SETTING} in the config file only if this Mac has that much memory to spare "
        f"({resume})"
    )


def follow_moved_audio(request: Fingerprinted) -> None:
    """Re-stamp `request.out_dir` when it holds this same request for a WAV path that has since changed.

    The worker then resumes the folder instead of refusing it as another request's output. Only for a caller that
    owns the folder exclusively, as the pipeline does under its per-source lock.
    """
    request = anchored(request)
    out = Path(request.out_dir)
    stored = stored_fingerprint(out)
    current = request.fingerprint()
    if stored is not None and stored != current and without_wav(stored) == without_wav(current):
        atomic.write_json(out / FINGERPRINT_FILE, current)


def stored_fingerprint(out: Path) -> JsonValue | None:
    """Return the fingerprint of the request whose output `out` holds, or None if it holds none readable."""
    try:
        stored: JsonValue = json.loads((out / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return stored


def without_wav(fingerprint: JsonValue) -> JsonValue:
    """Drop the WAV paths from a fingerprint: where the audio lies says nothing about what decoding it gives."""
    if not isinstance(fingerprint, dict):
        return fingerprint
    files = fingerprint.get("files")
    if not isinstance(files, list):
        return fingerprint
    kept: list[JsonValue] = [
        {k: v for k, v in job.items() if k != "wav"} if isinstance(job, dict) else job for job in files
    ]
    return {**fingerprint, "files": kept}


def _failure(label: str, out: Path, returncode: int, stderr: str) -> str:
    try:
        error = WorkerError.model_validate_json((out / ERROR_FILE).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return f"{label} worker exited with {returncode} and left no error report\n{stderr_preview(stderr)}"
    return f"{label} worker failed ({error.type}): {error.message}"


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes() / reader.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise MediaValidationError(f"not a readable WAV file: {exc}") from exc


def claim_output(out: Path, current: dict[str, JsonValue], *, existing: str | None) -> None:
    """Record which request this output directory belongs to, or refuse to resume one that belongs to another.

    `existing` names the output already in `out` when there is any; it is refused if no fingerprint says what made it.
    """
    path = out / FINGERPRINT_FILE
    if not path.exists():
        if existing is not None:
            raise EngineError(
                f"{out} holds {existing} but no {FINGERPRINT_FILE}, so what produced it is unknown; use a fresh "
                "output directory"
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


def bound_allocator(
    mlx: MlxAllocator, memory_cap_bytes: int | None, *, cache_limit_bytes: int = CACHE_LIMIT_BYTES
) -> None:
    """Keep MLX's freed-buffer cache small and steer its scheduling under the cap.

    MLX defaults both limits to about the whole of RAM, so without this the cache alone can grow until the machine
    stops. The memory limit is a soft scheduling guideline: past it MLX waits for queued work to finish before it
    allocates more, but it still allocates. The parent's watch is the hard stop.
    """
    mlx.set_cache_limit(cache_limit_bytes)
    if memory_cap_bytes is not None:
        mlx.set_memory_limit(int(memory_cap_bytes * MLX_LIMIT_FRACTION))


def mlx_allocator() -> MlxAllocator | None:
    # Without MLX installed no model can allocate through it, and the model's own load reports the missing extra.
    try:
        return cast(MlxAllocator, importlib.import_module("mlx.core"))
    except ImportError:
        return None


def peak_footprint_bytes() -> int | None:
    sample = sysmem.footprint(os.getpid())
    return None if sample is None else sample.peak


def peak_memory_bytes() -> int | None:
    # The module is loaded only where MLX is installed and `main` bounded it or a model used it; importing it here
    # would fail an in-process run without MLX, which has no peak to report.
    core = sys.modules.get("mlx.core")
    return None if core is None else cast(_MlxCore, core).get_peak_memory()


def run_child[R: Request](
    argv: Sequence[str] | None,
    *,
    module: str,
    noun: str,
    parse: Callable[[str], R],
    run_request: Callable[[R], object],
) -> int:
    """Entry point of a worker child: read the request, run it, and leave a typed `error.json` if that raises."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(f"usage: python -P -m {module} REQUEST.json", file=sys.stderr)
        return EXIT_BAD_REQUEST
    try:
        request = parse(Path(args[0]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"invalid {noun} request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST
    if request.hf_hub_cache is not None:
        # huggingface_hub reads this when it is first imported, which only happens inside an adapter.
        os.environ["HF_HUB_CACHE"] = request.hf_hub_cache
    try:
        run_request(request)
    except Exception as exc:
        error = WorkerError(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic.write_json(out / ERROR_FILE, error.model_dump(mode="json"))
        print(f"{error.type}: {error.message}", file=sys.stderr)
        return EXIT_FAILED
    return 0
