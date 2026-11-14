"""Parent side of the diarization worker: same lock, memory watch and failure reporting as the ASR worker.

It takes the ASR worker's machine-wide MLX lock, so a diarizer never runs while an ASR engine does. Nothing here
imports MLX or numpy.

`python -m ytverbatim.diarize.runner DIARIZER OUT_DIR WAV...` diarizes WAVs outside the pipeline, for smoke runs and
evals; it prints the metrics and each file's speaker count as JSON.
"""

import json
import resource
import sys
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from ytverbatim import atomic, proc
from ytverbatim.asr.models import WorkerError, without_wav
from ytverbatim.asr.runner import default_memory_cap_bytes, memory_fix, timeout_s, worker_lock
from ytverbatim.diarize import registry
from ytverbatim.diarize.protocol import (
    ERROR_FILE,
    FINGERPRINT_FILE,
    METRICS_FILE,
    REQUEST_FILE,
    DiarizeRequest,
    FileJob,
    RunMetrics,
    read_file_diarization,
)
from ytverbatim.errors import EngineError, EngineMemoryError, MediaValidationError, ProcMemoryExceeded, stderr_preview

WORKER_MODULE = "ytverbatim.diarize.worker"


@dataclass(frozen=True)
class DiarizeResult:
    out_dir: Path
    metrics: RunMetrics
    # ru_maxrss of the largest child this process has waited for, in bytes on macOS; an upper bound for this worker.
    children_max_rss_bytes: int
    elapsed_s: float


def worker_argv(request_path: Path, *, python: str | None = None) -> list[str]:
    return [python or sys.executable, "-P", "-m", WORKER_MODULE, str(request_path)]


def run_worker(request: DiarizeRequest, *, python: str | None = None) -> DiarizeResult:
    """Run the request in a worker subprocess and return where its output is, or raise `EngineError`.

    The worker is killed if its footprint passes `request.memory_cap_bytes` (default: half of physical memory) or the
    machine runs short of memory; the files it finished stay on disk for the next run to skip.
    """
    cap = request.memory_cap_bytes or default_memory_cap_bytes()
    request = request.model_copy(update={"memory_cap_bytes": cap})
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    request_path = out / REQUEST_FILE
    limit = timeout_s(sum(_wav_seconds(Path(job.wav)) for job in request.files))
    diarizer = request.diarizer.spec.id
    with worker_lock():
        # Under the lock, so a run waiting its turn cannot overwrite a running worker's request or read a report left
        # by an earlier run as this worker's failure.
        (out / ERROR_FILE).unlink(missing_ok=True)
        atomic.write_text(request_path, request.model_dump_json(indent=2) + "\n")
        try:
            result = proc.run(
                worker_argv(request_path, python=python), timeout_s=limit, memory_watch=proc.MemoryWatch(cap)
            )
        except ProcMemoryExceeded as exc:
            detail = f"{diarizer} worker stopped to protect the machine: {exc}"
            fix = memory_fix(exc, smaller="use the 8-bit diarizer", kept="finished files")
            raise EngineMemoryError(detail, fix, exc) from exc
    children_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if result.returncode != 0:
        raise EngineError(_failure(diarizer, out, result.returncode, result.stderr))
    try:
        metrics = RunMetrics.model_validate_json((out / METRICS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"{diarizer} worker exited cleanly but left no readable {METRICS_FILE}: {exc}") from exc
    return DiarizeResult(out, metrics, children_rss, result.elapsed_s)


def finished(request: DiarizeRequest) -> Path | None:
    """Return the output folder a completed run of this request left, or None; the WAV paths are not compared."""
    out = Path(request.out_dir)
    if not (out / METRICS_FILE).is_file():
        return None
    try:
        stored: JsonValue = json.loads((out / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if without_wav(stored) == without_wav(request.fingerprint()) else None


def _failure(diarizer: str, out: Path, returncode: int, stderr: str) -> str:
    try:
        error = WorkerError.model_validate_json((out / ERROR_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"{diarizer} worker exited with {returncode} and left no error report\n{stderr_preview(stderr)}"
    return f"{diarizer} worker failed ({error.type}): {error.message}"


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getnframes() / reader.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise MediaValidationError(f"not a readable WAV file: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3:  # noqa: PLR2004
        print("usage: python -m ytverbatim.diarize.runner DIARIZER OUT_DIR WAV...", file=sys.stderr)
        return 2
    definition = registry.get(args[0])
    registry.check_dependencies(definition)
    wavs = [Path(arg) for arg in args[2:]]
    jobs = tuple(FileJob(id=f"f{index}", wav=str(wav.resolve())) for index, wav in enumerate(wavs, start=1))
    result = run_worker(DiarizeRequest(diarizer=definition, out_dir=args[1], files=jobs))
    files = {
        str(wav): {"num_speakers": stored.num_speakers, "turns": len(stored.turns), "duration_s": stored.duration_s}
        for wav, stored in (
            (wav, read_file_diarization(result.out_dir, job.id)) for wav, job in zip(wavs, jobs, strict=True)
        )
    }
    summary = {"metrics": result.metrics.model_dump(mode="json"), "elapsed_s": result.elapsed_s, "files": files}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
