"""Parent side of the diarization worker: same lock, memory watch and failure reporting as the ASR worker.

It takes the machine-wide MLX lock the ASR worker also takes, so a diarizer never runs while an ASR engine does. Nothing
here imports MLX or numpy.

`python -m anyverbatim.diarize.runner DIARIZER OUT_DIR WAV...` diarizes WAVs outside the pipeline, for smoke runs and
evals; it prints the metrics and each file's speaker count as JSON.
"""

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from anyverbatim import mlx_worker
from anyverbatim.diarize import registry
from anyverbatim.diarize.protocol import (
    FINGERPRINT_FILE,
    METRICS_FILE,
    DiarizeRequest,
    DiarizerSpec,
    FileJob,
    RunMetrics,
    read_file_diarization,
)
from anyverbatim.mlx_worker import without_wav

WORKER_MODULE = "anyverbatim.diarize.worker"


@dataclass(frozen=True)
class DiarizeResult:
    out_dir: Path
    metrics: RunMetrics
    # ru_maxrss of the largest child this process has waited for, in bytes on macOS; an upper bound for this worker.
    children_max_rss_bytes: int
    elapsed_s: float


def run_worker(request: DiarizeRequest, *, python: str | None = None) -> DiarizeResult:
    """Run the request in a worker subprocess and return where its output is, or raise `EngineError`.

    The worker is killed if its footprint passes `request.memory_cap_bytes` (default: half of physical memory) or the
    machine runs short of memory; the files it finished stay on disk for the next run to skip.
    """
    done = mlx_worker.run(
        request,
        label=request.diarizer.spec.id,
        module=WORKER_MODULE,
        metrics=RunMetrics,
        python=python,
        smaller=_smaller(request.diarizer.spec),
        kept="finished files",
    )
    return DiarizeResult(done.out, done.metrics, done.children_max_rss_bytes, done.elapsed_s)


def _smaller(spec: DiarizerSpec) -> str:
    lighter = [d.spec for d in registry.DIARIZERS.values() if d.spec.weights_bytes < spec.weights_bytes]
    if lighter:
        return f"use {min(lighter, key=lambda s: s.weights_bytes).id}"
    return "split the recording into shorter files"


def finished(request: DiarizeRequest) -> Path | None:
    """Return the output folder a completed run of this request left, or None; the WAV paths are not compared."""
    out = Path(request.out_dir)
    if not (out / METRICS_FILE).is_file():
        return None
    try:
        stored: JsonValue = json.loads((out / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return out if without_wav(stored) == without_wav(request.fingerprint()) else None


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3:  # noqa: PLR2004
        sys.stderr.write("usage: python -m anyverbatim.diarize.runner DIARIZER OUT_DIR WAV...\n")
        return 2
    definition = registry.get(args[0])
    registry.check_dependencies(definition)
    wavs = [Path(arg) for arg in args[2:]]
    jobs = tuple(FileJob(id=f"f{index}", wav=str(wav.resolve())) for index, wav in enumerate(wavs, start=1))
    result = run_worker(DiarizeRequest(diarizer=definition, out_dir=args[1], files=jobs))
    files = {
        str(wav): {"num_speakers": stored.num_speakers, "turns": len(stored.turns), "duration_s": stored.duration_s}
        for wav, stored in (
            (wav, read_file_diarization(result.out_dir, job.id, definition.spec.id))
            for wav, job in zip(wavs, jobs, strict=True)
        )
    }
    summary = {"metrics": result.metrics.model_dump(mode="json"), "elapsed_s": result.elapsed_s, "files": files}
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
