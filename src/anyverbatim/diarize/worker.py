"""Child process that loads one diarizer and diarizes a request's files: `python -P -m anyverbatim.diarize.worker`.

Everything that needs MLX happens here, never in the parent. Each file's output is written whole, probabilities first
and turns last, so a file with a turns file is done and a rerun skips it without loading the model.
"""

import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from anyverbatim import atomic, mlx_worker
from anyverbatim.asr.audio import Audio, load_wav
from anyverbatim.diarize.protocol import (
    ERROR_FILE,
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
from anyverbatim.diarize.registry import build_diarizer
from anyverbatim.errors import EngineError
from anyverbatim.mlx_worker import MlxAllocator, bound_allocator

_PROB_DIMS = 2


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
        peak_memory_bytes=mlx_worker.peak_memory_bytes(),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_footprint_bytes=mlx_worker.peak_footprint_bytes(),
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
    existing = "diarization output" if any(out.glob(f"*{TURNS_SUFFIX}")) else None
    mlx_worker.claim_output(out, request.fingerprint(), existing=existing)


def main(argv: Sequence[str] | None = None) -> int:
    return mlx_worker.run_child(
        argv,
        module="anyverbatim.diarize.worker",
        noun="diarization",
        parse=DiarizeRequest.model_validate_json,
        run_request=lambda request: run_request(request, mlx=mlx_worker.mlx_allocator()),
    )


if __name__ == "__main__":
    sys.exit(main())
