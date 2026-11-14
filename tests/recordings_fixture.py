"""Fake recordings, ASR rows and a fake worker for the recordings evals; nothing here loads a model."""

import json
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from evals.recordings_data import (
    INDEX,
    OWNER,
    CorpusIndex,
    RefUtterance,
    wav_bytes,
    write_index,
    write_recording,
)
from ytverbatim import atomic
from ytverbatim.asr.models import (
    FINGERPRINT_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    REQUEST_FILE,
    ChunkHypothesis,
    RunMetrics,
    WorkerRequest,
)
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.contract.models import Word

SECONDS = 80


def silent(seconds: float) -> bytes:
    return wav_bytes(np.zeros(round(16_000 * seconds), dtype="<i2"))


def make_meeting(
    root: Path, corpus: str, name: str, utterances: Sequence[RefUtterance], seconds: float = SECONDS
) -> None:
    """Write one indexed recording (``audio.wav`` and references) under ``root/corpus``."""
    entry = write_recording(root / corpus, name, silent(seconds), utterances)
    index_path = root / corpus / INDEX
    known = (
        CorpusIndex.model_validate_json(index_path.read_text(encoding="utf-8")).recordings
        if index_path.exists()
        else ()
    )
    write_index(
        root / corpus, CorpusIndex(corpus=corpus, source="synthetic", license="none", recordings=(*known, entry))
    )


def make_owner(root: Path, entries: Sequence[tuple[str, float, dict[str, object]]]) -> None:
    """Write the owner index and a silent WAV for each ``(id, seconds, extra index fields)``."""
    folder = root / OWNER
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for recording_id, seconds, extra in entries:
        (folder / f"{recording_id}.wav").write_bytes(silent(seconds))
        records.append({"id": recording_id, "audio": f"{recording_id}.wav", **extra})
    (folder / INDEX).write_text(json.dumps({"recordings": records}), encoding="utf-8")


def make_clips(root: Path, clips: Sequence[tuple[str, str]]) -> None:
    """Write ASCEND-style clips ``(id, reference text)`` with their ``utterances.jsonl``."""
    folder = root / "ascend" / "utterances"
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for clip_id, text in clips:
        (folder / f"{clip_id}.wav").write_bytes(silent(3))
        lines.append(json.dumps({"id": clip_id, "language": "mixed", "text": text}, ensure_ascii=False))
    (folder / "utterances.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def spoken(
    hypothesis: str,
    file: str,
    pieces: Sequence[tuple[str, float, float]],
    *,
    chunk: int = 0,
    language: str | None = None,
) -> ChunkHypothesis:
    """One row whose words are ``(text, start, end)`` pieces."""
    words = tuple(Word(text=t, start=s, end=e) for t, s, e in pieces)
    return ChunkHypothesis(
        hypothesis=hypothesis,
        file=file,
        chunk=chunk,
        start=min(w.start for w in words),
        end=max(w.end for w in words),
        text="".join(w.text for w in words),
        language=language,
        words=words,
    )


class FakeWorker:
    """Stands in for ``run_worker``: writes what a finished worker leaves behind, from ``rows_for(request)``."""

    def __init__(self, rows_for: Callable[[WorkerRequest], Sequence[ChunkHypothesis]]) -> None:
        self.rows_for = rows_for
        self.requests: list[WorkerRequest] = []

    def __call__(self, request: WorkerRequest) -> WorkerResult:
        self.requests.append(request)
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = self.rows_for(request)
        (out / HYPOTHESES_FILE).write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
        atomic.write_text(out / REQUEST_FILE, request.model_dump_json(indent=2) + "\n")
        atomic.write_json(out / FINGERPRINT_FILE, request.fingerprint())
        metrics = RunMetrics(
            hypothesis=request.engine.spec.id,
            files=len(request.files),
            chunks_decoded=len(rows),
            chunks_skipped=0,
            load_s=1.0,
            decode_s=2.0,
            audio_s=10.0,
            peak_memory_bytes=None,
            max_rss_bytes=0,
        )
        atomic.write_json(out / METRICS_FILE, metrics.model_dump(mode="json"))
        return WorkerResult(out / HYPOTHESES_FILE, metrics, 0, 0.5)


def wav_seconds_of(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def pieces(text: str, start: float, step: float = 0.4) -> list[tuple[str, float, float]]:
    """Word pieces for ``text``: leading-space words for English, one piece per character for Chinese."""
    words = [f" {w}" for w in text.split()] if " " in text else list(text)
    return [(w, start + i * step, start + (i + 1) * step) for i, w in enumerate(words)]
