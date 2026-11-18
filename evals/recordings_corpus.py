"""The recordings the ASR evals decode and score, and where their runs live.

Every corpus becomes a list of ``Item`` (one audio file with an optional reference); the layout of the files is
``evals.recordings_data``'s. ASR runs go under ``<recordings>/asr/<corpus>/runs/<hypothesis>/<batch>/``, which is an
``evals.store.Layout`` rooted at ``<recordings>/asr/<corpus>``, so rows are read back the way the YouTube evals do.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from anyverbatim import atomic, legacy
from anyverbatim.asr import registry
from anyverbatim.asr.models import HYPOTHESES_FILE, METRICS_FILE, REQUEST_FILE, ChunkHypothesis, FileJob, WorkerRequest
from anyverbatim.asr.protocol import EngineDef
from anyverbatim.asr.runner import stored_fingerprint
from anyverbatim.errors import ConfigError
from anyverbatim.mlx_worker import without_wav
from evals.normalize import Language
from evals.recordings_data import (
    OWNER,
    REFERENCE_JSONL,
    RefUtterance,
    load_index,
    load_owner,
    recordings_dir,
    wav_seconds,
)
from evals.run_asr import ALLOWED_LANGUAGES
from evals.store import Layout, read_rows

Metric = Literal["wer", "cer", "mer"]

AMI = "ami"
ALIMEETING = "alimeeting"
ASCEND = "ascend"
ASCEND_UTTERANCES = "ascend-utterances"
ASR_CORPORA = (AMI, ALIMEETING, ASCEND, ASCEND_UTTERANCES, OWNER)
# What a command runs without --corpus: the public corpora. The owner's recordings run only when asked for, so a
# missing owner index never stops the public runs.
PUBLIC_CORPORA = (AMI, ALIMEETING, ASCEND, ASCEND_UTTERANCES)
# Corpora with one folder per recording, so a ``words.json`` can sit next to each audio file.
FOLDER_CORPORA = (AMI, ALIMEETING, ASCEND, OWNER)
METRICS: dict[str, Metric] = {AMI: "wer", ALIMEETING: "cer", ASCEND: "mer", ASCEND_UTTERANCES: "mer"}
ASR_DIR = "asr"
INPUTS_FILE = "inputs.json"
UTTERANCES = "utterances"


@dataclass(frozen=True, slots=True)
class Reference:
    text: str
    # Timed reference lines for a long recording; empty for a clip that is scored whole.
    utterances: tuple[RefUtterance, ...] = ()


@dataclass(frozen=True, slots=True)
class Item:
    corpus: str
    id: str
    wav: Path
    duration_s: float
    # The scoring view: mixed Mandarin-and-English speech is scored as Chinese, as in the YouTube evals.
    language: Language
    reference: Reference | None


def asr_layout(corpus: str, root: Path | None = None) -> Layout:
    return Layout((root or recordings_dir()) / ASR_DIR / corpus)


def batch_key(ids: Sequence[str]) -> str:
    """Name of the worker folder for these files: the id itself for one file, else a digest of the ids.

    A worker folder is bound to the one request that made it, so a batch must keep its folder whatever else is run;
    the digest changes whenever its membership does.
    """
    if len(ids) == 1:
        return ids[0]
    return "batch-" + hashlib.sha256("\0".join(ids).encode()).hexdigest()[:12]


def audio_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def recording_job(item: Item, definition: EngineDef) -> FileJob:
    """Build a job with no language hint; a language-ID hypothesis gets the allowed set instead."""
    detects = definition.language_policy() == "detect"
    return FileJob(id=item.id, wav=str(item.wav), languages=ALLOWED_LANGUAGES if detects else ())


def stamp_inputs(folder: Path, items: Sequence[Item]) -> None:
    """Record a digest of each item's audio in a worker folder, or fail if the folder holds another audio's run.

    The worker's own fingerprint leaves the audio path out, so replacing a file under the same id would otherwise
    resume or reuse the old audio's rows.
    """
    digests = {item.id: audio_digest(item.wav) for item in items}
    path = folder / INPUTS_FILE
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        atomic.write_text(path, json.dumps(digests, indent=2) + "\n")
        return
    check_stamp(folder, items, digests)


def check_stamp(folder: Path, items: Sequence[Item], digests: dict[str, str] | None = None) -> None:
    path = folder / INPUTS_FILE
    stored: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for item in items:
        digest = digests[item.id] if digests else audio_digest(item.wav)
        if stored.get(item.id) != digest:
            raise ConfigError(
                f"the audio of {item.id} is not what worker folder {folder.name} decoded (or the folder has no input "
                "stamp); move that folder aside to decode it again"
            )


def _as_keyed(definition: EngineDef) -> EngineDef:
    """Folders made before the rename name the adapter under the old package; that is the same engine."""
    return replace(definition, factory=legacy.fingerprint_factory(definition.factory))


def finished_folders(layout: Layout, hypothesis: str, items: Sequence[Item]) -> dict[str, Path]:
    """Map each selected item to the one finished worker folder that decoded it (a worker writes its metrics last).

    A folder is refused, loudly, when it was made by another engine definition or another job for one of the selected
    items, when the audio has changed since, or when a second finished folder holds the same item.
    """
    definition = registry.get(hypothesis)
    wanted = {item.id: item for item in items}
    found: dict[str, Path] = {}
    for metrics in sorted(layout.run_dir(hypothesis).glob(f"*/{METRICS_FILE}")):
        folder = metrics.parent
        request = WorkerRequest.model_validate_json((folder / REQUEST_FILE).read_text(encoding="utf-8"))
        stored = stored_fingerprint(folder)
        if (
            _as_keyed(request.engine) != _as_keyed(definition)
            or stored is None
            or without_wav(stored) != without_wav(request.fingerprint())
        ):
            raise ConfigError(f"worker folder {folder.name} was not made by the current {hypothesis} definition")
        mine = [job for job in request.files if job.id in wanted]
        for job in mine:
            expected = recording_job(wanted[job.id], definition)
            if job.model_copy(update={"wav": ""}) != expected.model_copy(update={"wav": ""}):
                raise ConfigError(f"{job.id} was decoded with a different job than {hypothesis} gets now")
        check_stamp(folder, [wanted[job.id] for job in mine])
        for job in mine:
            if job.id in found:
                raise ConfigError(
                    f"{job.id} is finished in two {hypothesis} worker folders ({found[job.id].name} and "
                    f"{folder.name}); move one aside"
                )
            found[job.id] = folder
    return found


def finished_rows(layout: Layout, hypothesis: str, items: Sequence[Item]) -> dict[str, list[ChunkHypothesis]]:
    """Rows of each finished item, read from the one folder ``finished_folders`` names for it."""
    folders = finished_folders(layout, hypothesis, items)
    cache: dict[Path, list[ChunkHypothesis]] = {}
    rows: dict[str, list[ChunkHypothesis]] = {}
    for item_id, folder in folders.items():
        if folder not in cache:
            cache[folder] = read_rows(folder / HYPOTHESES_FILE)
        rows[item_id] = sorted((r for r in cache[folder] if r.file == item_id), key=lambda r: r.chunk)
    return rows


def _utterances(path: Path) -> tuple[RefUtterance, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return tuple(RefUtterance(r["start"], r["end"], r["speaker"], r["text"]) for r in records)


def _indexed(corpus: str, root: Path) -> list[Item]:
    corpus_dir = root / corpus
    index = load_index(corpus_dir)
    language: Language = "en" if corpus == AMI else "zh"
    items: list[Item] = []
    for entry in index.recordings:
        folder = corpus_dir / Path(entry.audio).parent
        reference = Reference(
            (corpus_dir / entry.reference_text).read_text(encoding="utf-8").strip(),
            _utterances(folder / REFERENCE_JSONL),
        )
        items.append(Item(corpus, entry.id, corpus_dir / entry.audio, entry.duration_s, language, reference))
    return items


def _clips(root: Path) -> list[Item]:
    folder = root / ASCEND / UTTERANCES
    listing = folder / "utterances.jsonl"
    if not listing.is_file():
        raise ConfigError(f"no ASCEND utterances at {listing}; run `python -m evals.recordings_data ascend` first")
    records = [json.loads(line) for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        Item(
            ASCEND_UTTERANCES,
            r["id"],
            folder / f"{r['id']}.wav",
            wav_seconds(folder / f"{r['id']}.wav"),
            "zh",
            Reference(r["text"]),
        )
        for r in records
    ]


def _owner(root: Path) -> list[Item]:
    return [
        Item(OWNER, o.entry.id, o.wav, o.duration_s, "en" if o.entry.language == "en" else "zh", None)
        for o in load_owner(root / OWNER)
    ]


def load_items(corpus: str, only: Sequence[str] = (), root: Path | None = None) -> list[Item]:
    base = root or recordings_dir()
    if corpus not in ASR_CORPORA:
        raise ConfigError(f"unknown corpus {corpus!r}; known: {', '.join(ASR_CORPORA)}")
    items = _owner(base) if corpus == OWNER else _clips(base) if corpus == ASCEND_UTTERANCES else _indexed(corpus, base)
    chosen = [item for item in items if not only or item.id in only]
    missing = set(only) - {item.id for item in chosen}
    if missing:
        raise ConfigError(f"{corpus} has no recording(s) {', '.join(sorted(missing))}")
    return chosen
