"""The recordings the ASR evals decode and score, and where their runs live.

Every corpus becomes a list of ``Item`` (one audio file with an optional reference); the layout of the files is
``evals.recordings_data``'s. ASR runs go under ``<recordings>/asr/<corpus>/runs/<hypothesis>/<batch>/``, which is an
``evals.store.Layout`` rooted at ``<recordings>/asr/<corpus>``, so rows are read back the way the YouTube evals do.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from evals.store import Layout
from ytverbatim.asr.models import METRICS_FILE, REQUEST_FILE
from ytverbatim.errors import ConfigError

Metric = Literal["wer", "cer", "mer"]

AMI = "ami"
ALIMEETING = "alimeeting"
ASCEND = "ascend"
ASCEND_UTTERANCES = "ascend-utterances"
ASR_CORPORA = (AMI, ALIMEETING, ASCEND, ASCEND_UTTERANCES, OWNER)
# Corpora with one folder per recording, so a ``words.json`` can sit next to each audio file.
FOLDER_CORPORA = (AMI, ALIMEETING, ASCEND, OWNER)
METRICS: dict[str, Metric] = {AMI: "wer", ALIMEETING: "cer", ASCEND: "mer", ASCEND_UTTERANCES: "mer"}
ASR_DIR = "asr"
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


def finished_ids(layout: Layout, hypothesis: str) -> set[str]:
    """Ids of every recording whose worker ran to the end (it writes its metrics last)."""
    done: set[str] = set()
    for metrics in sorted(layout.run_dir(hypothesis).glob(f"*/{METRICS_FILE}")):
        request = json.loads((metrics.parent / REQUEST_FILE).read_text(encoding="utf-8"))
        done.update(job["id"] for job in request["files"])
    return done


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
