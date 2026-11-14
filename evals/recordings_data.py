"""Public meeting corpora for the recordings evals: download, verify and lay out under ``<eval dir>/recordings``.

``uv run python -m evals.recordings_data <ami|alimeeting|ascend|status>``. Everything lands outside the repository;
nothing derived from these corpora is ever committed.

Layout, one folder per recording with ``audio.wav`` (16 kHz mono), ``reference.rttm``, ``reference.txt`` and
``reference.jsonl`` (timed reference utterances), plus a corpus ``index.json`` listing them::

    recordings/ami/<meeting>/
    recordings/alimeeting/<session>/
    recordings/ascend/sessions/<session>/      long-form sessions rebuilt from utterances
    recordings/ascend/utterances/<id>.wav      the test split, with utterances.jsonl
"""

from __future__ import annotations

import io
import itertools
import json
import re
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
import wave
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from evals import disk
from evals.diarization import Turn, write_rttm
from evals.manifest import eval_dir
from ytverbatim import atomic
from ytverbatim.errors import ConfigError, YtverbatimError

SAMPLE_RATE = 16_000
MIN_FREE_BYTES = 10 * 10**9
RECORDINGS = "recordings"
RAW = "raw"
INDEX = "index.json"
AUDIO = "audio.wav"
REFERENCE_RTTM = "reference.rttm"
REFERENCE_TEXT = "reference.txt"
REFERENCE_JSONL = "reference.jsonl"
WORDS_FILE = "words.json"
CLIPS = "clips"
OWNER = "owner"

AMI_REPO = "diarizers-community/ami"
AMI_TEXT_REPO = "edinburghcstr/ami"
AMI_AUDIO_SHARD = "sdm/test-{index:05d}-of-00003.parquet"
AMI_TEXT_SHARDS = tuple(f"sdm/test-{i:05d}-of-00004.parquet" for i in range(4))
ASCEND_REPO = "CAiRE/ASCEND"
ASCEND_TEST = "main/test-00000-of-00001.parquet"
ALIMEETING_URL = "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"
ALIMEETING_FAR = "Eval_Ali/Eval_Ali_far/"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"

# Below this the transcript and the diarization turns are on different clocks or are different meetings, not noisy.
MIN_ALIGNED_FRACTION = 0.9
DURATION_TOLERANCE_S = 0.02
# File names round offsets and durations to four decimals, and speakers' neighbouring utterances can touch.
OFFSET_TOLERANCE_S = 1e-3
_OWNER_ID = re.compile(r"^rec-\d{2,}$")
_UNK = re.compile(r"\s*\[UNK\]\s*")
_BLANKS = re.compile(r"\s+")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class RecordingEntry(_Frozen):
    id: str
    audio: str
    duration_s: float
    speakers: int
    reference_rttm: str
    reference_text: str
    note: str = ""


class CorpusIndex(_Frozen):
    corpus: str
    source: str
    license: str
    recordings: tuple[RecordingEntry, ...]


class OwnerEntry(_Frozen):
    """One of the owner's own recordings: a generic label, no reference, and the speaker count only if they know it."""

    id: str
    # 16 kHz mono 16-bit WAV, relative to the owner folder or absolute; made with ffmpeg from the original.
    audio: str
    language: Literal["zh", "en", "zh+en"] = "zh+en"
    speakers: int | None = None

    @field_validator("id")
    @classmethod
    def _generic(cls, value: str) -> str:
        if not _OWNER_ID.match(value):
            raise ValueError(f"owner recording ids are generic labels such as rec-01, got {value!r}")
        return value


class OwnerIndex(_Frozen):
    recordings: tuple[OwnerEntry, ...]


@dataclass(frozen=True, slots=True)
class OwnerRecording:
    entry: OwnerEntry
    wav: Path
    duration_s: float


@dataclass(frozen=True, slots=True)
class RefUtterance:
    start: float
    end: float
    speaker: str
    text: str


def recordings_dir() -> Path:
    return eval_dir() / RECORDINGS


def clips_dir() -> Path:
    return recordings_dir() / CLIPS


def load_index(corpus_dir: Path) -> CorpusIndex:
    path = corpus_dir / INDEX
    if not path.is_file():
        raise ConfigError(f"no prepared data at {path}; run `python -m evals.recordings_data` for that corpus first")
    return CorpusIndex.model_validate_json(path.read_text(encoding="utf-8"))


def load_owner(owner_dir: Path) -> list[OwnerRecording]:
    """Read the private owner index and check each listed file is the 16 kHz mono 16-bit WAV the evals decode."""
    path = owner_dir / INDEX
    if not path.is_file():
        raise ConfigError(f"no owner index at {path}; list the owner's WAV files there (see OwnerIndex)")
    index = OwnerIndex.model_validate_json(path.read_text(encoding="utf-8"))
    ids = [e.id for e in index.recordings]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path} lists a recording id more than once")
    found: list[OwnerRecording] = []
    for entry in index.recordings:
        wav = owner_dir / entry.audio
        if not wav.is_file():
            raise ConfigError(f"{entry.id}: {wav} does not exist")
        found.append(OwnerRecording(entry, wav, wav_seconds(wav)))
    return found


def wav_seconds(path: Path) -> float:
    """Duration of a 16 kHz mono 16-bit WAV file; any other format is refused with the conversion to run."""
    try:
        with wave.open(str(path), "rb") as handle:
            shape = (handle.getframerate(), handle.getnchannels(), handle.getsampwidth())
            frames = handle.getnframes()
    except (wave.Error, EOFError) as exc:
        raise ConfigError(f"{path.name} is not a readable WAV file: {exc}") from exc
    if shape != (SAMPLE_RATE, 1, 2):
        raise ConfigError(
            f"{path.name} must be 16 kHz mono 16-bit; convert it with "
            "`ffmpeg -i IN -ac 1 -ar 16000 -c:a pcm_s16le OUT.wav`"
        )
    return frames / SAMPLE_RATE


def check_free_disk(path: Path, *, minimum: int = MIN_FREE_BYTES, free: Callable[[Path], int] = disk.free_bytes) -> int:
    available = free(path)
    if available < minimum:
        raise ConfigError(
            f"only {available / 1e9:.1f} GB is free on the volume of {path}; downloads need at least "
            f"{minimum / 1e9:.0f} GB free first"
        )
    return available


def download(url: str, destination: Path, *, free: Callable[[Path], int] = disk.free_bytes) -> Path:
    """Fetch ``url`` to ``destination`` after the free-disk check, resuming a partial download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    check_free_disk(destination.parent, free=free)
    part = destination.with_name(destination.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        with urllib.request.urlopen(request) as response:
            resumed = have > 0 and response.status == 206  # noqa: PLR2004
            with part.open("ab" if resumed else "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
    except urllib.error.URLError as exc:
        raise ConfigError(f"could not download {url}: {exc}") from exc
    part.replace(destination)
    return destination


def clean_text(text: str) -> str:
    """Drop ASCEND's ``[UNK]`` markers: annotation tokens, not words."""
    return _BLANKS.sub(" ", _UNK.sub(" ", text)).strip()


def ordered_text(utterances: Iterable[RefUtterance]) -> str:
    """Join reference text in start order; overlapping speech is interleaved by start time, as it is heard."""
    ordered = sorted(utterances, key=lambda u: (u.start, u.end, u.speaker))
    return " ".join(u.text for u in ordered if u.text.strip())


def wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.ascontiguousarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ConfigError(
                f"expected 16 kHz mono 16-bit audio, got {handle.getframerate()} Hz, "
                f"{handle.getnchannels()} channel(s), {8 * handle.getsampwidth()}-bit"
            )
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def write_recording(
    corpus_dir: Path,
    rel: str,
    audio: bytes,
    utterances: Sequence[RefUtterance],
    *,
    turns: Sequence[Turn] | None = None,
    note: str = "",
) -> RecordingEntry:
    """Write one recording folder; ``turns`` default to the utterance spans."""
    folder = corpus_dir / rel
    folder.mkdir(parents=True, exist_ok=True)
    ordered = sorted(utterances, key=lambda u: (u.start, u.end, u.speaker))
    reference_turns = turns if turns is not None else [Turn(u.start, u.end, u.speaker) for u in ordered]
    atomic.write_bytes(folder / AUDIO, audio)
    write_rttm(folder / REFERENCE_RTTM, folder.name, reference_turns)
    atomic.write_text(folder / REFERENCE_TEXT, ordered_text(ordered) + "\n")
    lines = (
        json.dumps({"start": u.start, "end": u.end, "speaker": u.speaker, "text": u.text}, ensure_ascii=False)
        for u in ordered
    )
    atomic.write_text(folder / REFERENCE_JSONL, "".join(f"{line}\n" for line in lines))
    return RecordingEntry(
        id=folder.name,
        audio=f"{rel}/{AUDIO}",
        duration_s=round(wav_duration(audio), 3),
        speakers=len({t.speaker for t in reference_turns}),
        reference_rttm=f"{rel}/{REFERENCE_RTTM}",
        reference_text=f"{rel}/{REFERENCE_TEXT}",
        note=note,
    )


def write_index(corpus_dir: Path, index: CorpusIndex) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    atomic.write_text(corpus_dir / INDEX, index.model_dump_json(indent=2) + "\n")


def ami_meeting_id(audio_path: str) -> str:
    """``EN2002a.Array1-01.wav`` is the first far-field channel of meeting ``EN2002a``."""
    return Path(audio_path).name.split(".")[0]


def ami_alignment(turns: Sequence[Turn], utterances: Sequence[RefUtterance]) -> float:
    """Fraction of the transcript's speech time that lies inside the same speaker's diarization turns.

    Raises when the two sources name different speakers: that is another meeting, not a time-base slip.
    """
    if {t.speaker for t in turns} != {u.speaker for u in utterances}:
        raise ConfigError("the diarization turns and the transcript name different speakers")
    by_speaker: dict[str, list[Turn]] = defaultdict(list)
    for turn in turns:
        by_speaker[turn.speaker].append(turn)
    total = covered = 0.0
    for utterance in utterances:
        span = utterance.end - utterance.start
        shared = sum(
            max(0.0, min(t.end, utterance.end) - max(t.start, utterance.start)) for t in by_speaker[utterance.speaker]
        )
        total += span
        covered += min(shared, span)
    return covered / total if total else 0.0


def prepare_ami(audio_shard: Path, text_shards: Sequence[Path], corpus_dir: Path) -> CorpusIndex:
    """Lay out every meeting of one sdm test shard: RTTM from its turns, reference text from the utterance corpus."""
    import pyarrow.parquet as pq

    transcripts: dict[str, list[RefUtterance]] = defaultdict(list)
    columns = ["meeting_id", "begin_time", "end_time", "speaker_id", "text"]
    for shard in text_shards:
        for row in pq.read_table(shard, columns=columns).to_pylist():
            transcripts[row["meeting_id"]].append(
                RefUtterance(row["begin_time"], row["end_time"], row["speaker_id"], row["text"])
            )
    entries: list[RecordingEntry] = []
    for batch in pq.ParquetFile(audio_shard).iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        meeting = ami_meeting_id(row["audio"]["path"])
        if meeting not in transcripts:
            raise ConfigError(f"meeting {meeting} has no transcript in the {AMI_TEXT_REPO} shards given")
        turns = [
            Turn(start, end, speaker)
            for start, end, speaker in zip(row["timestamps_start"], row["timestamps_end"], row["speakers"], strict=True)
        ]
        aligned = ami_alignment(turns, transcripts[meeting])
        if aligned < MIN_ALIGNED_FRACTION:
            raise ConfigError(f"{meeting}: only {aligned:.0%} of the transcript lies inside the diarization turns")
        entries.append(
            write_recording(
                corpus_dir,
                meeting,
                row["audio"]["bytes"],
                transcripts[meeting],
                turns=turns,
                note=f"{aligned:.1%} of the transcript's speech lies inside the same speaker's reference turns",
            )
        )
    index = CorpusIndex(
        corpus="ami-sdm-test",
        source=f"{AMI_REPO} (audio, turns), {AMI_TEXT_REPO} (text)",
        license="CC-BY-4.0",
        recordings=tuple(sorted(entries, key=lambda e: e.id)),
    )
    write_index(corpus_dir, index)
    return index


_ASCEND_NAME = re.compile(r"^ses(\d+)_spk(\d+)_L(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.wav$")


@dataclass(frozen=True, slots=True)
class AscendUtterance:
    id: str
    session: int
    speaker: int
    line: int
    start: float
    duration: float
    language: str
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def parse_ascend_name(name: str) -> tuple[int, int, int, float, float]:
    """Split ``ses<S>_spk<P>_L<line>_<start>_<duration>.wav`` into session, speaker, line, start and duration."""
    match = _ASCEND_NAME.match(Path(name).name)
    if match is None:
        raise ConfigError(f"unexpected ASCEND file name {name!r}")
    return int(match[1]), int(match[2]), int(match[3]), float(match[4]), float(match[5])


def check_ascend_offsets(utterances: Sequence[AscendUtterance]) -> None:
    """Fail unless the file-name offsets behave as positions on one session clock.

    Within one speaker, utterances must follow their line numbers in time and never overlap. The other speaker may
    overlap freely: that is real cross-talk.
    """
    by_speaker: dict[tuple[int, int], list[AscendUtterance]] = defaultdict(list)
    for utterance in utterances:
        by_speaker[(utterance.session, utterance.speaker)].append(utterance)
    for (session, speaker), items in by_speaker.items():
        in_time = sorted(items, key=lambda u: u.start)
        if [u.line for u in in_time] != sorted(u.line for u in items):
            raise ConfigError(f"session {session} speaker {speaker}: offsets do not follow the line order")
        for earlier, later in itertools.pairwise(in_time):
            if later.start < earlier.end - OFFSET_TOLERANCE_S:
                raise ConfigError(f"session {session} speaker {speaker}: {earlier.id} and {later.id} overlap")


@dataclass(frozen=True, slots=True)
class SessionPlan:
    session: int
    duration_s: float
    utterances: tuple[AscendUtterance, ...]

    def reference(self) -> list[RefUtterance]:
        return [RefUtterance(u.start, u.end, f"spk{u.speaker}", clean_text(u.text)) for u in self.utterances]


def plan_sessions(utterances: Sequence[AscendUtterance], tail_s: float = 1.0) -> list[SessionPlan]:
    """Group utterances by session and order each by start offset, then line, for a silent timeline."""
    grouped: dict[int, list[AscendUtterance]] = defaultdict(list)
    for utterance in utterances:
        grouped[utterance.session].append(utterance)
    return [
        SessionPlan(session, max(u.end for u in items) + tail_s, tuple(sorted(items, key=lambda u: (u.start, u.line))))
        for session, items in sorted(grouped.items())
    ]


def render_session(plan: SessionPlan, samples: Callable[[AscendUtterance], np.ndarray]) -> np.ndarray:
    """Mix each utterance onto silence at its offset. Overlapping speakers add, clipped to the 16-bit range."""
    mix = np.zeros(round(plan.duration_s * SAMPLE_RATE), dtype=np.int32)
    for utterance in plan.utterances:
        audio = samples(utterance)
        offset = round(utterance.start * SAMPLE_RATE)
        mix[offset : offset + len(audio)] += audio
    return np.clip(mix, -32768, 32767).astype("<i2")


def _ascend_rows(parquet: Path) -> Iterator[tuple[AscendUtterance, bytes]]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        session, speaker, line, start, duration = parse_ascend_name(row["path"])
        if session != row["session_id"] or speaker != row["original_speaker_id"]:
            raise ConfigError(f"{row['id']}: the file name disagrees with the session or speaker columns")
        data = row["audio"]["bytes"]
        if abs(wav_duration(data) - duration) > DURATION_TOLERANCE_S:
            raise ConfigError(f"{row['id']}: the audio length disagrees with the duration in its file name")
        yield (
            AscendUtterance(row["id"], session, speaker, line, start, duration, row["language"], row["transcription"]),
            data,
        )


def prepare_ascend(parquet: Path, corpus_dir: Path) -> CorpusIndex:
    """Write the test utterances, then rebuild each session as one long recording from its file-name offsets."""
    utterance_dir = corpus_dir / "utterances"
    utterance_dir.mkdir(parents=True, exist_ok=True)
    utterances: list[AscendUtterance] = []
    for utterance, data in _ascend_rows(parquet):
        atomic.write_bytes(utterance_dir / f"{utterance.id}.wav", data)
        utterances.append(utterance)
    check_ascend_offsets(utterances)
    records = (
        json.dumps(
            {
                "id": u.id,
                "session": u.session,
                "speaker": f"spk{u.speaker}",
                "language": u.language,
                "start": u.start,
                "duration": u.duration,
                "text": clean_text(u.text),
            },
            ensure_ascii=False,
        )
        for u in utterances
    )
    atomic.write_text(utterance_dir / "utterances.jsonl", "".join(f"{record}\n" for record in records))

    def samples(utterance: AscendUtterance) -> np.ndarray:
        return read_wav((utterance_dir / f"{utterance.id}.wav").read_bytes())

    entries = [
        write_recording(
            corpus_dir,
            f"sessions/ses{plan.session}",
            wav_bytes(render_session(plan, samples)),
            plan.reference(),
            note="approximate: utterance boundaries, and silence where the other speaker's utterances are not in the "
            "test split",
        )
        for plan in plan_sessions(utterances)
    ]
    index = CorpusIndex(
        corpus="ascend-test-sessions", source=ASCEND_REPO, license="CC-BY-SA-4.0", recordings=tuple(entries)
    )
    write_index(corpus_dir, index)
    return index


_INTERVAL = re.compile(r"^intervals \[\d+\]:")


def tier_utterances(textgrid: str) -> list[RefUtterance]:
    """Read the non-empty intervals of a long-format Praat TextGrid; every tier is one speaker. Raise if none."""
    utterances: list[RefUtterance] = []
    tier = ""
    start = end = 0.0
    inside = False
    for raw in textgrid.splitlines():
        line = raw.strip()
        if line.startswith("name ="):
            # A label with a space would split an RTTM line into extra fields.
            tier, inside = _BLANKS.sub("_", line.split("=", 1)[1].strip().strip('"')), False
        elif _INTERVAL.match(line):
            inside = True
        elif inside and line.startswith("xmin ="):
            start = float(line.split("=", 1)[1])
        elif inside and line.startswith("xmax ="):
            end = float(line.split("=", 1)[1])
        elif inside and line.startswith("text ="):
            text = line.split("=", 1)[1].strip()
            text = text[1:-1].replace('""', '"') if len(text) > 1 and text[0] == text[-1] == '"' else text
            if text.strip():
                utterances.append(RefUtterance(start, end, tier, text.strip()))
    if not utterances:
        raise ConfigError("the TextGrid has no non-empty intervals")
    return utterances


def first_channel(source: Path, destination: Path) -> None:
    """Write channel 1 of a multi-channel WAV as 16 kHz mono 16-bit."""
    part = destination.with_name(destination.name + ".part")
    command = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-af", "pan=mono|c0=c0", "-ar", str(SAMPLE_RATE)]
    result = subprocess.run([*command, "-c:a", "pcm_s16le", "-f", "wav", str(part)], capture_output=True, check=False)
    if result.returncode != 0:
        part.unlink(missing_ok=True)
        raise ConfigError(f"ffmpeg failed on {source.name}: {result.stderr.decode(errors='replace')[-300:]}")
    part.replace(destination)


def extract_far(archive: Path, staging: Path) -> list[str]:
    """Extract the far-field part of the Eval archive into a new empty ``staging``; return any RTTM names it holds."""
    if staging.exists() and any(staging.iterdir()):
        raise ConfigError(f"{staging} is not empty; downloaded archives are extracted into a fresh directory")
    staging.mkdir(parents=True, exist_ok=True)
    rttms: list[str] = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name.lower().endswith(".rttm"):
                rttms.append(member.name)
            if member.name.startswith(ALIMEETING_FAR):
                tar.extract(member, staging, filter="data")
    return rttms


def prepare_alimeeting(staging: Path, corpus_dir: Path, limit: int | None = None) -> CorpusIndex:
    """Lay out far-field channel 1 of each extracted session with its TextGrid-derived references."""
    far = staging / ALIMEETING_FAR
    entries: list[RecordingEntry] = []
    for textgrid in sorted((far / "textgrid_dir").glob("*.TextGrid"))[:limit]:
        session = textgrid.stem
        matches = sorted((far / "audio_dir").glob(f"{session}_*.wav"))
        if len(matches) != 1:
            raise ConfigError(f"{session}: expected one far-field WAV, found {len(matches)}")
        channel = corpus_dir / f".{session}.channel1.wav"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        first_channel(matches[0], channel)
        utterances = tier_utterances(textgrid.read_text(encoding="utf-8"))
        entries.append(
            write_recording(
                corpus_dir,
                session,
                channel.read_bytes(),
                utterances,
                note="far-field channel 1; references from the headset-derived TextGrid tiers (no RTTM in the archive)",
            )
        )
        channel.unlink()
    index = CorpusIndex(
        corpus="alimeeting-eval-far-ch1",
        source=ALIMEETING_URL,
        license="CC BY-SA 4.0",
        recordings=tuple(entries),
    )
    write_index(corpus_dir, index)
    return index


@click.group()
def main() -> None:
    """Public recordings eval data under $YTVERBATIM_EVAL_DIR/recordings (default ~/.local/share/ytverbatim/eval)."""


def _fetch(repo: str, name: str, folder: str) -> Path:
    return download(HF_URL.format(repo=repo, name=name), recordings_dir() / RAW / folder / name)


def _show(index: CorpusIndex) -> None:
    for entry in index.recordings:
        click.echo(f"{entry.id}: {entry.duration_s / 60:.1f} min, {entry.speakers} speakers")


@main.command()
@click.option(
    "--shard", default=0, show_default=True, type=click.IntRange(0, 2), help="sdm test shard (about 5 meetings)."
)
@click.option("--keep-raw", is_flag=True, help="Keep the downloaded parquet files.")
def ami(shard: int, keep_raw: bool) -> None:
    """AMI SDM test: one shard of meetings with RTTMs and reference text."""
    audio = _fetch(AMI_REPO, AMI_AUDIO_SHARD.format(index=shard), "ami-sdm")
    text = [_fetch(AMI_TEXT_REPO, name, "ami-text") for name in AMI_TEXT_SHARDS]
    _show(prepare_ami(audio, text, recordings_dir() / "ami"))
    if not keep_raw:
        for folder in ("ami-sdm", "ami-text"):
            shutil.rmtree(recordings_dir() / RAW / folder, ignore_errors=True)


@main.command()
@click.option("--keep-raw", is_flag=True, help="Keep the downloaded parquet file.")
def ascend(keep_raw: bool) -> None:
    """ASCEND test split: utterances plus long-form sessions."""
    parquet = _fetch(ASCEND_REPO, ASCEND_TEST, "ascend")
    _show(prepare_ascend(parquet, recordings_dir() / "ascend"))
    if not keep_raw:
        shutil.rmtree(recordings_dir() / RAW / "ascend", ignore_errors=True)


@main.command()
@click.option("--sessions", type=int, default=None, help="Keep only the first N sessions (default: all 8).")
@click.option("--keep-archive", is_flag=True, help="Keep the 3.67 GB archive.")
def alimeeting(sessions: int | None, keep_archive: bool) -> None:
    """AliMeeting Eval: far-field channel 1 as 16 kHz WAV plus references; the archive is deleted afterwards."""
    root = recordings_dir()
    archive = download(ALIMEETING_URL, root / RAW / "alimeeting" / "Eval_Ali.tar.gz")
    staging = root / RAW / "alimeeting-x"
    rttms = extract_far(archive, staging)
    click.echo(f"RTTM files in the archive: {len(rttms)}")
    _show(prepare_alimeeting(staging, root / "alimeeting", sessions))
    shutil.rmtree(staging)
    if not keep_archive:
        shutil.rmtree(root / RAW / "alimeeting")


@main.command()
def status() -> None:
    """Show what is prepared and the free disk."""
    root = recordings_dir()
    click.echo(f"{root}: {disk.free_bytes(root) / 1e9:.1f} GB free")
    owner = root / OWNER / INDEX
    click.echo(f"{OWNER}: {len(load_owner(owner.parent)) if owner.is_file() else 'not prepared'} recordings")
    for corpus in ("ami", "alimeeting", "ascend"):
        if (root / corpus / INDEX).is_file():
            index = load_index(root / corpus)
            hours = sum(e.duration_s for e in index.recordings) / 3600
            click.echo(f"{index.corpus}: {len(index.recordings)} recordings, {hours:.2f} h")
        else:
            click.echo(f"{corpus}: not prepared")


def run() -> None:
    try:
        main(standalone_mode=False)
    except YtverbatimError as exc:
        raise SystemExit(f"error: {exc}") from None
    except click.exceptions.Abort:
        raise SystemExit(130) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    run()
