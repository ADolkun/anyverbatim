"""Local recordings: probe a file, choose the audio stream to transcribe, and derive the recording's id.

The id hashes the chosen stream's encoded packets (`ffmpeg -c copy -f hash`), neither the file nor the decoded PCM: a
rename, a metadata edit or a remux into another container keeps it, and no ffmpeg decoder or resampler change can
move it. The source file is only read; nothing records where it lives, so audio needed on a rerun means passing the
file again.
"""

import json
import math
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ytverbatim import ids, proc
from ytverbatim.acquire.audio import DEFAULT_CONVERT_TIMEOUT_S, PROBE_TIMEOUT_S, to_wav16k
from ytverbatim.contract import RecordingSource
from ytverbatim.errors import MediaValidationError, stderr_preview
from ytverbatim.ytdlp.runner import redact

# Container date tags in the order they are trusted: QuickTime's local time with offset, then the muxing time.
_DATE_TAGS = ("com.apple.quicktime.creationdate", "creation_time")
# QuickTime and MP4 count from 1904 and Unix from 1970, so a zero date reads as one of those, never as a real one.
_FIRST_PLAUSIBLE_YEAR = 1971
_HASH_PREFIX = "SHA256="

Origin = Literal["file", "voice_memos"]


@dataclass(frozen=True)
class LocalRecording:
    """A probed local file: the audio stream that is transcribed, its packet hash, and what the context records."""

    path: Path
    # Index among the file's audio streams, as `-map 0:a:N` takes it.
    stream: int
    audio_sha256: str
    title: str
    recorded_at: datetime | None
    duration_s: float
    origin: Origin = "file"

    @property
    def id(self) -> str:
        return ids.recording_id(self.audio_sha256)

    @property
    def file_name(self) -> str:
        return self.path.name

    def source(self, language: str | None) -> RecordingSource:
        return RecordingSource(
            kind="recording",
            id=self.id,
            audio_sha256=self.audio_sha256,
            title=self.title,
            file_name=self.file_name,
            origin=self.origin,
            recorded_at=self.recorded_at,
            duration_s=self.duration_s,
            language=language,
        )


@dataclass(frozen=True)
class _Probe:
    streams: list[dict[str, object]]
    format: dict[str, object]


def open_recording(path: Path) -> LocalRecording:
    """Probe `path` and hash its transcribed stream; raises `MediaValidationError` for a file it cannot use."""
    probed = _probe(path)
    stream = pick_stream(path, probed.streams)
    return LocalRecording(
        path=path,
        stream=stream,
        audio_sha256=packet_sha256(path, stream),
        title=_title(probed, path),
        recorded_at=_recorded_at(probed, stream),
        duration_s=_duration(probed, stream, path),
    )


def recording_id(path: Path) -> str:
    """Return the `rec-<16 hex>` id of the audio in `path`; the hasher the Voice Memos cache takes."""
    stream = pick_stream(path, _probe(path).streams)
    return ids.recording_id(packet_sha256(path, stream))


def pick_stream(path: Path, streams: list[dict[str, object]]) -> int:
    """Choose the audio stream to transcribe: the first AAC one, else the first one ffmpeg can decode.

    AAC goes first because Voice Memos' spatial recordings carry an APAC track next to it that ffmpeg cannot decode.
    """
    if not streams:
        raise MediaValidationError(f"{path.name} has no audio stream")
    codecs = [stream.get("codec_name") for stream in streams]
    if "aac" in codecs:
        return codecs.index("aac")
    for index in range(len(streams)):
        if _decodes(path, index):
            return index
    named = ", ".join(str(codec) if codec else "unknown" for codec in codecs)
    raise MediaValidationError(f"ffmpeg cannot decode any audio stream of {path.name} ({named})")


def packet_sha256(path: Path, stream: int) -> str:
    result = proc.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            f"0:a:{stream}",
            "-c",
            "copy",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        timeout_s=DEFAULT_CONVERT_TIMEOUT_S,
    )
    line = result.stdout.strip()
    digest = line.removeprefix(_HASH_PREFIX).lower()
    if result.returncode != 0 or not line.startswith(_HASH_PREFIX):
        raise MediaValidationError(f"ffmpeg could not read the audio of {path.name}\n{_stderr(result)}")
    return digest


def to_wav(recording: LocalRecording, destination: Path) -> None:
    if not recording.path.is_file():
        error = MediaValidationError(
            f"{recording.file_name} is no longer at the path it was read from, and its audio is needed again"
        )
        error.fix = "pass the file again"
        raise error
    to_wav16k(recording.path, destination, expected_duration=recording.duration_s, stream=recording.stream)


def _probe(path: Path) -> _Probe:
    if not path.is_file():
        raise MediaValidationError(f"no such file: {path.name}")
    result = proc.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name,duration:stream_tags:format=duration:format_tags",
            "-of",
            "json",
            str(path),
        ],
        timeout_s=PROBE_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise MediaValidationError(f"{path.name} is not a readable audio or video file\n{_stderr(result)}")
    try:
        raw: object = json.loads(result.stdout or "{}")
    except ValueError as exc:
        raise MediaValidationError(f"ffprobe gave unreadable output for {path.name}") from exc
    data = raw if isinstance(raw, dict) else {}
    streams = data.get("streams")
    found = data.get("format")
    return _Probe(
        streams=[stream for stream in streams if isinstance(stream, dict)] if isinstance(streams, list) else [],
        format=found if isinstance(found, dict) else {},
    )


def _decodes(path: Path, stream: int) -> bool:
    result = proc.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{stream}", "-t", "1", "-f", "null", "-"],
        timeout_s=PROBE_TIMEOUT_S,
    )
    return result.returncode == 0


def _tags(section: dict[str, object]) -> dict[str, str]:
    # Vorbis comments come upper-case (TITLE), MP4 atoms lower-case.
    tags = section.get("tags")
    if not isinstance(tags, dict):
        return {}
    return {str(key).lower(): value for key, value in tags.items() if isinstance(value, str)}


def _title(probed: _Probe, path: Path) -> str:
    title = _printable(_tags(probed.format).get("title", ""))
    return title or path.stem


def _printable(text: str) -> str:
    """Titles reach a terminal and the Markdown header: control characters become spaces."""
    return " ".join("".join(" " if unicodedata.category(c) in {"Cc", "Zl", "Zp"} else c for c in text).split())


def _recorded_at(probed: _Probe, stream: int) -> datetime | None:
    """Read the recording time from the container tags; never the file's mtime, which copying and syncing change."""
    sections = [_tags(probed.format), _tags(probed.streams[stream])]
    for tag in _DATE_TAGS:
        for tags in sections:
            moment = _aware(tags.get(tag))
            if moment is not None:
                return moment
    return None


def _aware(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if moment.tzinfo is None or moment.year < _FIRST_PLAUSIBLE_YEAR:
        return None
    return moment


def _duration(probed: _Probe, stream: int, path: Path) -> float:
    for section in (probed.format, probed.streams[stream]):
        try:
            seconds = float(str(section.get("duration")))
        except ValueError:
            continue
        if math.isfinite(seconds) and seconds > 0:
            return seconds
    raise MediaValidationError(f"{path.name} has no readable duration")


def _stderr(result: proc.ProcResult) -> str:
    # ffmpeg echoes the absolute input path, which names the user.
    return stderr_preview(redact(result.stderr))
