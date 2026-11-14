import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from ytverbatim import proc
from ytverbatim.acquire import local
from ytverbatim.acquire.voicememos import Memo, MemoCache
from ytverbatim.errors import MediaValidationError

AAC = {"codec_name": "aac", "duration": "12.5"}
APAC = {"codec_name": "apac", "duration": "12.5"}


class FakeMedia:
    """Stands in for ffprobe and ffmpeg over synthetic files.

    A file's first line names its audio packets, so two files sharing it are the same audio in different containers;
    `probes` holds each file's ffprobe JSON and `undecodable` the audio stream indexes ffmpeg cannot decode.
    """

    def __init__(self) -> None:
        self.probes: dict[str, dict[str, object]] = {}
        self.undecodable: set[int] = set()
        self.calls: list[list[str]] = []

    def add(self, path: Path, packets: str, *, streams: list[dict[str, object]], tags: dict[str, str]) -> Path:
        path.write_text(f"{packets}\n{path.suffix} container\n", encoding="utf-8")
        self.probes[path.name] = {"streams": streams, "format": {"duration": "12.5", "tags": tags}}
        return path

    def __call__(self, argv: list[str], *, timeout_s: float, **_: object) -> proc.ProcResult:
        self.calls.append(argv)
        source = Path(argv[argv.index("-i") + 1]) if "-i" in argv else Path(argv[-1])
        if argv[0] == "ffprobe":
            return proc.ProcResult(0, json.dumps(self.probes[source.name]), "", 0.0)
        stream = int(argv[argv.index("-map") + 1].rsplit(":", 1)[1])
        if "hash" in argv:
            packets = source.read_text(encoding="utf-8").splitlines()[0]
            return proc.ProcResult(0, f"SHA256={hashlib.sha256(f'{packets}:{stream}'.encode()).hexdigest()}\n", "", 0.0)
        if "null" in argv:
            return proc.ProcResult(1 if stream in self.undecodable else 0, "", "", 0.0)
        raise AssertionError(f"unexpected command {argv}")


@pytest.fixture
def media(monkeypatch: pytest.MonkeyPatch) -> FakeMedia:
    fake = FakeMedia()
    monkeypatch.setattr(proc, "run", fake)
    return fake


def test_the_id_is_the_packet_hash_whatever_the_name_container_or_metadata(tmp_path: Path, media: FakeMedia) -> None:
    first = media.add(tmp_path / "meeting.m4a", "alpha", streams=[AAC], tags={"title": "Weekly sync"})
    renamed = media.add(tmp_path / "renamed.mov", "alpha", streams=[AAC], tags={"title": "Edited title"})
    other = media.add(tmp_path / "other.m4a", "beta", streams=[AAC], tags={})

    a, b, c = (local.open_recording(path) for path in (first, renamed, other))
    digest = hashlib.sha256(b"alpha:0").hexdigest()
    assert a.audio_sha256 == digest and a.id == f"rec-{digest[:16]}"
    assert b.id == a.id and c.id != a.id
    assert local.recording_id(renamed) == a.id
    hashed = next(argv for argv in media.calls if "hash" in argv)
    assert hashed[hashed.index("-map") + 1] == "0:a:0" and "copy" in hashed


def test_aac_is_preferred_over_an_undecodable_spatial_track(tmp_path: Path, media: FakeMedia) -> None:
    memo = media.add(tmp_path / "memo.qta", "alpha", streams=[APAC, AAC], tags={})
    media.undecodable = {0}
    recording = local.open_recording(memo)
    assert recording.stream == 1
    assert recording.audio_sha256 == hashlib.sha256(b"alpha:1").hexdigest()
    assert not any("null" in argv for argv in media.calls), "an AAC stream needs no trial decode"


def test_without_aac_the_first_decodable_stream_is_taken(tmp_path: Path, media: FakeMedia) -> None:
    clip = media.add(tmp_path / "clip.mkv", "alpha", streams=[APAC, {"codec_name": "opus"}], tags={})
    media.undecodable = {0}
    assert local.open_recording(clip).stream == 1
    media.undecodable = {0, 1}
    with pytest.raises(MediaValidationError, match=r"cannot decode any audio stream of clip.mkv \(apac, opus\)"):
        local.open_recording(clip)


def test_a_file_without_audio_is_an_error(tmp_path: Path, media: FakeMedia) -> None:
    silent = media.add(tmp_path / "screen.mov", "alpha", streams=[], tags={})
    with pytest.raises(MediaValidationError, match=r"screen\.mov has no audio stream"):
        local.open_recording(silent)


def test_a_missing_file_is_an_error(tmp_path: Path, media: FakeMedia) -> None:
    with pytest.raises(MediaValidationError, match=r"no such file: gone\.m4a"):
        local.open_recording(tmp_path / "gone.m4a")


def test_metadata_precedence_and_no_date_rather_than_the_mtime(tmp_path: Path, media: FakeMedia) -> None:
    plain = local.open_recording(media.add(tmp_path / "Team call.m4a", "a", streams=[AAC], tags={}))
    assert plain.title == "Team call" and plain.recorded_at is None
    assert plain.duration_s == 12.5 and plain.file_name == "Team call.m4a"

    both = {
        "creation_time": "2026-11-03T06:05:00.000000Z",
        "com.apple.quicktime.creationdate": "2026-11-03T14:05:00+0800",
    }
    tagged = local.open_recording(media.add(tmp_path / "x.mov", "b", streams=[AAC], tags={"TITLE": "Sync\n1", **both}))
    assert tagged.title == "Sync 1"
    assert tagged.recorded_at == datetime(2026, 11, 3, 14, 5, tzinfo=timezone(timedelta(hours=8)))

    muxed = local.open_recording(
        media.add(tmp_path / "y.m4a", "c", streams=[AAC], tags={"creation_time": "2026-11-03T06:05:00.000000Z"})
    )
    assert muxed.recorded_at == datetime(2026, 11, 3, 6, 5, tzinfo=UTC)

    for unusable in ("1904-01-01T00:00:00.000000Z", "2026-11-03T14:05:00", "garbage"):
        path = media.add(tmp_path / "z.m4a", "d", streams=[AAC], tags={"creation_time": unusable})
        assert local.open_recording(path).recorded_at is None, unusable


def test_a_stream_date_is_read_when_the_container_has_none(tmp_path: Path, media: FakeMedia) -> None:
    stream = {**AAC, "tags": {"creation_time": "2026-11-03T06:05:00Z"}}
    path = media.add(tmp_path / "s.m4a", "a", streams=[stream], tags={})
    assert local.open_recording(path).recorded_at == datetime(2026, 11, 3, 6, 5, tzinfo=UTC)


def test_the_wav_of_a_vanished_source_asks_for_the_file_again(tmp_path: Path, media: FakeMedia) -> None:
    path = media.add(tmp_path / "meeting.m4a", "alpha", streams=[AAC], tags={})
    recording = local.open_recording(path)
    path.unlink()
    with pytest.raises(MediaValidationError, match=r"meeting\.m4a is no longer") as raised:
        local.to_wav(recording, tmp_path / "audio.wav")
    assert raised.value.fix == "pass the file again"


def test_the_memo_cache_takes_the_recording_hasher(tmp_path: Path, media: FakeMedia) -> None:
    path = media.add(tmp_path / "memo.m4a", "alpha", streams=[AAC], tags={})
    stat = path.stat()
    memo = Memo("UNIQUE-1", None, None, None, path, "local", stat.st_size, stat.st_mtime_ns)
    cache = MemoCache(tmp_path / "voicememos.json")
    expected = local.open_recording(path).id
    assert cache.resolve(memo, local.recording_id) == expected
    assert MemoCache(tmp_path / "voicememos.json").lookup(memo) == expected
