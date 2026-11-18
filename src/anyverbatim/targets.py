"""What a `run` or `process` target names: a YouTube video, playlist or channel, a local file or folder, or memos."""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from anyverbatim import ids
from anyverbatim.acquire.voicememos import access_fix
from anyverbatim.errors import ConfigError
from anyverbatim.paths import default_output_root
from anyverbatim.ytdlp.runner import redact

MEMOS_PREFIX = "memos:"
EXTENSIONS = frozenset(
    {
        ".m4a",
        ".mp3",
        ".wav",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".mp4",
        ".m4v",
        ".mov",
        ".mkv",
        ".webm",
        ".qta",
        ".caf",
        ".aif",
        ".aiff",
        ".amr",
        ".3gp",
    }
)


@dataclass(frozen=True)
class MemoSelector:
    """A `memos:` target: `new`, `latest`, `all` or an id prefix."""

    selector: str


type Target = ids.YouTubeRef | Path | MemoSelector


def classify(target: str | os.PathLike[str]) -> Target:
    """Read one target; raises `ValueError` for text that names nothing, or both a file and a YouTube id.

    A path object is always local. Text is YouTube when it is a URL on a YouTube host, local when it exists on disk,
    and YouTube again when it is a bare video id.
    """
    if isinstance(target, os.PathLike):
        return Path(target)
    if target.startswith(MEMOS_PREFIX):
        return MemoSelector(target.removeprefix(MEMOS_PREFIX))
    text = target.strip()
    bare_id = ids.is_video_id(text)
    ref = ids.parse_youtube(text)
    if ref is not None and not bare_id:
        return ref
    path = Path(target).expanduser()
    if path.exists():
        if bare_id:
            raise ValueError(f"{target!r} is both a file and a YouTube id; write ./{target} or the watch URL")
        return path
    if ref is not None:
        return ref
    raise ValueError(f"{target!r} is not a YouTube URL or id, and there is no such file or folder")


def expand(paths: Iterable[Path], *, exclude: Path | None = None) -> list[Path]:
    """Replace each folder with the media files under it, sorted by path; files are kept as given.

    Hidden files and folders are skipped, and so is anyverbatim's own output: `exclude` (the output root) and the
    default videos root wherever they lie inside a folder, and every `rec-…` folder, so a run over the folder that
    holds the output root never picks up its own WAVs. Raises `ConfigError` for a folder it cannot read, rather than
    report it empty.
    """
    skipped = {root.resolve() for root in (exclude, default_output_root()) if root is not None}
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found += sorted(_media(path, skipped))
        else:
            found.append(path)
    return found


def _media(folder: Path, skipped: set[Path]) -> Iterable[Path]:
    for root, dirs, files in os.walk(folder, onerror=_unreadable):
        here = Path(root)
        dirs[:] = [name for name in dirs if not _own_or_hidden(here / name, skipped)]
        for name in files:
            if not name.startswith(".") and Path(name).suffix.lower() in EXTENSIONS:
                yield here / name


def _own_or_hidden(folder: Path, skipped: set[Path]) -> bool:
    name = folder.name
    return name.startswith(".") or ids.is_recording_id(name) or folder.resolve() in skipped


def _unreadable(error: OSError) -> None:
    where = redact(str(Path(error.filename).absolute())) if error.filename else "a folder"
    if isinstance(error, PermissionError):
        raised = ConfigError(f"cannot read {where}: permission denied")
        raised.fix = access_fix()
        raise raised from error
    raise ConfigError(f"cannot read {where}: {error.strerror or error}") from error
