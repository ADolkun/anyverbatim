"""What a `run` or `process` target names: a YouTube video, playlist or channel, a local file or folder, or memos."""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ytverbatim import ids

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
    ref = ids.classify(text)
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

    Hidden files and folders are skipped, and so is `exclude` (the output root) when it lies inside a folder.
    """
    skipped = exclude.resolve() if exclude is not None else None
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found += sorted(_media(path, skipped))
        else:
            found.append(path)
    return found


def _media(folder: Path, skipped: Path | None) -> Iterable[Path]:
    for root, dirs, files in os.walk(folder):
        here = Path(root)
        dirs[:] = [name for name in dirs if not name.startswith(".") and (here / name).resolve() != skipped]
        for name in files:
            if not name.startswith(".") and Path(name).suffix.lower() in EXTENSIONS:
                yield here / name
