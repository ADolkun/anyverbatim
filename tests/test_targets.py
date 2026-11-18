from pathlib import Path

import pytest

from anyverbatim import ids
from anyverbatim.errors import ConfigError
from anyverbatim.targets import MemoSelector, classify, expand

VIDEO = "pubVideo001"


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("target", "kind", "key"),
    [
        (VIDEO, ids.RefKind.VIDEO, VIDEO),
        (f"https://www.youtube.com/watch?v={VIDEO}", ids.RefKind.VIDEO, VIDEO),
        (f"youtu.be/{VIDEO}", ids.RefKind.VIDEO, VIDEO),
        ("https://www.youtube.com/playlist?list=PLsynthetic", ids.RefKind.PLAYLIST, "PLsynthetic"),
        ("https://www.youtube.com/@synthetic", ids.RefKind.CHANNEL, "@synthetic"),
    ],
)
def test_youtube_targets(here: Path, target: str, kind: ids.RefKind, key: str) -> None:
    found = classify(target)
    assert isinstance(found, ids.YouTubeRef) and found.kind is kind and found.id == key


def test_local_targets(here: Path) -> None:
    (here / "meeting.m4a").write_bytes(b"x")
    (here / "folder").mkdir()
    (here / VIDEO).write_bytes(b"x")
    assert classify("meeting.m4a") == Path("meeting.m4a")
    assert classify("folder") == Path("folder")
    assert classify(Path("missing.m4a")) == Path("missing.m4a"), "a path object is always a file"
    assert classify(f"./{VIDEO}") == Path(f"./{VIDEO}")


def test_a_home_relative_path_is_expanded(here: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(here))
    (here / "memo.m4a").write_bytes(b"x")
    assert classify("~/memo.m4a") == here / "memo.m4a"


def test_a_url_wins_over_a_same_named_file(here: Path) -> None:
    assert isinstance(classify(f"https://youtu.be/{VIDEO}"), ids.YouTubeRef)


def test_memo_selectors(here: Path) -> None:
    assert classify("memos:new") == MemoSelector("new")
    assert classify("memos:1a2b") == MemoSelector("1a2b")


def test_a_bare_id_that_is_also_a_file_is_refused(here: Path) -> None:
    (here / VIDEO).write_bytes(b"x")
    with pytest.raises(ValueError, match=f"is both a file and a YouTube id; write ./{VIDEO} or the watch URL"):
        classify(VIDEO)


def test_text_that_names_nothing_is_refused(here: Path) -> None:
    with pytest.raises(ValueError, match="not a YouTube URL or id, and there is no such file or folder"):
        classify("missing.m4a")
    with pytest.raises(ValueError, match="no such file"):
        classify("https://example.com/watch?v=pubVideo001")


def test_folders_expand_recursively_to_sorted_media_files(tmp_path: Path) -> None:
    root = tmp_path / "in"
    for name in ("b.m4a", "a.MP3", "notes.txt", ".hidden.m4a", "sub/c.mov", ".private/d.m4a", "out/rec.m4a"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    single = tmp_path / "chosen.txt"
    single.write_bytes(b"x")

    found = expand([root, single], exclude=root / "out")
    assert found == [root / "a.MP3", root / "b.m4a", root / "sub" / "c.mov", single]
    assert root / "out" / "rec.m4a" in expand([root])


def test_expansion_skips_its_own_output_even_when_run_over_the_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANYVERBATIM_HOME", str(tmp_path / "home"))
    (tmp_path / "meeting.m4a").write_bytes(b"x")
    for leftover in ("rec-0123456789abcdef/audio.wav", "home/videos/pubVideo001/audio.webm"):
        path = tmp_path / leftover
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    assert expand([tmp_path], exclude=tmp_path) == [tmp_path / "meeting.m4a"]


def test_an_unreadable_folder_is_an_error_with_the_access_fix(tmp_path: Path) -> None:
    locked = tmp_path / "Recordings"
    locked.mkdir()
    locked.chmod(0)
    try:
        with pytest.raises(ConfigError, match="Recordings: permission denied") as raised:
            expand([tmp_path])
    finally:
        locked.chmod(0o700)
    assert "Full Disk Access" in raised.value.fix
