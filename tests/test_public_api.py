"""Pins the declared library surface: any change to a name or a signature here is a deliberate API change.

`docs/library.md` explains the surface and the semver policy that this file enforces.
"""

import dataclasses
import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import ytverbatim
from ytverbatim import config, contract, errors, ids
from ytverbatim.asr import registry
from ytverbatim.errors import DependencyError

EXPECTED_ALL = [
    "AsrOptions",
    "AuthOptions",
    "ContextDocument",
    "LLMClient",
    "LLMOptions",
    "LLMRequest",
    "LLMResponse",
    "Options",
    "ProgressEvent",
    "QKind",
    "QToken",
    "RefKind",
    "Result",
    "SUPPORTED_BROWSERS",
    "TextOptions",
    "Usage",
    "YouTubeRef",
    "YtdlpOptions",
    "__version__",
    "check_dependencies",
    "expand",
    "load_document",
    "parse_youtube",
    "process",
    "quantity_tokens",
    "video_file",
    "video_url",
]


def shown(signature: inspect.Signature) -> str:
    return re.sub(r"(?:[a-z_]+\.)+(?=[A-Za-z_])", "", str(signature))


def test_the_top_level_names_are_pinned() -> None:
    assert sorted(ytverbatim.__all__) == sorted(EXPECTED_ALL)
    for name in ytverbatim.__all__:
        assert getattr(ytverbatim, name) is not None, name


def test_importing_the_package_loads_none_of_the_library() -> None:
    import subprocess
    import sys

    code = "import sys, ytverbatim; sys.exit(any(m.startswith('ytverbatim.pipeline') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


FUNCTIONS: dict[str, str] = {
    "process": (
        "(target: str | PathLike[str], options: Options | None = None, *, llm: LLMClient | None = None, "
        "out: Path | None = None, auth: str | None = None, captions: bool | None = None, force: bool = False, "
        "cookies_first: bool | None = None, interactive: bool = False, "
        "progress: Callable[[ProgressEvent], None] | None = None) -> Result"
    ),
    "expand": "(target: str, *, limit: int = 0, auth: str | None = None, options: Options | None = None) -> list[str]",
    "video_file": "(video_id: str, out: Path | None = None) -> Path | None",
    "parse_youtube": "(text: str) -> YouTubeRef | None",
    "video_url": "(video_id: str) -> str",
    "check_dependencies": (
        "(options: Options | None = None, *, source: Literal['youtube', 'recording'] = 'youtube') -> None"
    ),
    "load_document": "(source: Path | str) -> ContextDocument",
    "quantity_tokens": "(text: str) -> list[QToken]",
}


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_function_signatures_are_pinned(name: str) -> None:
    function: Callable[..., Any] = getattr(ytverbatim, name)
    assert shown(inspect.signature(function)) == FUNCTIONS[name]


DATACLASSES: dict[str, list[str]] = {
    "Result": [
        "id: str",
        "kind: Literal['youtube', 'recording']",
        "document: ContextDocument",
        "json_bytes: bytes",
        "directory: Path",
        "markdown_path: Path",
        "video_path: Path | None",
        "reused: tuple[str, ...]",
        "notices: tuple[str, ...]",
    ],
    "ProgressEvent": ["id: str", "stage: str", "detail: str"],
    "YouTubeRef": ["kind: RefKind", "id: str", "url: str"],
    "Usage": ["input_tokens: int", "output_tokens: int"],
    "LLMRequest": [
        "system: str",
        "user: str",
        "json_schema: Mapping[str, object] | None",
        "max_output_tokens: int",
        "temperature: float | None",
    ],
    "LLMResponse": ["text: str", "usage: Usage | None", "elapsed_s: float"],
}


def type_text(annotation: object) -> str:
    text = annotation.__name__ if isinstance(annotation, type) else str(annotation)
    return re.sub(r"(?:[A-Za-z_]+\.)+(?=[A-Za-z_])", "", text).replace("typing.", "")


@pytest.mark.parametrize("name", sorted(DATACLASSES))
def test_dataclass_fields_are_pinned_and_frozen(name: str) -> None:
    cls = getattr(ytverbatim, name)
    assert [f"{f.name}: {type_text(f.type)}" for f in dataclasses.fields(cls)] == DATACLASSES[name]
    assert cls.__dataclass_params__.frozen


def test_the_stable_option_classes_are_the_config_ones() -> None:
    for name in ("Options", "AuthOptions", "YtdlpOptions", "AsrOptions", "TextOptions", "LLMOptions"):
        assert getattr(ytverbatim, name) is getattr(config, name)
    assert ytverbatim.ContextDocument is contract.ContextDocument
    assert ytverbatim.RefKind is ids.RefKind


def test_errors_export_every_error_class_with_the_auth_vocabulary() -> None:
    classes = {
        name
        for name, value in vars(errors).items()
        if isinstance(value, type) and issubclass(value, errors.YtverbatimError)
    }
    assert set(errors.__all__) == classes | {"AuthState", "Signal"}
    for name in classes:
        error = getattr(errors, name)
        assert error.fix == "" and error.stage is None, "both default to unset, for the caller to read"


def test_a_members_playlist_is_told_from_other_references() -> None:
    members = ytverbatim.parse_youtube("https://www.youtube.com/playlist?list=UUMOabcdefgh")
    ordinary = ytverbatim.parse_youtube("https://www.youtube.com/playlist?list=PLabcdefgh")
    video = ytverbatim.parse_youtube("https://youtu.be/abcDEF_1234")
    assert members is not None and members.kind is ytverbatim.RefKind.PLAYLIST and members.members_only
    assert ordinary is not None and not ordinary.members_only
    assert video is not None and video.kind is ytverbatim.RefKind.VIDEO and not video.members_only
    assert video.url == ytverbatim.video_url("abcDEF_1234")
    assert ytverbatim.parse_youtube("https://example.com/watch?v=abcDEF_1234") is None


@pytest.fixture
def libraries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> set[str]:
    """The top-level modules the engines import, mutable: remove a name to make that library missing.

    Also points HOME at an empty folder, so `check_dependencies()` reads no real config file.
    """
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    present = {"mlx_whisper", "mlx_audio"}
    monkeypatch.setattr(registry, "find_spec", lambda name: object() if name in present else None)
    return present


def test_check_dependencies_passes_when_the_engine_is_installed(libraries: set[str]) -> None:
    ytverbatim.check_dependencies()
    ytverbatim.check_dependencies(source="recording")


def test_check_dependencies_names_the_extra_the_youtube_engine_needs(libraries: set[str]) -> None:
    libraries.discard("mlx_whisper")
    with pytest.raises(DependencyError, match=r"install ytverbatim\[whisper\]"):
        ytverbatim.check_dependencies()


def test_a_recording_needs_both_extras(libraries: set[str]) -> None:
    libraries.clear()
    with pytest.raises(DependencyError, match=r"install ytverbatim\[whisper,asr\]"):
        ytverbatim.check_dependencies(source="recording")
    libraries.add("mlx_whisper")
    with pytest.raises(DependencyError, match=r"install ytverbatim\[asr\]"):
        ytverbatim.check_dependencies(source="recording")
    ytverbatim.check_dependencies()


def test_check_dependencies_checks_the_chosen_engine_and_the_second_opinion(libraries: set[str]) -> None:
    libraries.discard("mlx_audio")
    chosen = config.load_options(overrides={"asr": {"engine": "w-vad"}})
    with pytest.raises(DependencyError, match=r"ytverbatim\[asr\]"):
        ytverbatim.check_dependencies(chosen)
    second = config.load_options(overrides={"text": {"numeric_second_opinion": "w-vad"}})
    with pytest.raises(DependencyError, match=r"ytverbatim\[asr\]"):
        ytverbatim.check_dependencies(second)
