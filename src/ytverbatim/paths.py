import os
from pathlib import Path

from ytverbatim.ids import is_recording_id, is_video_id

_APP = "ytverbatim"
_PRIVATE_MODE = 0o700


def _xdg(variable: str, fallback: Path) -> Path:
    value = os.environ.get(variable)
    # The XDG spec says relative values are invalid and must be ignored.
    if value and Path(value).is_absolute():
        return Path(value)
    return fallback


def _private(path: Path) -> Path:
    path.mkdir(mode=_PRIVATE_MODE, parents=True, exist_ok=True)
    if path.stat().st_mode & 0o777 != _PRIVATE_MODE:
        path.chmod(_PRIVATE_MODE)
    return path


def config_file() -> Path:
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / _APP / "config.toml"


def _state_root() -> Path:
    return _xdg("XDG_STATE_HOME", Path.home() / ".local" / "state") / _APP


def state_dir() -> Path:
    return _private(_state_root())


def auth_dir() -> Path:
    return _private(state_dir() / "auth")


def auth_source_file() -> Path:
    return auth_dir() / "source.json"


def preflight_cache_file() -> Path:
    return auth_dir() / "preflight.json"


def cooldown_file() -> Path:
    return auth_dir() / "cooldown.json"


def snapshots_dir() -> Path:
    return _private(auth_dir() / "snapshots")


def chatgpt_auth_dir(*, create_parents: bool = False) -> Path:
    """Home of the ChatGPT credentials, never created here: reading the sign-in status must leave no trace.

    With `create_parents`, the private directories above it are made; the store creates the directory itself and
    refuses a symlink in its place.
    """
    if create_parents:
        return _private(state_dir() / "llm") / "chatgpt"
    return _state_root() / "llm" / "chatgpt"


def chatgpt_credentials_file() -> Path:
    return chatgpt_auth_dir() / "credentials.json"


def chatgpt_lock_file() -> Path:
    return chatgpt_auth_dir() / "credentials.lock"


def locks_dir() -> Path:
    return _private(state_dir() / "locks")


def voicememos_cache_file(*, create_parents: bool = False) -> Path:
    """Never created here, so listing memos leaves no trace; a write passes `create_parents`."""
    return (state_dir() if create_parents else _state_root()) / "voicememos.json"


def cache_dir() -> Path:
    return _private(Path.home() / "Library" / "Caches" / _APP)


def pins_dir() -> Path:
    return _private(cache_dir() / "pins")


def llm_dir(provider: str) -> Path:
    return _private(_private(cache_dir() / "llm") / provider)


def llm_workdir(provider: str) -> Path:
    """Empty cwd for an LLM CLI child, so no project files or instructions are in reach."""
    return _private(llm_dir(provider) / "cwd")


def home_dir() -> Path:
    value = os.environ.get("YTVERBATIM_HOME")
    return Path(value).expanduser() if value else Path.home() / ".local" / "share" / _APP


def default_output_root() -> Path:
    return home_dir() / "videos"


def recordings_root() -> Path:
    return home_dir() / "recordings"


def video_dir(video_id: str, out: Path | None = None) -> Path:
    if not is_video_id(video_id):
        raise ValueError(f"not a YouTube video id: {video_id!r}")
    return (out if out is not None else default_output_root()) / video_id


def source_dir(source_id: str, out: Path | None = None) -> Path:
    """Return a video's or a recording's folder; without `out`, under the default root of its kind."""
    if is_video_id(source_id):
        return video_dir(source_id, out)
    if not is_recording_id(source_id):
        raise ValueError(f"not a YouTube video id or a recording id: {source_id!r}")
    return (out if out is not None else recordings_root()) / source_id
