from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ytverbatim.acquire.session import expand
    from ytverbatim.acquire.video import video_file
    from ytverbatim.auth.chromium import SUPPORTED_BROWSERS
    from ytverbatim.config import AsrOptions, AuthOptions, LLMOptions, Options, TextOptions, YtdlpOptions
    from ytverbatim.contract import ContextDocument, load_document
    from ytverbatim.ids import RefKind, YouTubeRef, parse_youtube, video_url
    from ytverbatim.llm.protocol import LLMClient, LLMRequest, LLMResponse, Usage
    from ytverbatim.pipeline import ProgressEvent, Result, check_dependencies, process
    from ytverbatim.text.numbers import QKind, QToken, quantity_tokens

__version__ = version("ytverbatim")

# The library surface is imported on first use so `ytverbatim --version` and the lighter commands never pay for
# yt-dlp, pydantic and the pipeline.
_LAZY = {
    "AsrOptions": "ytverbatim.config",
    "AuthOptions": "ytverbatim.config",
    "ContextDocument": "ytverbatim.contract",
    "LLMClient": "ytverbatim.llm.protocol",
    "LLMOptions": "ytverbatim.config",
    "LLMRequest": "ytverbatim.llm.protocol",
    "LLMResponse": "ytverbatim.llm.protocol",
    "Options": "ytverbatim.config",
    "ProgressEvent": "ytverbatim.pipeline",
    "QKind": "ytverbatim.text.numbers",
    "QToken": "ytverbatim.text.numbers",
    "RefKind": "ytverbatim.ids",
    "Result": "ytverbatim.pipeline",
    "SUPPORTED_BROWSERS": "ytverbatim.auth.chromium",
    "TextOptions": "ytverbatim.config",
    "Usage": "ytverbatim.llm.protocol",
    "YouTubeRef": "ytverbatim.ids",
    "YtdlpOptions": "ytverbatim.config",
    "check_dependencies": "ytverbatim.pipeline",
    "expand": "ytverbatim.acquire.session",
    "load_document": "ytverbatim.contract",
    "parse_youtube": "ytverbatim.ids",
    "process": "ytverbatim.pipeline",
    "quantity_tokens": "ytverbatim.text.numbers",
    "video_file": "ytverbatim.acquire.video",
    "video_url": "ytverbatim.ids",
}

__all__ = [
    "SUPPORTED_BROWSERS",
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


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    # Cached so the next lookup is an ordinary attribute and never reaches __getattr__ again.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
