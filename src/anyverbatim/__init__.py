from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anyverbatim.acquire.session import expand
    from anyverbatim.acquire.video import video_file
    from anyverbatim.auth.chromium import SUPPORTED_BROWSERS
    from anyverbatim.config import (
        AsrOptions,
        AuthOptions,
        DiarizeOptions,
        LLMOptions,
        Options,
        TextOptions,
        YtdlpOptions,
    )
    from anyverbatim.contract import ContextDocument, load_document
    from anyverbatim.ids import RefKind, YouTubeRef, parse_youtube, video_url
    from anyverbatim.llm.protocol import LLMClient, LLMRequest, LLMResponse, Usage
    from anyverbatim.pipeline import (
        LLM_STAGES,
        STAGES,
        ProgressEvent,
        ProgressStage,
        Result,
        check_dependencies,
        process,
    )
    from anyverbatim.text.numbers import QKind, QToken, quantity_tokens

__version__ = version("anyverbatim")

# The library surface is imported on first use so `anyverbatim --version` and the lighter commands never pay for
# yt-dlp, pydantic and the pipeline.
_LAZY = {
    "AsrOptions": "anyverbatim.config",
    "AuthOptions": "anyverbatim.config",
    "ContextDocument": "anyverbatim.contract",
    "DiarizeOptions": "anyverbatim.config",
    "LLMClient": "anyverbatim.llm.protocol",
    "LLM_STAGES": "anyverbatim.pipeline",
    "LLMOptions": "anyverbatim.config",
    "LLMRequest": "anyverbatim.llm.protocol",
    "LLMResponse": "anyverbatim.llm.protocol",
    "Options": "anyverbatim.config",
    "ProgressEvent": "anyverbatim.pipeline",
    "ProgressStage": "anyverbatim.pipeline",
    "QKind": "anyverbatim.text.numbers",
    "QToken": "anyverbatim.text.numbers",
    "RefKind": "anyverbatim.ids",
    "Result": "anyverbatim.pipeline",
    "STAGES": "anyverbatim.pipeline",
    "SUPPORTED_BROWSERS": "anyverbatim.auth.chromium",
    "TextOptions": "anyverbatim.config",
    "Usage": "anyverbatim.llm.protocol",
    "YouTubeRef": "anyverbatim.ids",
    "YtdlpOptions": "anyverbatim.config",
    "check_dependencies": "anyverbatim.pipeline",
    "expand": "anyverbatim.acquire.session",
    "load_document": "anyverbatim.contract",
    "parse_youtube": "anyverbatim.ids",
    "process": "anyverbatim.pipeline",
    "quantity_tokens": "anyverbatim.text.numbers",
    "video_file": "anyverbatim.acquire.video",
    "video_url": "anyverbatim.ids",
}

__all__ = [
    "LLM_STAGES",
    "STAGES",
    "SUPPORTED_BROWSERS",
    "AsrOptions",
    "AuthOptions",
    "ContextDocument",
    "DiarizeOptions",
    "LLMClient",
    "LLMOptions",
    "LLMRequest",
    "LLMResponse",
    "Options",
    "ProgressEvent",
    "ProgressStage",
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
