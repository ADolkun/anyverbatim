"""Engine that answers from a script, so guard tests control exactly what each decode returns."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from anyverbatim.asr.protocol import AsrContext, EngineSpec, RawChunk, Window
from anyverbatim.contract.models import Word

LOOP_TEXT = " ".join(["repeat risk"] * 10)


def looping(tokens: int = 64) -> RawChunk:
    return RawChunk(text=LOOP_TEXT, generated_tokens=tokens, max_tokens=64)


def sound(text: str = "a sound decode", tokens: int = 5) -> RawChunk:
    return RawChunk(text=text, generated_tokens=tokens, max_tokens=64)


def timed(texts: Sequence[str], start: float, end: float) -> tuple[Word, ...]:
    """Words with exactly these texts, spread evenly over `start`-`end`."""
    step = (end - start) / max(len(texts), 1)
    return tuple(Word(text=text, start=start + i * step, end=start + (i + 1) * step) for i, text in enumerate(texts))


def spaced(text: str) -> list[str]:
    """English words as Whisper times them: each with its leading space."""
    return [f" {word}" for word in text.split()]


def said(*texts: str, seconds: float = 2.0, language: str | None = None) -> RawChunk:
    """A decode of `seconds` with exactly these word texts, its text the words joined as Whisper's is."""
    return RawChunk(text="".join(texts).strip(), language=language, words=timed(texts, 0.0, seconds))


def windowed(*windows: tuple[float, float, Sequence[str]], language: str | None = "en") -> RawChunk:
    """A whole-call decode made of one window per `(start, end, word texts)`."""
    words = tuple(word for start, end, texts in windows for word in timed(texts, start, end))
    return RawChunk(
        text="".join(word.text for word in words).strip(),
        language=language,
        words=words,
        windows=tuple(Window(start, end, len(texts)) for start, end, texts in windows),
    )


class ScriptedEngine:
    """Replies in order and repeats the last reply once the script runs out; keeps what it was asked to decode."""

    def __init__(self, script: Sequence[RawChunk]) -> None:
        self.script = script
        self.heard: list[NDArray[np.float32]] = []

    def load(self) -> None:
        pass

    def transcribe(self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None) -> RawChunk:
        self.heard.append(audio)
        return self.script[min(len(self.heard), len(self.script)) - 1]


class WindowedEngine(ScriptedEngine):
    """A whole-call engine that also decodes single windows, answering those from `windows` the same way."""

    def __init__(self, script: Sequence[RawChunk], windows: Sequence[RawChunk]) -> None:
        super().__init__(script)
        self.windows = windows
        self.windows_heard: list[tuple[NDArray[np.float32], str | None]] = []

    def transcribe_window(self, audio: NDArray[np.float32], *, language: str | None) -> RawChunk:
        self.windows_heard.append((audio, language))
        return self.windows[min(len(self.windows_heard), len(self.windows)) - 1]


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> ScriptedEngine:
    """A worker-loadable engine whose first decode loops; with `always`, every decode does.

    With `windowed` it decodes whole calls: the second of two 5 s windows loops, and decoding it again as halves
    gives sound text when `always` is off.
    """
    loop = looping()
    if variant.get("windowed"):
        decode = windowed((0.0, 5.0, spaced("a sound decode")), (5.0, 10.0, spaced(LOOP_TEXT)))
        halves = [said(*spaced(LOOP_TEXT))] if variant.get("always") else [said(" left"), said(" right")]
        return WindowedEngine([decode], halves)
    return ScriptedEngine([loop] if variant.get("always") else [loop, sound()])
