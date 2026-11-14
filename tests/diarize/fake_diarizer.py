"""Deterministic stand-in diarizer, named by dotted path (`fake_diarizer.create`) so the worker imports it lazily."""

import time
import wave
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from ytverbatim.diarize.protocol import DiarizerDef, DiarizerSpec, RawDiarization, Turn

RATE = 16_000
CHANNELS = 8
FRAME_S = 0.01


def write_silence(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes(b"\x00\x00" * round(seconds * RATE))
    return path


def definition(**variant: object) -> DiarizerDef:
    spec = DiarizerSpec("d-fake", "fake", "example/fake", "0" * 40, 1, CHANNELS, RATE)
    return DiarizerDef(spec, "fake_diarizer.create", {"threshold": 0.5, **variant})  # type: ignore[arg-type]


class FakeDiarizer:
    """Speaker 0 for the first half of the audio, speaker 1 for the second."""

    def __init__(self, variant: Mapping[str, JsonValue]) -> None:
        self.variant = variant

    def _log(self, line: str) -> None:
        target = self.variant.get("calls_log")
        if isinstance(target, str):
            with Path(target).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def load(self) -> None:
        self._log("load")

    def diarize(self, audio: NDArray[np.float32]) -> RawDiarization:
        self._log(f"diarize {len(audio)}")
        pause = self.variant.get("sleep_s")
        if isinstance(pause, float):
            time.sleep(pause)
        if self.variant.get("fail"):
            raise RuntimeError("fake diarizer exploded")
        frames = round(len(audio) / RATE / FRAME_S)
        half = frames // 2
        probs = np.zeros((frames, CHANNELS), dtype=np.float32)
        probs[:half, 0] = 0.9
        probs[half:, 1] = 0.8
        bad = self.variant.get("bad_prob")
        if isinstance(bad, float):
            probs[0, 0] = bad
        turns = (
            Turn(start=0.0, end=half * FRAME_S, speaker=0),
            Turn(start=half * FRAME_S, end=frames * FRAME_S, speaker=1),
        )
        return RawDiarization(turns=turns, probs=probs, frame_s=FRAME_S)


def create(spec: DiarizerSpec, variant: Mapping[str, JsonValue]) -> FakeDiarizer:
    return FakeDiarizer(variant)
