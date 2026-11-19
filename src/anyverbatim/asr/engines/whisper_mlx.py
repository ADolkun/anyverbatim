"""The Whisper control (`w-prod`, `w-vad`) and its language-ID variants: mlx-whisper decoding as production does.

Every mode makes the same call. `whole_file` hands over the entire recording and lets mlx-whisper window it;
`chunk` hands over one planned chunk (at most 30 s); `runs` hands over one same-language run of chunks, which
mlx-whisper windows like a whole file.

Only `whole_file` sends the Mandarin punctuation prompt, and only when the language is known to be Mandarin. With
`condition_on_previous_text=False` mlx-whisper feeds the prompt to the first 30 s window of each call only. When that
window has little clear speech (a music intro, crosstalk, short audio), Whisper often transcribes the prompt instead,
losing the window's speech, however long the file. A whole-file decode has one such window, so a decode whose first
window echoes the prompt is decoded again, whole and once, without it and flagged `prompt_echo`: twice the ASR time
for that file, traded for its accuracy. `chunk` and `runs` make a call per chunk or run, so every one would be exposed;
they never send the prompt. Their text gets punctuation from the `punctuate` stage and, under the default
`[text] script` of `zh-Hans`, Simplified characters from `normalize`.

A whole-file or run decode is many 30 s windows, and a decoder loop stays inside one. Every decode therefore reports
the windows mlx-whisper decoded, read from the `seek` each segment records, and `transcribe_window` decodes one
window's audio on its own, so the quality guard can decode a looped window again instead of the whole call. It
makes the same call as every mode, never with the prompt.
"""

import importlib
import logging
import re
import unicodedata
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

from pydantic import JsonValue

from anyverbatim.asr.protocol import AsrContext, ChunkFlag, EngineSpec, RawChunk, Window
from anyverbatim.contract.models import Word
from anyverbatim.errors import ConfigError, DependencyError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

Mode = Literal["whole_file", "chunk", "runs"]

# A single fixed temperature would disable Whisper's fallback re-decode and let a repetition loop run unchecked.
TEMPERATURE_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ZH_INITIAL_PROMPT = "以下是普通话内容，请使用简体中文并加上标点符号。"
# The part of the prompt every echo seen in evaluation contains, including the full prompt, a dropped leading 请 and
# paraphrases around it.
_PROMPT_ECHO = "使用简体中文并加上标点符号"
_WEIGHT_FILES = ("weights.safetensors", "weights.npz")


class _MlxCore(Protocol):
    def set_cache_limit(self, limit: int) -> int: ...
    def clear_cache(self) -> None: ...


class _WhisperWord(TypedDict, total=False):
    word: str
    start: float | None
    end: float | None
    probability: float


class _WhisperSegment(TypedDict, total=False):
    seek: int
    text: str
    words: list[_WhisperWord]


class _Dims(Protocol):
    n_mels: int


class _WhisperModel(Protocol):
    dims: _Dims

    def detect_language(self, mel: Any) -> tuple[Any, dict[str, float]]: ...


class _MelArray(Protocol):
    def astype(self, dtype: object) -> _MelArray: ...


class _AudioModule(Protocol):
    SAMPLE_RATE: int
    N_SAMPLES: int
    N_FRAMES: int
    FRAMES_PER_SECOND: int

    def log_mel_spectrogram(self, audio: Any, n_mels: int, padding: int) -> _MelArray: ...
    def pad_or_trim(self, array: _MelArray, length: int, *, axis: int) -> _MelArray: ...


class _WhisperModule(Protocol):
    def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]: ...


class _ModelHolder(Protocol):
    def get_model(self, model_path: str, dtype: object) -> _WhisperModel: ...


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> WhisperEngine:
    mode = variant.get("mode")
    if mode not in ("whole_file", "chunk", "runs"):
        raise ConfigError(f"{spec.id}: Whisper mode must be 'whole_file', 'chunk' or 'runs', got {mode!r}")
    if (mode == "whole_file") != (spec.max_chunk_s is None):
        raise ConfigError(f"{spec.id}: mode {mode!r} contradicts max_chunk_s={spec.max_chunk_s}")
    # Spelled out in the chunked variants so their decode keys differ from those made while they still sent it.
    zh_prompt = variant.get("zh_prompt", True)
    if zh_prompt is not (mode == "whole_file"):
        raise ConfigError(
            f"{spec.id}: only whole_file mode sends the Mandarin prompt; mode {mode!r} got zh_prompt={zh_prompt!r}"
        )
    return WhisperEngine(spec, mode)


def snapshot_path(spec: EngineSpec) -> Path:
    """Return the pinned snapshot in the default Hugging Face cache.

    The worker points `HF_HUB_CACHE` at its own candidates directory; that variable is deliberately ignored so the
    existing 2.9 GB snapshot is reused instead of downloaded again.
    """
    constants = importlib.import_module("huggingface_hub.constants")
    cache = Path(cast(str, constants.default_cache_path))
    path = cache / f"models--{spec.repo.replace('/', '--')}" / "snapshots" / spec.revision
    if not (path / "config.json").is_file() or not any((path / name).is_file() for name in _WEIGHT_FILES):
        raise DependencyError(
            f"the {spec.id} model snapshot is missing at {path}; download it once into the default Hugging Face "
            f"cache with: hf download {spec.repo} --revision {spec.revision}"
        )
    return path


class WhisperEngine:
    def __init__(self, spec: EngineSpec, mode: Mode) -> None:
        self.spec = spec
        self.mode = mode
        self._model_path: str | None = None
        self._whisper: _WhisperModule | None = None
        self._mx: _MlxCore | None = None
        self._model: _WhisperModel | None = None
        self._audio: _AudioModule | None = None
        self._float16: object = None

    def load(self) -> None:
        path = snapshot_path(self.spec)
        try:
            whisper = importlib.import_module("mlx_whisper")
            mx = importlib.import_module("mlx.core")
            holder = cast(_ModelHolder, importlib.import_module("mlx_whisper.transcribe").ModelHolder)
            audio = importlib.import_module("mlx_whisper.audio")
        except ImportError as exc:
            raise DependencyError("install anyverbatim[whisper]") from exc
        # transcribe() reuses the holder's model for the same path and dtype, so this moves the load out of the first
        # decode and times it as load, and language detection shares the one copy.
        self._model = holder.get_model(str(path), mx.float16)
        self._model_path = str(path)
        self._whisper = cast(_WhisperModule, whisper)
        self._mx = cast(_MlxCore, mx)
        self._audio = cast(_AudioModule, audio)
        self._float16 = mx.float16

    def detect_language(self, audio: NDArray[np.float32]) -> dict[str, float]:
        """Language probabilities for the first 30 s of `audio`, computed exactly as mlx-whisper's own detect-once."""
        if self._model is None or self._audio is None or self._mx is None:
            raise RuntimeError("load() must be called before detect_language()")
        mel_audio = self._audio
        with _bounded_mlx_cache(self._mx):
            mel = mel_audio.log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels, padding=mel_audio.N_SAMPLES)
            segment = mel_audio.pad_or_trim(mel, mel_audio.N_FRAMES, axis=-2).astype(self._float16)
            _, probabilities = self._model.detect_language(segment)
        return {code: float(p) for code, p in probabilities.items()}

    def transcribe(self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None) -> RawChunk:
        self._loaded()
        hint = _normalize_language(language)
        prompted = self.mode == "whole_file" and hint is not None and hint.startswith("zh")
        result = self._decode(audio, hint, ZH_INITIAL_PROMPT if prompted else None)
        flags: tuple[ChunkFlag, ...] = ()
        if prompted and echoes_prompt(result):
            result = self._decode(audio, hint, None)
            flags = ("prompt_echo",)
        return self._raw(audio, result, flags)

    def transcribe_window(self, audio: NDArray[np.float32], *, language: str | None) -> RawChunk:
        """Decode at most one window's audio exactly as `transcribe` decodes a window, but never with the prompt."""
        return self._raw(audio, self._decode(audio, _normalize_language(language), None), ())

    def _raw(self, audio: NDArray[np.float32], result: Mapping[str, Any], flags: tuple[ChunkFlag, ...]) -> RawChunk:
        if self._audio is None:
            raise RuntimeError("load() must be called before transcribe()")
        segments = cast(Sequence[_WhisperSegment], result.get("segments") or ())
        words = _words(segments)
        probabilities = [w.conf for w in words if w.conf is not None]
        return RawChunk(
            text=str(result.get("text", "")).strip(),
            language=cast("str | None", result.get("language")),
            confidence=sum(probabilities) / len(probabilities) if probabilities else None,
            words=words,
            flags=flags,
            windows=_windows(
                segments,
                frames_per_second=self._audio.FRAMES_PER_SECOND,
                window_s=self._audio.N_FRAMES / self._audio.FRAMES_PER_SECOND,
                duration=len(audio) / self._audio.SAMPLE_RATE,
            ),
        )

    def _loaded(self) -> tuple[_WhisperModule, _MlxCore, str]:
        if self._whisper is None or self._mx is None or self._model_path is None:
            raise RuntimeError("load() must be called before transcribe()")
        return self._whisper, self._mx, self._model_path

    def _decode(self, audio: NDArray[np.float32], hint: str | None, prompt: str | None) -> dict[str, Any]:
        whisper, mx, model_path = self._loaded()
        with _bounded_mlx_cache(mx):
            return whisper.transcribe(
                audio,
                path_or_hf_repo=model_path,
                word_timestamps=True,
                language=hint,
                initial_prompt=prompt,
                condition_on_previous_text=False,
                temperature=TEMPERATURE_SCHEDULE,
            )


def echoes_prompt(result: Mapping[str, Any]) -> bool:
    """Whether the first window of a transcribe() result echoes the prompt, ignoring punctuation and whitespace.

    Only window 0 sees the prompt, so a speaker saying the sentence later is not an echo. A result whose segments
    carry no `seek` is checked whole.
    """
    segments = cast(Sequence[_WhisperSegment], result.get("segments") or ())
    if any("seek" in segment for segment in segments):
        text = "".join(segment.get("text", "") for segment in segments if segment.get("seek") == 0)
    else:
        text = str(result.get("text", ""))
    return _PROMPT_ECHO in _bare(text)


def _bare(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c)[0] not in "PZC")


def _normalize_language(code: str | None) -> str | None:
    """Reduce a tag such as `zh-Hans` to the primary code Whisper accepts.

    An unknown tag auto-detects with a warning, as in production.
    """
    if not code or code.strip().lower() == "auto":
        return None
    primary = re.split(r"[-_]", code.strip(), maxsplit=1)[0].lower()
    if not primary:
        return None
    tokenizer = importlib.import_module("mlx_whisper.tokenizer")
    known = set(cast(Mapping[str, str], tokenizer.LANGUAGES)) | set(
        cast(Mapping[str, str], tokenizer.TO_LANGUAGE_CODE).values()
    )
    if primary not in known:
        log.warning("unknown language %r; letting Whisper auto-detect", code)
        return None
    return primary


def _words(segments: Sequence[_WhisperSegment]) -> tuple[Word, ...]:
    return tuple(
        Word(text=w["word"], start=w["start"], end=w["end"], conf=w.get("probability"))
        for segment in segments
        for w in segment.get("words") or ()
        if w.get("start") is not None and w.get("end") is not None
    )


def _windows(
    segments: Sequence[_WhisperSegment], *, frames_per_second: int, window_s: float, duration: float
) -> tuple[Window, ...]:
    """Return the decode windows behind `segments`, read from the `seek` frame each segment records.

    A window ends where the next one starts, but never after its 30 s or the audio: mlx-whisper leaves no segment
    for a window it judged silent, which must not stretch the window before it.
    """
    if not segments or not all("seek" in segment for segment in segments):
        return ()
    seeks: list[int] = []
    counts: list[int] = []
    for segment in segments:
        count = len(_words((segment,)))
        if seeks and seeks[-1] == segment["seek"]:
            counts[-1] += count
        else:
            seeks.append(segment["seek"])
            counts.append(count)
    starts = [seek / frames_per_second for seek in seeks]
    return tuple(
        Window(start, min(following, start + window_s, duration), count)
        for start, following, count in zip(starts, [*starts[1:], duration], counts, strict=True)
    )


@contextmanager
def _bounded_mlx_cache(mx: _MlxCore) -> Generator[None]:
    """Run inference with unused-buffer caching off so the allocator does not hold a whole decode's temporaries."""
    previous = mx.set_cache_limit(0)
    failed = False
    try:
        mx.clear_cache()
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            mx.clear_cache()
        except Exception:
            if not failed:
                raise
            log.warning("MLX cache cleanup failed while preserving the inference error", exc_info=True)
        finally:
            mx.set_cache_limit(previous)
