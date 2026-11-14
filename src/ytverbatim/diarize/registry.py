import importlib
from collections.abc import Mapping
from importlib.util import find_spec

from pydantic import JsonValue

from ytverbatim.diarize.protocol import Diarizer, DiarizerDef, DiarizerFactory, DiarizerSpec
from ytverbatim.errors import ConfigError, DependencyError

# The model's own default, reported as is rather than tuned on test data.
_NEMOTRON_THRESHOLD = 0.5
_NEMOTRON = "ytverbatim.diarize.nemotron.create"
_EXTRA = "asr"
_MODULES = ("mlx_audio",)

DIARIZERS: Mapping[str, DiarizerDef] = {
    definition.spec.id: definition
    for definition in (
        DiarizerDef(
            DiarizerSpec(
                id="d-nemotron-8bit",
                family="nemotron-diarization",
                repo="mlx-community/Nemotron-3-Diarization-8bit",
                revision="dd8b8ce3d69a80540a7e50150e2beed1c34fee37",
                weights_bytes=106_618_515,
                max_speakers=8,
                sample_rate=16_000,
            ),
            _NEMOTRON,
            {"threshold": _NEMOTRON_THRESHOLD},
        ),
        DiarizerDef(
            DiarizerSpec(
                id="d-nemotron-bf16",
                family="nemotron-diarization",
                repo="mlx-community/Nemotron-3-Diarization",
                revision="59ed2dbfc1346dcea9d423c71306a3a2499c568f",
                weights_bytes=198_636_662,
                max_speakers=8,
                sample_rate=16_000,
            ),
            _NEMOTRON,
            {"threshold": _NEMOTRON_THRESHOLD},
        ),
    )
}


def get(diarizer_id: str) -> DiarizerDef:
    try:
        return DIARIZERS[diarizer_id]
    except KeyError:
        raise ConfigError(f"unknown diarizer {diarizer_id!r}; known: {', '.join(DIARIZERS)}") from None


def build_diarizer(definition: DiarizerDef) -> Diarizer:
    """Import the adapter named by `definition.factory` and construct the diarizer; this is the lazy import."""
    module_name, _, attribute = definition.factory.rpartition(".")
    try:
        factory: DiarizerFactory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise DependencyError(f"cannot load the {definition.spec.id} adapter {definition.factory}: {exc}") from exc
    variant: Mapping[str, JsonValue] = definition.variant
    return factory(definition.spec, variant)


def check_dependencies(definition: DiarizerDef) -> None:
    """Raise `DependencyError` naming the extra to install when the diarizer's libraries are not importable."""
    missing = [module for module in _MODULES if find_spec(module) is None]
    if missing:
        raise DependencyError(
            f"the {definition.spec.id} diarizer needs {', '.join(missing)}, which is not installed; "
            f"install ytverbatim[{_EXTRA}] (in a checkout: uv sync --extra {_EXTRA})"
        )
