import json
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from ytverbatim.contract.models import SCHEMA_VERSION, ContextDocument
from ytverbatim.errors import ContractError

SCHEMA_PATH = Path(__file__).with_name("context.v2.schema.json")
_CURRENT_MAJOR, _CURRENT_MINOR = (int(part) for part in SCHEMA_VERSION.split("."))
# The 1.x minors that were released; each is upgraded in memory to the current version.
_UPGRADABLE_1X = frozenset({"1.0", "1.1"})
_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def export_schema() -> dict[str, Any]:
    return {"$schema": _JSON_SCHEMA_DIALECT, **ContextDocument.model_json_schema()}


def dump_document(doc: ContextDocument) -> str:
    """Field-declaration order, with free-form maps already key-sorted by the models: output is byte-stable."""
    return json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


def load_document(source: Path | str) -> ContextDocument:
    """Read a document from a file (`Path`) or from JSON text (`str`).

    A 1.0 or 1.1 document is upgraded in memory: `video` and `access` become a YouTube `source`, it lists no speakers,
    and its one language is its `transcript.languages`. Within major 2, an older minor version loads as it is and is
    reported as the current one, since a minor version only adds optional fields. A newer minor version loads with
    the fields this reader does not know dropped, which is why it is reported as the current version afterwards;
    re-saving it discards them. Any other version is refused.
    """
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError("a context document must be a JSON object")
    data = cast("dict[str, Any]", raw)
    major, minor = _check_version(data.get("schema_version"))
    if major == 1:
        data = _upgrade_1x(data)
    elif minor > _CURRENT_MINOR:
        schema = export_schema()
        data = _prune_unknown(data, schema, schema["$defs"])
    if (major, minor) != (_CURRENT_MAJOR, _CURRENT_MINOR):
        data = {**data, "schema_version": SCHEMA_VERSION}
    try:
        return ContextDocument.model_validate(data)
    except ValidationError as exc:
        raise ContractError(f"invalid context document:\n{exc}") from exc


def _check_version(version: object) -> tuple[int, int]:
    if not isinstance(version, str):
        raise ContractError(f"missing or non-string schema_version: {version!r}")
    parts = version.split(".")
    if len(parts) != 2 or not all(part.isascii() and part.isdigit() for part in parts):  # noqa: PLR2004
        raise ContractError(f"malformed schema_version {version!r}; expected '<major>.<minor>'")
    major, minor = int(parts[0]), int(parts[1])
    if major != _CURRENT_MAJOR and version not in _UPGRADABLE_1X:
        supported = ", ".join(sorted(_UPGRADABLE_1X))
        raise ContractError(
            f"unsupported context.json schema version {version!r}; this reader supports {supported} and "
            f"{_CURRENT_MAJOR}.x"
        )
    return major, minor


def _upgrade_1x(data: dict[str, Any]) -> dict[str, Any]:
    """Map a 1.x document onto the current shape; fields it does not know stay, so validation still rejects them."""
    upgraded = {key: value for key, value in data.items() if key not in ("video", "access")}
    video = data.get("video")
    if isinstance(video, dict):
        upgraded["source"] = {"kind": "youtube", **cast("dict[str, Any]", video), "access": data.get("access")}
    transcript = data.get("transcript")
    if isinstance(transcript, dict):
        old = cast("dict[str, Any]", transcript)
        language = old.get("language")
        upgraded["transcript"] = {**old, "languages": [language] if language else [], "diarization": None}
    upgraded["speakers"] = list[Any]()
    return upgraded


def _prune_unknown(value: object, node: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Drop object keys that the schema node does not declare, following $ref, discriminators, anyOf and array items."""
    if "$ref" in node:
        return _prune_unknown(value, defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    if isinstance(value, dict) and "properties" in node:
        properties: dict[str, Any] = node["properties"]
        return {key: _prune_unknown(item, properties[key], defs) for key, item in value.items() if key in properties}
    if isinstance(value, list) and "items" in node:
        return [_prune_unknown(item, node["items"], defs) for item in value]
    if "discriminator" in node and isinstance(value, dict):
        tag = cast("dict[str, Any]", value).get(node["discriminator"]["propertyName"])
        target = node["discriminator"]["mapping"].get(tag) if isinstance(tag, str) else None
        if target is not None:
            return _prune_unknown(value, defs[target.rsplit("/", 1)[-1]], defs)
    for branch in node.get("anyOf", ()):
        if _matches_shape(value, branch, defs):
            return _prune_unknown(value, branch, defs)
    return value


def _matches_shape(value: object, node: dict[str, Any], defs: dict[str, Any]) -> bool:
    if "$ref" in node:
        return _matches_shape(value, defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    declared = node.get("type")
    if declared == "object":
        return isinstance(value, dict)
    if declared == "array":
        return isinstance(value, list)
    return False
