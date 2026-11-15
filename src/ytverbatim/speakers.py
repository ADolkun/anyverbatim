"""Speaker names: `speakers.json` beside `context.json`, bound to the attribution the names were given for.

Ids (`S1`, `S2`, ...) come from the diarization and are never renumbered; a name is set afterwards by `ytverbatim
speakers`, and two ids with one name are one person. The file stores the digest of the speaker-attributed stretches
(`Transcribed.assignment`), which a text-stage rerun leaves alone and any change to diarization, attribution or ASR
moves: names for another attribution are then dropped with a quality warning, never applied to the wrong person.
Names stay out of every LLM request: the text stages never see them, only `build_document` does.
"""

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from ytverbatim import atomic
from ytverbatim.contract import ContextDocument
from ytverbatim.errors import ContractError

SPEAKERS_JSON = "speakers.json"
STALE_WARNING = (
    "the speaker names saved for this recording belong to an earlier speaker attribution and were not applied; "
    "set them again with `ytverbatim speakers`"
)

_ID = re.compile(r"S[1-9][0-9]*")
_FIRST_LINE_CHARS = 60
# Control characters and line or paragraph separators would break the Markdown and the listing a name appears in.
_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})


class SpeakerNames(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment: str
    names: dict[str, str]

    @field_validator("names")
    @classmethod
    def _names(cls, names: dict[str, str]) -> dict[str, str]:
        for speaker, name in names.items():
            if not _ID.fullmatch(speaker):
                raise ValueError(f"{speaker!r} is not a speaker id")
            if not name.strip():
                raise ValueError(f"{speaker} has a blank name")
            _check_characters(speaker, name)
        return names


def _check_characters(speaker: str, name: str) -> None:
    for char in name:
        if unicodedata.category(char) in _FORBIDDEN_CATEGORIES:
            raise ValueError(f"the name of {speaker} contains the control character U+{ord(char):04X}")


@dataclass(frozen=True)
class Bound:
    """The saved names that apply to an attribution; `stale` when names exist for another one."""

    names: dict[str, str]
    stale: bool


@dataclass(frozen=True)
class SpeakerLine:
    id: str
    name: str | None
    speech_s: float
    start: float
    first_line: str


def path(directory: Path) -> Path:
    return directory / SPEAKERS_JSON


def digest(directory: Path) -> str | None:
    """Return the digest of the names file as it is on disk, or None without one.

    The context record keeps the digest the document was written with; a run that finds another one rebuilds the
    document from its stored stages, so a names file edited or removed by hand still takes effect.
    """
    try:
        return hashlib.sha256(path(directory).read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def read(directory: Path) -> SpeakerNames | None:
    try:
        text = path(directory).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return SpeakerNames.model_validate(json.loads(text))
    except ValueError as exc:
        error = ContractError(f"{directory.name}/{SPEAKERS_JSON} is not a valid speaker names file: {exc}")
        error.fix = f"fix or delete {SPEAKERS_JSON}, then set the names again with `ytverbatim speakers`"
        raise error from exc


def bound(directory: Path, assignment: str | None) -> Bound:
    """Return the saved names when they were given for `assignment`; a source with no attribution ignores them."""
    saved = read(directory)
    if saved is None or assignment is None:
        return Bound({}, stale=False)
    if saved.assignment != assignment:
        return Bound({}, stale=True)
    return Bound(dict(saved.names), stale=False)


def write(directory: Path, assignment: str, names: Mapping[str, str]) -> None:
    """Store `names` for `assignment`, atomically; no names at all removes the file, as if none were ever given."""
    if not names:
        path(directory).unlink(missing_ok=True)
        return
    ordered = dict(sorted(names.items(), key=lambda item: int(item[0][1:])))
    atomic.write_json(path(directory), {"assignment": assignment, "names": ordered})


def parse(assignments: Iterable[str]) -> dict[str, str | None]:
    """Parse `S1=Alice` arguments; `S2=` (an empty name) clears the name. Ids are matched without regard to case."""
    changes: dict[str, str | None] = {}
    for item in assignments:
        speaker, separator, name = item.partition("=")
        speaker = speaker.strip().upper()
        if not separator or not _ID.fullmatch(speaker):
            raise ContractError(f"{item!r} is not ID=NAME; for example S1=Alice, or S1= to clear a name")
        if speaker in changes:
            raise ContractError(f"{speaker} is given twice; give each speaker one name")
        try:
            _check_characters(speaker, name)
        except ValueError as exc:
            raise ContractError(f"{exc}; a name is one line of plain text") from exc
        changes[speaker] = name.strip() or None
    return changes


def changed(document: ContextDocument, current: Mapping[str, str], changes: Mapping[str, str | None]) -> dict[str, str]:
    """Return `current` with `changes` applied, refusing an unknown id or a name that is another id's default label."""
    listed = {speaker.id: speaker for speaker in document.speakers}
    unknown = [speaker for speaker in changes if speaker not in listed]
    if unknown:
        raise ContractError(f"no speaker {', '.join(unknown)}; this recording has {', '.join(listed)}")
    names = {speaker: name for speaker, name in current.items() if speaker in listed}
    for speaker, name in changes.items():
        if name is None:
            names.pop(speaker, None)
        else:
            names[speaker] = name
    for speaker, name in names.items():
        owner = document.default_label_owner(name)
        if owner is not None and owner != speaker:
            raise ContractError(f"{name!r} is the default label of {owner}, so it cannot name {speaker}")
    return names


def named(document: ContextDocument, names: Mapping[str, str]) -> ContextDocument:
    """Return `document` with its speakers named from `names`, and without the warning that names were dropped."""
    data = document.model_dump(mode="json")
    for item in data["speakers"]:
        item["name"] = names.get(item["id"])
    quality = data["quality"]
    quality["warnings"] = [warning for warning in quality["warnings"] if warning != STALE_WARNING]
    try:
        return ContextDocument.model_validate(data)
    except ValidationError as exc:
        raise ContractError(f"the names do not fit the contract:\n{exc}") from exc


def lines(document: ContextDocument) -> list[SpeakerLine]:
    """Return each speaker's id, name, talk time and first sentence (shortened), in id order."""
    first = {}
    for segment in document.segments:
        if segment.speaker is not None and segment.speaker not in first:
            first[segment.speaker] = segment
    return [
        SpeakerLine(item.id, item.name, item.speech_s, first[item.id].start, _short(first[item.id].text))
        for item in document.speakers
    ]


def _short(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _FIRST_LINE_CHARS else flat[: _FIRST_LINE_CHARS - 1].rstrip() + "…"
