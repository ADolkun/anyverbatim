"""Disputes to the owner's grouped multiple-choice questions, within the privacy limit on what an agent may read.

``uv run python -m evals.questions build DISPUTES.jsonl --out DIR (--names-file FILE | --no-names)``. Claude agents read
``questions.jsonl`` only, so each item carries at most two sentences of context, generic labels and no names. The
option-to-hypothesis key goes to a separate ``questions.key.jsonl`` that the orchestrator does not open.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import click
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from anyverbatim import atomic
from anyverbatim.errors import AnyverbatimError, ConfigError

MAX_SENTENCES = 2
MAX_ANCHOR_CHARS = 80
MAX_OPTION_CHARS = 120
MIN_ANCHOR_CHARS = 12
MAX_EXAMPLES = 3
MAX_ATTRIBUTION_OPTIONS = 2
SPOT = "⟪?⟫"
OTHER = "someone else"
UNCLEAR = "can't tell"
NO_PREFERENCE = "none of these"

_CLOSERS = re.escape("\"'”’)）]」』")
_SENTENCE_END = re.compile(rf"[.!?]+[{_CLOSERS}]*(?=\s|$)|[。！？…]+[{_CLOSERS}]*")
_WHITESPACE = re.compile(r"\s+")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TextDispute(_Frozen):
    """Candidates differ at one spot; ``before`` and ``after`` are the transcript text flanking it."""

    type: Literal["text"] = "text"
    id: str
    recording: str
    start: float
    end: float
    before: str
    after: str
    # Hypothesis id to the text it produced at the spot.
    candidates: Mapping[str, str]
    group: str | None = None
    clip: str | None = None


class AttributionLine(_Frozen):
    """A transcript line with the speaker each candidate gave it, in one shared label space."""

    start: float
    end: float
    text: str
    speakers: Mapping[str, str]


class AttributionDispute(_Frozen):
    """Candidates name different speakers for a stretch of at least two seconds."""

    type: Literal["attribution"] = "attribution"
    id: str
    recording: str
    start: float
    end: float
    # Candidate id to the speaker it named over the stretch, in the same label space as ``lines``.
    speakers: Mapping[str, str]
    lines: tuple[AttributionLine, ...]
    clip: str | None = None


Dispute = Annotated[TextDispute | AttributionDispute, Field(discriminator="type")]


class Example(_Frozen):
    # The text with the spot marked; empty for an attribution item, whose options carry the anchor lines.
    excerpt: str
    start: float
    end: float
    clip: str | None
    options: tuple[str, ...]


class Question(_Frozen):
    id: str
    kind: Literal["policy", "text", "attribution"]
    group: str | None
    recording: str
    prompt: str
    examples: tuple[Example, ...]


class KeyEntry(_Frozen):
    question: str
    sources: tuple[str, ...]
    # Per example and option, the hypotheses (text) or speaker labels (attribution) the option stands for; empty for
    # the trailing "can't tell" style options.
    options: tuple[tuple[tuple[str, ...], ...], ...]


class Built(_Frozen):
    questions: tuple[Question, ...]
    key: tuple[KeyEntry, ...]
    skipped: tuple[tuple[str, str], ...]


def split_sentences(text: str) -> list[str]:
    pieces: list[str] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        pieces.append(text[cursor : match.end()])
        cursor = match.end()
    pieces.append(text[cursor:])
    return [piece.strip() for piece in pieces if piece.strip()]


def sentence_count(text: str) -> int:
    return len(split_sentences(text.replace(SPOT, " ")))


def limit_sentences(text: str, limit: int = MAX_SENTENCES) -> str:
    return " ".join(split_sentences(text)[:limit])


def _boundary_before(text: str) -> int:
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    return ends[-1] if ends else 0


def _boundary_after(text: str) -> int:
    match = _SENTENCE_END.search(text)
    return match.end() if match else len(text)


def spot_excerpt(before: str, after: str, limit: int = MAX_SENTENCES) -> str:
    """Show the disputed sentence with ``⟪?⟫`` at the spot, plus one neighbour while ``limit`` allows.

    The neighbour is the sentence before when there is one, otherwise the sentence after.
    """
    cut = _boundary_before(before)
    head, lead = before[:cut], before[cut:]
    stop = _boundary_after(after)
    trail, rest = after[:stop], after[stop:]
    current = f"{_WHITESPACE.sub(' ', lead).lstrip()}{SPOT}{_WHITESPACE.sub(' ', trail).rstrip()}"
    previous, following = split_sentences(head), split_sentences(rest)
    pieces = [current]
    if limit > 1:
        if previous:
            pieces.insert(0, previous[-1])
        elif following:
            pieces.append(following[0])
    return " ".join(pieces)


def redact(text: str, names: Iterable[str]) -> str:
    """Replace each participant name with ``[name]``; Latin names match whole words, any case."""
    for name in sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True):
        if name.isascii():
            text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "[name]", text, flags=re.IGNORECASE)
        else:
            text = text.replace(name, "[name]")
    return text


def clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def anchor_text(text: str, names: Iterable[str] = ()) -> str:
    """First sentence of a line, redacted before it is cut so a name is never left half showing."""
    sentence = (split_sentences(redact(text, names)) or [""])[0]
    return _shorten(sentence, MAX_ANCHOR_CHARS)


def option_text(text: str, names: Iterable[str] = ()) -> str:
    """Redact a candidate's text, then hold it to the same sentence and character limits as the context."""
    return _shorten(limit_sentences(redact(text, names)), MAX_OPTION_CHARS)


def pick_anchors(
    lines: Sequence[AttributionLine],
    labels: Iterable[str],
    start: float,
    end: float,
    candidates: Sequence[str],
    names: Iterable[str] = (),
) -> dict[str, AttributionLine]:
    """For each speaker label, the line every candidate attributes to that speaker that is nearest the dispute.

    A line qualifies only if all ``candidates`` agree on its speaker, it lies outside the disputed stretch, and its
    first sentence is long enough to recognise. Two speakers never share an anchor sentence.
    """
    middle = (start + end) / 2
    chosen: dict[str, AttributionLine] = {}
    used: set[str] = set()
    for label in labels:
        options = [
            line
            for line in lines
            if (line.end <= start or line.start >= end)
            and all(line.speakers.get(c) == label for c in candidates)
            and len(anchor_text(line.text, names).rstrip("…")) >= MIN_ANCHOR_CHARS
        ]
        for line in sorted(options, key=lambda line: abs((line.start + line.end) / 2 - middle)):
            text = anchor_text(line.text, names)
            if text not in used:
                chosen[label] = line
                used.add(text)
                break
    return chosen


_Item = tuple[Example, tuple[tuple[str, ...], ...]]


def _attribution(item: AttributionDispute, names: Sequence[str]) -> _Item | str:
    candidates = list(item.speakers)
    labels = [label for label, _ in Counter(item.speakers.values()).most_common()]
    anchors = pick_anchors(item.lines, labels, item.start, item.end, candidates, names)
    usable = [label for label in labels if label in anchors][:MAX_ATTRIBUTION_OPTIONS]
    if len(usable) < MAX_ATTRIBUTION_OPTIONS:
        return "fewer than two speakers have an anchor line every candidate agrees on"
    options = tuple(
        f"the person who says '{anchor_text(anchors[label].text, names)}' at {clock(anchors[label].start)}"
        for label in usable
    )
    example = Example(excerpt="", start=item.start, end=item.end, clip=item.clip, options=(*options, OTHER, UNCLEAR))
    return example, (*((label,) for label in usable), (), ())


def _text(item: TextDispute, names: Sequence[str], rng: random.Random) -> _Item:
    by_text: dict[str, list[str]] = defaultdict(list)
    for hypothesis, text in item.candidates.items():
        by_text[option_text(text, names)].append(hypothesis)
    texts = sorted(by_text)
    rng.shuffle(texts)
    example = Example(
        excerpt=redact(spot_excerpt(item.before, item.after), names),
        start=item.start,
        end=item.end,
        clip=item.clip,
        options=(*(text or "(nothing)" for text in texts), NO_PREFERENCE, UNCLEAR),
    )
    return example, (*(tuple(sorted(by_text[t])) for t in texts), (), ())


def build_questions(
    disputes: Sequence[TextDispute | AttributionDispute], names: Sequence[str] = (), seed: int = 0
) -> Built:
    """Turn disputes into questions: one per policy group first, then single items in recording and time order."""
    rng = random.Random(seed)
    singles: list[tuple[str, str, str, _Item]] = []
    grouped: dict[str, list[tuple[str, str, _Item]]] = defaultdict(list)
    skipped: list[tuple[str, str]] = []
    for item in sorted(disputes, key=lambda d: (d.recording, d.start, d.id)):
        built = _attribution(item, names) if isinstance(item, AttributionDispute) else _text(item, names, rng)
        if isinstance(built, str):
            skipped.append((item.id, built))
        elif isinstance(item, TextDispute) and item.group:
            grouped[item.group].append((item.id, item.recording, built))
        else:
            kind = "attribution" if isinstance(item, AttributionDispute) else "text"
            singles.append((kind, item.recording, item.id, built))
    questions: list[Question] = []
    key: list[KeyEntry] = []

    def add(
        kind: Literal["policy", "text", "attribution"],
        group: str | None,
        recording: str,
        prompt: str,
        parts: Sequence[tuple[str, _Item]],
    ) -> None:
        question_id = f"q-{len(questions) + 1:03d}"
        questions.append(
            Question(
                id=question_id,
                kind=kind,
                group=group,
                recording=recording,
                prompt=prompt,
                examples=tuple(example for _, (example, _) in parts),
            )
        )
        key.append(
            KeyEntry(
                question=question_id,
                sources=tuple(source for source, _ in parts),
                options=tuple(options for _, (_, options) in parts),
            )
        )

    for group, parts in sorted(grouped.items()):
        kept = parts[:MAX_EXAMPLES]
        recordings = {recording for _, recording, _ in kept}
        add(
            "policy",
            group,
            recordings.pop() if len(recordings) == 1 else "multiple",
            f"Policy '{group}': for each example, which form is right? Your answers set the rule.",
            [(source, item) for source, _, item in kept],
        )
    for kind, recording, source, item in singles:
        if kind == "attribution":
            prompt = f"Who is speaking between {clock(item[0].start)} and {clock(item[0].end)}?"
            add("attribution", None, recording, prompt, [(source, item)])
        else:
            add("text", None, recording, "Which is what was said at the marked spot?", [(source, item)])
    for question in questions:
        if any(sentence_count(example.excerpt) > MAX_SENTENCES for example in question.examples):
            raise ConfigError(f"{question.id}: an excerpt exceeds {MAX_SENTENCES} sentences")
    return Built(questions=tuple(questions), key=tuple(key), skipped=tuple(skipped))


def load_disputes(path: Path) -> list[TextDispute | AttributionDispute]:
    adapter: TypeAdapter[TextDispute | AttributionDispute] = TypeAdapter(Dispute)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [adapter.validate_json(line) for line in lines]


def write_questions(built: Built, directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    questions = directory / "questions.jsonl"
    key = directory / "questions.key.jsonl"
    atomic.write_text(questions, "".join(q.model_dump_json() + "\n" for q in built.questions))
    atomic.write_text(key, "".join(k.model_dump_json() + "\n" for k in built.key))
    return questions, key


def _names(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@click.group()
def main() -> None:
    """Owner questions from disputes (privacy-limited excerpts)."""


@main.command()
@click.argument("disputes", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--names-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Participant names to redact, one per line.",
)
@click.option(
    "--no-names", is_flag=True, help="Assert that there are no names to redact (the alternative to --names-file)."
)
@click.option("--seed", default=0, show_default=True)
def build(disputes: Path, out_dir: Path, names_file: Path | None, *, no_names: bool, seed: int) -> None:
    """Write questions.jsonl (agent-readable) and questions.key.jsonl (private) under OUT."""
    if (names_file is not None) == no_names:
        raise click.UsageError("pass exactly one of --names-file and --no-names, so an unredacted run is deliberate")
    built = build_questions(load_disputes(disputes), _names(names_file), seed)
    questions, key = write_questions(built, out_dir)
    click.echo(f"{len(built.questions)} questions -> {questions}; key -> {key}")
    for item, reason in built.skipped:
        click.echo(f"skipped {item}: {reason}")


def run() -> None:
    try:
        main(standalone_mode=False)
    except AnyverbatimError as exc:
        raise SystemExit(f"error: {exc}") from None
    except click.exceptions.Abort:
        raise SystemExit(130) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    run()
