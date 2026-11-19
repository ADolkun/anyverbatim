"""Build ``disputes.jsonl`` for the owner questions (``evals.questions``): ``uv run python -m evals.disputes build``.

Two sources, both over recordings the owner supplied:

- E1 attribution disputes: stretches of two seconds or more where finished diarization runs
  (``<recordings>/runs/diarization/<diarizer>/<corpus>/<id>/hypothesis.rttm``) name different speakers, shown against
  the transcript lines of the recording's ``words.json``;
- E2 text disputes: spots where finished ASR runs of ``evals.run_recordings_asr`` word the same speech differently,
  with a policy group where the spot is a number or a Latin-script word.

``disputes.jsonl`` holds the full disputed text and unredacted lines: it is an intermediate for ``evals.questions
build``, it stays under the recordings folder, and agents must not read it. Only ``questions.jsonl`` is for them.

Per recording, at most ``MAX_EXAMPLES`` examples of each policy group and ``--limit`` single items of each kind are
written, and the clip each one points at is cut from the local audio. Names are redacted and excerpts held to two
sentences later, by ``evals.questions build``.
"""

from __future__ import annotations

import random
import re
import wave
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import click

from anyverbatim import atomic
from anyverbatim.asr.models import ChunkHypothesis
from anyverbatim.contract.models import Word
from anyverbatim.errors import AnyverbatimError, ConfigError
from evals.consensus import find_disputes
from evals.diarization import Turn, align_labels, disagreement_runs, read_rttm
from evals.questions import MAX_EXAMPLES, AttributionDispute, AttributionLine, TextDispute
from evals.recordings_corpus import FOLDER_CORPORA, Item, asr_layout, finished_rows, load_items
from evals.recordings_data import CLIPS, OWNER, SAMPLE_RATE, WORDS_FILE, recordings_dir
from evals.run_diarization import load_words
from evals.run_recordings_asr import E2_HYPOTHESES
from evals.store import hypothesis_units

DISPUTES_FILE = "disputes.jsonl"
DEFAULT_LIMIT = 6
LINE_GAP_S = 0.8
MAX_LINE_WORDS = 30
# Anchor lines come from near the dispute, so the owner is asked about speech they can find quickly.
ANCHOR_WINDOW_S = 300.0
CLIP_PAD_S = 2.0
MAX_CLIP_S = 30.0
NUMERAL_GROUP = "numeral form"
LATIN_GROUP = "Latin spelling and case"
_BLANKS = re.compile(r"\s+")

ClipFor = Callable[[str, float, float], str]


def transcript_lines(words: Sequence[Word]) -> list[tuple[float, float, str]]:
    """Group words into lines, breaking at a pause of ``LINE_GAP_S`` or after ``MAX_LINE_WORDS`` words."""
    lines: list[tuple[float, float, str]] = []
    current: list[Word] = []

    def flush() -> None:
        text = _BLANKS.sub(" ", "".join(w.text for w in current)).strip()
        if text:
            lines.append((current[0].start, max(w.end for w in current), text))
        current.clear()

    for word in sorted(words, key=lambda w: (w.start, w.end)):
        if current and (word.start - current[-1].end >= LINE_GAP_S or len(current) >= MAX_LINE_WORDS):
            flush()
        current.append(word)
    if current:
        flush()
    return lines


def label_over(turns: Sequence[Turn], start: float, end: float) -> str | None:
    """Return the speaker with the most time inside ``[start, end]``, or None when nobody speaks there."""
    seconds: dict[str, float] = {}
    for turn in turns:
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > 0:
            seconds[turn.speaker] = seconds.get(turn.speaker, 0.0) + overlap
    return max(seconds, key=lambda s: (seconds[s], s)) if seconds else None


def attribution_disputes(
    recording: str,
    candidates: Mapping[str, Sequence[Turn]],
    words: Sequence[Word],
) -> list[AttributionDispute]:
    """One dispute per stretch where the candidates name different speakers, in the first candidate's labels."""
    mapped = align_labels(candidates)
    lines = transcript_lines(words)
    disputes: list[AttributionDispute] = []
    for index, run in enumerate(disagreement_runs(candidates)):
        middle = (run.start + run.end) / 2
        nearby = [line for line in lines if abs((line[0] + line[1]) / 2 - middle) <= ANCHOR_WINDOW_S]
        disputes.append(
            AttributionDispute(
                id=f"{recording}-a{index:04d}",
                recording=recording,
                start=run.start,
                end=run.end,
                speakers={name: "+".join(sorted(labels)) for name, labels in run.speakers.items() if labels},
                lines=tuple(
                    AttributionLine(
                        start=start,
                        end=end,
                        text=text,
                        speakers={
                            name: label for name, turns in mapped.items() if (label := label_over(turns, start, end))
                        },
                    )
                    for start, end, text in nearby
                ),
            )
        )
    return disputes


def text_disputes(
    item: Item,
    rows: Mapping[str, Sequence[ChunkHypothesis]],
    pivot: str,
) -> list[TextDispute]:
    """One dispute per spot where the hypotheses word the recording differently."""
    units = {name: hypothesis_units(rows[name], item.language, "raw") for name in rows}
    found = find_disputes(units, item.language, pivot=pivot)
    return [
        TextDispute(
            id=f"{item.id}-t{d.index:04d}",
            recording=item.id,
            start=d.start_s,
            end=d.end_s,
            before=d.left_context,
            after=d.right_context,
            candidates={r.hypothesis: r.text for r in d.readings},
            group=NUMERAL_GROUP if d.touches_quantity else LATIN_GROUP if d.touches_latin else None,
        )
        for d in found
    ]


def select_text(disputes: Sequence[TextDispute], limit: int, seed: int) -> list[TextDispute]:
    """Up to ``MAX_EXAMPLES`` of each policy group, plus a seeded random sample of ``limit`` ungrouped disputes."""
    rng = random.Random(seed)
    chosen: list[TextDispute] = []
    for group in sorted({d.group for d in disputes if d.group}):
        members = [d for d in disputes if d.group == group]
        chosen += rng.sample(members, min(MAX_EXAMPLES, len(members)))
    singles = [d for d in disputes if d.group is None]
    chosen += rng.sample(singles, min(len(singles), limit))
    return sorted(chosen, key=lambda d: (d.start, d.id))


def select_attribution(disputes: Sequence[AttributionDispute], limit: int, seed: int) -> list[AttributionDispute]:
    """Sample ``limit`` disputes at random (seeded).

    The sample carries neither the number of disputes it came from nor their durations, so a time-weighted win rate
    has to weight each item by its own duration.
    """
    chosen = random.Random(seed).sample(list(disputes), min(limit, len(disputes)))
    return sorted(chosen, key=lambda d: (d.start, d.id))


def cut_clip(wav: Path, start: float, end: float, destination: Path) -> None:
    """Write ``[start - pad, end + pad]`` of a 16 kHz mono WAV to ``destination``, at most ``MAX_CLIP_S`` long."""
    first = max(round((start - CLIP_PAD_S) * SAMPLE_RATE), 0)
    last = first + min(round((end - start + 2 * CLIP_PAD_S) * SAMPLE_RATE), round(MAX_CLIP_S * SAMPLE_RATE))
    with wave.open(str(wav), "rb") as source:
        source.setpos(min(first, source.getnframes()))
        frames = source.readframes(last - first)
        params = source.getparams()
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    with wave.open(str(part), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)
    part.replace(destination)


def clip_maker(item: Item, directory: Path) -> ClipFor:
    def clip_for(kind: str, start: float, end: float) -> str:
        destination = directory / item.corpus / item.id / f"{kind}-{round(start * 1000)}-{round(end * 1000)}.wav"
        if not destination.is_file():
            cut_clip(item.wav, start, end, destination)
        return str(destination)

    return clip_for


def _hypothesis_turns(diarizer: str, item: Item, root: Path) -> list[Turn]:
    path = root / "runs" / "diarization" / diarizer / item.corpus / item.id / "hypothesis.rttm"
    if not path.is_file():
        raise ConfigError(f"{diarizer} has not diarized {item.corpus}/{item.id}; run evals.run_diarization first")
    return read_rttm(path).get(item.id, [])


def _words(item: Item) -> list[Word]:
    words = load_words(item.wav.parent / WORDS_FILE)
    if words is None:
        raise ConfigError(f"no {WORDS_FILE} for {item.corpus}/{item.id}; run `evals.run_recordings_asr words` first")
    return words


def build_disputes(
    items: Sequence[Item],
    *,
    asr_hypotheses: Sequence[str],
    pivot: str | None,
    diarizers: Sequence[str],
    limit: int,
    seed: int,
    root: Path | None = None,
    clips: Path | None = None,
) -> list[TextDispute | AttributionDispute]:
    base = root or recordings_dir()
    out: list[TextDispute | AttributionDispute] = []
    for item in items:
        clip_for = clip_maker(item, clips or base / CLIPS)
        if asr_hypotheses:
            layout = asr_layout(item.corpus, base)
            rows = {name: finished_rows(layout, name, [item]) for name in asr_hypotheses}
            for name, finished in rows.items():
                if item.id not in finished:
                    raise ConfigError(f"{name} is not finished for {item.corpus}/{item.id}; run it first")
            texts = {name: finished[item.id] for name, finished in rows.items()}
            found = text_disputes(item, texts, pivot or asr_hypotheses[0])
            out += [
                d.model_copy(update={"clip": clip_for("text", d.start, d.end)}) for d in select_text(found, limit, seed)
            ]
        if diarizers:
            turns = {name: _hypothesis_turns(name, item, base) for name in diarizers}
            found_attribution = attribution_disputes(item.id, turns, _words(item))
            out += [
                d.model_copy(update={"clip": clip_for("attribution", d.start, d.end)})
                for d in select_attribution(found_attribution, limit, seed)
            ]
    return out


def write_disputes(path: Path, disputes: Sequence[TextDispute | AttributionDispute]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic.write_text(path, "".join(d.model_dump_json() + "\n" for d in disputes))


@click.group()
def main() -> None:
    """Disputes for the owner questions (data under $ANYVERBATIM_EVAL_DIR/recordings)."""


@main.command()
@click.option("--corpus", type=click.Choice(FOLDER_CORPORA), default=OWNER, show_default=True)
@click.option("--recordings", help="Comma-separated recording ids (default: every one in the corpus).")
@click.option("--asr-hyps", help="E2: comma-separated finished ASR hypotheses to compare (text disputes).")
@click.option("--pivot", default=None, help="Hypothesis the others align to (default: the first of --asr-hyps).")
@click.option("--diarizers", help="E1: comma-separated finished diarizers to compare (attribution disputes).")
@click.option("--limit", default=DEFAULT_LIMIT, show_default=True, help="Single items kept per recording and kind.")
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the full-text intermediate (default and only allowed place: the recordings folder).",
)
def build(
    corpus: str,
    recordings: str | None,
    asr_hyps: str | None,
    pivot: str | None,
    diarizers: str | None,
    limit: int,
    seed: int,
    out_path: Path | None,
) -> None:
    """Write disputes.jsonl, a full-text intermediate that agents must not read; prints counts only.

    Feed it to ``evals.questions build``; only that command's ``questions.jsonl`` is for agents.
    """
    hypotheses = [h for h in (asr_hyps or "").split(",") if h]
    named = [d for d in (diarizers or "").split(",") if d]
    if not hypotheses and not named:
        raise click.UsageError("pass --asr-hyps, --diarizers or both")
    if len(hypotheses) == 1 or len(named) == 1:
        raise click.UsageError("a dispute needs at least two hypotheses or two diarizers")
    if pivot is not None and pivot not in hypotheses:
        raise click.UsageError("--pivot must be one of --asr-hyps")
    unknown = set(hypotheses) - set(E2_HYPOTHESES)
    if unknown:
        raise click.UsageError(f"unknown ASR hypotheses: {', '.join(sorted(unknown))}")
    path = out_path or recordings_dir() / DISPUTES_FILE
    if not path.resolve().is_relative_to(recordings_dir().resolve()):
        raise click.UsageError("--out must be inside the recordings folder: the file holds unredacted text")
    items = load_items(corpus, [r for r in (recordings or "").split(",") if r])
    disputes = build_disputes(items, asr_hypotheses=hypotheses, pivot=pivot, diarizers=named, limit=limit, seed=seed)
    write_disputes(path, disputes)
    text = sum(isinstance(d, TextDispute) for d in disputes)
    click.echo(f"{text} text and {len(disputes) - text} attribution disputes -> {path}")


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
