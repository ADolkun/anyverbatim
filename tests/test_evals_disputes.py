import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from recordings_fixture import FakeWorker, make_meeting, make_owner, pieces, silent, spoken, wav_seconds_of

from evals import disputes
from evals.diarization import Turn, write_rttm
from evals.disputes import (
    LATIN_GROUP,
    NUMERAL_GROUP,
    attribution_disputes,
    build_disputes,
    cut_clip,
    label_over,
    select_attribution,
    select_text,
    text_disputes,
    transcript_lines,
    write_disputes,
)
from evals.manifest import EVAL_DIR_ENV
from evals.questions import (
    MAX_ANCHOR_CHARS,
    MAX_EXAMPLES,
    MAX_OPTION_CHARS,
    MAX_SENTENCES,
    TextDispute,
    build_questions,
    load_disputes,
    sentence_count,
)
from evals.recordings_corpus import Item, asr_layout, load_items
from evals.recordings_data import OWNER
from evals.run_recordings_asr import run_corpus
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError


def word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def spoken_line(text: str, start: float) -> list[Word]:
    return [word(f" {w}", start + i * 0.4, start + (i + 1) * 0.4) for i, w in enumerate(text.split())]


WORDS = [
    *spoken_line("the quarterly numbers look quite strong overall", 0),
    *spoken_line("I would like to review the budget plan first", 10),
    *spoken_line("our hiring plan needs a closer look next week", 16),
]
A = [Turn(0, 10, "s1"), Turn(10, 20, "s2")]
B = [Turn(0, 10, "u"), Turn(10, 14, "v"), Turn(14, 20, "u")]


def test_lines_break_at_pauses_and_at_the_word_cap():
    lines = transcript_lines(WORDS)
    assert [round(start) for start, _, _ in lines] == [0, 10, 16]
    assert lines[0][2] == "the quarterly numbers look quite strong overall"
    many = [word(f" w{i}", i * 0.1, i * 0.1 + 0.1) for i in range(65)]
    assert [len(text.split()) for _, _, text in transcript_lines(many)] == [30, 30, 5]
    assert transcript_lines([]) == []


def test_a_span_is_labelled_by_the_speaker_with_most_time_in_it():
    assert label_over(A, 8, 11) == "s1"
    assert label_over(A, 9, 15) == "s2"
    assert label_over(A, 30, 31) is None


def test_attribution_disputes_name_each_candidates_speaker_in_one_label_space():
    (dispute,) = attribution_disputes("rec-01", {"nemotron": A, "other": B}, WORDS)
    assert (dispute.id, dispute.recording) == ("rec-01-a0000", "rec-01")
    assert (dispute.start, dispute.end) == (14, 20)
    assert dispute.speakers == {"nemotron": "s2", "other": "s1"}
    labelled = {round(line.start): dict(line.speakers) for line in dispute.lines}
    assert labelled[0] == {"nemotron": "s1", "other": "s1"}
    assert labelled[10] == {"nemotron": "s2", "other": "s2"}
    assert labelled[16] == {"nemotron": "s2", "other": "s1"}


def test_attribution_lines_far_from_the_dispute_are_left_out():
    far = [*WORDS, *spoken_line("a line that comes much much later", 900)]
    (dispute,) = attribution_disputes("rec-01", {"a": A, "b": B}, far)
    assert [round(line.start) for line in dispute.lines] == [0, 10, 16]


def test_attribution_disputes_become_anchored_questions():
    disputes_ = attribution_disputes("rec-01", {"a": A, "b": B}, WORDS)
    built = build_questions(disputes_)
    (question,) = built.questions
    assert question.kind == "attribution"
    assert question.examples[0].options[0].startswith("the person who says '")
    assert not built.skipped


def rows(hypothesis: str, text: str):
    return [spoken(hypothesis, "m1", pieces(text, 1.0), chunk=0)]


def test_text_disputes_carry_every_readings_wording_and_a_policy_group_for_numbers():
    base = "we closed the quarter with twenty five new customers in the region"
    rows_by = {
        "w-runs-lid": rows("w-runs-lid", base),
        "w-vad-lid": rows("w-vad-lid", base.replace("twenty five", "twenty four")),
        "w-prod": rows("w-prod", base),
    }
    item = Item("owner", "rec-01", Path("a.wav"), 60.0, "en", None)
    (dispute,) = text_disputes(item, rows_by, "w-runs-lid")
    assert dispute.group == NUMERAL_GROUP
    assert dispute.recording == "rec-01" and dispute.id == "rec-01-t0000"
    assert dispute.candidates["w-vad-lid"] == "four"
    assert dispute.candidates["w-runs-lid"] == "five"
    assert "closed the quarter" in dispute.before and "customers" in dispute.after


def test_a_latin_word_in_chinese_speech_is_a_latin_group_dispute():
    item = Item("owner", "rec-01", Path("a.wav"), 60.0, "zh", None)
    rows_by = {
        "a": [spoken("a", "rec-01", pieces("我们今天开 Zoom 会议", 0.0, 0.3))],
        "b": [spoken("b", "rec-01", pieces("我们今天开 zoom 会议", 0.0, 0.3))],
    }
    (dispute,) = text_disputes(item, rows_by, "a")
    assert dispute.group == LATIN_GROUP
    assert set(dispute.candidates.values()) == {"Zoom", "zoom"}


def many_text(n: int, group: str | None) -> list[TextDispute]:
    return [
        TextDispute(
            id=f"r-t{group or 's'}{i:03d}",
            recording="rec-01",
            start=float(i),
            end=i + 1.0,
            before="a b",
            after="c d",
            candidates={"x": "p", "y": "q"},
            group=group,
        )
        for i in range(n)
    ]


def test_selection_keeps_three_examples_per_group_and_a_seeded_sample_of_singles():
    pool = [*many_text(10, NUMERAL_GROUP), *many_text(10, LATIN_GROUP), *many_text(30, None)]
    chosen = select_text(pool, 4, seed=1)
    assert sum(d.group == NUMERAL_GROUP for d in chosen) == MAX_EXAMPLES
    assert sum(d.group == LATIN_GROUP for d in chosen) == MAX_EXAMPLES
    assert sum(d.group is None for d in chosen) == 4
    assert chosen == select_text(pool, 4, seed=1)
    assert chosen != select_text(pool, 4, seed=2)
    assert select_text(many_text(2, None), 6, seed=0) == many_text(2, None)


def test_attribution_selection_is_a_seeded_sample():
    runs = attribution_disputes("rec-01", {"a": A, "b": B}, WORDS)
    assert select_attribution(runs, 5, 0) == runs
    assert select_attribution(runs, 0, 0) == []


def test_built_questions_stay_inside_the_excerpt_limits():
    long = " ".join(f"word{i}" for i in range(80))
    items = [
        TextDispute(
            id=f"r-{i}",
            recording="rec-01",
            start=float(i),
            end=i + 1.0,
            before=long + ". " + long,
            after=long + ". " + long,
            candidates={"x": long + " one", "y": long + " two"},
            group=NUMERAL_GROUP if i < 5 else None,
        )
        for i in range(8)
    ]
    built = build_questions(items)
    assert built.questions
    for question in built.questions:
        assert len(question.examples) <= MAX_EXAMPLES
        for example in question.examples:
            assert sentence_count(example.excerpt) <= MAX_SENTENCES
            assert all(len(o) <= MAX_OPTION_CHARS + 1 for o in example.options)
    attribution = build_questions(attribution_disputes("rec-01", {"a": A, "b": B}, WORDS)).questions[0]
    assert all(len(o) <= MAX_ANCHOR_CHARS + 80 for o in attribution.examples[0].options)


def test_a_clip_holds_the_dispute_with_two_seconds_of_padding_and_is_capped(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(silent(100))
    cut_clip(wav, 20, 22, tmp_path / "clips" / "c.wav")
    assert wav_seconds_of(tmp_path / "clips" / "c.wav") == pytest.approx(6.0)
    cut_clip(wav, 20, 80, tmp_path / "clips" / "long.wav")
    assert wav_seconds_of(tmp_path / "clips" / "long.wav") == pytest.approx(30.0)


def test_disputes_are_built_from_finished_runs_and_loaded_back_within_the_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(EVAL_DIR_ENV, str(tmp_path / "eval"))
    root = tmp_path / "eval" / "recordings"
    make_owner(root, [("rec-01", 60, {"language": "en"})])
    (item,) = load_items(OWNER, root=root)
    base = "we closed the quarter with twenty five new customers in the region"
    texts = {"w-runs-lid": base, "w-vad-lid": base.replace("twenty five", "twenty four"), "w-prod": base}

    def rows_for(request):
        name = request.engine.spec.id
        return [spoken(name, "rec-01", pieces(texts[name], 1.0))]

    for name in texts:
        run_corpus(asr_layout(OWNER, root), name, [item], worker=FakeWorker(rows_for))
    for name, turns in (("d-a", A), ("d-b", B)):
        folder = root / "runs" / "diarization" / name / OWNER / "rec-01"
        folder.mkdir(parents=True)
        write_rttm(folder / "hypothesis.rttm", "rec-01", turns)
    (root / OWNER / "words.json").write_text(
        json.dumps([{"text": w.text, "start": w.start, "end": w.end} for w in WORDS]), encoding="utf-8"
    )
    found = build_disputes(
        [item],
        asr_hypotheses=list(texts),
        pivot="w-runs-lid",
        diarizers=["d-a", "d-b"],
        limit=3,
        seed=0,
        root=root,
    )
    assert sorted(type(d).__name__ for d in found) == ["AttributionDispute", "TextDispute"]
    assert all(d.clip and Path(d.clip).is_file() for d in found)
    path = root / "disputes.jsonl"
    write_disputes(path, found)
    assert load_disputes(path) == found
    built = build_questions(found)
    assert {q.kind for q in built.questions} == {"policy", "attribution"}


def test_building_needs_finished_runs_words_and_diarizations(tmp_path: Path):
    make_meeting(tmp_path, "ami", "m1", [])
    (item,) = load_items("ami", root=tmp_path)
    with pytest.raises(ConfigError, match="not finished"):
        build_disputes(
            [item], asr_hypotheses=["w-prod", "w-vad"], pivot=None, diarizers=[], limit=3, seed=0, root=tmp_path
        )
    with pytest.raises(ConfigError, match="has not diarized"):
        build_disputes([item], asr_hypotheses=[], pivot=None, diarizers=["d-a", "d-b"], limit=3, seed=0, root=tmp_path)
    folder = tmp_path / "runs" / "diarization"
    for name in ("d-a", "d-b"):
        (folder / name / "ami" / "m1").mkdir(parents=True)
        write_rttm(folder / name / "ami" / "m1" / "hypothesis.rttm", "m1", A)
    with pytest.raises(ConfigError, match=r"words\.json"):
        build_disputes([item], asr_hypotheses=[], pivot=None, diarizers=["d-a", "d-b"], limit=3, seed=0, root=tmp_path)


def test_the_command_refuses_a_single_hypothesis_and_an_empty_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(EVAL_DIR_ENV, str(tmp_path / "eval"))
    runner = CliRunner()
    assert runner.invoke(disputes.main, ["build"]).exit_code != 0
    single = runner.invoke(disputes.main, ["build", "--asr-hyps", "w-prod"])
    assert single.exit_code != 0 and "at least two" in single.output
    unknown = runner.invoke(disputes.main, ["build", "--asr-hyps", "w-prod,zzz"])
    assert "unknown ASR hypotheses" in unknown.output
    off_pivot = runner.invoke(disputes.main, ["build", "--asr-hyps", "w-prod,w-vad", "--pivot", "w-runs-lid"])
    assert "--pivot" in off_pivot.output
