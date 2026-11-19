import json
import shutil
from dataclasses import replace
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from recordings_fixture import FakeWorker, make_clips, make_meeting, make_owner, pieces, silent, spoken

from anyverbatim.asr import registry
from anyverbatim.asr.models import AnomalySpan, ChunkHypothesis, FileJob, WorkerRequest
from anyverbatim.errors import ConfigError
from evals import run_diarization, run_recordings_asr
from evals.manifest import EVAL_DIR_ENV
from evals.recordings_corpus import (
    ASR_CORPORA,
    PUBLIC_CORPORA,
    asr_layout,
    batch_key,
    finished_folders,
    finished_rows,
    load_items,
    recording_job,
    stamp_inputs,
)
from evals.recordings_data import RefUtterance
from evals.recordings_rescore import rescore_item, total_repair
from evals.recordings_scoring import ASCEND_POOLED, MEETINGS
from evals.run_diarization import load_words
from evals.run_recordings_asr import E2_HYPOTHESES, _load_all, run_corpus, scored
from evals.store import Layout

EN_LINES = (RefUtterance(0, 3, "A", "hello there everyone"), RefUtterance(70, 73, "B", "now we begin"))
ZH_LINES = (RefUtterance(0, 4, "A", "今天开会讨论预算"), RefUtterance(70, 73, "B", "明天继续"))


def transcripts(table: dict[tuple[str, str], list[list[tuple[str, float, float]]]]):
    """Rows per request: ``table[(hypothesis, file)]`` lists one pieces list per row."""

    def rows_for(request: WorkerRequest) -> list[ChunkHypothesis]:
        hypothesis = request.engine.spec.id
        return [
            spoken(hypothesis, job.id, row, chunk=i)
            for job in request.files
            for i, row in enumerate(table.get((hypothesis, job.id), [[(" ", 0.0, 0.1)]]))
        ]

    return rows_for


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(EVAL_DIR_ENV, str(tmp_path / "eval"))
    recordings = tmp_path / "eval" / "recordings"
    make_meeting(recordings, "ami", "m1", EN_LINES)
    make_meeting(recordings, "alimeeting", "z1", ZH_LINES)
    return recordings


def test_recording_jobs_carry_no_language_hint_and_only_language_id_hypotheses_the_allowed_set(root: Path) -> None:
    (item,) = load_items("ami", root=root)
    jobs = {h: recording_job(item, registry.get(h)) for h in E2_HYPOTHESES}
    assert all(job.language is None and job.chunks is None and job.context is None for job in jobs.values())
    assert {h: job.languages for h, job in jobs.items()} == {
        "w-prod": (),
        "w-vad": (),
        "w-vad-lid": ("zh", "en"),
        "w-runs-lid": ("zh", "en"),
    }


def test_each_long_recording_gets_its_own_worker_and_a_finished_one_is_not_decoded_again(root: Path) -> None:
    items = load_items("ami", root=root) + load_items("alimeeting", root=root)
    worker = FakeWorker(transcripts({}))
    layout = asr_layout("ami", root)
    first = run_corpus(layout, "w-prod", items[:1], worker=worker)
    assert (first.files_decoded, first.files_resumed) == (1, 0)
    second = run_corpus(layout, "w-prod", items[:1], worker=worker)
    assert (second.files_decoded, second.files_resumed) == (0, 1)
    assert len(worker.requests) == 1
    assert Path(worker.requests[0].out_dir) == layout.run_dir("w-prod") / "m1"
    assert set(finished_folders(layout, "w-prod", items)) == {"m1"}
    assert finished_folders(layout, "w-vad", items) == {}


def test_a_changed_request_is_decoded_again_in_a_fresh_folder_name_only_when_the_batch_changes(root: Path) -> None:
    make_clips(root, [("u1", "你好"), ("u2", "ok"), ("u3", "好的")])
    clips = load_items("ascend-utterances", root=root)
    worker = FakeWorker(transcripts({}))
    layout = asr_layout("ascend-utterances", root)
    run_corpus(layout, "w-vad-lid", clips, batch_size=2, worker=worker)
    assert [len(r.files) for r in worker.requests] == [2, 1]
    assert {j.languages for r in worker.requests for j in r.files} == {("zh", "en")}
    assert set(finished_folders(layout, "w-vad-lid", clips)) == {"u1", "u2", "u3"}
    run_corpus(layout, "w-vad-lid", clips, batch_size=2, worker=worker)
    assert len(worker.requests) == 2
    assert batch_key(["u1", "u2"]) == batch_key(["u1", "u2"]) != batch_key(["u1", "u3"])
    assert batch_key(["u1"]) == "u1"


def test_scoring_waits_for_a_finished_run(root: Path) -> None:
    (item,) = load_items("ami", root=root)
    with pytest.raises(ConfigError, match="not finished for 1 of 1 ami"):
        scored("ami", "w-prod", "itn", [item], root)


def run_both(root: Path) -> None:
    table = {
        ("w-prod", "m1"): [pieces("hello there everyone", 0.2) + pieces("now we begin", 70.2)],
        ("w-runs-lid", "m1"): [pieces("hello there everybody", 0.2) + pieces("now we begin", 70.2)],
        ("w-prod", "z1"): [pieces("今天开会讨论预算", 0.2) + pieces("明天继续", 70.2)],
        ("w-runs-lid", "z1"): [pieces("今天开会讨论预算", 0.2) + pieces("明天继", 70.2)],
    }
    worker = FakeWorker(transcripts(table))
    for hypothesis in ("w-prod", "w-runs-lid"):
        for corpus in ("ami", "alimeeting"):
            run_corpus(asr_layout(corpus, root), hypothesis, load_items(corpus, root=root), worker=worker)


def invoke(*args: str) -> dict[str, object]:
    result = CliRunner().invoke(run_recordings_asr.main, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_score_prints_the_metric_per_corpus_without_recording_ids(root: Path) -> None:
    run_both(root)
    output = CliRunner().invoke(
        run_recordings_asr.main, ["score", "--hyp", "w-runs-lid", "--corpus", "ami", "--corpus", "alimeeting"]
    )
    assert output.exit_code == 0, output.output
    report = json.loads(output.output)["corpora"]
    assert (report["ami"]["metric"], report["alimeeting"]["metric"]) == ("wer", "cer")
    assert report["ami"]["errors"] == 1
    assert report["ami"]["reference_units"] == 6
    assert report["alimeeting"]["errors"] == 1
    assert report["alimeeting"]["reference_units"] == 12
    assert report["ami"]["windows"] == 2
    assert "m1" not in output.output
    assert "z1" not in output.output


def test_compare_pairs_each_corpus_and_pools_the_two_meeting_corpora(root: Path) -> None:
    run_both(root)
    output = CliRunner().invoke(
        run_recordings_asr.main,
        ["compare", "--a", "w-runs-lid", "--b", "w-prod", "--corpus", "ami", "--corpus", "alimeeting"],
    )
    assert output.exit_code == 0, output.output
    results = json.loads(output.output)["results"]
    assert set(results) == {"ami", "alimeeting", MEETINGS}
    assert results["ami"]["difference"] == pytest.approx(1 / 6)
    assert results[MEETINGS]["windows"] == 4
    assert results[MEETINGS]["difference"] == pytest.approx(2 / 18)


def choices(command: str, option: str) -> tuple[str, ...]:
    (param,) = (p for p in run_recordings_asr.main.commands[command].params if p.name == option)
    assert isinstance(param.type, click.Choice)
    return tuple(param.type.choices)


def test_every_e2_hypothesis_is_a_choice_of_every_command_that_names_one() -> None:
    for command, option in (
        ("run", "hypothesis"),
        ("score", "hypothesis"),
        ("compare", "first"),
        ("compare", "second"),
    ):
        assert choices(command, option) == E2_HYPOTHESES


def test_score_counts_rows_holding_a_special_token(root: Path) -> None:
    table = {("w-runs-lid", "m1"): [pieces("hello there", 0.2), pieces("<|zh|> now we begin", 70.2)]}
    run_corpus(
        asr_layout("ami", root), "w-runs-lid", load_items("ami", root=root), worker=FakeWorker(transcripts(table))
    )
    report = invoke("score", "--hyp", "w-runs-lid", "--corpus", "ami")["corpora"]
    assert report["ami"]["special_token_rows"] == 1  # type: ignore[index]
    assert report["ami"]["multi_language_recordings"] == 0  # type: ignore[index]
    assert report["ami"]["chunk_incidents"]["clean_in_mixed_recordings"] == 0  # type: ignore[index]


def test_words_writes_a_file_wder_can_read(root: Path) -> None:
    run_both(root)
    output = CliRunner().invoke(run_recordings_asr.main, ["words", "--hyp", "w-runs-lid", "--corpus", "ami"])
    assert output.exit_code == 0, output.output
    words = load_words(root / "ami" / "m1" / "words.json")
    assert words is not None
    assert [w.text.strip() for w in words][:3] == ["hello", "there", "everybody"]


def test_the_owner_corpus_has_no_reference_and_reports_only_what_it_decoded(root: Path) -> None:
    make_owner(root, [("rec-01", 60, {"language": "zh+en"})])
    (item,) = load_items("owner", root=root)
    assert item.reference is None
    worker = FakeWorker(
        lambda request: [
            spoken("w-runs-lid", "rec-01", pieces("你好吗", 1.0), language="zh"),
            spoken("w-runs-lid", "rec-01", pieces("how are you", 5.0), chunk=1, language="en"),
        ]
    )
    run_corpus(asr_layout("owner", root), "w-runs-lid", [item], worker=worker)
    report = invoke("score", "--hyp", "w-runs-lid", "--corpus", "owner")["corpora"]
    assert report["owner"]["recordings"] == 1  # type: ignore[index]
    assert report["owner"]["decoded_recordings"] == 1  # type: ignore[index]
    assert set(report["owner"]["language_seconds"]) == {"zh", "en"}  # type: ignore[index]
    with pytest.raises(ConfigError, match="owner"):
        scored("owner", "w-runs-lid", "itn", [item], root)


def test_unknown_corpora_and_recordings_are_refused(root: Path) -> None:
    assert "owner" in ASR_CORPORA
    with pytest.raises(ConfigError, match="unknown corpus"):
        load_items("nope", root=root)
    with pytest.raises(ConfigError, match="no recording"):
        load_items("ami", ["zzz"], root=root)
    with pytest.raises(ConfigError, match="no prepared data"):
        load_items("ascend", root=root)
    with pytest.raises(ConfigError, match="no ASCEND utterances"):
        load_items("ascend-utterances", root=root)


def test_a_different_batch_size_never_decodes_a_finished_clip_again(root: Path) -> None:
    make_clips(root, [("u1", "你好"), ("u2", "ok"), ("u3", "好的")])
    clips = load_items("ascend-utterances", root=root)
    worker = FakeWorker(transcripts({}))
    layout = asr_layout("ascend-utterances", root)
    run_corpus(layout, "w-prod", clips[:2], batch_size=2, worker=worker)
    summary = run_corpus(layout, "w-prod", clips, batch_size=3, worker=worker)
    assert (summary.files_decoded, summary.files_resumed) == (1, 2)
    assert [len(r.files) for r in worker.requests] == [2, 1]
    assert set(finished_rows(layout, "w-prod", clips)) == {"u1", "u2", "u3"}


def test_a_clip_finished_in_two_worker_folders_fails_loud_and_is_never_read_twice(root: Path) -> None:
    make_clips(root, [("u1", "你好"), ("u2", "ok")])
    clips = load_items("ascend-utterances", root=root)
    layout = asr_layout("ascend-utterances", root)
    run_corpus(layout, "w-prod", clips, batch_size=2, worker=FakeWorker(transcripts({})))
    (folder,) = [p for p in layout.run_dir("w-prod").iterdir() if p.is_dir()]
    shutil.copytree(folder, layout.run_dir("w-prod") / "elsewhere")
    with pytest.raises(ConfigError, match="finished in two"):
        finished_rows(layout, "w-prod", clips)


def test_replaced_audio_under_the_same_id_is_refused_not_scored_with_the_old_rows(root: Path) -> None:
    items = load_items("ami", root=root)
    layout = asr_layout("ami", root)
    worker = FakeWorker(transcripts({}))
    run_corpus(layout, "w-prod", items, worker=worker)
    (root / "ami" / "m1" / "audio.wav").write_bytes(silent(79))
    with pytest.raises(ConfigError, match="not what worker folder"):
        finished_rows(layout, "w-prod", items)
    with pytest.raises(ConfigError, match="not what worker folder"):
        run_corpus(layout, "w-prod", items, worker=worker)
    assert len(worker.requests) == 1


def test_a_partial_run_of_other_audio_is_not_resumed(root: Path) -> None:
    items = load_items("ami", root=root)
    layout = asr_layout("ami", root)
    folder = layout.run_dir("w-prod") / "m1"
    folder.mkdir(parents=True)
    (folder / "inputs.json").write_text(json.dumps({"m1": "0" * 64}), encoding="utf-8")
    worker = FakeWorker(transcripts({}))
    with pytest.raises(ConfigError, match="not what worker folder"):
        run_corpus(layout, "w-prod", items, worker=worker)
    assert worker.requests == []


def test_a_folder_decoded_with_another_job_is_refused(root: Path) -> None:
    items = load_items("ami", root=root)
    layout = asr_layout("ami", root)
    definition = registry.get("w-vad-lid")
    stale = WorkerRequest(
        engine=definition,
        out_dir=str(layout.run_dir("w-vad-lid") / "m1"),
        files=(FileJob(id="m1", wav=str(items[0].wav)),),
    )
    FakeWorker(transcripts({}))(stale)
    stamp_inputs(layout.run_dir("w-vad-lid") / "m1", items)
    with pytest.raises(ConfigError, match="different job"):
        finished_folders(layout, "w-vad-lid", items)


def test_a_folder_made_by_another_engine_definition_is_refused(root: Path) -> None:
    items = load_items("ami", root=root)
    layout = asr_layout("ami", root)
    run_corpus(layout, "w-prod", items, worker=FakeWorker(transcripts({})))
    request_path = layout.run_dir("w-prod") / "m1" / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["engine"]["variant"]["mode"] = "chunk"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ConfigError, match="not made by the current"):
        finished_folders(layout, "w-prod", items)


def test_the_owner_corpus_is_not_a_default_anywhere(root: Path) -> None:
    assert "owner" not in PUBLIC_CORPORA
    for command in (
        run_recordings_asr.run,
        run_recordings_asr.score,
        run_recordings_asr.compare_command,
        run_recordings_asr.words,
    ):
        corpus = next(p for p in command.params if p.name == "corpora")
        assert "owner" not in corpus.default
    assert (
        tuple(next(p for p in run_diarization.run.params if p.name == "corpora").default)
        == run_diarization.DEFAULT_CORPORA
    )
    assert "owner" not in run_diarization.DEFAULT_CORPORA


def test_every_selected_corpus_is_loaded_before_anything_runs(root: Path) -> None:
    with pytest.raises(ConfigError, match="no prepared data"):
        _load_all(["ami", "ascend"], ())
    assert list(_load_all(["ami", "alimeeting"], ())) == ["ami", "alimeeting"]


def test_compare_names_the_decision_interval_and_reports_the_meetings_macro_average(root: Path) -> None:
    run_both(root)
    results = invoke("compare", "--a", "w-runs-lid", "--b", "w-prod", "--corpus", "ami", "--corpus", "alimeeting")
    assert results["decision_interval"] == "decision"
    pooled = results["results"][MEETINGS]  # type: ignore[index]
    assert pooled["macro_difference"] == pytest.approx(
        (results["results"]["ami"]["difference"] + results["results"]["alimeeting"]["difference"]) / 2  # type: ignore[index]
    )
    assert "dominate" in pooled["note"]
    assert pooled["recordings"] == 2
    assert set(pooled["by_window"]) == {"low", "high", "excludes_zero"}


ASCEND_LINES = (
    RefUtterance(0, 5, "A", "我们今天开会讨论预算问题"),
    RefUtterance(10, 13, "B", "we shipped it today"),
    RefUtterance(20, 24, "A", "明天我们继续开会讨论"),
)


def decode_ascend(root: Path, hypothesis: str, translated: bool) -> None:
    """Sessions and a clip that share the id ``m1``; ``translated`` renders the first line and the clip in English."""
    first = pieces("we discuss the budget today", 0.2) if translated else pieces("我们今天开会讨论预算问题", 0.2, 0.3)
    session = [first + pieces("we shipped it today", 10.2) + pieces("明天我们继续开会讨论", 20.2, 0.3)]
    clip = [pieces("how are you friends", 0.2) if translated else pieces("你好吗朋友 ok now", 0.2)]
    for corpus, rows in (("ascend", session), ("ascend-utterances", clip)):
        items = load_items(corpus, root=root)
        run_corpus(
            asr_layout(corpus, root), hypothesis, items, worker=FakeWorker(transcripts({(hypothesis, "m1"): rows}))
        )


def test_compare_pairs_incidents_per_corpus_and_pools_the_ascend_sessions_with_the_clips(root: Path) -> None:
    make_meeting(root, "ascend", "m1", ASCEND_LINES)
    make_clips(root, [("m1", "你好吗朋友 ok now")])
    decode_ascend(root, "w-runs-lid", translated=False)
    decode_ascend(root, "w-prod", translated=True)
    results = invoke(
        "compare", "--a", "w-runs-lid", "--b", "w-prod", "--corpus", "ascend", "--corpus", "ascend-utterances"
    )["results"]
    assert set(results) == {"ascend", "ascend-utterances", ASCEND_POOLED}  # type: ignore[arg-type]
    sessions, clips, pooled = (results[k] for k in ("ascend", "ascend-utterances", ASCEND_POOLED))  # type: ignore[index]
    assert (sessions["incidents"]["fixed"], clips["incidents"]["fixed"]) == (1, 1)
    assert pooled["incidents"] == {
        "lines": 4,
        "a_incidents": 0,
        "b_incidents": 2,
        "fixed": 2,
        "introduced": 0,
        "p": 0.25,
    }
    assert pooled["windows"] == sessions["windows"] + clips["windows"]
    assert pooled["recordings"] == 2


def test_compare_leaves_out_incidents_for_corpora_without_a_mixed_recording(root: Path) -> None:
    run_both(root)
    results = invoke("compare", "--a", "w-runs-lid", "--b", "w-prod", "--corpus", "ami", "--corpus", "alimeeting")[
        "results"
    ]
    assert all("incidents" not in entry for entry in results.values())  # type: ignore[attr-defined]


LOOP_LINES = (
    RefUtterance(0, 3, "A", "hello there everyone"),
    RefUtterance(10, 14, "B", "please repeat that slowly"),
    RefUtterance(70, 73, "B", "now we begin"),
)
LOOP = " ".join(["alpha", "beta"] * 10)


def looped_before() -> list[tuple[str, float, float]]:
    return (
        pieces("hello there everyone", 0.2)
        + pieces("okay fine", 5.0)
        + pieces(LOOP, 10.2)
        + pieces("now we begin", 70.2)
    )


def repaired_after(window: str = "please repeat that slowly") -> list[tuple[str, float, float]]:
    return (
        pieces("hello there everybody", 0.2)
        + pieces("okay fine", 5.0)
        + pieces(window, 10.2)
        + pieces("now we begin", 70.2)
    )


def decode_rows(root: Path, layout: Layout, rows: dict[str, ChunkHypothesis]) -> None:
    items = load_items("ami", list(rows), root=root)
    worker = FakeWorker(lambda request: [rows[job.id] for job in request.files])
    run_corpus(layout, "w-prod", items, worker=worker)


def row(pieces_: list[tuple[str, float, float]], *flags: str, spans: tuple[AnomalySpan, ...] = ()) -> ChunkHypothesis:
    return spoken("w-prod", "m2", pieces_).model_copy(update={"flags": flags, "anomaly_spans": spans})


def rescore_output(root: Path, tmp_path: Path, before: ChunkHypothesis, after: ChunkHypothesis) -> str:
    make_meeting(root, "ami", "m2", LOOP_LINES)
    before_dir = tmp_path / "before"
    decode_rows(root, Layout(before_dir / "ami"), {"m2": before})
    decode_rows(root, asr_layout("ami", root), {"m2": after})
    result = CliRunner().invoke(
        run_recordings_asr.main,
        ["rescore", "--hyp", "w-prod", "--corpus", "ami", "--recordings", "m2", "--before", str(before_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return result.output


def rescored(root: Path, tmp_path: Path, before: ChunkHypothesis, after: ChunkHypothesis) -> dict[str, object]:
    return json.loads(rescore_output(root, tmp_path, before, after))


def test_rescore_counts_errors_before_and_after_and_pairs_the_windows(root: Path, tmp_path: Path) -> None:
    report = rescored(root, tmp_path, row(looped_before()), row(repaired_after(), "redecoded"))
    (file,) = report["files"]
    assert (report["metric"], file["item"]) == ("wer", "m2")
    assert (file["before"]["errors"], file["after"]["errors"], file["before"]["reference_units"]) == (22, 3, 10)
    assert file["before"]["rate"] == pytest.approx(2.2)
    assert report["after"]["errors"] == 3
    assert report["difference_is_after_minus_before"] is True
    assert report["paired"]["windows"] == 2
    assert report["paired"]["difference"] == pytest.approx(-1.9)
    assert (file["before_rows"]["redecoded"], file["after_rows"]["redecoded"]) == (0, 1)


def test_rescore_finds_the_repaired_window_and_its_loop_and_leaves_other_differences_outside(
    root: Path, tmp_path: Path
) -> None:
    report = rescored(root, tmp_path, row(looped_before()), row(repaired_after(), "redecoded"))
    (file,) = report["files"]
    (window,) = file["repaired_windows"]
    assert window["span"]["start"] == pytest.approx(10.2)
    assert window["span"]["end"] == pytest.approx(18.2)
    assert (window["before_words"], window["after_words"]) == (20, 4)
    assert window["loop"]["start"] == pytest.approx(10.2)
    assert window["loop"]["end"] == pytest.approx(18.2)
    assert window["loop_offset_s"] == pytest.approx(0.0)
    assert window["still_flagged"] is False
    assert window["seam_duplicate"] is False
    # "everyone" became "everybody" in a window that never looped: a different decode, not a repair.
    assert file["outside"] == {"words": 8, "identical": 7, "share": pytest.approx(7 / 8)}
    assert report["repair"]["looped_windows"] == 1
    assert report["repair"]["still_flagged"] == 0
    assert report["unavailable"]


def test_rescore_needs_a_redecoded_row_to_call_a_difference_a_repair(root: Path, tmp_path: Path) -> None:
    report = rescored(root, tmp_path, row(looped_before()), row(repaired_after()))
    assert report["files"][0]["repaired_windows"] == []


def test_rescore_counts_a_seam_duplicate_only_when_the_repeated_word_follows_at_once(
    root: Path, tmp_path: Path
) -> None:
    close_before = (
        pieces("hello there everyone", 0.2)
        + pieces("okay fine", 5.0)
        + pieces(LOOP, 10.2)
        + pieces("now we begin", 18.4)
    )
    close_after = (
        pieces("hello there everybody", 0.2)
        + pieces("okay fine", 5.0)
        + pieces("please repeat that now", 10.2, step=2.0)
        + pieces("now we begin", 18.4)
    )
    report = rescored(root, tmp_path, row(close_before), row(close_after, "redecoded"))
    (window,) = report["files"][0]["repaired_windows"]
    assert window["seam_duplicate"] is True
    assert (report["repair"]["seams_checked"], report["repair"]["seam_duplicates"]) == (1, 1)
    assert report["rules"]["rule_3"] == {"repaired_windows": 1, "seam_duplicates": 1, "share": 1.0, "plan_dedupe": True}


def test_rescore_does_not_call_the_same_word_much_later_a_seam_duplicate() -> None:
    after = row(repaired_after("please repeat that now"), "redecoded")
    (window,) = rescore_item("m2", [row(looped_before())], [after], [], []).repaired_windows
    assert window.seam_duplicate is False


def test_rescore_reports_the_anomaly_rows_spans_and_what_the_repair_left_flagged(root: Path, tmp_path: Path) -> None:
    before = row(looped_before(), "decoder_anomaly")
    spans = (AnomalySpan(start=10.0, end=19.0), AnomalySpan(start=40.0, end=45.0))
    after = row(repaired_after(), "redecoded", "decoder_anomaly", spans=spans)
    report = rescored(root, tmp_path, before, after)
    (file,) = report["files"]
    (flagged_before,) = file["before_rows"]["decoder_anomaly"]
    assert flagged_before["localized"] is False
    assert flagged_before["spans"] == []
    (flagged_after,) = file["after_rows"]["decoder_anomaly"]
    assert flagged_after["localized"] is True
    assert len(flagged_after["spans"]) == 2
    assert file["repaired_windows"][0]["still_flagged"] is True
    assert file["unrepaired_spans"] == [{"start": 40.0, "end": 45.0}]
    assert (report["repair"]["looped_windows"], report["repair"]["still_flagged"]) == (2, 2)


def test_rescore_flags_a_whole_row_anomaly_as_unlocalized(root: Path, tmp_path: Path) -> None:
    after = row(repaired_after(), "redecoded", "decoder_anomaly")
    report = rescored(root, tmp_path, row(looped_before()), after)
    (flagged,) = report["files"][0]["after_rows"]["decoder_anomaly"]
    assert flagged["localized"] is False
    assert report["files"][0]["repaired_windows"][0]["still_flagged_unlocalized"] is True


def test_rescore_prints_no_transcript_text(root: Path, tmp_path: Path) -> None:
    output = rescore_output(root, tmp_path, row(looped_before()), row(repaired_after(), "redecoded"))
    for text in ("hello", "everybody", "alpha", "please", "slowly", "okay"):
        assert text not in output


def test_rescore_scores_alimeeting_by_characters(root: Path, tmp_path: Path) -> None:
    before_dir = tmp_path / "before"
    short = spoken("w-prod", "z1", pieces("今天开会讨论预", 0.2) + pieces("明天继续", 70.2))
    full = spoken("w-prod", "z1", pieces("今天开会讨论预算", 0.2) + pieces("明天继续", 70.2))
    for layout, made in ((Layout(before_dir / "alimeeting"), short), (asr_layout("alimeeting", root), full)):
        items = load_items("alimeeting", root=root)
        run_corpus(layout, "w-prod", items, worker=FakeWorker(lambda _, made=made: [made]))
    result = CliRunner().invoke(
        run_recordings_asr.main,
        ["rescore", "--hyp", "w-prod", "--corpus", "alimeeting", "--recordings", "z1", "--before", str(before_dir)],
        catch_exceptions=False,
    )
    report = json.loads(result.output)
    assert (report["metric"], report["before"]["errors"], report["after"]["errors"]) == ("cer", 1, 0)
    assert report["files"][0]["repaired_windows"] == []


def test_rescore_refuses_windows_the_two_runs_were_not_both_scored_on(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_meeting(root, "ami", "m2", LOOP_LINES)
    before_dir = tmp_path / "before"
    decode_rows(root, Layout(before_dir / "ami"), {"m2": row(looped_before())})
    decode_rows(root, asr_layout("ami", root), {"m2": row(repaired_after(), "redecoded")})
    real = run_recordings_asr.score_items
    calls: list[int] = []

    def short_first(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(1)
        return replace(result, windows=result.windows[1:]) if len(calls) == 1 else result

    monkeypatch.setattr(run_recordings_asr, "score_items", short_first)
    items = load_items("ami", ["m2"], root=root)
    with pytest.raises(ConfigError, match="different windows"):
        run_recordings_asr.rescored("ami", "w-prod", "itn", items, before_dir)


def test_rescore_needs_the_flagged_recordings_named(root: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        run_recordings_asr.main,
        ["rescore", "--hyp", "w-prod", "--corpus", "ami", "--recordings", ",", "--before", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "needs at least one id" in result.output


def at(text: str, start: float, step: float = 0.4) -> list[tuple[str, float, float]]:
    return pieces(text, start, step)


def probe_row(
    words: list[tuple[str, float, float]], *flags: str, spans: tuple[AnomalySpan, ...] = ()
) -> ChunkHypothesis:
    return spoken("w-prod", "m2", words).model_copy(update={"flags": flags, "anomaly_spans": spans})


HEAD = at("one two three four", 0.0)
TAIL = at("now we begin", 70.0)
LOOP_A = at(LOOP, 12.0)
FIXED_A = at("please repeat that slowly", 12.0, 1.0)
LOOP_B = at(LOOP, 21.0)


def test_windows_that_touch_at_a_seam_do_not_overlap_so_an_unrepaired_neighbour_is_its_own_looped_window() -> None:
    before = probe_row(HEAD + LOOP_A + LOOP_B + TAIL, "decoder_anomaly")
    after = probe_row(
        HEAD + FIXED_A + LOOP_B + TAIL, "redecoded", "decoder_anomaly", spans=(AnomalySpan(start=20.0, end=30.0),)
    )
    file = rescore_item("m2", [before], [after], [], [])
    (window,) = file.repaired_windows
    assert (window.span.start, window.span.end) == (12.0, pytest.approx(20.0))
    assert window.still_flagged is False
    assert [(s.start, s.end) for s in file.unrepaired_spans] == [(20.0, 30.0)]
    repair = total_repair([file])
    assert (repair.looped_windows, repair.still_flagged) == (2, 1)


def test_a_window_only_a_whole_row_flag_covers_is_unlocalized_and_left_out_of_rule_2(
    root: Path, tmp_path: Path
) -> None:
    before = probe_row(looped_before(), "decoder_anomaly")
    after = probe_row(repaired_after(), "redecoded", "decoder_anomaly")
    report = rescored(root, tmp_path, before, after)
    (window,) = report["files"][0]["repaired_windows"]
    assert (window["still_flagged"], window["still_flagged_unlocalized"]) == (False, True)
    assert report["repair"]["still_flagged_unlocalized"] == 1
    rules = report["rules"]
    assert rules["rule_2"]["judged_windows"] == 0
    assert rules["rule_2"]["share"] is None
    assert rules["rule_2"]["stop"] is False
    assert rules["rule_4"]["unlocalized_rows"] == [{"item": "m2", "chunk": 0, "start": 0.2, "end": 71.4}]


def test_rules_name_their_inputs_and_outcomes(root: Path, tmp_path: Path) -> None:
    spans = (AnomalySpan(start=10.0, end=19.0), AnomalySpan(start=40.0, end=45.0))
    after = probe_row(repaired_after(), "redecoded", "decoder_anomaly", spans=spans)
    rules = rescored(root, tmp_path, probe_row(looped_before(), "decoder_anomaly"), after)["rules"]
    assert rules["rule_1"]["before_rate"] == pytest.approx(2.2)
    assert rules["rule_1"]["after_rate"] == pytest.approx(0.3)
    assert rules["rule_1"]["difference"] < 0
    assert rules["rule_1"]["window_ci"]["low"] < rules["rule_1"]["window_ci"]["high"]
    assert rules["rule_1"]["worse_with_ci_above_zero"] is False
    assert rules["rule_1"]["ships"] is True
    assert rules["rule_2"] == {
        "looped_windows": 2,
        "still_flagged_unlocalized": 0,
        "judged_windows": 2,
        "still_flagged": 2,
        "share": 1.0,
        "stop": True,
    }
    assert rules["rule_3"]["plan_dedupe"] is False
    assert rules["rule_3"]["share"] == 0.0
    assert rules["rule_4"]["unlocalized_rows"] == []


def test_a_difference_the_guard_would_not_judge_a_failure_is_not_a_repair_and_is_listed_with_its_triggers() -> None:
    varied = at(
        "we looked at several numbers this quarter and the team agreed on a new plan for hiring next year "
        "while costs stayed flat across most regions so margins held up better than many expected",
        40.0,
        0.3,
    )
    repeated = at("the market rallied on strong earnings " * 4, varied[-1][2] + 0.1, 0.3)
    other = at(
        "a completely different reading of that passage came out on this fresh decode today", repeated[0][1], 0.3
    )
    before = probe_row(HEAD + LOOP_A + varied + repeated + TAIL, "decoder_anomaly")
    after = probe_row(HEAD + FIXED_A + varied + other + TAIL, "redecoded")
    file = rescore_item("m2", [before], [after], [], [])
    assert len(file.repaired_windows) == 1
    (rest,) = file.other_differences
    assert rest.triggers == ()
    assert rest.in_redecoded_row is True
    assert total_repair([file]).other_differences == 1


def test_a_loop_the_new_words_keep_is_not_a_repair_even_when_one_word_of_it_changed() -> None:
    legit_old = at("yes " * 12, 50.0, 0.3)
    legit_new = at("yes " * 11 + "right", 50.0, 0.3)
    before = probe_row(HEAD + LOOP_A + legit_old + TAIL, "decoder_anomaly")
    after = probe_row(HEAD + FIXED_A + legit_new + TAIL, "redecoded")
    file = rescore_item("m2", [before], [after], [], [])
    assert [(w.span.start, w.span.end) for w in file.repaired_windows] == [(12.0, pytest.approx(20.0))]
    assert len(file.other_differences) == 1


def test_score_counts_recordings_whose_rows_hold_more_than_one_language(root: Path) -> None:
    def rows_for(request: WorkerRequest) -> list[ChunkHypothesis]:
        (job,) = request.files
        second = "en" if job.id == "m1" else "zh"
        return [
            spoken("w-runs-lid", job.id, pieces("hello there", 0.2), language="zh"),
            spoken("w-runs-lid", job.id, pieces("now we begin", 70.2), chunk=1, language=second),
        ]

    items = load_items("ami", root=root)
    run_corpus(asr_layout("ami", root), "w-runs-lid", items, worker=FakeWorker(rows_for))
    report = invoke("score", "--hyp", "w-runs-lid", "--corpus", "ami")["corpora"]
    assert report["ami"]["multi_language_recordings"] == 1  # type: ignore[index]
