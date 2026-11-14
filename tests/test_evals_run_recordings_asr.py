import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from recordings_fixture import FakeWorker, make_clips, make_meeting, make_owner, pieces, silent, spoken

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
from evals.recordings_scoring import MEETINGS
from evals.run_diarization import load_words
from evals.run_recordings_asr import E2_HYPOTHESES, _load_all, run_corpus, scored
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.errors import ConfigError

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


def test_recording_jobs_carry_no_language_hint_and_only_language_id_hypotheses_the_allowed_set(root: Path):
    (item,) = load_items("ami", root=root)
    jobs = {h: recording_job(item, registry.get(h)) for h in E2_HYPOTHESES}
    assert all(job.language is None and job.chunks is None and job.context is None for job in jobs.values())
    assert {h: job.languages for h, job in jobs.items()} == {
        "w-prod": (),
        "w-vad": (),
        "w-vad-lid": ("zh", "en"),
        "w-runs-lid": ("zh", "en"),
    }


def test_each_long_recording_gets_its_own_worker_and_a_finished_one_is_not_decoded_again(root: Path):
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


def test_a_changed_request_is_decoded_again_in_a_fresh_folder_name_only_when_the_batch_changes(root: Path):
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


def test_scoring_waits_for_a_finished_run(root: Path):
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


def test_score_prints_the_metric_per_corpus_without_recording_ids(root: Path):
    run_both(root)
    output = CliRunner().invoke(
        run_recordings_asr.main, ["score", "--hyp", "w-runs-lid", "--corpus", "ami", "--corpus", "alimeeting"]
    )
    assert output.exit_code == 0, output.output
    report = json.loads(output.output)["corpora"]
    assert (report["ami"]["metric"], report["alimeeting"]["metric"]) == ("wer", "cer")
    assert report["ami"]["errors"] == 1 and report["ami"]["reference_units"] == 6
    assert report["alimeeting"]["errors"] == 1 and report["alimeeting"]["reference_units"] == 12
    assert report["ami"]["windows"] == 2
    assert "m1" not in output.output and "z1" not in output.output


def test_compare_pairs_each_corpus_and_pools_the_two_meeting_corpora(root: Path):
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


def test_words_writes_a_file_wder_can_read(root: Path):
    run_both(root)
    output = CliRunner().invoke(run_recordings_asr.main, ["words", "--hyp", "w-runs-lid", "--corpus", "ami"])
    assert output.exit_code == 0, output.output
    words = load_words(root / "ami" / "m1" / "words.json")
    assert words is not None
    assert [w.text.strip() for w in words][:3] == ["hello", "there", "everybody"]


def test_the_owner_corpus_has_no_reference_and_reports_only_what_it_decoded(root: Path):
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


def test_unknown_corpora_and_recordings_are_refused(root: Path):
    assert "owner" in ASR_CORPORA
    with pytest.raises(ConfigError, match="unknown corpus"):
        load_items("nope", root=root)
    with pytest.raises(ConfigError, match="no recording"):
        load_items("ami", ["zzz"], root=root)
    with pytest.raises(ConfigError, match="no prepared data"):
        load_items("ascend", root=root)
    with pytest.raises(ConfigError, match="no ASCEND utterances"):
        load_items("ascend-utterances", root=root)


def test_a_different_batch_size_never_decodes_a_finished_clip_again(root: Path):
    make_clips(root, [("u1", "你好"), ("u2", "ok"), ("u3", "好的")])
    clips = load_items("ascend-utterances", root=root)
    worker = FakeWorker(transcripts({}))
    layout = asr_layout("ascend-utterances", root)
    run_corpus(layout, "w-prod", clips[:2], batch_size=2, worker=worker)
    summary = run_corpus(layout, "w-prod", clips, batch_size=3, worker=worker)
    assert (summary.files_decoded, summary.files_resumed) == (1, 2)
    assert [len(r.files) for r in worker.requests] == [2, 1]
    assert set(finished_rows(layout, "w-prod", clips)) == {"u1", "u2", "u3"}


def test_a_clip_finished_in_two_worker_folders_fails_loud_and_is_never_read_twice(root: Path):
    make_clips(root, [("u1", "你好"), ("u2", "ok")])
    clips = load_items("ascend-utterances", root=root)
    layout = asr_layout("ascend-utterances", root)
    run_corpus(layout, "w-prod", clips, batch_size=2, worker=FakeWorker(transcripts({})))
    (folder,) = [p for p in layout.run_dir("w-prod").iterdir() if p.is_dir()]
    shutil.copytree(folder, layout.run_dir("w-prod") / "elsewhere")
    with pytest.raises(ConfigError, match="finished in two"):
        finished_rows(layout, "w-prod", clips)


def test_replaced_audio_under_the_same_id_is_refused_not_scored_with_the_old_rows(root: Path):
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


def test_a_partial_run_of_other_audio_is_not_resumed(root: Path):
    items = load_items("ami", root=root)
    layout = asr_layout("ami", root)
    folder = layout.run_dir("w-prod") / "m1"
    folder.mkdir(parents=True)
    (folder / "inputs.json").write_text(json.dumps({"m1": "0" * 64}), encoding="utf-8")
    worker = FakeWorker(transcripts({}))
    with pytest.raises(ConfigError, match="not what worker folder"):
        run_corpus(layout, "w-prod", items, worker=worker)
    assert worker.requests == []


def test_a_folder_decoded_with_another_job_is_refused(root: Path):
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


def test_a_folder_made_by_another_engine_definition_is_refused(root: Path):
    items = load_items("ami", root=root)
    layout = asr_layout("ami", root)
    run_corpus(layout, "w-prod", items, worker=FakeWorker(transcripts({})))
    request_path = layout.run_dir("w-prod") / "m1" / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["engine"]["variant"]["mode"] = "chunk"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ConfigError, match="not made by the current"):
        finished_folders(layout, "w-prod", items)


def test_the_owner_corpus_is_not_a_default_anywhere(root: Path):
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


def test_every_selected_corpus_is_loaded_before_anything_runs(root: Path):
    with pytest.raises(ConfigError, match="no prepared data"):
        _load_all(["ami", "ascend"], ())
    assert list(_load_all(["ami", "alimeeting"], ())) == ["ami", "alimeeting"]


def test_compare_names_the_decision_interval_and_reports_the_meetings_macro_average(root: Path):
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
