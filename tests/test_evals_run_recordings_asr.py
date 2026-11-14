import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from recordings_fixture import FakeWorker, make_clips, make_meeting, make_owner, pieces, spoken

from evals import run_recordings_asr
from evals.manifest import EVAL_DIR_ENV
from evals.recordings_corpus import ASR_CORPORA, asr_layout, batch_key, finished_ids, load_items
from evals.recordings_data import RefUtterance
from evals.recordings_scoring import MEETINGS
from evals.run_diarization import load_words
from evals.run_recordings_asr import (
    E2_HYPOTHESES,
    recording_job,
    run_corpus,
    scored_windows,
)
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis, WorkerRequest
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
    assert finished_ids(layout, "w-prod") == {"m1"}
    assert finished_ids(layout, "w-vad") == set()


def test_a_changed_request_is_decoded_again_in_a_fresh_folder_name_only_when_the_batch_changes(root: Path):
    make_clips(root, [("u1", "你好"), ("u2", "ok"), ("u3", "好的")])
    clips = load_items("ascend-utterances", root=root)
    worker = FakeWorker(transcripts({}))
    layout = asr_layout("ascend-utterances", root)
    run_corpus(layout, "w-vad-lid", clips, batch_size=2, worker=worker)
    assert [len(r.files) for r in worker.requests] == [2, 1]
    assert {j.languages for r in worker.requests for j in r.files} == {("zh", "en")}
    assert finished_ids(layout, "w-vad-lid") == {"u1", "u2", "u3"}
    run_corpus(layout, "w-vad-lid", clips, batch_size=2, worker=worker)
    assert len(worker.requests) == 2
    assert batch_key(["u1", "u2"]) == batch_key(["u1", "u2"]) != batch_key(["u1", "u3"])
    assert batch_key(["u1"]) == "u1"


def test_scoring_waits_for_a_finished_run(root: Path):
    (item,) = load_items("ami", root=root)
    with pytest.raises(ConfigError, match="not finished for 1 of 1 ami"):
        scored_windows("ami", "w-prod", "itn", [item], root)


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
        scored_windows("owner", "w-runs-lid", "itn", [item], root)


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
