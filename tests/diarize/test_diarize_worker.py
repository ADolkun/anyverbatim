import fcntl
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fake_diarizer import CHANNELS, FRAME_S, definition, write_silence

from ytverbatim import mlx_worker, paths, proc
from ytverbatim.contract.models import Word
from ytverbatim.diarize import protocol, registry, runner, worker
from ytverbatim.diarize.attribution import assign
from ytverbatim.diarize.protocol import (
    ERROR_FILE,
    FINGERPRINT_FILE,
    METRICS_FILE,
    REQUEST_FILE,
    DiarizeRequest,
    FileJob,
    RunMetrics,
    probs_path,
    turns_path,
)
from ytverbatim.errors import EngineError, EngineMemoryError, ProcMemoryExceeded

HERE = Path(__file__).parent
GIB = 1 << 30


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(HERE))
    return home


def make_request(tmp_path: Path, *seconds: float, **variant: object) -> DiarizeRequest:
    jobs = tuple(
        FileJob(id=f"f{i}", wav=str(write_silence(tmp_path / f"f{i}.wav", s))) for i, s in enumerate(seconds or (4.0,))
    )
    return DiarizeRequest(diarizer=definition(**variant), out_dir=str(tmp_path / "out"), files=jobs)


class FakeMlx:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def set_cache_limit(self, limit: int) -> int:
        self.calls.append(f"cache_limit {limit}")
        return 0

    def set_memory_limit(self, limit: int) -> int:
        self.calls.append(f"memory_limit {limit}")
        return 0

    def clear_cache(self) -> None:
        self.calls.append("clear_cache")


def test_fake_output_becomes_turns_and_float16_probabilities(tmp_path: Path) -> None:
    request = make_request(tmp_path, 4.0)
    metrics = worker.run_request(request)
    out = tmp_path / "out"
    stored = protocol.read_file_diarization(out, "f0", "d-fake")
    assert stored.diarizer == "d-fake"
    assert (stored.frames, stored.channels, stored.frame_s) == (400, CHANNELS, FRAME_S)
    assert stored.num_speakers == 2
    assert [(t.speaker, t.start, t.end) for t in stored.turns] == [(0, 0.0, 2.0), (1, 2.0, 4.0)]
    assert probs_path(out, "f0").stat().st_size == 400 * CHANNELS * 2
    loaded = protocol.load(out, "f0", "d-fake")
    assert loaded.prob(0, 0) == pytest.approx(0.9, abs=1e-3)
    assert loaded.prob(399, 1) == pytest.approx(0.8, abs=1e-3)
    assert loaded.prob(399, 0) == 0.0
    spoken = [Word(text="a", start=0.5, end=1.0), Word(text="b", start=2.5, end=3.0)]
    assert assign(spoken, loaded).speakers == ("S1", "S2")
    assert (metrics.files_diarized, metrics.files_skipped) == (1, 0)
    assert metrics.audio_s == pytest.approx(4.0)
    assert RunMetrics.model_validate_json((out / METRICS_FILE).read_text(encoding="utf-8")) == metrics


def test_the_allocator_is_bounded_before_loading_and_emptied_after_every_file(tmp_path: Path) -> None:
    mlx = FakeMlx()
    request = make_request(tmp_path, 2.0, 2.0).model_copy(update={"memory_cap_bytes": 10 * GIB})
    worker.run_request(request, mlx=mlx)
    assert mlx.calls == ["cache_limit 0", f"memory_limit {int(10 * GIB * 0.85)}", "clear_cache", "clear_cache"]


def test_a_rerun_skips_finished_files_without_loading_the_model(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    request = make_request(tmp_path, 2.0, 3.0, calls_log=str(log))
    worker.run_request(request)
    turns_path(tmp_path / "out", "f1").unlink()
    again = worker.run_request(request)
    assert (again.files_diarized, again.files_skipped) == (1, 1)
    assert log.read_text(encoding="utf-8").splitlines() == [
        "load",
        "diarize 32000",
        "diarize 48000",
        "load",
        "diarize 48000",
    ]
    log.unlink()
    third = worker.run_request(request)
    assert (third.files_diarized, third.files_skipped, third.load_s) == (0, 2, 0.0)
    assert not log.exists()


def test_output_of_another_request_is_refused(tmp_path: Path) -> None:
    worker.run_request(make_request(tmp_path, 2.0))
    with pytest.raises(EngineError, match="diarizer changed"):
        worker.run_request(make_request(tmp_path, 2.0, threshold=0.6))


def test_output_without_a_fingerprint_is_refused(tmp_path: Path) -> None:
    worker.run_request(make_request(tmp_path, 2.0))
    (tmp_path / "out" / FINGERPRINT_FILE).unlink()
    with pytest.raises(EngineError, match=r"no fingerprint\.json"):
        worker.run_request(make_request(tmp_path, 2.0))


def test_a_diarizer_failure_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(EngineError, match="d-fake failed on f0: fake diarizer exploded"):
        worker.run_request(make_request(tmp_path, 2.0, fail=True))


def test_file_ids_differing_only_in_case_are_refused(tmp_path: Path) -> None:
    jobs = (FileJob(id="Ab", wav="a.wav"), FileJob(id="aB", wav="b.wav"))
    with pytest.raises(ValueError, match="ignoring case"):
        DiarizeRequest(diarizer=definition(), out_dir=str(tmp_path / "out"), files=jobs)


def test_a_turns_file_written_for_another_id_is_refused(tmp_path: Path) -> None:
    worker.run_request(make_request(tmp_path, 2.0))
    out = tmp_path / "out"
    turns_path(out, "f0").rename(turns_path(out, "g0"))
    probs_path(out, "f0").rename(probs_path(out, "g0"))
    with pytest.raises(EngineError, match="for file 'f0', not d-fake output for 'g0'"):
        protocol.load(out, "g0", "d-fake")
    with pytest.raises(EngineError, match="holds d-fake output"):
        protocol.read_file_diarization(out, "g0", "d-other")


def test_a_turns_file_without_its_probabilities_is_not_skipped(tmp_path: Path) -> None:
    request = make_request(tmp_path, 2.0)
    worker.run_request(request)
    probs_path(tmp_path / "out", "f0").unlink()
    with pytest.raises(EngineError, match=r"f0\.probs\.f16 is unreadable"):
        worker.run_request(request)


@pytest.mark.parametrize("bad", [float("nan"), 1.5, -0.1])
def test_probabilities_outside_zero_and_one_are_refused(tmp_path: Path, bad: float) -> None:
    with pytest.raises(EngineError, match="outside"):
        worker.run_request(make_request(tmp_path, 2.0, bad_prob=bad))
    assert not probs_path(tmp_path / "out", "f0").exists()


def test_a_file_id_that_is_not_a_plain_name_is_refused() -> None:
    with pytest.raises(ValueError, match="names output files"):
        FileJob(id="../x", wav="a.wav")


def test_a_threshold_outside_zero_and_one_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="threshold"):
        make_request(tmp_path, 2.0, threshold=1.5)


def test_a_real_worker_subprocess_diarizes_and_resumes(tmp_path: Path) -> None:
    request = make_request(tmp_path, 2.0)
    first = runner.run_worker(request)
    assert first.out_dir == tmp_path / "out"
    assert (first.metrics.files_diarized, first.metrics.files_skipped) == (1, 0)
    assert first.metrics.peak_footprint_bytes is not None
    assert runner.finished(request) == tmp_path / "out"
    moved = request.model_copy(update={"files": (FileJob(id="f0", wav=str(tmp_path / "elsewhere.wav")),)})
    assert runner.finished(moved) == tmp_path / "out"
    assert runner.finished(make_request(tmp_path, 2.0, threshold=0.6)) is None
    again = runner.run_worker(request)
    assert (again.metrics.files_diarized, again.metrics.files_skipped) == (0, 1)


def test_the_error_report_travels_to_the_parent(tmp_path: Path) -> None:
    with pytest.raises(EngineError, match=r"d-fake worker failed \(EngineError\): d-fake failed on f0"):
        runner.run_worker(make_request(tmp_path, 2.0, fail=True))
    assert json.loads((tmp_path / "out" / ERROR_FILE).read_text(encoding="utf-8"))["type"] == "EngineError"


def test_the_worker_runs_in_its_output_folder_under_the_asr_lock_and_a_memory_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], *, memory_watch: proc.MemoryWatch, cwd: Path, **_: object) -> proc.ProcResult:
        seen.update(argv=argv, cap=memory_watch.cap_bytes, cwd=cwd)
        fd = os.open(paths.locks_dir() / mlx_worker.LOCK_NAME, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        return proc.ProcResult(1, "", "boom", 0.1)

    monkeypatch.setattr(proc, "run", fake_run)
    monkeypatch.setattr("ytverbatim.mlx_worker.sysmem.physical_memory_bytes", lambda: 24 * GIB)
    monkeypatch.chdir(tmp_path)
    request = make_request(tmp_path, 2.0).model_copy(update={"out_dir": "out"})
    with pytest.raises(EngineError, match="exited with 1"):
        runner.run_worker(request)
    assert seen["cap"] == 12 * GIB
    assert seen["cwd"] == (tmp_path / "out").resolve()
    assert seen["argv"][1:4] == ["-P", "-m", "ytverbatim.diarize.worker"]  # type: ignore[index]
    written = DiarizeRequest.model_validate_json((tmp_path / "out" / REQUEST_FILE).read_text(encoding="utf-8"))
    assert Path(written.out_dir).is_absolute()
    assert all(Path(job.wav).is_absolute() for job in written.files)


def test_a_diarizer_waits_for_an_asr_worker_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    entered = threading.Event()

    def take() -> None:
        with mlx_worker.mlx_lock():
            entered.set()

    with caplog.at_level(logging.INFO), mlx_worker.mlx_lock():
        thread = threading.Thread(target=take)
        thread.start()
        assert not entered.wait(0.3)
    thread.join(timeout=10)
    assert entered.is_set()
    assert "waiting for another MLX worker (ASR or diarization)" in caplog.text


def test_the_memory_advice_names_a_lighter_diarizer_only_when_there_is_one() -> None:
    assert runner._smaller(registry.get("d-nemotron-bf16").spec) == "use d-nemotron-8bit"
    assert runner._smaller(registry.get("d-nemotron-8bit").spec) == "split the recording into shorter files"


@pytest.mark.parametrize(
    ("trigger", "fix"),
    [
        ("footprint", "split the recording into shorter files, or raise asr.memory_cap_gib"),
        ("available", "close memory-heavy apps"),
    ],
)
def test_a_memory_kill_becomes_an_engine_memory_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trigger: str, fix: str
) -> None:
    def fake_run(argv: list[str], **_: object) -> proc.ProcResult:
        raise ProcMemoryExceeded(
            "python3",
            trigger,  # type: ignore[arg-type]
            footprint_bytes=9 * GIB,
            cap_bytes=8 * GIB,
            available_percent=40 if trigger == "footprint" else 6,
            floor_percent=10,
        )

    monkeypatch.setattr(proc, "run", fake_run)
    with pytest.raises(EngineMemoryError) as caught:
        runner.run_worker(make_request(tmp_path, 2.0))
    assert caught.value.trigger == trigger
    assert fix in caught.value.fix
    assert "finished files are kept" in caught.value.fix
    assert caught.value.detail.startswith("d-fake worker stopped to protect the machine")
    assert isinstance(caught.value.__cause__, ProcMemoryExceeded)


def test_a_real_worker_over_its_cap_is_killed_by_the_watch(tmp_path: Path) -> None:
    request = make_request(tmp_path, 2.0, sleep_s=5.0).model_copy(update={"memory_cap_bytes": 1})
    with pytest.raises(EngineMemoryError) as caught:
        runner.run_worker(request)
    assert caught.value.trigger == "footprint"
    assert not turns_path(tmp_path / "out", "f0").exists()


def test_the_parent_imports_neither_mlx_nor_numpy_nor_an_adapter() -> None:
    code = (
        "import sys, ytverbatim.diarize.runner, ytverbatim.diarize.registry, ytverbatim.diarize.attribution\n"
        "heavy = {'mlx', 'mlx_audio', 'numpy'}\n"
        "loaded = {name.split('.')[0] for name in sys.modules} & heavy\n"
        "loaded |= {name for name in sys.modules if name == 'ytverbatim.diarize.nemotron'}\n"
        "print(sorted(loaded))\n"
        "sys.exit(bool(loaded))"
    )
    result = subprocess.run([sys.executable, "-P", "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
