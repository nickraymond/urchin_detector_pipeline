"""S4 bite 4 (D36): resource limits: the settings file and form, each job's recorded limits,
nice and the thread cap on the worker, the pause between items, the memory guard, and the
CPU-fallback warning. Fake job kind and fake SAM only."""

import io
import json
import os
import time
import warnings

import pytest

from app.jobctl import JobControl
from app.jobctl import limits
from app.jobctl import worker as jw
from app.sam import worker as sam_worker
from app.stages import label
from tests.test_app_skeleton import client_for, write_config
from tests.test_jobctl import fake_kind, params, wait_for, wait_state, writes


@pytest.fixture
def jc(tmp_path):
    c = JobControl(tmp_path / "jobs")
    c.register(fake_kind())
    yield c
    c.close()


def good(**kw):
    return {**limits.defaults(), **kw}


def test_defaults_check_and_round_trip(tmp_path):
    d = limits.defaults()
    assert d["threads"] == max(1, (os.cpu_count() or 2) // 2) and d["low_priority"]
    assert d["device"] == "auto" and d["pause_s"] == 0 and d["batch"] == 1
    for bad, msg in [({"device": "tpu"}, "Device"), ({"threads": 0}, "CPU threads"),
                     ({"threads": "x"}, "whole number"), ({"pause_s": -1}, "Pause"),
                     ({"memory_guard_gb": 0}, "memory threshold")]:
        with pytest.raises(limits.LimitsError, match=msg):
            limits.check(good(**bad))
    root = tmp_path / "jobs"
    root.mkdir()
    assert limits.load(root) == d                               # no file: the defaults
    limits.save(root, good(threads="2", low_priority="", pause_s="1.5", batch=8))
    got = limits.load(root)
    assert (got["threads"], got["low_priority"], got["pause_s"], got["batch"]) == (2, False, 1.5, 1)
    limits.path(root).write_text("{not json")
    assert limits.load(root) == d                               # unreadable: the defaults


def test_jobs_page_settings_form(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    client = client_for(write_config(tmp_path, data))
    page = client.get("/jobs").text
    assert 'action="/jobs/settings"' in page and "free now:" in page and "fixed at 1" in page
    r = client.post("/jobs/settings", data={"device": "cpu", "threads": "3", "pause_s": "2",
                                            "memory_guard": "1", "memory_guard_gb": "6"})
    assert "Saved." in r.text
    saved = json.loads((data / "jobs" / "settings.json").read_text())
    assert saved == {"device": "cpu", "threads": 3, "low_priority": False, "pause_s": 2.0,
                     "memory_guard": True, "memory_guard_gb": 6.0, "batch": 1}
    r = client.post("/jobs/settings", data={"device": "cpu", "threads": "999",
                                            "pause_s": "0", "memory_guard_gb": "6"})
    assert "CPU threads must be from 1" in r.text
    assert json.loads((data / "jobs" / "settings.json").read_text())["threads"] == 3   # unchanged


def test_a_job_runs_with_and_records_its_limits(jc, tmp_path):
    limits.save(jc.root, good(threads=2, low_priority=True, pause_s=0.3))
    jid = jc.submit("fake", params(tmp_path, n=3))
    job = wait_state(jc, jid, "done")
    env = json.loads((tmp_path / "out" / "env.json").read_text())
    assert env["nice"] == os.nice(0) + 10                        # nice -n 10
    assert (env["omp"], env["mkl"]) == ("2", "2") and env["limits"]["threads"] == 2
    assert job["limits"]["threads"] == 2 and job["limits"]["pause_s"] == 0.3
    stamps = [float(l.split()[2]) for l in (tmp_path / "out" / "writes.log").read_text().splitlines()]
    assert all(b - a >= 0.3 for a, b in zip(stamps, stamps[1:]))   # the pause between items
    # a job started after a change uses the new limits; the old row keeps its own
    limits.save(jc.root, good(threads=1, low_priority=False))
    (tmp_path / "out" / "writes.log").unlink()
    for f in (tmp_path / "out").glob("item_*"):
        f.unlink()
    jid2 = jc.submit("fake", params(tmp_path, n=1))
    wait_state(jc, jid2, "done")
    env = json.loads((tmp_path / "out" / "env.json").read_text())
    assert env["nice"] == os.nice(0) and env["omp"] == "1"
    assert jc.status(jid)["limits"]["threads"] == 2


def test_memory_guard_pauses_the_job_and_says_why(jc, tmp_path):
    limits.save(jc.root, good(memory_guard=True, memory_guard_gb=4))
    jc.runner.free_gb = lambda: 1.25
    jc.runner.guard_every_s = 0.1
    jid = jc.submit("fake", params(tmp_path, n=40, delay=0.1))
    job = wait_state(jc, jid, "paused")
    assert job["stop_reason"] == "Paused by the memory guard: 1.2 GB free (limit 4 GB)"
    assert len(writes(tmp_path)) < 40
    assert "memory guard" in (jc.root / f"{jid}.log").read_text()


def test_a_guard_pause_resumes_by_itself_once_memory_stays_free(jc, tmp_path):
    """D36c: twice the limit free for recover_s -> resumed; the summary counts both."""
    limits.save(jc.root, good(memory_guard=True, memory_guard_gb=2))
    free = {"gb": 1.0}
    jc.runner.free_gb = lambda: free["gb"]
    jc.runner.guard_every_s = 0.1
    jc.recover_s = 0.3
    jid = jc.submit("fake", params(tmp_path, n=6, delay=0.1))
    wait_state(jc, jid, "paused")
    assert jc.auto_resume() == []                  # 1 GB: still short
    free["gb"] = 3.9                               # above the limit, below twice it
    assert jc.auto_resume() == []
    free["gb"] = 4.0
    assert jc.auto_resume() == []                  # the clock starts now
    time.sleep(0.4)
    limits.save(jc.root, good(memory_guard=False))   # so it can finish
    assert jc.auto_resume() == [jid]
    job = wait_state(jc, jid, "done")
    assert job["summary"]["memory_pauses"] == 1 and job["summary"]["auto_resumes"] == 1
    assert "memory recovered" in (jc.root / f"{jid}.log").read_text()


def test_a_pause_by_hand_never_resumes_by_itself(jc, tmp_path):
    jc.runner.free_gb = lambda: 100.0
    jc.recover_s = 0
    jid = jc.submit("fake", params(tmp_path, n=20, delay=0.1))
    wait_for(lambda: writes(tmp_path))
    jc.pause(jid)
    wait_state(jc, jid, "paused")
    assert jc.auto_resume() == [] and jc.status(jid)["state"] == "paused"


def test_memory_guard_off_does_not_pause(jc, tmp_path):
    limits.save(jc.root, good(memory_guard=False))
    jc.runner.free_gb = lambda: 0.1
    jc.runner.guard_every_s = 0.05
    jid = jc.submit("fake", params(tmp_path, n=4, delay=0.1))
    assert wait_state(jc, jid, "done")["stop_reason"] == "finished"


def test_warnings_are_kept_once_and_shown(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=1))
    wait_state(jc, jid, "done")
    for _ in range(2):
        jc.runner._on_event(jid, {"event": "warning", "message": "Apple GPU: x runs on the CPU"})
    assert jc.status(jid)["warnings"] == ["Apple GPU: x runs on the CPU"]


def fallback_events(monkeypatch, ops):
    buf = io.StringIO()
    monkeypatch.setattr(jw, "OUT", buf)
    before = warnings.showwarning
    try:
        sam_worker.report_cpu_fallback()
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            for op in ops:
                warnings.warn(f"The operator '{op}' is not currently supported on the MPS "
                              "backend and will fall back to run on the CPU.", UserWarning)
    finally:
        warnings.showwarning = before
    return [json.loads(l[3:]) for l in buf.getvalue().splitlines()]


def test_cpu_fallback_becomes_one_warning_per_op(monkeypatch):
    assert fallback_events(monkeypatch, ["aten::foo"] * 3) == [
        {"event": "warning",
         "message": "One step (aten::foo) runs on the CPU (slower); SAM itself runs on the "
                    "Apple GPU"}]


def test_a_harmless_fallback_goes_to_the_log_only(monkeypatch, jc, tmp_path):
    """D36b: aten::_assert_async is a check with no maths: a note in the log, no warning."""
    events = fallback_events(monkeypatch, ["aten::_assert_async"] * 2)
    assert [e["event"] for e in events] == ["note"]
    jid = jc.submit("fake", params(tmp_path, n=1))
    wait_state(jc, jid, "done")
    jc.runner._on_event(jid, events[0])
    assert not jc.status(jid)["warnings"]
    assert "note: Apple GPU: aten::_assert_async" in (jc.root / f"{jid}.log").read_text()


def test_sam_device_comes_from_the_limits():
    assert label.pick_device({"device": "cpu"}) == "cpu"
    assert label.pick_device({"device": "mps"}) == "mps"
