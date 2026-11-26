"""S4 bite 1a: the job module core (D32, D33, D34), end to end with real worker processes of a
fake job kind (tests/fake_job.py). No SAM, no frames."""

import ast
import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from app.jobctl import JobControl, JobError, Kind
from app.jobctl import worker as jw

REPO = Path(__file__).resolve().parents[1]


def fake_kind() -> Kind:
    def progress(params):
        out = Path(params["out"])
        return len(list(out.glob("item_*.txt"))), params["n"]
    return Kind("fake", module="tests.fake_job", progress=progress, cwd=REPO)


@pytest.fixture
def jc(tmp_path):
    c = JobControl(tmp_path / "jobs")
    c.register(fake_kind())
    yield c
    c.close()


def params(tmp_path, n=6, **kw):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return {"out": str(out), "n": n, **kw}


def wait_for(fn, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("timed out")


def wait_state(jc, jid, *states, timeout=30.0):
    return wait_for(lambda: jc.status(jid)["state"] in states and jc.status(jid), timeout)


def writes(tmp_path) -> list[int]:
    log = tmp_path / "out" / "writes.log"
    return [int(l.split()[0]) for l in log.read_text().splitlines()] if log.exists() else []


def assert_each_once(tmp_path, n):
    w = writes(tmp_path)
    assert sorted(w) == list(range(1, n + 1)), f"items written: {w}"
    assert sorted(p.name for p in (tmp_path / "out").glob("item_*.txt")) == \
        sorted(f"item_{i}.txt" for i in range(1, n + 1))


def test_jobctl_imports_nothing_labeling_specific():
    """D4 / D32: app.jobctl stands alone."""
    for f in (REPO / "app" / "jobctl").glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert not name.startswith("app.") or name.startswith("app.jobctl"), \
                    f"{f.name} imports {name}"
                assert name not in ("app",), f"{f.name} imports app"


def test_runs_to_done_with_record_and_log(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=4), project="p1")
    job = wait_state(jc, jid, "done")
    assert (job["done"], job["total"], job["project"]) == (4, 4, "p1")
    assert job["started"] and job["ended"] and job["stop_reason"] == "finished"
    assert_each_once(tmp_path, 4)
    log = (tmp_path / "jobs" / f"{jid}.log").read_text()
    assert "start: kind fake" in log and "item " in log and "done in" in log


def test_unknown_kind_refused(jc):
    with pytest.raises(JobError, match="No job kind"):
        jc.submit("nope", {})


def test_pause_then_resume_repeats_and_skips_nothing(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=8, delay=0.2))
    wait_for(lambda: len(writes(tmp_path)) >= 2)
    jc.pause(jid)
    job = wait_state(jc, jid, "paused")
    assert job["stop_reason"] == "paused"
    done_at_pause = len(writes(tmp_path))
    assert 2 <= done_at_pause < 8
    time.sleep(0.5)
    assert len(writes(tmp_path)) == done_at_pause          # the worker really stopped
    jc.resume(jid)
    wait_state(jc, jid, "done")
    assert_each_once(tmp_path, 8)
    pids = {l.split()[1] for l in (tmp_path / "out" / "writes.log").read_text().splitlines()}
    assert len(pids) == 2                                   # resume = a new worker process


def test_cancel_running_keeps_finished_items(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=8, delay=0.2))
    wait_for(lambda: len(writes(tmp_path)) >= 1)
    jc.cancel(jid)
    job = wait_state(jc, jid, "cancelled")
    assert job["ended"] and job["stop_reason"] == "cancelled"
    kept = writes(tmp_path)
    assert 1 <= len(kept) < 8 and sorted(kept) == list(range(1, len(kept) + 1))
    with pytest.raises(JobError, match="only a paused or interrupted"):
        jc.resume(jid)
    with pytest.raises(JobError, match="already cancelled"):
        jc.cancel(jid)


def test_one_job_at_a_time_and_queued_pause_cancel(jc, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = jc.submit("fake", params(a, n=3, delay=0.2))
    second = jc.submit("fake", params(b, n=3, delay=0.2))
    third = jc.submit("fake", params(tmp_path / "c", n=2))
    assert jc.status(second)["state"] == "queued"
    jc.pause(second)
    jc.cancel(third)
    wait_state(jc, first, "done")
    time.sleep(0.5)
    assert jc.status(second)["state"] == "paused" and not writes(b)
    assert jc.status(third)["state"] == "cancelled"
    jc.resume(second)
    wait_state(jc, second, "done")
    end_a = max(float(l.split()[2]) for l in (a / "out" / "writes.log").read_text().splitlines())
    start_b = min(float(l.split()[2]) for l in (b / "out" / "writes.log").read_text().splitlines())
    assert start_b > end_a


def test_worker_crash_is_an_error_with_the_log(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=5, crash_at=3))
    job = wait_state(jc, jid, "error")
    assert "exit code 3" in job["stop_reason"] and f"{jid}.log" in job["stop_reason"]
    assert "fake crash at item 3" in (tmp_path / "jobs" / f"{jid}.log").read_text()
    assert writes(tmp_path) == [1, 2]


def test_worker_that_ignores_pause_is_ended(jc, tmp_path):
    jc.runner.pause_timeout_s = 0.5
    jid = jc.submit("fake", params(tmp_path, n=50, delay=0.2, ignore_pause=True))
    wait_for(lambda: len(writes(tmp_path)) >= 1)
    jc.pause(jid)
    job = wait_state(jc, jid, "paused")
    assert "before saying it paused" in job["stop_reason"]
    assert "did not pause within" in (tmp_path / "jobs" / f"{jid}.log").read_text()


APP = textwrap.dedent("""
    import sys, time
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    from app.jobctl import JobControl
    from tests.test_jobctl import fake_kind
    jc = JobControl(Path({root!r}))
    jc.register(fake_kind())
    print(jc.submit("fake", {params!r}), flush=True)
    time.sleep(600)
""")


def test_app_killed_mid_job_shows_interrupted_and_resumes(tmp_path):
    """The app process dies with kill -9; its worker reads end of file on stdin, finishes the
    item in progress and exits. The next start shows the job interrupted; Resume finishes it
    with no item repeated or skipped."""
    root = tmp_path / "jobs"
    p = params(tmp_path, n=10, delay=0.3)
    app = subprocess.Popen([sys.executable, "-c", APP.format(repo=str(REPO), root=str(root),
                                                               params=p)],
                           stdout=subprocess.PIPE, text=True, cwd=REPO)
    jid = int(app.stdout.readline())
    wait_for(lambda: len(writes(tmp_path)) >= 2)
    app.send_signal(signal.SIGKILL)
    app.wait()
    time.sleep(1.0)   # the orphaned worker finishes its item and exits
    left = len(writes(tmp_path))
    time.sleep(0.8)
    assert len(writes(tmp_path)) == left and left < 10   # it did stop

    jc = JobControl(root)
    jc.register(fake_kind())
    try:
        assert jc.interrupted == [jid]
        job = jc.status(jid)
        assert job["state"] == "interrupted" and "app quit" in job["stop_reason"]
        log = (root / f"{jid}.log").read_text()
        assert "interrupted: the app quit" in log and "Traceback" not in log
        jc.resume(jid)
        job = wait_state(jc, jid, "done")
        assert (job["done"], job["total"]) == (10, 10)
        assert_each_once(tmp_path, 10)
    finally:
        jc.close()


def test_worker_control_pause_line_and_end_of_file():
    ctl = jw.listen(io.StringIO("hello\npause\n"))
    assert wait_for(ctl.pause_requested, 2)
    ctl = jw.listen(io.StringIO(""))          # end of file = the app is gone
    assert wait_for(ctl.pause_requested, 2)
    r, w = os.pipe()
    ctl = jw.listen(os.fdopen(r))
    t = time.monotonic()
    assert ctl.rest(0.2) is False and time.monotonic() - t >= 0.2
    os.write(w, b"pause\n")
    assert ctl.rest(5) is True
    os.close(w)


def test_emit_writes_one_json_line(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(jw, "OUT", buf)
    jw.emit("item", i=1, n=2)
    line = buf.getvalue()
    assert line.startswith("@@ ") and json.loads(line[3:]) == {"event": "item", "i": 1, "n": 2}
