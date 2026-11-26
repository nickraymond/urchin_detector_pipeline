"""S4 bite 5b (D38): keep the laptop awake while a keep-awake job waits or runs (with a fake
command: tests never run the real keep-awake tool or change power settings), the battery
warning, one summary per job across Pause / Resume, the dashboard's morning summary and the
log page. Fake job kind and fake SAM only."""

import os
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

from app.jobctl import JobControl, awake
from app.stages import label
from app.web import server
from tests.test_clips import wait_for
from tests.test_jobctl import fake_kind, params, wait_state, writes
from tests.test_preview import make_project
from tests.test_run import run_and_wait, status
from tests.test_sam_runner import FAKE

FAKE_AWAKE = [sys.executable, "-c", "import time; time.sleep(600)"]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                          text=True).stdout.strip()[:1] not in ("", "Z")


def test_the_real_command_waits_for_the_app_and_prevents_idle_and_system_sleep():
    cmd = awake.default_cmd()
    assert cmd[1:] == ["-i", "-s", "-w", str(os.getpid())]


def test_one_process_started_and_only_it_ended():
    other = subprocess.Popen(FAKE_AWAKE)          # someone else's process with the same command
    a = awake.Awake(FAKE_AWAKE)
    try:
        a.update(True)
        pid = a.pid()
        assert pid and alive(pid)
        a.update(True)
        assert a.pid() == pid                      # not a second one
        a.update(False)
        assert a.pid() is None and wait_for(lambda: not alive(pid), 5)
        assert alive(other.pid)                    # ended by its own PID only (D38)
    finally:
        other.kill()
        other.wait()
        a.stop()


def test_a_missing_command_is_reported_not_raised():
    a = awake.Awake(["no-such-keep-awake-command-xyz"])
    a.update(True)
    assert a.pid() is None and "cannot keep the laptop awake" in a.error


def test_battery_reading(monkeypatch):
    def fake(out):
        return lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout=out, stderr="")
    monkeypatch.setattr(awake.subprocess, "run", fake("Now drawing from 'Battery Power'\n"))
    assert awake.on_battery() is True
    monkeypatch.setattr(awake.subprocess, "run", fake("Now drawing from 'AC Power'\n"))
    assert awake.on_battery() is False
    monkeypatch.setattr(awake.subprocess, "run", fake(""))
    assert awake.on_battery() is None


@pytest.fixture
def jc(tmp_path):
    c = JobControl(tmp_path / "jobs", awake_cmd=FAKE_AWAKE)
    c.register(fake_kind())
    yield c
    c.close()


def test_awake_while_a_keep_awake_job_waits_or_runs(jc, tmp_path):
    later = datetime.now() + timedelta(hours=3)
    plain = jc.submit("fake", params(tmp_path / "a", n=1))
    wait_state(jc, plain, "done")
    assert jc.awake.pid() is None                  # no keep-awake job: no process
    jid = jc.submit("fake", params(tmp_path / "b", n=1), start_at=later, keep_awake=True)
    pid = jc.awake.pid()
    assert pid and jc.status(jid)["keep_awake"]   # on while it waits for its start
    jc.cancel(jid)
    assert wait_for(lambda: jc.awake.pid() is None, 6) and wait_for(lambda: not alive(pid), 5)
    jid = jc.submit("fake", params(tmp_path / "c", n=4, delay=0.3), keep_awake=True)
    assert jc.awake.pid()
    wait_state(jc, jid, "done")
    assert wait_for(lambda: jc.awake.pid() is None, 6)   # released when no such job is left


def test_one_summary_per_job_across_pause_and_resume(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=8, delay=0.2))
    wait_for(lambda: len(writes(tmp_path)) >= 2)
    jc.pause(jid)
    job = wait_state(jc, jid, "paused")
    s = job["summary"]
    assert s["state"] == "paused" and s["stop_reason"] == "paused"
    assert s["items"] == len(writes(tmp_path)) == s["counts"]["thing"]
    assert jc.last_summary()["id"] == jid          # a paused job's summary counts as last
    jc.resume(jid)
    wait_state(jc, jid, "done")
    # the runner writes the summary just after the state (a race seen in the suite, PR 45)
    assert wait_for(lambda: (jc.status(jid)["summary"] or {}).get("state") == "done", 10)
    job = jc.status(jid)
    s = job["summary"]
    assert (s["items"], s["counts"], s["errors"], s["state"]) == (8, {"thing": 8}, 0, "done")
    assert s["stop_reason"] == "finished" and s["seconds"] >= 1
    assert '"items": 8' in (jc.root / f"{jid}.log").read_text()


def test_keep_awake_from_the_run_form_morning_summary_and_log(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "AWAKE_CMD", FAKE_AWAKE)
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path, n=3)
    page = client.get("/p/p/labeling").text
    assert 'name="keep_awake"' in page and "lid open" in page
    later = (datetime.now() + timedelta(hours=3)).strftime("%H:%M")
    client.post("/p/p/labeling/run", data={"start": later, "keep_awake": "1"})
    st = status(client)
    assert st["state"] == "scheduled" and st["keep_awake"]
    jobs = client.get("/jobs").text
    assert "Keeping the laptop awake (process" in jobs and "keep awake" in jobs
    client.post(f"/jobs/{st['id']}/start-now", data={"back": "/jobs"})
    assert wait_for(lambda: status(client)["state"] == "done", 60)
    home = client.get("/").text
    assert "Last job:" in home and "3 frames done" in home
    found = home.split("Found:")[1]
    assert found.index("urchin") < found.index("sea star")    # urchins first
    log = client.get(f"/jobs/{st['id']}/log")
    assert log.status_code == 200 and "summary" in log.text
    assert client.get("/jobs/999/log").status_code == 404
    jc = client.app.state.jobctl
    from app import config
    assert wait_for(lambda: jc(config.load(tmp_path / "config.local.toml")).awake.pid() is None, 6)
