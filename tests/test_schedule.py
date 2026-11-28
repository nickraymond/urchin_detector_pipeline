"""S4 bite 5a (D37): Start at / Stop by. Clock-time rules, scheduled jobs firing, Start now,
missed starts after a restart, Stop by pausing a running or waiting job, and the pages.
Fake job kind and fake SAM only."""

from datetime import datetime, timedelta

import pytest

from app.jobctl import JobControl, JobError, schedule
from app.stages import label
from tests.test_clips import wait_for
from tests.test_jobctl import assert_each_once, fake_kind, params, wait_state, writes
from tests.test_preview import make_project
from tests.test_run import status
from tests.test_sam_runner import FAKE


@pytest.fixture
def jc(tmp_path):
    c = JobControl(tmp_path / "jobs")
    c.register(fake_kind())
    yield c
    c.close()


def test_clock_times_mean_the_next_time_they_come():
    now = datetime(2026, 11, 25, 19, 30)
    assert schedule.plan("02:00", "06:00", now) == (datetime(2026, 11, 26, 2, 0),
                                                    datetime(2026, 11, 26, 6, 0))
    assert schedule.plan("22:00", "01:00", now) == (datetime(2026, 11, 25, 22, 0),
                                                    datetime(2026, 11, 26, 1, 0))
    assert schedule.plan("", "20:00", now) == (None, datetime(2026, 11, 25, 20, 0))
    assert schedule.plan("19:30", "", now)[0] == datetime(2026, 11, 26, 19, 30)  # now = next day
    assert schedule.plan("", "", now) == (None, None)
    with pytest.raises(schedule.ScheduleError, match="HH:MM"):
        schedule.plan("25:99", "", now)
    assert schedule.show("2026-11-26T02:00:00", now) == "Thu 2:00 AM"
    assert schedule.show("2026-11-25T22:05:00", now) == "10:05 PM"


def test_a_scheduled_job_waits_then_runs(jc, tmp_path):
    at = datetime.now().replace(microsecond=0) + timedelta(seconds=2)
    jid = jc.submit("fake", params(tmp_path, n=2), start_at=at)
    job = jc.status(jid)
    assert job["state"] == "scheduled" and job["message"].startswith("starts at")
    assert not writes(tmp_path)
    wait_state(jc, jid, "done", timeout=20)
    first = float((tmp_path / "out" / "writes.log").read_text().split()[2])
    assert first >= at.timestamp()


def test_start_now_and_cancel_a_scheduled_job(jc, tmp_path):
    later = datetime.now() + timedelta(hours=5)
    a = jc.submit("fake", params(tmp_path / "a", n=1), start_at=later)
    b = jc.submit("fake", params(tmp_path / "b", n=1), start_at=later)
    jc.start_now(a)
    wait_state(jc, a, "done")
    jc.cancel(b)
    assert jc.status(b)["state"] == "cancelled"
    with pytest.raises(JobError, match="only a scheduled job"):
        jc.start_now(a)
    with pytest.raises(JobError, match="Stop by must come after"):
        jc.submit("fake", {}, start_at=later, stop_by=later)


def test_after_a_restart_a_passed_start_runs_unless_stop_by_passed_too(tmp_path):
    root = tmp_path / "jobs"
    c = JobControl(root, start=False)
    c.register(fake_kind())
    past, future = datetime.now() - timedelta(hours=2), datetime.now() + timedelta(hours=2)
    late = c.submit("fake", params(tmp_path / "late", n=1), start_at=past, stop_by=future)
    gone = c.submit("fake", params(tmp_path / "gone", n=1), start_at=past,
                    stop_by=past + timedelta(minutes=30))
    c.close()                                            # the app was closed at the start time
    c = JobControl(root, start=False)                    # as the app does: kinds first, then
    c.register(fake_kind())                              # start (a due job must find its kind)
    c.start()
    try:
        wait_state(c, late, "done")
        job = c.status(gone)
        assert job["state"] == "missed" and "passed while the app was not running" in job["stop_reason"]
        assert not writes(tmp_path / "gone")
    finally:
        c.close()


def test_stop_by_pauses_a_running_job_and_resume_carries_on(jc, tmp_path):
    stop = datetime.now().replace(microsecond=0) + timedelta(seconds=2)
    jid = jc.submit("fake", params(tmp_path, n=30, delay=0.2), stop_by=stop)
    job = wait_state(jc, jid, "paused", timeout=20)
    assert job["stop_reason"].startswith("stop by ")
    assert 1 <= len(writes(tmp_path)) < 30
    jc.resume(jid)                                       # Resume later: Stop by no longer applies
    assert jc.status(jid)["stop_by"] is None
    wait_state(jc, jid, "done", timeout=30)
    assert_each_once(tmp_path, 30)


def test_a_pause_and_resume_before_the_stop_by_keeps_it(jc, tmp_path):
    """QE PR 37: Resume clears a Stop by only once it has passed."""
    stop = datetime.now().replace(microsecond=0) + timedelta(seconds=4)
    jid = jc.submit("fake", params(tmp_path, n=60, delay=0.2), stop_by=stop)
    wait_for(lambda: len(writes(tmp_path)) >= 2)
    jc.pause(jid)
    wait_state(jc, jid, "paused")
    jc.resume(jid)
    assert jc.status(jid)["stop_by"] == stop.isoformat()     # still holds
    job = wait_state(jc, jid, "paused", timeout=30)
    assert job["stop_reason"].startswith("stop by ") and len(writes(tmp_path)) < 60


def test_a_stop_by_that_passes_while_queued_leaves_it_paused(jc, tmp_path):
    first = jc.submit("fake", params(tmp_path / "a", n=10, delay=0.3))
    stop = datetime.now() + timedelta(seconds=1)
    second = jc.submit("fake", params(tmp_path / "b", n=2), stop_by=stop)
    job = wait_state(jc, second, "paused", timeout=30)
    assert "before it could start" in job["stop_reason"] and not writes(tmp_path / "b")
    wait_state(jc, first, "done", timeout=30)


def test_run_at_a_clock_time_through_the_page(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path, n=3)
    later = (datetime.now() + timedelta(hours=3)).strftime("%H:%M")
    r = client.post("/p/p/labeling/run", data={"start": "25:99"})
    assert "is not a clock time" in r.text
    client.post("/p/p/labeling/run", data={"start": later})
    st = status(client)
    assert st["state"] == "scheduled" and st["start_show"]
    page = client.get("/p/p/labeling").text
    assert f"starts at <b>{st['start_show']}</b>" in page and f"/jobs/{st['id']}/start-now" in page
    r = client.post("/p/p/labeling/run", data={})
    assert "is scheduled" in r.text                      # not a second run
    assert "Start now" in client.get("/jobs").text
    client.post(f"/jobs/{st['id']}/start-now", data={"back": "/p/p/labeling"})
    assert wait_for(lambda: status(client)["state"] == "done", 60)


# S8 Demo fixes (2026-11-27): Stop by can be set, changed or cleared on a job that has started.
def test_a_stop_by_set_on_a_running_job_pauses_it(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=60, delay=0.2))
    wait_for(lambda: len(writes(tmp_path)) >= 2)
    assert jc.status(jid)["state"] == "running"
    jc.set_stop_by(jid, datetime.now().replace(microsecond=0) + timedelta(seconds=2))
    job = wait_state(jc, jid, "paused", timeout=30)
    assert job["stop_reason"].startswith("stop by ") and len(writes(tmp_path)) < 60


def test_a_stop_by_cleared_on_a_running_job_no_longer_pauses_it(jc, tmp_path):
    stop = datetime.now().replace(microsecond=0) + timedelta(seconds=3)
    jid = jc.submit("fake", params(tmp_path, n=25, delay=0.2), stop_by=stop)
    wait_for(lambda: len(writes(tmp_path)) >= 1)
    jc.set_stop_by(jid, None)
    assert jc.status(jid)["stop_by"] is None
    wait_state(jc, jid, "done", timeout=30)
    assert_each_once(tmp_path, 25)


def test_stop_by_must_come_later_and_only_on_a_job_not_ended(jc, tmp_path):
    jid = jc.submit("fake", params(tmp_path, n=1))
    wait_state(jc, jid, "done", timeout=30)
    with pytest.raises(JobError, match="only to a job"):
        jc.set_stop_by(jid, datetime.now() + timedelta(hours=1))
    later = datetime.now() + timedelta(hours=2)
    sched = jc.submit("fake", params(tmp_path / "s", n=1), start_at=later)
    with pytest.raises(JobError, match="after Start at"):
        jc.set_stop_by(sched, later - timedelta(minutes=5))
    jc.cancel(sched)
    busy = jc.submit("fake", params(tmp_path / "b", n=60, delay=0.2))
    wait_for(lambda: len(writes(tmp_path / "b")) >= 1)
    with pytest.raises(JobError, match="still to come"):
        jc.set_stop_by(busy, datetime.now() - timedelta(seconds=1))
    jc.cancel(busy)


def test_stop_by_through_the_page_on_a_running_job(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path, n=3)
    later = (datetime.now() + timedelta(hours=3)).strftime("%H:%M")
    client.post("/p/p/labeling/run", data={"start": later})       # scheduled: it has a panel
    st = status(client)
    page = client.get("/p/p/labeling").text
    assert f'action="/jobs/{st["id"]}/stop-by"' in page and "none set; at that time" in page
    r = client.post(f"/jobs/{st['id']}/stop-by", data={"stop": "7:7x", "back": "/p/p/labeling"})
    assert "is not a clock time" in r.text
    stop = (datetime.now() + timedelta(hours=5)).strftime("%H:%M")
    client.post(f"/jobs/{st['id']}/stop-by", data={"stop": stop, "back": "/p/p/labeling"})
    assert status(client)["stop_by"][11:16] == stop
    assert f'value="{stop}"' in client.get("/p/p/labeling").text
    client.post(f"/jobs/{st['id']}/stop-by", data={"stop": "", "back": "/p/p/labeling"})
    assert status(client)["stop_by"] is None
    client.post(f"/jobs/{st['id']}/cancel", data={"back": "/p/p/labeling"})


def test_cancel_asks_first_and_stands_apart_from_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path, n=3)
    later = (datetime.now() + timedelta(hours=3)).strftime("%H:%M")
    client.post("/p/p/labeling/run", data={"start": later})
    page = client.get("/p/p/labeling").text
    assert 'class="inline job-cancel" data-confirm="Cancel job' in page
    js = client.get("/static/live.js").text
    assert "form[data-confirm]" in js and "window.confirm" in js
    assert ".job-actions .job-cancel { margin-left" in client.get("/static/style.css").text
    client.post(f"/jobs/{status(client)['id']}/cancel", data={"back": "/p/p/labeling"})
