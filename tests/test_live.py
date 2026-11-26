"""S4 bite 6b (Nick's S4 test): job boxes on every page that update in place (/live/...
fragments, live.js, no page reloads), Resume easy to find (buttons next to the state, a banner
for a waiting job), a progress bar with a running sign and "done by", a memory-guard pause that
says why, and the project page's stage rows showing their job. Fake SAM, synthetic frames."""

from datetime import datetime

from app import config
from app.jobctl.runner import GUARD_REASON
from app.stages import label
from tests.test_clips import wait_for
from tests.test_preview import make_project
from tests.test_run import SLOW, status


def jc_of(client, tmp_path):
    return client.app.state.jobctl(config.load(tmp_path / "config.local.toml"))


def test_a_running_job_shows_bar_sign_and_time_left_everywhere(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client, folder = make_project(tmp_path, n=8)
    client.post("/p/p/labeling/run", data={})
    assert wait_for(lambda: status(client).get("eta_s"), 60)
    panel = client.get("/p/p/live/sam?back=/p/p/labeling").text
    assert 'data-state="running"' in panel and 'class="running-sign"' in panel
    assert '<progress max="8"' in panel and "left, done by " in panel
    assert "/pause" in panel and "/cancel" in panel and "Resume" not in panel
    # the project page's SAM labeling row shows the same job, compact and live
    proj = client.get("/p/p").text
    assert 'data-live="/p/p/live/sam?compact=1' in proj and 'class="running-sign"' in proj
    assert 'data-live="/p/p/live/extract?compact=1' in proj
    # the Jobs page: the table is a live fragment; buttons sit in the State cell
    jobs = client.get("/jobs").text
    assert 'data-live="/live/jobs"' in jobs and "location.reload" not in jobs
    table = client.get("/live/jobs").text
    state_cell = table.split('<b class="job-state">running</b>')[1].split("</td>")[0]
    assert "/pause" in state_cell and "/cancel" in state_cell
    for page in ("/p/p/labeling", "/p/p/extract", "/", "/jobs", "/p/p"):
        assert "live.js" in client.get(page).text, page
    jid = status(client)["id"]
    client.post(f"/jobs/{jid}/cancel", data={"back": "/jobs"})
    assert wait_for(lambda: status(client)["state"] == "cancelled", 30)


def test_a_paused_job_stands_out_with_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client, folder = make_project(tmp_path, n=8)
    client.post("/p/p/labeling/run", data={})
    assert wait_for(lambda: status(client).get("done", 0) >= 1, 60)
    jid = status(client)["id"]
    client.post(f"/jobs/{jid}/pause", data={"back": "/jobs"})
    pausing = client.get("/live/banner").text
    if 'data-state="pausing"' in pausing:          # item 4: say what it waits for
        assert "Resume appears in" in pausing
    assert wait_for(lambda: status(client)["state"] == "paused", 60)
    done = status(client)["done"]
    for page in ("/p/p/labeling", "/jobs", "/", "/p/p"):
        text = client.get(page).text
        assert f"<b>Job {jid}</b> (SAM labeling," in text, page
        assert f"is <b>paused</b> at frame {done} of 8" in text, page
        assert f'action="/jobs/{jid}/resume"' in text and 'class="small primary">Resume' in text
    assert f"<b>Job {jid}</b>" not in client.get("/live/banner?project=other").text


def test_a_memory_guard_pause_says_why_and_that_it_resumes(tmp_path):
    client, folder = make_project(tmp_path, n=3)
    jc = jc_of(client, tmp_path)
    jid = jc.submit("sam", {"folder": str(folder)}, project=str(folder),
                    start_at=datetime(2099, 1, 1))          # never starts on its own
    reason = f"{GUARD_REASON}: 1.8 GB free (limit 2 GB)"
    jc.store.update(jid, state="paused", stop_reason=reason, message=reason, done=5, total=21,
                    limits={"memory_guard": True, "memory_guard_gb": 2})
    panel = client.get("/p/p/live/sam").text
    assert reason in panel and "Close other apps, or lower" in panel
    assert "resumes by itself once 4 GB has stayed free for 1 min" in panel
    banner = client.get("/live/banner").text
    assert "is <b>paused</b> at frame 5 of 21: " + reason in banner
    assert "resumes by itself once 4 GB" in client.get("/live/jobs").text
