"""S3 bite 4: Run on every kept frame through the page, re-run only what changed (SPEC Q7),
the final class map overlay. S4 bite 1b: SAM on the job module, Pause / Resume / Cancel,
interrupted after an app quit. Fake SAM, synthetic frames only."""

import json
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from app.sam import classes, results
from app.stages import label
from tests.test_clips import wait_for
from tests.test_preview import make_project
from tests.fake_sam import run_sam
from tests.test_sam_runner import FAKE

SLOW = "tests.fake_sam:SlowFakePredictor"
REPO = Path(__file__).resolve().parents[1]


def run_and_wait(client):
    assert client.post("/p/p/labeling/run", data={}).status_code == 200
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "done", 60)
    return client.get("/p/p/labeling/status").json()["message"]


def test_run_labels_every_kept_frame_then_only_a_new_class(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path)
    page = client.get("/p/p/labeling").text
    assert "<tr><td>urchin</td><td>4</td></tr>" in page
    assert "Takes: not known yet: press Estimate" in page
    assert run_and_wait(client) == "labeled 4 frames, 0 already current"
    assert all(results.read(folder, f"v_aaaaaa_{i}.png") for i in range(4))
    page = client.get("/p/p/labeling").text
    assert "Nothing to run: every class is current on all 4 frames." in page
    before = results.read(folder, "v_aaaaaa_1.png")["units"]["urchin"]["run_at"]
    # add a class: only it will run, and the others' results stay as they are
    text = (folder / "classes.yaml").read_text().rstrip("\n") + \
        "\n  - {id: 10, name: boulder, kind: area, prompts: [rock]}\n"
    assert client.post("/p/p/labeling/classes", data={"text": text}).status_code == 200
    page = client.get("/p/p/labeling").text
    assert "<tr><td>boulder</td><td>4</td></tr>" in page
    assert "<tr><td>urchin</td><td>unchanged</td></tr>" in page
    assert "<b>4 of 4</b>" in page
    assert run_and_wait(client) == "labeled 4 frames, 0 already current"
    raw = results.read(folder, "v_aaaaaa_1.png")
    assert raw["units"]["urchin"]["run_at"] == before and "boulder" in raw["units"]


def status(client):
    return client.get("/p/p/labeling/status").json()


def run_at(folder, files):
    """{frame: when its urchin unit was labeled} for the frames that have results."""
    out = {}
    for f in files:
        raw = results.read(folder, f)
        if raw:
            out[f] = raw["units"]["urchin"]["run_at"]
    return out


def test_pause_then_resume_through_the_pages_repeats_and_skips_no_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client, folder = make_project(tmp_path, n=8)
    files = [f"v_aaaaaa_{i}.png" for i in range(8)]
    client.post("/p/p/labeling/run", data={})
    assert wait_for(lambda: status(client).get("done", 0) >= 2, 60)
    job = status(client)
    assert job["message"].startswith("frame ")              # QE PR 33: frames, not "items"
    page = client.get("/p/p/labeling").text
    assert f"/jobs/{job['id']}/pause" in page and "/labeling/stop" not in page
    r = client.post(f"/jobs/{job['id']}/pause", data={"back": "/p/p/labeling"})
    assert r.status_code == 200 and r.url.path == "/p/p/labeling"
    assert wait_for(lambda: status(client)["state"] == "paused", 60)
    before = run_at(folder, files)
    assert 2 <= len(before) < 8
    time.sleep(1.0)
    assert run_at(folder, files) == before                    # the worker really ended
    page = client.get("/p/p/labeling").text
    assert f"/jobs/{job['id']}/resume" in page and f"/jobs/{job['id']}/cancel" in page
    client.post(f"/jobs/{job['id']}/resume", data={"back": "/p/p/labeling"})
    assert wait_for(lambda: status(client)["state"] == "done", 60)
    after = run_at(folder, files)
    assert set(after) == set(files)
    assert all(after[f] == t for f, t in before.items())     # no frame labeled twice
    st = status(client)
    assert (st["done"], st["total"]) == (8, 8)
    assert f"labeled {8 - len(before)} frames, {len(before)} already current" in st["message"]
    jobs = client.get("/jobs").text
    assert "SAM labeling" in jobs and "done" in jobs and "8 / 8" in jobs


def test_cancel_keeps_finished_frames_and_run_does_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client, folder = make_project(tmp_path, n=6)
    files = [f"v_aaaaaa_{i}.png" for i in range(6)]
    client.post("/p/p/labeling/run", data={})
    assert wait_for(lambda: status(client).get("done", 0) >= 1, 60)
    jid = status(client)["id"]
    client.post(f"/jobs/{jid}/cancel", data={"back": "/jobs"})
    assert wait_for(lambda: status(client)["state"] == "cancelled", 60)
    kept = run_at(folder, files)
    assert 1 <= len(kept) < 6
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    msg = run_and_wait(client)
    assert msg == f"labeled {6 - len(kept)} frames, {len(kept)} already current"
    assert all(run_at(folder, files)[f] == t for f, t in kept.items())


def test_job_actions_refuse_bad_requests(tmp_path):
    client, _ = make_project(tmp_path)
    r = client.post("/jobs/99/pause", data={"back": "//evil.example/x"})
    assert r.url.path == "/jobs" and "No job 99" in r.text      # no redirect off the app
    assert client.post("/jobs/1/explode").status_code == 404


APP = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {repo!r})
    from app.stages import label
    from tests.test_app_skeleton import client_for
    label.PREDICTOR = {slow!r}
    client = client_for({config!r})
    client.post("/p/p/labeling/run", data={{}})
    print("started", flush=True)
    time.sleep(600)
""")


def test_app_killed_mid_run_shows_unfinished_job_and_resumes(tmp_path, monkeypatch):
    """TRACKER S4 1b: the app process dies (kill -9) during a Run. The next start shows the
    job interrupted, the dashboard offers Resume, and Resume repeats or skips no frame."""
    client, folder = make_project(tmp_path, n=8)
    files = [f"v_aaaaaa_{i}.png" for i in range(8)]
    config = str(tmp_path / "config.local.toml")
    client.app.state.close_jobs()                # this app has quit
    app = subprocess.Popen([sys.executable, "-c", APP.format(repo=str(REPO), slow=SLOW,
                                                             config=config)],
                           stdout=subprocess.PIPE, text=True, cwd=REPO)
    assert app.stdout.readline().strip() == "started"
    assert wait_for(lambda: len(run_at(folder, files)) >= 2, 60)
    app.send_signal(signal.SIGKILL)
    app.wait()
    time.sleep(2.0)                              # its worker finishes the frame in progress
    before = run_at(folder, files)
    assert 2 <= len(before) < 8

    from tests.test_app_skeleton import client_for
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client = client_for(Path(config))
    home = client.get("/").text
    st = status(client)
    assert st["state"] == "interrupted" and "the app quit or crashed during it" in home
    assert f"/jobs/{st['id']}/resume" in home
    client.post(f"/jobs/{st['id']}/resume", data={"back": "/"})
    assert wait_for(lambda: status(client)["state"] == "done", 60)
    after = run_at(folder, files)
    assert set(after) == set(files) and all(after[f] == t for f, t in before.items())
    assert "the app quit or crashed during it" not in client.get("/").text


def test_plan_estimate_is_a_fixed_cost_per_frame_plus_a_cost_per_unit(tmp_path):
    from tests.test_sam_runner import project_with_frames
    folder, files = project_with_frames(tmp_path, n=2)
    run_sam(folder, files[:1], preview=True)
    preview = label.read_preview(folder)
    # pretend SAM's timing as measured on the M1 Max: 62 s per frame, units ~5 s, the first 13 s
    raw = results.read(folder, files[0])
    for k, u in raw["units"].items():
        u["seconds"] = 13.0 if k == "urchin" else 5.0
    (folder / "sam_raw" / f"{files[0]}.json").write_text(json.dumps(raw))
    preview["frames"][0]["seconds"] = 62.0
    fixed, unit = label.frame_costs(folder, preview)
    assert unit == 5.0 and fixed == 62.0 - 48.0 + 13.0 - 5.0          # 22 s per frame
    cfg = classes.ensure(folder)
    units = classes.run_units(cfg["classes"])
    plan = label.run_plan(folder, files, units, cfg["settings"], preview)
    assert plan["frames"] == 1 and plan["total"] == 2 and set(plan["units"].values()) == {1}
    assert plan["estimate_s"] == round(22 + 8 * 5)                   # one frame, all 8 units
    assert label.run_plan(folder, files, units, cfg["settings"], None)["estimate_s"] is None


def test_final_class_map_overlay(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path)
    client.post("/p/p/labeling/preview", data={})   # the toggle sits with the preview frames
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "done", 60)
    url = "/p/p/labeling/overlay/v_aaaaaa_0.png"
    plain, mapped = client.get(url), client.get(url, params={"map": "1"})
    assert mapped.status_code == 200 and mapped.headers["content-type"] == "image/jpeg"
    assert mapped.content != plain.content
    assert "final class map" in client.get("/p/p/labeling").text


def test_an_unreadable_results_file_does_not_stop_run(tmp_path):
    """QE PR 24: a corrupt sam_raw file is left alone, named, and the other frames labeled."""
    _, folder = make_project(tmp_path)
    files = [f"v_aaaaaa_{i}.png" for i in range(4)]
    results.raw_dir(folder).mkdir(exist_ok=True)
    results.raw_path(folder, files[0]).write_text("{not json")
    msg = run_sam(folder, files)["done"]["message"]
    assert "labeled 3 frames" in msg and "cannot be read" in msg and files[0] in msg
    assert all(results.read(folder, f) for f in files[1:])
    assert results.raw_path(folder, files[0]).read_text() == "{not json"   # left as it was


def test_pause_is_offered_for_a_preview_too(tmp_path, monkeypatch):
    """S3 offered Stop for a Run only (QE PR 24); a Preview is a job like any other now and
    pauses the same way. After a Preview, Run skips its frames."""
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client, folder = make_project(tmp_path, n=4)
    client.post("/p/p/labeling/preview", data={})
    assert wait_for(lambda: status(client)["state"] == "running", 30)
    st = status(client)
    assert st["params"]["preview"] and f"/jobs/{st['id']}/pause" in client.get("/p/p/labeling").text
    assert wait_for(lambda: status(client)["state"] == "done", 60)
    assert status(client)["message"].startswith("preview: labeled 3 frames")
    assert label.read_preview(folder)["seconds_per_frame"]
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    assert run_and_wait(client) == "labeled 1 frames, 3 already current"
