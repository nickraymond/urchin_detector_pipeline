"""S3 bite 4: Run on every kept frame through the page, Stop and carry on, re-run only what
changed (SPEC Q7), and the final class map overlay. Fake SAM, synthetic frames only."""

import threading

from app.sam import classes, results
from app.stages import label
from tests.test_clips import wait_for
from tests.test_preview import make_project
from tests.test_sam_runner import FAKE


def run_and_wait(client):
    assert client.post("/p/p/labeling/run", data={}).status_code == 200
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "done", 60)
    return client.get("/p/p/labeling/status").json()["message"]


def test_run_labels_every_kept_frame_then_only_a_new_class(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, folder = make_project(tmp_path)
    page = client.get("/p/p/labeling").text
    assert "Will run: " in page and "<b>urchin</b> (4 frames)" in page
    assert "press Preview first" in page
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
    assert "Will run: <b>boulder</b> (4 frames)." in page and "Unchanged: " in page
    assert "4 of 4 frames have work" in page
    assert run_and_wait(client) == "labeled 4 frames, 0 already current"
    raw = results.read(folder, "v_aaaaaa_1.png")
    assert raw["units"]["urchin"]["run_at"] == before and "boulder" in raw["units"]


def test_stop_keeps_finished_frames_and_run_carries_on(tmp_path):
    from tests.test_sam_runner import project_with_frames
    folder, files = project_with_frames(tmp_path, n=2)
    stop = threading.Event()
    stop.set()                                   # Stop pressed before the worker got going
    msg = label.run_all(folder, files, lambda p, m: None, stop=stop, predictor=FAKE)
    assert msg.startswith("stopped after") and "carries on" in msg
    msg = label.run_all(folder, files, lambda p, m: None, predictor=FAKE)
    assert msg.startswith("labeled ") and msg.endswith("already current")
    assert all(results.read(folder, f) for f in files)


def test_plan_estimate_from_the_preview_scales_with_the_units_that_run(tmp_path):
    from tests.test_sam_runner import project_with_frames
    folder, files = project_with_frames(tmp_path, n=2)
    label.run(folder, files[:1], lambda p, m: None, predictor=FAKE)
    cfg = classes.ensure(folder)
    units = classes.run_units(cfg["classes"])
    plan = label.run_plan(folder, files, units, cfg["settings"], {"seconds_per_frame": 80})
    assert plan["frames"] == 1 and plan["total"] == 2 and plan["estimate_s"] == 80
    assert set(plan["units"].values()) == {1}


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
