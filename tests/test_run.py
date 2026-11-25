"""S3 bite 4: Run on every kept frame through the page, Stop and carry on, re-run only what
changed (SPEC Q7), and the final class map overlay. Fake SAM, synthetic frames only."""

import json
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


def test_plan_estimate_is_a_fixed_cost_per_frame_plus_a_cost_per_unit(tmp_path):
    from tests.test_sam_runner import project_with_frames
    folder, files = project_with_frames(tmp_path, n=2)
    preview = label.run_preview(folder, files[:1], lambda p, m: None, predictor=FAKE)
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
    msg = label.run_all(folder, files, lambda p, m: None, predictor=FAKE)
    assert "labeled 3 frames" in msg and "cannot be read" in msg and files[0] in msg
    assert all(results.read(folder, f) for f in files[1:])
    assert results.raw_path(folder, files[0]).read_text() == "{not json"   # left as it was


def test_stop_is_offered_for_a_run_not_a_preview(tmp_path, monkeypatch):
    """QE PR 24: Stop did nothing during a Preview, so it is shown only while a Run goes."""
    client, folder = make_project(tmp_path)
    gate = threading.Event()

    def slow(folder, files, report, **kw):
        gate.wait(10)
        return {}
    monkeypatch.setattr(label, "run_preview", slow)
    monkeypatch.setattr(label, "run_all", lambda *a, stop=None, **kw: gate.wait(10) and "done")
    client.post("/p/p/labeling/preview", data={})
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "running", 10)
    assert "/labeling/stop" not in client.get("/p/p/labeling").text
    gate.set()
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "done", 10)
    gate.clear()
    client.post("/p/p/labeling/run")
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "running", 10)
    assert "/labeling/stop" in client.get("/p/p/labeling").text
    gate.set()
    wait_for(lambda: client.get("/p/p/labeling/status").json().get("state") == "done", 10)
