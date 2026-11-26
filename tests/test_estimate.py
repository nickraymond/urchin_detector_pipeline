"""S4 bite 3 (D35): Estimate on the Labeling page (1 warm-up + 5 frames spread over the frames
Run would label, results kept) and the remaining time of any job (rolling mean of the last 20
items). Fake SAM, synthetic frames only."""

import json

from app.jobctl import JobControl
from app.sam import results
from app.stages import label
from tests.test_clips import wait_for
from tests.test_preview import make_project
from tests.test_run import SLOW, run_and_wait, status
from tests.test_sam_runner import FAKE


def test_estimate_frames_are_spread_over_the_frames_with_work():
    work = [f"f{i:02d}" for i in range(20)]
    assert label.estimate_frames(work) == ["f00", "f04", "f08", "f11", "f15", "f19"]
    assert label.estimate_frames(work[:4]) == work[:4]
    assert label.estimate_frames([]) == []


def test_estimate_through_the_page_keeps_its_results_and_times_five_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client, folder = make_project(tmp_path, n=12)
    files = sorted(f"v_aaaaaa_{i}.png" for i in range(12))      # name order, as the app
    page = client.get("/p/p/labeling").text
    assert "/labeling/estimate" in page and "Takes: not known yet: press Estimate" in page
    client.post("/p/p/labeling/estimate", data={})
    assert wait_for(lambda: status(client)["state"] == "done", 60)
    st = status(client)
    assert st["params"]["estimate"] and st["total"] == 6
    assert st["message"] == "estimate: labeled 6 frames, 0 already current"
    picked = st["params"]["files"]
    assert picked == label.estimate_frames(files)
    done = [f for f in files if results.read(folder, f)]
    assert done == sorted(picked, key=files.index)            # only those 6 were labeled
    est = label.read_estimate(folder)
    assert est["warmup"]["file"] == picked[0] and len(est["frames"]) == 5
    assert est["seconds_per_frame"] > 0
    page = client.get("/p/p/labeling").text
    assert f"Estimate: <b>{est['seconds_per_frame']} s / frame</b>" in page
    assert "<b>6 of 12</b>" in page and "from Estimate" in page
    # bite 6 (Nick): how long, then when it finishes, then what runs (the table)
    assert page.index("Takes: about") < page.index("Finishes: about") < page.index('<table class="plan">')
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    assert run_and_wait(client) == "labeled 6 frames, 6 already current"   # Run skips them


def test_an_estimate_goes_out_of_date_when_the_classes_change(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", SLOW)
    client, folder = make_project(tmp_path, n=8)
    client.post("/p/p/labeling/estimate", data={})
    assert wait_for(lambda: status(client)["state"] == "done", 60)
    assert "from Estimate" in client.get("/p/p/labeling").text
    text = (folder / "classes.yaml").read_text().replace("prompts: [fish]", "prompts: [rock]")
    assert client.post("/p/p/labeling/classes", data={"text": text}).status_code == 200
    page = client.get("/p/p/labeling").text
    assert "made with other classes or settings" in page and "from Estimate" not in page


def test_nothing_to_estimate_when_every_frame_is_current(tmp_path, monkeypatch):
    monkeypatch.setattr(label, "PREDICTOR", FAKE)
    client, _ = make_project(tmp_path, n=3)
    run_and_wait(client)
    r = client.post("/p/p/labeling/estimate", data={})
    assert "Nothing to estimate" in r.text


def test_remaining_time_is_the_rolling_mean_of_timed_items(tmp_path):
    jc = JobControl(tmp_path / "jobs", start=False)
    try:
        jid = jc.store.add("fake", {}, state="running")
        ev = lambda **kw: jc.runner._on_event(jid, {"event": "item", "n": 10, **kw})
        ev(i=1, seconds=2.0)
        assert jc.status(jid)["eta_s"] == 18                     # 2 s × 9 left
        ev(i=2, skipped=True, seconds=100.0)                     # a skipped item is not timed
        ev(i=3, seconds=4.0)
        assert jc.status(jid)["eta_s"] == 21                     # mean 3 s × 7 left
        for i in range(4, 30):
            jc.runner._on_event(jid, {"event": "item", "i": i, "n": 40, "seconds": 1.0})
        assert jc.status(jid)["eta_s"] == 11                     # last 20 only: 1 s × 11
    finally:
        jc.close()


def test_after_a_resume_the_warm_up_is_the_first_frame_labeled_again(tmp_path):
    """QE PR 35: a resumed Estimate skips the frames done before the pause; the first frame it
    labels after the model loads again is the warm-up, not a timed one."""
    from app.sam.worker import write_estimate
    recs = [{"file": "a", "skipped": True}, {"file": "b", "skipped": True},
            {"file": "c", "seconds": 90.0}, {"file": "d", "seconds": 60.0},
            {"file": "e", "seconds": 62.0}]
    write_estimate(tmp_path / "e.json", 13.0, recs, [], {})
    est = json.loads((tmp_path / "e.json").read_text())
    assert est["warmup"]["file"] == "c" and [f["file"] for f in est["frames"]] == ["d", "e"]
    assert est["seconds_per_frame"] == 61.0
