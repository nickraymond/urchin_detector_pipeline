"""S8 bite 5 (first part): the Train page with a fake model type and the fake trainer (no
torch): the form, Run as a job, the recipe that is written, the curves' data, the model list."""

import json

import pytest

from app.train import job as train_job
from app.train import models
from tests.test_clips import wait_for
from tests.test_dataset_loaders import frozen
from tests.test_preview import make_project

FAKE_TYPE = {"name": "Fake-Tiny", "kind": "detector", "trainer": "fake", "family": "fake",
             "size": "tiny", "variant": "", "inputs": [64, 96], "presets": ["none", "blur + haze"],
             "defaults": {"epochs": 3, "input": 64, "augment": "blur + haze", "seed": 7},
             "why": {"type": "a stand-in for tests.", "input": "small and fast.",
                     "epochs": "three is enough here.", "augment": "turbid water."},
             "licence": "none (test)"}


@pytest.fixture
def page(tmp_path, monkeypatch):
    monkeypatch.setitem(train_job.TRAINERS, "fake", "tests.fake_trainer:FakeTrainer")
    monkeypatch.setattr(train_job, "MODEL_TYPES", {"fake-tiny": FAKE_TYPE})
    client, folder = make_project(tmp_path, 1)
    return client, folder


def run(client, **data):
    form = {"version": 1, "model_type": "fake-tiny", "classes": ["1"], "epochs": "3",
            "input": "64", "augment": "blur + haze", "seed": "7", **data}
    return client.post("/p/p/train/run", data=form, follow_redirects=False).headers["location"]


def test_page_needs_a_version_and_a_model_type(tmp_path, monkeypatch):
    client, folder = make_project(tmp_path, 1)
    assert "No dataset version yet" in client.get("/p/p/train").text
    assert 'href="/p/p/train"' in client.get("/p/p").text
    frozen(folder)
    monkeypatch.setattr(train_job, "MODEL_TYPES", {})
    assert "No model type can be trained yet" in client.get("/p/p/train").text


def test_run_writes_the_recipe_trains_as_a_job_and_lists_the_model(page):
    client, folder = page
    frozen(folder)
    html = client.get("/p/p/train").text
    assert "Fake-Tiny (detector)" in html and "three is enough here." in html
    assert 'name="classes" value="1" checked' in html and 'name="classes" value="3" checked' in html
    assert "YOLOX COCO weights" in html and "No model yet." in html
    where = run(client)
    assert where == "/p/p/train?model=m1-fake-tiny-64-c1"
    mdir = folder / "models" / "m1-fake-tiny-64-c1"
    assert wait_for(lambda: len(models.read_metrics(mdir)) == 3, 60)
    rec = json.loads((mdir / "recipe.json").read_text())
    manifest = json.loads((folder / "datasets" / "v1" / "manifest.json").read_text())
    assert rec["arch"] == {"kind": "detector", "family": "fake", "size": "tiny", "variant": "",
                           "input": [64, 64], "classes": ["purple urchin"],
                           "licence": "none (test)"}
    t = rec["train"]
    assert t["dataset"] == {"version": 1, "hash": manifest["hash"], "outside": []}
    assert (t["class_ids"], t["epochs"], t["augment"], t["seed"], t["tile"]) == \
        ([1], 3, "blur + haze", 7, 64)
    assert t["start"] == {"kind": "scratch", "licences": []} and t["recorded"][0]["code"]["commit"]
    got = client.get("/p/p/train/metrics/m1-fake-tiny-64-c1").json()
    assert got["epochs"] == 3 and [r["epoch"] for r in got["rows"]] == [0, 1, 2]
    assert all("loss" in r for r in got["rows"])
    assert wait_for(lambda: "<td>done</td>" in client.get("/p/p/train").text, 30)
    html = client.get("/p/p/train").text
    assert "Model m1-fake-tiny-64-c1" in html and "<td>3 / 3</td>" in html and "train.js" in html
    assert "Training" in client.get("/jobs").text
    assert client.get("/p/p/train/metrics/m9-nope").status_code == 404
    assert client.get("/p/p/train/metrics/..%2Fx").status_code == 404
    # a second model gets the next number; the list shows both, newest first
    assert run(client, input="96", classes=["1", "3"]) == "/p/p/train?model=m2-fake-tiny-96-c2"
    assert wait_for(lambda: len(models.read_metrics(folder / "models" / "m2-fake-tiny-96-c2")) == 3, 60)
    html = client.get("/p/p/train").text
    assert html.index("m2-fake-tiny-96-c2") < html.index('train?model=m1-fake-tiny-64-c1')


def test_run_is_refused_with_a_reason_and_makes_no_model(page):
    client, folder = page
    frozen(folder)
    for data, msg in (({"classes": ["9"]}, "not+object+classes"),
                      ({"model_type": "nope"}, "Pick+a+model+type"),
                      ({"epochs": "0"}, "between+1+and+1000"),
                      ({"input": "128"}, "Input+size+must+be+one+of"),
                      ({"augment": "magic"}, "Unknown+augmentation"),
                      ({"version": 9}, "cannot+be+read"),
                      ({"start_kind": "published"}, "No+published+weights"),
                      ({"start_kind": "pretrained", "start_model": "m1-x"}, "No+model"),
                      ({"start": "25:99"}, "error=")):
        where = run(client, **data)
        assert msg in where, (data, where)
    # sea stars only: the version has none that training would see
    assert "gives+training+no+object" in run(client, classes=["3"])
    assert not (folder / "models").exists()


def test_one_training_job_per_project_at_a_time(page, monkeypatch):
    client, folder = page
    frozen(folder)
    monkeypatch.setattr(train_job, "MODEL_TYPES", {"fake-tiny": FAKE_TYPE})
    assert "model=m1-" in run(client, epochs="200")
    assert "waiting+or+running" in run(client)
    assert models.list_ids(folder) == ["m1-fake-tiny-64-c1"]
    jid = client.app.state.jobctl(__import__("app.config", fromlist=["x"]).load(
        folder.parents[2] / "config.local.toml")).latest("train", str(folder))["id"]
    client.post(f"/jobs/{jid}/cancel", data={"back": "/p/p/train"})
    assert wait_for(lambda: "model=m2-" in run(client), 30)


def test_estimate_times_a_few_steps_as_a_job_and_makes_no_model(page):
    """Bite 5 part 2 (spec stage 8, the D35 pattern): Estimate runs the form's settings for a
    few steps, shows what the whole run would take, and leaves no model behind."""
    client, folder = page
    frozen(folder)
    html = client.get("/p/p/train").text
    assert 'formaction="/p/p/train/estimate"' in html and 'id="estimate"' not in html
    form = {"version": 1, "model_type": "fake-tiny", "classes": ["1"], "epochs": "40",
            "input": "96", "augment": "none", "seed": "3"}
    r = client.post("/p/p/train/estimate", data=form, follow_redirects=False)
    assert r.headers["location"] == "/p/p/train?version=1"
    assert wait_for(lambda: train_job.read_estimate(folder) is not None, 60)
    est = train_job.read_estimate(folder)
    # the fake trainer: 3 steps per epoch at 0.25 s + 0.5 s validation = 1.25 s per epoch
    assert (est["steps_per_epoch"], est["seconds_per_step"], est["validation_seconds"]) == (3, 0.25, 0.5)
    assert est["seconds_per_epoch"] == 1.25 and est["total_seconds"] == 50 and est["epochs"] == 40
    assert est["steps_timed"] == train_job.ESTIMATE_STEPS and est["recipe"]["train"]["tile"] == 96
    assert models.list_ids(folder) == []                       # no model, no number used
    edir = folder / "models" / ".estimate"
    assert not (edir / "train.jsonl").exists() and not list(edir.glob("checkpoints/*"))
    assert wait_for(lambda: "estimate: 1.25 s per epoch" in client.get("/p/p/train").text, 60)
    html = client.get("/p/p/train").text
    assert "Estimate: about <b>1 min</b> for 40 epochs" in html and "(1.25 s per epoch)" in html and "Training (estimate)" in html
    # the form now holds what was estimated, so Run runs the same settings
    assert 'name="epochs" type="number" min="1" max="1000" value="40"' in html
    assert "<option selected>96</option>" in html and "<option selected>none</option>" in html
    # a refused estimate says why and keeps nothing new
    r = client.post("/p/p/train/estimate", data={**form, "input": "128"}, follow_redirects=False)
    assert "Input+size+must+be+one+of" in r.headers["location"]
    # Run after Estimate makes model 1 as usual
    assert run(client, epochs="2") == "/p/p/train?model=m1-fake-tiny-64-c1"


def test_validation_frames_are_served_and_listed_with_their_epoch(page):
    from PIL import Image
    client, folder = page
    frozen(folder)
    where = run(client)
    mdir = folder / "models" / "m1-fake-tiny-64-c1"
    assert wait_for(lambda: len(models.read_metrics(mdir)) == 3, 60)
    got = client.get("/p/p/train/metrics/m1-fake-tiny-64-c1").json()
    assert got["val_frames"] == [] and got["val_epoch"] is None      # this trainer draws none
    (mdir / "val_frames").mkdir()
    for i in (0, 1):
        Image.new("RGB", (8, 8), (i, 0, 0)).save(mdir / "val_frames" / f"val_{i}.png")
    (mdir / "val_frames" / "epoch.txt").write_text("3\n")
    got = client.get("/p/p/train/metrics/m1-fake-tiny-64-c1").json()
    assert got["val_frames"] == [0, 1] and got["val_epoch"] == 3
    r = client.get("/p/p/train/val/m1-fake-tiny-64-c1/1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "no-store" and r.content[:4] == b"\x89PNG"
    assert client.get("/p/p/train/val/m1-fake-tiny-64-c1/7").status_code == 404
    assert client.get("/p/p/train/val/..%2F..%2Fx/0").status_code == 404
    html = client.get(where).text
    assert 'data-val="/p/p/train/val/m1-fake-tiny-64-c1/"' in html and "val-frames" in html
    assert "White</b> boxes are the labels" in html


def test_the_form_carries_every_model_types_defaults_and_whys(page, monkeypatch):
    """The size, epochs, preset and "why" of the form follow the model type picked (train.js
    reads them from the page); the server uses the picked type's defaults for empty fields."""
    client, folder = page
    other = {**FAKE_TYPE, "name": "Fake-Big", "size": "big", "inputs": [128],
             "defaults": {"epochs": 9, "input": 128, "augment": "none", "seed": 1},
             "why": {**FAKE_TYPE["why"], "epochs": "nine for the big one."}}
    monkeypatch.setattr(train_job, "MODEL_TYPES", {"fake-tiny": FAKE_TYPE, "fake-big": other})
    frozen(folder)
    html = client.get("/p/p/train").text
    data = json.loads(html.split('<script type="application/json" id="model-types">')[1].split("</script>")[0])
    assert data["fake-big"]["defaults"]["epochs"] == 9 and data["fake-big"]["inputs"] == [128]
    assert 'data-why="epochs">three is enough here.' in html and 'id="model-type"' in html
    where = client.post("/p/p/train/run", data={"version": 1, "model_type": "fake-big",
                                                "classes": ["1"]}, follow_redirects=False).headers["location"]
    assert where == "/p/p/train?model=m1-fake-big-128-c1"
    assert models.read_recipe(folder / "models" / "m1-fake-big-128-c1")["train"]["epochs"] == 9
