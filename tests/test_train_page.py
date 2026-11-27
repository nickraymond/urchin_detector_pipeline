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
