"""S9 bite 1 (D90): the evaluate job on a frozen synthetic version with a fake predictor, and the tiling path
with a stand-in detector on painted squares. No torch."""

import json
import time

import numpy as np
import pytest
from PIL import Image

from app.datasets import versions
from app.evaluate import run as runs
from app.evaluate import worker as eval_worker
from app.jobctl import JobControl, JobError
from app.train import models, tiles
from tests.dataset_fixtures import dive, fake_export

FAKE = "tests.fake_predictor:FakePredictor"


def model(folder, mid_hint="m", best=True, **train):
    recipe = {"arch": {"kind": "detector", "family": "fake", "classes": ["purple urchin"]},
              "train": {"trainer": "fake", "dataset": {"version": 1, "hash": "h"}, "class_ids": [1], "tile": 32,
                        "epochs": 3, "val_every": 1, **train}}
    mdir = models.create(folder, f"fake-{mid_hint}-32-c1", recipe)
    if best:
        (mdir / "checkpoints").mkdir(exist_ok=True)
        (mdir / "checkpoints" / "best.pt").write_bytes(b"weights")
    return mdir


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "projects" / "p"
    folder.mkdir(parents=True)
    fake_export(folder, dive("reef", 1, 8) + dive("cove", 2, 4) + dive("wall", 3, 3))
    assert versions.freeze(folder)["new"]
    return folder


@pytest.fixture
def jc(tmp_path):
    j = JobControl(tmp_path / "jobs")
    j.register(runs.evaluate_kind())
    yield j
    j.close()


def wait(jc, jid, *states, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if jc.status(jid)["state"] in states:
            return jc.status(jid)
        time.sleep(0.05)
    raise AssertionError(jc.status(jid))


def test_a_model_is_evaluated_once_per_test_set(project, jc):
    mdir = model(project)
    p = runs.params(project, mdir.name, predictor=FAKE)
    n = len(runs.test_frames(project, 1, [1]))
    assert n >= 1 and p["frames"] == n and p["key"].startswith("v1-") and "worked out" in p["source"]
    job = wait(jc, jc.submit("evaluate", p), "done", "error")
    assert job["state"] == "done", job
    out = runs.eval_dir(project, mdir.name, p["key"])
    lines = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines()]
    # the fake finds one box in the middle of every tile (close ones merge across seams), class id 1 (not index 0)
    assert len(lines) == n and all(1 <= len(l["boxes"]) <= l["tiles"] and {b[0] for b in l["boxes"]} == {1} for l in lines)
    ev = json.loads((out / "eval.json").read_text())
    assert ev["frames"] == n and ev["threshold"] == 0.3 and ev["score_floor"] == 0.01 and ev["test_set"] == p["key"]
    assert ev["weights_sha256"] == versions.sha256_file(mdir / "checkpoints" / "best.pt")
    assert ev["eval_overlap_capped"] is False and ev["eval_scale"]["factor"] and ev["code"]
    m = json.loads((out / "metrics.json").read_text())        # bite 2 (D91, D92): every figure with its interval
    groups = runs.test_groups(project, 1)
    assert m["frames"] == n and m["intervals"]["groups"] == len({groups[l["file"]] for l in lines})
    assert m["intervals"]["unit"] == "frames" and "dives only" in m["intervals"]["note"]   # under 5 dives in test
    assert m["classes"]["1"]["recall"]["interval"] is not None and "rank_floor" in m["classes"]["1"]
    with pytest.raises(JobError, match="already evaluated"):
        runs.params(project, mdir.name, predictor=FAKE)


def test_refusals(project):
    with pytest.raises(JobError, match="has no best.pt: it is taken only from validated epochs.*after epoch 2"):
        runs.params(project, model(project, "a", best=False, val_every=2).name, predictor=FAKE)
    with pytest.raises(JobError, match="pre-trained on outside data"):
        runs.params(project, model(project, "b", data="outside").name, predictor=FAKE)
    with pytest.raises(JobError, match="No evaluation is built for trainer 'fake'"):
        runs.params(project, model(project, "c").name)


def test_a_test_split_without_a_fully_checked_frame_is_refused(tmp_path):
    folder = tmp_path / "projects" / "q"
    folder.mkdir(parents=True)
    fake_export(folder, dive("reef", 1, 8, checked=[1]) + dive("cove", 2, 4, checked=[1]) + dive("wall", 3, 3, checked=[1]))
    assert versions.freeze(folder)["new"]
    with pytest.raises(JobError, match="no frame with every class"):
        runs.params(folder, model(folder, "d", class_ids=[1, 3]).name, predictor=FAKE)     # sea stars never checked


def test_the_recipes_eval_scale_is_used_and_the_cap_is_carried(project, jc):
    stored = {"raw": 0.5, "factor": 0.5, "median_side_px": 88, "boxes": 1}
    mdir = model(project, "e", eval_scale=stored, eval_overlap=16, eval_overlap_capped=True)
    p = runs.params(project, mdir.name, predictor=FAKE)
    assert (p["eval_scale"], p["eval_overlap"], p["eval_overlap_capped"], p["source"]) == (stored, 16, True, "recipe")
    assert wait(jc, jc.submit("evaluate", p), "done", "error")["state"] == "done"
    ev = json.loads((runs.eval_dir(project, mdir.name, p["key"]) / "eval.json").read_text())
    assert ev["eval_overlap_capped"] is True and ev["eval_scale"] == stored


def test_pause_between_frames_and_resume_scores_each_frame_once(project, jc):
    mdir = model(project, "f")
    p = {**runs.params(project, mdir.name, predictor=FAKE), "fake_delay": 0.15}
    jid = jc.submit("evaluate", p)
    out = runs.eval_dir(project, mdir.name, p["key"])
    end = time.monotonic() + 20
    while len(runs.done_files(out)) < 1 and time.monotonic() < end:
        time.sleep(0.02)
    jc.pause(jid)
    wait(jc, jid, "paused", "done")
    assert not (out / "eval.json").exists() or jc.status(jid)["state"] == "done"
    jc.resume(jid)
    assert wait(jc, jid, "done", "error")["state"] == "done"
    files = runs.done_files(out)
    assert len(files) == len(set(files)) == p["frames"]


def test_tiles_and_seams_put_boxes_back_in_native_pixels(tmp_path):
    """A stand-in detector finds painted squares lying whole in a tile; the frame is 2x the eval scale, so the
    native boxes must come back at the painted positions and sizes, each once."""
    from scipy import ndimage

    class Squares:
        batch = 3

        def predict(self, imgs):
            out = []
            for img in imgs:
                lab, _ = ndimage.label(np.all(np.abs(img.astype(int) - (230, 200, 60)) < 12, axis=-1))
                rows = [[(x.start + x.stop) / 2, (y.start + y.stop) / 2, x.stop - x.start, y.stop - y.start, .9, 0]
                        for y, x in ndimage.find_objects(lab)
                        if x.start > 0 and y.start > 0 and x.stop < img.shape[1] and y.stop < img.shape[0]]
                out.append(np.array(rows, np.float32).reshape(-1, 6))
            return out
    a = np.full((300, 400, 3), (20, 60, 120), np.uint8)
    truth = [(20, 30, 40), (150, 140, 40), (300, 220, 40), (190, 20, 40)]
    for x, y, side in truth:
        a[y:y + side, x:x + side] = (230, 200, 60)
    Image.fromarray(a).save(tmp_path / "f.png")
    s = {"file": "f.png", "image": str(tmp_path / "f.png"), "width": 400, "height": 300, "boxes": [], "ignore": []}
    ov = tiles.overlap([{"boxes": [[0, 0, 0, 40, 40]]}], 0.5, 64)
    boxes, n = eval_worker.frame(s, 0.5, 64, ov, Squares())
    assert n == len(tiles.plan(200, 150, 64, ov)) and len(boxes) == 4
    got = sorted((round(b[1]), round(b[2]), round(b[3])) for b in boxes)
    assert all(abs(g[0] - t[0]) <= 2 and abs(g[1] - t[1]) <= 2 and abs(g[2] - t[2]) <= 3
               for g, t in zip(got, sorted(truth))), got


def test_numpy_nms_keeps_the_best_per_class():
    det = np.array([[10, 10, 10, 10, .9, 0], [11, 10, 10, 10, .8, 0], [11, 10, 10, 10, .7, 1], [50, 50, 5, 5, .6, 0]],
                   np.float32)
    kept = tiles.nms(det, 0.45)
    assert kept[:, 4].tolist() == pytest.approx([.9, .7, .6])


def test_a_version_split_by_video_has_no_dive_groups(project, monkeypatch):
    groups = runs.test_groups(project, 1)
    assert groups and len(set(groups.values())) == 3                     # reef, cove, wall: site + date
    real = runs.loaders.load
    monkeypatch.setattr(runs.loaders, "load", lambda f, v: ({**real(f, v)[0], "grouped_by": "video"}, {}))
    assert runs.test_groups(project, 1) is None
