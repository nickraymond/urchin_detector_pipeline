"""S8 bite 4b: the YOLOX trainer (app/train/yolox.py) on the CPU with synthetic frames. Needs
Megvii's YOLOX checkout under tools_dir (D51); skipped where it is not there. Small on purpose:
YOLOX-Nano, 96 px canvases, 8 frames, 2 epochs."""

import numpy as np
import pytest
from PIL import Image

from app import modeldefs
from app.modeldefs import yolox as ydef

pytestmark = pytest.mark.skipif(
    not (modeldefs.tools_dir() / "third_party" / "YOLOX" / "yolox" / "__init__.py").is_file(),
    reason="YOLOX checkout not found under tools_dir/third_party (set the tools up)")


def frames(folder, n, split):
    out = []
    for i in range(n):
        rng = np.random.default_rng(100 * (split == "valid") + i)
        a = np.full((240, 320, 3), (20, 60, 120), np.uint8)
        boxes = []
        for k in range(3):
            x, y = int(rng.integers(10, 250)), int(rng.integers(10, 170))
            a[y:y + 50, x:x + 50] = (230, 200, 60) if k < 2 else (200, 60, 200)
            boxes.append([1 if k < 2 else 3, x, y, 50, 50])
        p = folder / f"{split}_{i}.png"
        Image.fromarray(a).save(p)
        out.append({"file": p.name, "image": str(p), "sha256": "", "width": 320, "height": 240,
                    "boxes": boxes, "checked": [1, 3], "ignore": [[0, 0, 30, 30]] if i == 0 else []})
    return out


@pytest.fixture
def run(tmp_path, monkeypatch):
    from app.datasets import loaders
    from app.train import models
    data = {s: frames(tmp_path, n, s) for s, n in (("train", 8), ("valid", 4))}
    monkeypatch.setattr(loaders, "detector_samples",
                        lambda folder, version, split, ids=None: {"samples": data[split], "left_out": []})
    monkeypatch.setattr(loaders, "verify_image", lambda s: None)
    recipe = {"arch": ydef.arch("nano", 96, ["purple urchin", "sea star"]),
              "train": {"dataset": {"version": 1, "hash": "x"}, "class_ids": [1, 3], "tile": 96,
                        "epochs": 2, "augment": "standard + blur + haze", "seed": 4, "batch": 4}}

    def make(name):
        mdir = models.create(tmp_path / name, modeldefs.arch_key(recipe["arch"]), recipe)
        return mdir, recipe
    return make


def trainer(mdir, recipe):
    from app.train.yolox import Trainer
    return Trainer(mdir, recipe, {"limits": {"device": "cpu", "threads": 2}})


def finish(t, mdir, epoch, step, stop=lambda: False):
    """The worker's loop in short: line first, then the checkpoint."""
    from app.train import worker
    rows = [r for r in __import__("app.train.models", fromlist=["x"]).read_metrics(mdir)]
    while epoch < 2:
        res = t.run_epoch(epoch, step, stop)
        if not res["done"]:
            t.save(epoch, res["step"])
            return rows
        rows.append({"epoch": epoch, **res["metrics"]})
        worker.write_metrics(mdir, rows)
        t.save(epoch + 1, 0)
        epoch, step = epoch + 1, 0
    return rows


def test_two_epochs_write_metrics_checkpoints_and_validation_frames(run):
    mdir, recipe = run("a")
    t = trainer(mdir, recipe)
    assert t.load() == (0, 0) and t.steps == 2
    d = t.describe()
    assert d["device"] == "cpu" and d["libraries"]["yolox"] == ydef.CODE["commit"] and d["train_frames"] == 8
    assert d["outside_used"] == [] and d["start_used"] == "scratch"
    assert d["frames_left_out_not_fully_checked"] == {"train": 0, "valid": 0}
    rows = finish(t, mdir, 0, 0)
    assert [r["epoch"] for r in rows] == [0, 1]
    for r in rows:
        assert np.isfinite(r["loss"]) and r["loss"] > 0 and r["val_objects"] > 0
        assert 0 <= r["val_recall"] <= 1 and 0 <= r["val_precision"] <= 1 and r["lr"] > 0
        assert {"iou_loss", "conf_loss", "cls_loss"} <= set(r)
    ck = mdir / "checkpoints"
    assert (ck / "last.pt").is_file() and (ck / "best.pt").is_file() and not list(ck.glob(".*.tmp"))
    assert (mdir / "val_frames" / "val_0.png").is_file()
    assert (mdir / "val_frames" / "epoch.txt").read_text().strip() == "2"
    with Image.open(mdir / "val_frames" / "val_0.png") as im:
        assert im.size == (96, 96)
    t.unload()
    assert trainer(mdir, recipe).load() == (2, 0)          # where a Resume would continue


def test_a_run_paused_mid_epoch_ends_with_the_same_weights(run):
    """Pause after the first step, save, start a new trainer, finish: the same weights as a run
    that was never paused (same seed, CPU). No step repeated or skipped, same samples."""
    import torch
    a_dir, recipe = run("a")
    a = trainer(a_dir, recipe)
    a.load()
    finish(a, a_dir, 0, 0)
    b_dir, _ = run("b")
    b = trainer(b_dir, recipe)
    b.load()
    calls = iter([False, True])
    finish(b, b_dir, 0, 0, stop=lambda: next(calls))
    assert not (b_dir / "train.jsonl").exists()            # no epoch finished before the pause
    b.unload()
    b2 = trainer(b_dir, recipe)
    assert b2.load() == (0, 1)
    rows_b = finish(b2, b_dir, 0, 1)
    rows_a = __import__("app.train.models", fromlist=["x"]).read_metrics(a_dir)
    assert rows_b[0]["loss"] == pytest.approx(rows_a[0]["loss"], abs=2e-4)   # the whole epoch's mean
    assert rows_b[0]["cls_loss"] == pytest.approx(rows_a[0]["cls_loss"], abs=2e-4)
    wa = torch.load(a_dir / "checkpoints" / "last.pt", weights_only=True)
    wb = torch.load(b_dir / "checkpoints" / "last.pt", weights_only=True)
    assert wa["ema_updates"] == wb["ema_updates"] == 4
    for k, v in wa["model"].items():
        assert torch.allclose(v.float(), wb["model"][k].float(), atol=1e-6), k


def test_a_box_of_a_class_not_fully_checked_is_painted_over(run):
    mdir, recipe = run("a")
    t = trainer(mdir, recipe)
    sample = {**t.train[0], "boxes": [], "ignore": [[100, 60, 120, 120]]}
    img, boxes = t._example(sample, 0, 0, train=False)
    from app.train import samples
    got = samples.make(sample, 96, 4, 0, 0, "crop only")
    (cx, cy, w, h), = got["ignore"]
    assert boxes == [] and (img[int(cy), int(cx)] == samples.GREY).all()


def test_yolox_is_offered_with_defaults_and_whys():
    from app.train import job as train_job
    assert train_job.TRAINERS["yolox"] == "app.train.yolox:Trainer"
    nano = train_job.MODEL_TYPES["yolox-nano-conv"]
    assert nano["defaults"]["input"] == 256 and nano["defaults"]["augment"] in nano["presets"]
    assert all(nano["why"][k] for k in ("type", "input", "epochs", "augment"))
    manifest = {"number": 3, "hash": "h", "classes": [{"id": 1, "name": "purple urchin"},
                                                      {"id": 3, "name": "sea star"}]}
    key, recipe = train_job.build_recipe("yolox-nano-conv", {}, manifest, [1, 3],
                                         {"kind": "scratch", "licences": []})
    assert key == "yolox-nano-conv-256-c2" and recipe["arch"]["variant"] == {"stem": "conv"}
    assert recipe["arch"]["code"]["commit"] == ydef.CODE["commit"] and recipe["train"]["epochs"] == 120
    assert modeldefs.check(recipe["arch"]) == []


def test_a_loss_that_is_not_a_number_stops_the_run(run, monkeypatch):
    import torch
    mdir, recipe = run("a")
    t = trainer(mdir, recipe)
    t.load()
    real = t.model.forward
    monkeypatch.setattr(t.model, "forward", lambda *a, **k: {
        **real(*a, **k), "total_loss": torch.tensor(float("nan"), requires_grad=True)})
    with pytest.raises(RuntimeError, match="stopped being a number at epoch 1, step 1"):
        t.run_epoch(0, 0, lambda: False)
    assert not (mdir / "train.jsonl").exists()


def test_the_batch_is_contiguous_nchw(run):
    """D65: a channels-last batch is what gave wrong gradients on the Apple GPU."""
    from concurrent.futures import ThreadPoolExecutor
    mdir, recipe = run("a")
    t = trainer(mdir, recipe)
    with ThreadPoolExecutor(2) as pool:
        imgs, targets, made = t._batch([0, 1, 2], pool, 0, t.train, True)
    assert imgs.shape == (3, 3, 96, 96) and imgs.is_contiguous()
    assert np.array_equal(imgs[1].permute(1, 2, 0).numpy(), made[1][0].astype(np.float32))


def _mps() -> bool:
    import torch
    return torch.backends.mps.is_available()


@pytest.mark.skipif(not _mps(), reason="needs the Apple GPU")
def test_the_apple_gpu_is_used_when_its_gradients_match_the_cpu(run):
    from app.train.yolox import GPU_CHECK_LIMIT, Trainer
    mdir, recipe = run("a")
    t = Trainer(mdir, recipe, {"limits": {"device": "auto", "threads": 2}})
    t.load()
    d = t.describe()
    assert d["device"] == "mps" and d["warning"] == ""
    assert d["gpu_check"]["worst_relative_difference"] < GPU_CHECK_LIMIT / 2
    assert d["gpu_check"]["frames"] == [f"train_{i}.png" for i in range(8)]   # which frames it used
    res = t.run_epoch(0, 0, lambda: False)
    assert res["done"] and np.isfinite(res["metrics"]["loss"])


@pytest.mark.skipif(not _mps(), reason="needs the Apple GPU")
def test_wrong_apple_gpu_gradients_send_the_run_to_the_cpu_with_a_warning(run, monkeypatch):
    """The fault of PR 72, brought back on purpose: a channels-last batch. load() must see it."""
    from app.train.yolox import Trainer
    mdir, recipe = run("a")
    t = Trainer(mdir, recipe, {"limits": {"device": "mps", "threads": 2}})
    real = t._batch

    def channels_last(*a, **k):
        imgs, targets, made = real(*a, **k)
        return imgs.cpu().permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2).to(imgs.device), targets, made
    monkeypatch.setattr(t, "_batch", channels_last)
    t.load()
    d = t.describe()
    assert d["device"] == "cpu" and "trains on the CPU" in d["warning"]
    assert d["gpu_check"]["worst_relative_difference"] > 1


def test_validation_keeps_at_most_100_boxes_per_canvas():
    """An untrained head can score thousands of cells above the cut-off; decode caps them."""
    import torch
    from app.train import yolox
    maps = [torch.cat([torch.rand(1, 4, n, n), torch.ones(1, 3, n, n)], 1) for n in (32, 16, 8)]
    (det,) = yolox.decode(maps)
    assert det.shape == (yolox.MAX_DETECTIONS, 6) and (det[:, 4] >= yolox.VAL_SCORE).all()


def test_only_frames_with_every_run_class_fully_checked_are_used(run, monkeypatch):
    """QE PR 72: a frame where one of the run's classes is not fully checked would teach its
    unlabeled animals as background, so it is left out (training and validation) and counted."""
    from app.datasets import loaders
    from app.train import job as train_job
    mdir, recipe = run("a")
    data = {s: loaders.detector_samples(None, 1, s)["samples"] for s in ("train", "valid")}
    data["train"][0]["checked"] = [1]
    data["train"][1]["checked"] = [3]
    data["valid"][0]["checked"] = [1]
    t = trainer(mdir, recipe)
    assert len(t.train) == 6 and len(t.valid) == 3
    assert t.describe()["frames_left_out_not_fully_checked"] == {"train": 2, "valid": 1}
    assert all(s["checked"] == [1, 3] for s in t.train + t.valid)
    one = [dict(s, checked=[c for c in s["checked"] if c == 1]) for s in data["train"]]
    assert train_job.usable(one, [1]) == ([s for s in one if s["checked"] == [1]], 1)


def test_a_start_other_than_scratch_is_refused_and_outside_data_is_not_claimed(run):
    from app.train import job as train_job
    from app.train.yolox import Trainer
    mdir, recipe = run("a")
    pre = {**recipe, "train": {**recipe["train"], "start": {"kind": "pretrained", "model": "m1-x"}}}
    with pytest.raises(RuntimeError, match="cannot start from pretrained weights yet"):
        Trainer(mdir, pre, {"limits": {"device": "cpu"}})
    manifest = {"number": 3, "hash": "h", "classes": [{"id": 1, "name": "purple urchin"}],
                "outside": [{"key": "urchinbot", "licence": "CC-BY-4.0"}]}
    with pytest.raises(ValueError, match="can only start from scratch for now"):
        train_job.build_recipe("yolox-nano-conv", {}, manifest, [1],
                               {"kind": "pretrained", "model": "m1-x", "licences": []})
    _, rec = train_job.build_recipe("yolox-nano-conv", {}, manifest, [1],
                                    {"kind": "scratch", "licences": []})
    assert rec["train"]["dataset"]["outside"] == [{"key": "urchinbot", "licence": "CC-BY-4.0"}]
    assert trainer(mdir, recipe).describe()["outside_used"] == []     # what the run really used
