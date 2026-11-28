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


def test_recipes_for_outside_data_and_pretrained_starts():
    from app.train import job as train_job
    manifest = {"number": 3, "hash": "h", "classes": [{"id": 1, "name": "purple urchin"}],
                "outside": [{"key": "urchinbot", "licence": "CC-BY-4.0"}]}
    scratch = {"kind": "scratch", "licences": []}
    key, rec = train_job.build_recipe("yolox-nano-conv", {}, manifest, [1], scratch)
    assert rec["train"]["data"] == "project" and key == "yolox-nano-conv-256-c1"
    assert rec["train"]["dataset"]["outside"] == [{"key": "urchinbot", "licence": "CC-BY-4.0"}]
    # a pre-training run: one class "urchin", whatever classes the form ticked
    key, rec = train_job.build_recipe("yolox-nano-conv", {"data": "outside"}, manifest, [1], scratch)
    assert rec["arch"]["classes"] == ["urchin"] and rec["train"]["class_ids"] == []
    assert rec["train"]["data"] == "outside" and key == "yolox-nano-conv-256-c1"
    with pytest.raises(ValueError, match="holds no outside data"):
        train_job.build_recipe("yolox-nano-conv", {"data": "outside"}, {**manifest, "outside": []}, [1], scratch)
    with pytest.raises(ValueError, match="Train on"):
        train_job.build_recipe("yolox-nano-conv", {"data": "the moon"}, manifest, [1], scratch)
    # YOLOX may now start from a model pre-trained here; published weights stay refused
    _, rec = train_job.build_recipe("yolox-nano-conv", {}, manifest, [1],
                                    {"kind": "pretrained", "model": "m1-x", "licences": ["CC-BY-4.0"]})
    assert rec["train"]["start"]["model"] == "m1-x"
    with pytest.raises(ValueError, match="can only start from scratch for now"):
        train_job.build_recipe("yolox-nano-conv", {}, manifest, [1], {"kind": "published"})


def test_a_run_on_outside_data_and_a_run_that_starts_from_it(run, tmp_path, monkeypatch):
    """Spec stage 7: outside data is for pre-training only, one class; a later run starts from
    that model. The run record says what was really used and what was loaded."""
    import torch
    from app.datasets import loaders, outside
    from app.train import models
    from app.train.yolox import Trainer
    mdir, recipe = run("a")                                   # the project run, made first
    assert trainer(mdir, recipe).describe()["outside_used"] == []
    rows = {s: [{"file": f["file"], "image": f["image"], "bytes": __import__("os").path.getsize(f["image"]),
                 "width": 320, "height": 240, "boxes": [b[1:] for b in f["boxes"]]}
                for f in frames(tmp_path, n, s)] for s, n in (("train", 8), ("valid", 4))}
    from app.datasets import versions
    rows_file = versions.root(tmp_path / "a") / "v1" / "outside" / "urchinbot.jsonl"
    rows_file.parent.mkdir(parents=True)
    rows_file.write_text("frozen rows\n")
    entry = {"key": "urchinbot", "licence": "CC-BY-4.0", "folder": "urchinbot", "rows": "urchinbot.jsonl",
             "sha256": versions.sha256_file(rows_file)}
    monkeypatch.setattr(loaders, "load", lambda folder, number: ({"number": 1, "outside": [entry]}, {}))
    monkeypatch.setattr(outside, "samples", lambda root, vdir, e, split: rows[split])
    pre_recipe = {"arch": ydef.arch("nano", 96, ["urchin"]),
                  "train": {**recipe["train"], "data": "outside", "class_ids": []}}
    pre_dir = models.create(tmp_path / "a", "yolox-nano-conv-96-c1", pre_recipe)
    with pytest.raises(RuntimeError, match="outside_data_dir is not set"):
        Trainer(pre_dir, pre_recipe, {"limits": {"device": "cpu", "threads": 2}})
    job = {"limits": {"device": "cpu", "threads": 2}, "outside_dir": str(tmp_path)}
    pre = Trainer(pre_dir, pre_recipe, job)
    d = pre.describe()
    assert d["outside_used"] == [{"key": "urchinbot", "licence": "CC-BY-4.0"}]
    assert d["train_frames"] == 8 and d["valid_frames"] == 4 and d["start_used"] == "scratch"
    pre.load()
    finish(pre, pre_dir, 0, 0)
    best = torch.load(pre_dir / "checkpoints" / "best.pt", weights_only=True)["model"]
    rows["train"][0]["bytes"] += 1                             # the folder changed since the freeze
    with pytest.raises(RuntimeError, match="no longer the size the version froze"):
        Trainer(pre_dir, pre_recipe, job)
    rows["train"][0]["bytes"] -= 1
    rows_file.write_text("edited by hand\n")                  # the version's own rows file changed
    with pytest.raises(RuntimeError, match="no longer matches the hash in the version's manifest"):
        Trainer(pre_dir, pre_recipe, job)
    rows_file.write_text("frozen rows\n")
    # the project run (2 classes) starts from it: everything but the class-score layers is loaded
    start = {**recipe, "train": {**recipe["train"], "start": {"kind": "pretrained", "model": pre_dir.name}}}
    t = trainer(mdir, start)
    t.load()
    assert t.describe()["start_used"] == "pretrained"
    used = t.describe()["start_loaded"]
    assert used["kind"] == "pretrained" and used["model"] == pre_dir.name
    assert used["tensors_fresh"] == 6 and all(k.startswith("head.cls_preds.") for k in used["fresh"])
    assert used["its_classes"] == ["urchin"]
    got = t.model.state_dict()
    assert torch.equal(got["backbone.backbone.stem.conv.weight"], best["backbone.backbone.stem.conv.weight"])
    assert got["head.cls_preds.0.weight"].shape[0] == 2 and best["head.cls_preds.0.weight"].shape[0] == 1
    assert torch.equal(t.ema["head.reg_preds.0.weight"], best["head.reg_preds.0.weight"])   # EMA starts there too
    t.save(0, 1)
    again = trainer(mdir, start)
    assert again.load() == (0, 1) and again.describe()["start_loaded"]["note"] == "loaded at the run's first start"
    # one class to one class, but another class: the class scores are not taken over (QE PR 90)
    star = {"arch": ydef.arch("nano", 96, ["sea star"]),
            "train": {**recipe["train"], "class_ids": [3], "start": {"kind": "pretrained", "model": pre_dir.name}}}
    star_dir = models.create(tmp_path / "a", "yolox-nano-conv-96-c1", star)
    monkeypatch.undo()                                         # the project's own loader again ...
    monkeypatch.setattr(loaders, "detector_samples", lambda folder, version, split, ids=None: {
        "samples": [{**f, "boxes": [b for b in f["boxes"] if b[0] in ids], "checked": list(ids)}
                    for f in frames(tmp_path, 8 if split == "train" else 4, split)], "left_out": []})
    monkeypatch.setattr(loaders, "verify_image", lambda s: None)
    t3 = trainer(star_dir, star)
    t3.load()
    assert t3.describe()["start_loaded"]["tensors_fresh"] == 6
    assert not torch.equal(t3.model.state_dict()["head.cls_preds.0.weight"], best["head.cls_preds.0.weight"])
    same = {"arch": ydef.arch("nano", 96, ["urchin"]), "train": star["train"]}
    same_dir = models.create(tmp_path / "a", "yolox-nano-conv-96-c1", same)
    t4 = trainer(same_dir, same)
    t4.load()
    assert t4.describe()["start_loaded"]["tensors_fresh"] == 0     # the same class list: all of it
    # missing weights and other kinds of start stop the run with the reason
    gone = {**recipe, "train": {**recipe["train"], "start": {"kind": "pretrained", "model": "m9-none"}}}
    other, _ = run("b")
    with pytest.raises(RuntimeError, match="starting model's weights are missing"):
        trainer(other, gone).load()
    with pytest.raises(RuntimeError, match="cannot start from published weights"):
        trainer(other, {**recipe, "train": {**recipe["train"], "start": {"kind": "published"}}})


def test_estimate_times_steps_and_validation_and_saves_nothing(run):
    """Bite 5 part 2: Estimate = a few timed steps after one warm-up step + one validation pass."""
    mdir, recipe = run("a")
    t = trainer(mdir, recipe)
    t.load()
    got = t.estimate(2)
    assert got["steps_timed"] == 2 and got["seconds_per_step"] > 0 and got["validation_seconds"] > 0
    assert t.ema_updates == 3                               # 1 warm-up + 2 timed steps ran
    assert not list((mdir / "checkpoints").iterdir()) and not (mdir / "train.jsonl").exists()
