"""The YOLOX trainer behind the train job (S8 bite 4b; the interface is in app/train/worker.py).

The loop and its numbers are Nick's earlier trainer's, the known-good start
(`ADIN_SPI_OpenMV/ml/yolox_urchin/train.py` @ 994b8c2): YOLOX's own loss (SimOTA; `model(imgs,
targets)` in train mode), SGD + Nesterov, peak lr 0.01 / 64 x batch, linear warm-up over 3 epochs
(at most 1500 steps), cosine decay to 5 %, weight decay 5e-4 on conv / linear weights only, EMA
with ramped decay, BatchNorm eps 1e-3 and momentum 0.03.

The model is S6's definition (`app.modeldefs.yolox.build_model`: Nano / Tiny with the plain first
layer, the variant that compiled for both NPUs). Samples come from a frozen dataset version
(app/datasets/loaders.py) through app/train/samples.py; their order and augmentation are fixed by
(seed, epoch, index), so a run paused mid-epoch continues with exactly the steps it had left.

Review flags (spec "What gets trained"): frames with split check "no" and classes not fully
checked are already left out by the loader. A box of a class that is not fully checked on a frame
is painted over in grey before the frame is used, so the model is never taught that an unlabeled
neighbour of it is background there. That alone is not enough: the animals nobody labeled are
still in the picture. So a frame is used, for training and for validation, only when every class
of the run is fully checked on it (app/train/job.py `usable`; QE PR 72); the frames left out are
counted in the run record. (Finer, and keeping those frames: masking that class's loss. Not done.)

Pre-training (spec stage 7; TRACKER S8 bite 3): a run whose recipe says `data: "outside"`
trains on the version's public sources only (app/datasets/outside.py `samples`; their train and
valid splits, one class, "urchin") and records them in `outside_used`. Another run may start from
such a model (`start.kind: "pretrained"`): at its first start the EMA weights of that model's
`checkpoints/best.pt` are loaded into every tensor of the same name and shape; with a different
class list the class-score layers keep their fresh values. What was loaded is in the run record.
Published weights are never loaded (none is offered, D62).

Each epoch ends with a check on fixed validation canvases (same crops every epoch, no colour,
blur or haze): recall and precision at IoU 0.5, score 0.3, all classes together, from the EMA
weights. That is a training-progress signal on crops, not the evaluation of S9 (tiles on whole
test frames). A few of those canvases are saved with boxes drawn in `val_frames/`.

Device (D65): the Apple GPU when it is there. Its gradients were wrong at first (PR 72) because
the batch was laid out channels-last in memory (`from_numpy(NHWC).permute(...).float()` keeps
that layout) and torch's Apple GPU convolution gives wrong weight gradients for such an input
(measured 2026-11-27 on torch 2.10.0 and 2.13.0; right on the CPU). `_batch` now hands over a
contiguous tensor, and `load()` checks one small batch on both devices before each start: if the
gradients differ, the run uses the CPU and says so.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from app import modeldefs
from app.datasets import loaders
from app.modeldefs import yolox as ydef
from app.train import samples, yolox_mps

MAX_BOXES = 120          # targets per canvas (the earlier trainer's cap)
VAL_SCORE, VAL_IOU, NMS_IOU = 0.3, 0.5, 0.45
VAL_FRAMES = 4
GPU_CHECK_FRAMES, GPU_CHECK_LIMIT = 8, 0.05   # load(): devices agreeing give about 0.002
MAX_DETECTIONS = 100     # per canvas, as YOLOX's own evaluation keeps; an untrained head can fire
                         # on thousands of cells, which made one validation pass take 58 s (seen)
GREY = samples.GREY


def pick_device(wanted: str | None) -> str:
    if wanted in ("cpu", "mps", "cuda"):
        return wanted
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of [cx, cy, w, h] boxes: (len(a), len(b))."""
    a1, a2 = a[:, None, :2] - a[:, None, 2:] / 2, a[:, None, :2] + a[:, None, 2:] / 2
    b1, b2 = b[None, :, :2] - b[None, :, 2:] / 2, b[None, :, :2] + b[None, :, 2:] / 2
    inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None).prod(-1)
    return inter / (a[:, None, 2:].prod(-1) + b[None, :, 2:].prod(-1) - inter + 1e-9)


def decode(maps, strides=ydef.STRIDES, score=VAL_SCORE) -> list[np.ndarray]:
    """The exported raw maps (app.modeldefs.yolox.export_module) -> per image an array of
    [cx, cy, w, h, score, class index], after NMS per class."""
    from torchvision.ops import batched_nms
    rows = []
    for out, stride in zip(maps, strides):
        out = out.float().cpu()
        b, ch, h, w = out.shape
        ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        grid = torch.stack((xs, ys), 0).to(out.dtype)
        rows.append(torch.cat([(out[:, :2] + grid) * stride, out[:, 2:4].exp() * stride,
                               out[:, 4:]], 1).flatten(2))
    pred = torch.cat(rows, 2).permute(0, 2, 1)            # (B, anchors, 5 + classes)
    found = []
    for p in pred:
        conf, cls = (p[:, 4:5] * p[:, 5:]).max(1)
        keep = conf >= score
        p, conf, cls = p[keep], conf[keep], cls[keep]
        if len(conf) > 10 * MAX_DETECTIONS:           # keep NMS itself cheap
            top = conf.topk(10 * MAX_DETECTIONS).indices
            p, conf, cls = p[top], conf[top], cls[top]
        xyxy = torch.cat([p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2], 1)
        k = batched_nms(xyxy, conf, cls, NMS_IOU)[:MAX_DETECTIONS]   # best scores first
        found.append(torch.cat([p[k, :4], conf[k, None], cls[k, None].float()], 1).numpy())
    return found


class Trainer:
    def __init__(self, model_dir, recipe: dict, job: dict):
        self.dir, self.arch, self.t = Path(model_dir), recipe["arch"], recipe["train"]
        limits = job.get("limits") or {}
        self.device = pick_device(limits.get("device"))
        self.warning, self.gpu_check = "", None
        self.threads = int(limits.get("threads") or 4)
        torch.set_num_threads(self.threads)
        self.seed, self.canvas = int(self.t["seed"]), int(self.t["tile"])
        self.batch = int(self.t.get("batch") or 32)
        self.preset = self.t["augment"]
        folder, version = self.dir.parents[1], self.t["dataset"]["version"]
        self.ids = list(self.t["class_ids"])
        self.start = self.t.get("start") or {"kind": "scratch"}
        if self.start.get("kind") not in ("scratch", "pretrained"):
            raise RuntimeError(f"This trainer cannot start from {self.start.get('kind')} weights; "
                               "the recipe asks for them. Start from scratch or from a model "
                               "pre-trained here.")
        self.start_used = {"kind": "scratch"}
        self.outside_used = []
        if self.t.get("data") == "outside":
            self._outside_samples(folder, version, job.get("outside_dir"))
        else:
            from app.train.job import usable
            self.train, out_train = usable(
                loaders.detector_samples(folder, version, "train", self.ids)["samples"], self.ids)
            self.valid, out_valid = usable(
                loaders.detector_samples(folder, version, "valid", self.ids)["samples"], self.ids)
            self.left_out = {"train": out_train, "valid": out_valid}
            if not self.train:
                raise RuntimeError(f"Dataset version {version} has no training frame for these "
                                   "classes (see the Datasets page: what training will see).")
            for s in self.train[:1] + self.valid[:1]:
                loaders.verify_image(s)
        self.steps = max(1, len(self.train) // self.batch) if len(self.train) >= self.batch \
            else 1                                    # a tiny dataset: one short batch per epoch
        self.epochs = int(self.t["epochs"])
        self.model = self.opt = self.ema = None
        self.ema_updates = 0
        self.best = -1.0
        self.partial = None      # loss sums of an epoch paused part-way (kept in the checkpoint)

    def _outside_samples(self, folder: Path, version: int, outside_dir) -> None:
        """A pre-training run: every outside source of the version, as one class (index 0)."""
        from app.datasets import outside, versions
        manifest, _ = loaders.load(folder, version)
        entries = manifest.get("outside") or []
        if not entries or not outside_dir:
            raise RuntimeError(
                f"Dataset version {version} holds no outside data, or outside_data_dir is not set "
                "in config.local.toml: a run on outside data needs both.")
        vdir = versions.root(folder) / f"v{int(version)}"
        self.ids, got = [0], {"train": [], "valid": []}
        for e in entries:
            if not outside.BY_KEY.get(e["key"], {}).get("allowed"):
                raise RuntimeError(f"{e['key']} is not an allowed outside source.")
            for split in got:
                for r in outside.samples(Path(outside_dir), vdir, e, split):
                    got[split].append({**r, "boxes": [[0, *b] for b in r["boxes"]],
                                       "checked": [0], "ignore": []})
            self.outside_used.append({"key": e["key"], "licence": e["licence"]})
        self.train, self.valid = got["train"], got["valid"]
        self.left_out = {"train": 0, "valid": 0}
        if not self.train:
            raise RuntimeError(f"The outside data of version {version} has no training image.")
        for s in self.train[:1] + self.valid[:1]:     # frozen by size in bytes (DESIGN, bite 3)
            p = Path(s["image"])
            if not p.is_file() or p.stat().st_size != s["bytes"]:
                raise RuntimeError(f"{p} is missing or no longer the size the version froze: "
                                   "the outside data folder was changed.")

    def _load_start(self) -> None:
        """A fresh run that starts from a model pre-trained here: its best EMA weights go into
        every tensor of the same name and shape."""
        src = self.dir.parent / str(self.start.get("model")) / "checkpoints" / "best.pt"
        if not src.is_file():
            raise RuntimeError(f"The starting model's weights are missing: {src}.")
        have = torch.load(src, map_location="cpu", weights_only=True)["model"]
        mine = self.model.state_dict()
        take = {k: v.to(mine[k].dtype) for k, v in have.items()
                if k in mine and mine[k].shape == v.shape}
        self.model.load_state_dict(take, strict=False)
        fresh = sorted(k for k in mine if k not in take)
        self.start_used = {"kind": "pretrained", "model": self.start["model"],
                           "tensors_loaded": len(take), "tensors_fresh": len(fresh),
                           "fresh": fresh[:12]}

    def describe(self) -> dict:
        import torchvision
        return {"device": self.device, "warning": self.warning, "threads": self.threads,
                "batch": self.batch, "gpu_check": self.gpu_check,
                "steps_per_epoch": self.steps, "train_frames": len(self.train),
                "valid_frames": len(self.valid),
                "frames_left_out_not_fully_checked": self.left_out,
                "start_used": self.start_used, "outside_used": self.outside_used,
                "libraries": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                              "numpy": np.__version__, "yolox": ydef.CODE["commit"]}}

    # ---- model, optimizer, schedule ----

    def load(self) -> tuple[int, int]:
        torch.manual_seed(self.seed)
        model = ydef.build_model(self.arch)
        yolox_mps.apply()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
        if self.device == "mps":
            worst = self._gpu_gradients_differ_by(model)
            self.gpu_check = {"worst_relative_difference": round(worst, 5),
                              "limit": GPU_CHECK_LIMIT, "frames": min(GPU_CHECK_FRAMES, len(self.train))}
            if not worst <= GPU_CHECK_LIMIT:          # also true for NaN
                self.device = "cpu"
                self.warning = (
                    "This run trains on the CPU (slower): one test batch gave different gradients "
                    f"on the Apple GPU and on the CPU (worst difference {worst:.3g} of the CPU's "
                    f"value; up to {GPU_CHECK_LIMIT} is rounding). The run's results are right; "
                    "tell the team so the GPU fault can be found.")
        self.model = model.to(self.device).train()
        decay, other = [], []
        for name, p in self.model.named_parameters():
            (decay if name.endswith("weight") and "bn" not in name else other).append(p)
        self.lr = float(self.t.get("lr") or 0.01 / 64 * self.batch)
        self.opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4},
                                    {"params": other, "weight_decay": 0.0}],
                                   lr=self.lr, momentum=0.9, nesterov=True)
        last = self.dir / "checkpoints" / "last.pt"
        if self.start.get("kind") == "pretrained":
            if last.is_file():        # a resume: the weights are in last.pt; keep the first record
                self.start_used = {"kind": "pretrained", "model": self.start["model"],
                                   "note": "loaded at the run's first start"}
            else:
                self._load_start()
        self.ema = {k: v.detach().clone().float() for k, v in self.model.state_dict().items()}
        if not last.is_file():
            return 0, 0
        ck = torch.load(last, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.ema = {k: v.to(self.device) for k, v in ck["ema"].items()}
        self.ema_updates, self.best = ck["ema_updates"], ck.get("best", -1.0)
        self.partial = ck.get("partial")
        return ck["epoch"], ck["step"]

    def _gpu_gradients_differ_by(self, model) -> float:
        """One small batch through a copy of the model on the CPU and on the Apple GPU: the
        largest gradient difference of any parameter, as a share of that parameter's largest CPU
        gradient (parameters whose gradient is next to nothing are measured against the largest
        gradient of the whole model). Agreeing devices give about 0.002 (measured)."""
        import copy
        n = min(GPU_CHECK_FRAMES, len(self.train))
        with ThreadPoolExecutor(self.threads) as pool:
            imgs, targets, _ = self._batch(list(range(n)), pool, 0, self.train, True)
        grads = {}
        for dev in ("cpu", self.device):
            m = copy.deepcopy(model).to(dev).train()
            m(imgs.to(dev), targets.to(dev))["total_loss"].backward()
            grads[dev] = [p.grad.detach().cpu() for p in m.parameters() if p.grad is not None]
        top = max(float(g.abs().max()) for g in grads["cpu"])
        return max(float((c - g).abs().max()) / (float(c.abs().max()) + 1e-3 * top + 1e-30)
                   for c, g in zip(grads["cpu"], grads[self.device]))

    def _lr(self, it: int) -> float:
        # the earlier rule (3 epochs, at most 1500 steps) assumed hundreds of steps per epoch;
        # with a small dataset 3 epochs can be 3 steps. So: at least 300 steps, but never more
        # than a third of the run. A guess, not measured: the NaN run that prompted it (42
        # frames, 1 step per epoch) was on the Apple GPU while its gradients were wrong (D65).
        total = self.epochs * self.steps
        warm = min(1500, max(3 * self.steps, min(300, total // 3)), max(1, total))
        if it < warm:
            return self.lr * (it + 1) / warm
        t = (it - warm) / max(1, total - warm)
        return self.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    @torch.no_grad()
    def _ema_update(self, decay=0.9998, tau=2000) -> None:
        self.ema_updates += 1
        d = decay * (1 - math.exp(-self.ema_updates / tau))
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.ema[k].copy_(v)

    # ---- data ----

    def _example(self, sample: dict, epoch: int, index: int, train: bool):
        got = samples.make(sample, self.canvas, self.seed, epoch, index,
                           self.preset if train else "crop only")
        img = got["image"].copy()
        for cx, cy, w, h in got["ignore"]:            # not fully checked here: no lesson from it
            img[max(0, int(cy - h / 2)):int(cy + h / 2) + 1,
                max(0, int(cx - w / 2)):int(cx + w / 2) + 1] = GREY
        boxes = [[self.ids.index(c), cx, cy, w, h] for c, cx, cy, w, h in got["boxes"]][:MAX_BOXES]
        return img, boxes

    def _batch(self, picks: list[int], pool, epoch: int, source: list[dict], train: bool):
        made = list(pool.map(lambda i: self._example(source[i], epoch, i, train), picks))
        # NCHW and contiguous: a channels-last layout (what permute() of the stacked images
        # leaves) gets wrong convolution gradients on the Apple GPU (module docstring, D65)
        imgs = torch.from_numpy(np.ascontiguousarray(
            np.stack([m[0] for m in made]).transpose(0, 3, 1, 2))).float()
        targets = torch.zeros((len(made), MAX_BOXES, 5))
        for n, (_, boxes) in enumerate(made):
            if boxes:
                targets[n, :len(boxes)] = torch.tensor(boxes, dtype=torch.float32)
        return imgs.to(self.device), targets.to(self.device), made

    # ---- one epoch ----

    def _step(self, picks: list[int], pool, epoch: int, it: int) -> dict:
        """One training step on the frames `picks`; `it` = steps since the run began (the
        learning-rate schedule's clock). Returns YOLOX's loss dict."""
        imgs, targets, _ = self._batch(picks, pool, epoch, self.train, True)
        for g in self.opt.param_groups:
            g["lr"] = self._lr(it)
        out = self.model(imgs, targets)
        self.opt.zero_grad(set_to_none=True)
        out["total_loss"].backward()
        self.opt.step()
        self._ema_update()
        return out

    def estimate(self, steps: int) -> dict:
        """Estimate (spec stage 8): time `steps` training steps after one warm-up step, then one
        validation pass. Call after load(). The weights change in memory only: nothing is saved
        by an estimate (the worker never calls save())."""
        import time
        order = samples.order(len(self.train), self.seed, 0)
        times = []
        with ThreadPoolExecutor(self.threads) as pool:
            for i in range(steps + 1):
                at = (i % self.steps) * self.batch
                t = time.perf_counter()
                float(self._step(order[at:at + self.batch], pool, 0, i)["total_loss"].detach())
                times.append(time.perf_counter() - t)      # float() waits for the GPU
            t = time.perf_counter()
            self._validate(0, pool)
            val = time.perf_counter() - t
        return {"seconds_per_step": sum(times[1:]) / steps, "validation_seconds": val,
                "warm_up_seconds": times[0], "steps_timed": steps}

    def run_epoch(self, epoch: int, start_step: int, stop) -> dict:
        order = samples.order(len(self.train), self.seed, epoch)
        total, parts = 0.0, {"iou_loss": 0.0, "conf_loss": 0.0, "cls_loss": 0.0}
        if start_step and self.partial and self.partial["at"] == [epoch, start_step]:
            total, parts = self.partial["total"], dict(self.partial["parts"])   # resumed mid-epoch
            done_before = start_step
        else:
            done_before = 0
        with ThreadPoolExecutor(self.threads) as pool:
            for step in range(start_step, self.steps):
                if stop():
                    self.partial = {"at": [epoch, step], "total": total, "parts": parts}
                    return {"done": False, "step": step}
                out = self._step(order[step * self.batch:(step + 1) * self.batch], pool,
                                 epoch, epoch * self.steps + step)
                loss = float(out["total_loss"].detach())
                if not math.isfinite(loss):           # fail loudly, never write NaN lines
                    raise RuntimeError(
                        f"The loss stopped being a number at epoch {epoch + 1}, step {step + 1} "
                        f"(learning rate {self._lr(epoch * self.steps + step):.5f}). The last "
                        "checkpoint is from before this epoch. Start a new run with more "
                        "epochs (a longer warm-up) or more frames.")
                total += loss
                for k in parts:
                    parts[k] += float(torch.as_tensor(out[k]).detach())
            # the epoch's mean; after a resume without the earlier steps' sums (an old
            # checkpoint), the mean of this worker's steps only
            n = max(1, self.steps - start_step + done_before)
            self.partial = None
            metrics = {"loss": round(total / n, 4),
                       **{k: round(v / n, 4) for k, v in parts.items()},
                       "lr": round(self._lr(epoch * self.steps + self.steps - 1), 6)}
            metrics.update(self._validate(epoch, pool))
        return {"done": True, "metrics": metrics}

    @torch.no_grad()
    def _validate(self, epoch: int, pool) -> dict:
        if not self.valid:
            return {}
        live = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict({k: v.to(live[k].dtype) for k, v in self.ema.items()})
        export = ydef.export_module(self.model)       # eval mode, raw maps
        tp = n_pred = n_true = 0
        out_dir = self.dir / "val_frames"
        out_dir.mkdir(exist_ok=True)
        try:
            for at in range(0, len(self.valid), self.batch):
                picks = list(range(at, min(at + self.batch, len(self.valid))))
                imgs, _, made = self._batch(picks, pool, 0, self.valid, False)   # epoch 0: fixed
                for n, det in enumerate(decode(export(imgs))):
                    truth = np.array([b[1:] for b in made[n][1]], np.float32).reshape(-1, 4)
                    cls_t = [b[0] for b in made[n][1]]
                    used = set()
                    for d in det[np.argsort(-det[:, 4])] if len(det) else []:
                        iou = box_iou(d[None, :4], truth)[0] if len(truth) else np.zeros(0)
                        ok = [j for j in np.argsort(-iou) if iou[j] >= VAL_IOU and j not in used
                              and cls_t[j] == int(d[5])]
                        if ok:
                            used.add(ok[0])
                    tp, n_pred, n_true = tp + len(used), n_pred + len(det), n_true + len(truth)
                    if picks[n] < VAL_FRAMES:
                        self._draw(made[n][0], truth, det, out_dir / f"val_{picks[n]}.png")
        finally:
            self.model.load_state_dict(live)
            self.model.train()
        recall = tp / n_true if n_true else 0.0
        precision = tp / n_pred if n_pred else 0.0
        (out_dir / "epoch.txt").write_text(f"{epoch + 1}\n")
        return {"val_recall": round(recall, 4), "val_precision": round(precision, 4),
                "val_objects": n_true}

    @staticmethod
    def _draw(img: np.ndarray, truth: np.ndarray, det: np.ndarray, path: Path) -> None:
        """Ground truth in white, predictions in yellow."""
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        for boxes, colour in ((truth, (255, 255, 255)), (det[:, :4] if len(det) else [], (255, 220, 0))):
            for cx, cy, w, h in boxes:
                d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=colour)
        tmp = path.with_name(f".{path.name}.tmp")
        im.save(tmp, format="PNG")
        os.replace(tmp, path)

    # ---- checkpoints ----

    def save(self, epoch: int, step: int) -> None:
        ck = {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "ema": self.ema,
              "ema_updates": self.ema_updates, "epoch": epoch, "step": step, "best": self.best,
              "partial": self.partial if step else None}
        last = self.dir / "checkpoints" / "last.pt"
        tmp = last.with_name(".last.pt.tmp")
        torch.save(ck, tmp)
        os.replace(tmp, last)
        if step == 0 and epoch > 0:                   # an epoch just ended: keep the best EMA
            rows = [r for r in _metrics(self.dir) if r["epoch"] == epoch - 1]
            score = (rows[0].get("val_recall", 0) + rows[0].get("val_precision", 0)) if rows else 0
            if score > self.best or not (self.dir / "checkpoints" / "best.pt").is_file():
                self.best = max(self.best, score)
                tmp = last.with_name(".best.pt.tmp")
                torch.save({"model": self.ema, "epoch": epoch, "score": score}, tmp)
                os.replace(tmp, self.dir / "checkpoints" / "best.pt")

    def unload(self) -> None:
        self.model = self.opt = self.ema = None
        if self.device == "mps":
            torch.mps.empty_cache()


def _metrics(mdir: Path) -> list[dict]:
    from app.train import models
    return models.read_metrics(mdir)
