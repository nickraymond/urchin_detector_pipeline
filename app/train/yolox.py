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
neighbour of it is background there. (Finer: masking only that class's loss. Not done.)

Each epoch ends with a check on fixed validation canvases (same crops every epoch, no colour,
blur or haze): recall and precision at IoU 0.5, score 0.3, all classes together, from the EMA
weights. That is a training-progress signal on crops, not the evaluation of S9 (tiles on whole
test frames). A few of those canvases are saved with boxes drawn in `val_frames/`.
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
        self.warning = ""
        if self.device == "mps":
            # wrong class-branch gradients on the Apple GPU with torch 2.10.0 (yolox_mps.py)
            self.device = "cpu"
            self.warning = ("YOLOX trains on the CPU for now (slower): on the Apple GPU its "
                            "gradients come out wrong with this PyTorch version and the loss "
                            "turns into NaN. Seen 2026-11-27; the cause is not found yet.")
        self.threads = int(limits.get("threads") or 4)
        torch.set_num_threads(self.threads)
        self.seed, self.canvas = int(self.t["seed"]), int(self.t["tile"])
        self.batch = int(self.t.get("batch") or 32)
        self.preset = self.t["augment"]
        folder, version = self.dir.parents[1], self.t["dataset"]["version"]
        self.ids = list(self.t["class_ids"])
        got = loaders.detector_samples(folder, version, "train", self.ids)
        self.train = got["samples"]
        self.valid = loaders.detector_samples(folder, version, "valid", self.ids)["samples"]
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

    def describe(self) -> dict:
        import torchvision
        return {"device": self.device, "warning": self.warning, "threads": self.threads,
                "batch": self.batch,
                "steps_per_epoch": self.steps, "train_frames": len(self.train),
                "valid_frames": len(self.valid),
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
        self.model = model.to(self.device).train()
        decay, other = [], []
        for name, p in self.model.named_parameters():
            (decay if name.endswith("weight") and "bn" not in name else other).append(p)
        self.lr = float(self.t.get("lr") or 0.01 / 64 * self.batch)
        self.opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4},
                                    {"params": other, "weight_decay": 0.0}],
                                   lr=self.lr, momentum=0.9, nesterov=True)
        self.ema = {k: v.detach().clone().float() for k, v in self.model.state_dict().items()}
        last = self.dir / "checkpoints" / "last.pt"
        if not last.is_file():
            return 0, 0
        ck = torch.load(last, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["opt"])
        self.ema = {k: v.to(self.device) for k, v in ck["ema"].items()}
        self.ema_updates, self.best = ck["ema_updates"], ck.get("best", -1.0)
        return ck["epoch"], ck["step"]

    def _lr(self, it: int) -> float:
        # the earlier rule (3 epochs, at most 1500 steps) assumed hundreds of steps per epoch;
        # with a small dataset 3 epochs can be 3 steps, and the run diverged to NaN on the
        # Apple GPU (seen 2026-11-27: 42 frames, 1 step per epoch, NaN from step 5). So: at
        # least 300 steps, but never more than a third of the run.
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
        imgs = torch.from_numpy(np.stack([m[0] for m in made])).permute(0, 3, 1, 2).float()
        targets = torch.zeros((len(made), MAX_BOXES, 5))
        for n, (_, boxes) in enumerate(made):
            if boxes:
                targets[n, :len(boxes)] = torch.tensor(boxes, dtype=torch.float32)
        return imgs.to(self.device), targets.to(self.device), made

    # ---- one epoch ----

    def run_epoch(self, epoch: int, start_step: int, stop) -> dict:
        order = samples.order(len(self.train), self.seed, epoch)
        total, parts = 0.0, {"iou_loss": 0.0, "conf_loss": 0.0, "cls_loss": 0.0}
        with ThreadPoolExecutor(self.threads) as pool:
            for step in range(start_step, self.steps):
                if stop():
                    return {"done": False, "step": step}
                picks = order[step * self.batch:(step + 1) * self.batch]
                imgs, targets, _ = self._batch(picks, pool, epoch, self.train, True)
                for g in self.opt.param_groups:
                    g["lr"] = self._lr(epoch * self.steps + step)
                out = self.model(imgs, targets)
                self.opt.zero_grad(set_to_none=True)
                out["total_loss"].backward()
                self.opt.step()
                self._ema_update()
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
            n = max(1, self.steps - start_step)       # after a resume: the steps of this worker
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
              "ema_updates": self.ema_updates, "epoch": epoch, "step": step, "best": self.best}
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
