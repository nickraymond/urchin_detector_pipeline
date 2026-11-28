"""The YOLOX predictor for the evaluate job (S9 bite 1): best.pt's weights (the EMA kept by validation) in the
model S6 defines, inference only. `predict(tiles)` takes N canvas tiles (N x H x W x 3 uint8 RGB) and returns per
tile an array [cx, cy, w, h, score, class index] in tile pixels, after NMS per class, every score down to the
job's floor. Uses the trainer's own decode, so validation and evaluation read the raw maps the same way."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.evaluate import run as runs
from app.modeldefs import yolox as ydef
from app.train import yolox as trainer


class Predictor:
    def __init__(self, model_dir, recipe: dict, job: dict):
        limits = job.get("limits") or {}
        self.device = trainer.pick_device(limits.get("device"))
        torch.set_num_threads(int(limits.get("threads") or 4))
        self.batch = int(recipe["train"].get("batch") or 32)
        model = ydef.build_model(recipe["arch"])
        weights = torch.load(Path(model_dir) / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)["model"]
        mine = model.state_dict()
        model.load_state_dict({k: v.to(mine[k].dtype) for k, v in weights.items()})
        self.export = ydef.export_module(model.to(self.device))          # eval mode, raw maps

    @torch.no_grad()
    def predict(self, tiles: np.ndarray) -> list[np.ndarray]:
        # NCHW and contiguous, as the trainer feeds it (D65)
        imgs = torch.from_numpy(np.ascontiguousarray(tiles.transpose(0, 3, 1, 2))).float().to(self.device)
        return trainer.decode(self.export(imgs), score=runs.SCORE_FLOOR)
