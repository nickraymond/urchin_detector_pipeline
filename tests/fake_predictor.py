"""A fake predictor for the evaluate job's tests (no torch): one box of class 0 in the middle of every tile,
score 0.9, so a frame's merged result is known; `fake_delay` (s per tile batch) lets a test pause mid-run."""

import time

import numpy as np


class FakePredictor:
    device = "none (fake)"

    def __init__(self, model_dir, recipe, job):
        self.batch, self.delay = 2, float(job.get("fake_delay") or 0)

    def predict(self, tiles):
        time.sleep(self.delay)
        c = tiles.shape[1] / 2
        return [np.array([[c, c, 10, 10, 0.9, 0]], np.float32) for _ in tiles]
