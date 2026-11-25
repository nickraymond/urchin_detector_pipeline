"""A stand-in for SAM in tests (no torch, weights or real frames): each prompt finds the
connected blobs of one pure colour in a synthetic image."""

import numpy as np
from scipy import ndimage

COLOURS = {"urchin": (200, 30, 200), "sea star": (250, 140, 0), "kelp": (20, 160, 40),
           "rock": (120, 120, 120)}


class FakePredictor:
    def __init__(self, job):
        self.calls = 0

    def set_image(self, image):
        self.calls += 1
        return np.asarray(image.convert("RGB")).astype(int)

    def find(self, state, prompt):
        colour = COLOURS.get(prompt)
        if colour is None:
            return []
        hit = (np.abs(state - colour).sum(axis=2) < 30)
        labels, n = ndimage.label(hit)
        return [(labels == i, 0.9 - i / 100) for i in range(1, n + 1)]
