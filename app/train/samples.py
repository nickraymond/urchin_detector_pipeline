"""Training samples for the detector (S8 bite 4b, first part): one frame -> one square canvas with
its boxes, with the recipe's augmentation preset. numpy + Pillow + scipy only (no torch, no
OpenCV), so it is tested on synthetic images.

Adapted from Nick's earlier trainer, `ADIN_SPI_OpenMV/ml/yolox_urchin/data.py` @ 994b8c2
(`_pick_scale`, `_single`, `_clip_boxes`, `_hsv`, the blur step and its own random stream):

- scale: a training frame is resized so that its median box's shorter side lands in the 24-64 px
  band the cameras see (BAND), never above 1.5x; a frame without boxes, or any frame in
  evaluation, is only shrunk to fit the canvas;
- crop: a canvas-sized window aimed at a random box with jitter (15 % of the time anywhere, so
  backgrounds are seen); outside the frame the canvas is grey (114);
- a box cut by the canvas edge is clipped; one left under 8 px (MIN_BOX) on a side is dropped;
- flip left-right half the time; colour jitter in HSV;
- blur (a Gaussian, sigma 0.3-2.5 px) and, new in Build 2, haze (the frame fades toward one
  veil colour and loses contrast, as in turbid water) each use their own random stream, so
  turning one on or off never changes where the boxes of a given seed end up.
  The earlier work left haze out on purpose (it hurt YOLOX-Nano in its test); TRACKER S8 bite 4b
  asks for a preset with blur and haze, so it is a preset, and its effect is to be measured.

Not here yet: mosaic (four frames quilted together; the earlier v2 recipe used it).

A sample is what app/datasets/loaders.py gives: {"image": path, "boxes": [[class id, x, y, w, h]],
"ignore": [[x, y, w, h]]}, boxes in frame pixels, x / y the top-left corner. Outside-data
samples (one class) are passed with class id 0. Everything is repeatable: the same (seed, epoch,
index) always gives the same canvas and boxes, so a run resumed mid-epoch continues exactly.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image
from scipy import ndimage

from app import images

BAND = (24, 64)          # px: the shorter side of a typical target on the cameras (earlier work)
MIN_BOX = 8              # px: a box under this on a side after cropping is dropped
GREY = 114
PRESETS = {              # name -> which steps are on
    "none": {"geometry": False, "colour": False, "blur": 0.0, "haze": 0.0},
    "crop only": {"geometry": True, "colour": False, "blur": 0.0, "haze": 0.0},   # validation views
    "standard": {"geometry": True, "colour": True, "blur": 0.0, "haze": 0.0},
    "standard + blur + haze": {"geometry": True, "colour": True, "blur": 0.3, "haze": 0.3},
}
DEFAULT_PRESET = "standard + blur + haze"
BLUR_SIGMA = (0.3, 2.5)
HAZE_STRENGTH = (0.15, 0.55)   # share of the veil colour mixed in
HAZE_VEILS = ((70, 120, 110), (60, 105, 125), (95, 125, 95))   # green, blue-green, yellow-green


def order(n: int, seed: int, epoch: int) -> list[int]:
    """The order of the n samples in one epoch: a shuffle fixed by (seed, epoch)."""
    idx = list(range(n))
    random.Random(f"order-{seed}-{epoch}").shuffle(idx)
    return idx


def _load(path: str, size: tuple[int, int]) -> np.ndarray:
    """The upright frame as RGB uint8 at `size` (w, h). A JPEG that is shrunk a lot is decoded
    at a reduced size first (the earlier work's fast path for 4000 px frames)."""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (size[0] * 2, size[1] * 2))   # never below twice the target
        im = images.upright(im).convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.BOX if size[0] < im.size[0] else Image.BILINEAR)
        return np.asarray(im)


def _scale(sample: dict, canvas: int, rng: random.Random, train: bool, full: tuple[int, int]) -> float:
    boxes = sample["boxes"]
    if not train or not boxes:
        return min(1.0, canvas / max(full))
    sides = sorted(min(b[3], b[4]) for b in boxes)
    return max(0.03, min(rng.uniform(*BAND) / max(sides[len(sides) // 2], 1), 1.5))


def _clip(boxes: list[list[float]], canvas: int, min_side: float) -> list[list[float]]:
    """[..., x0, y0, w, h] -> the part inside the canvas as [..., cx, cy, w, h]; small ones dropped."""
    out = []
    for *head, x0, y0, w, h in boxes:
        x1, y1 = max(0.0, x0), max(0.0, y0)
        x2, y2 = min(float(canvas), x0 + w), min(float(canvas), y0 + h)
        if x2 - x1 >= min_side and y2 - y1 >= min_side:
            out.append([*head, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return out


def _hsv(img: np.ndarray, rng: random.Random, gains=(0.010, 0.4, 0.3)) -> np.ndarray:
    r = [1 + rng.uniform(-1, 1) * g for g in gains]
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] * r[0]) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return np.asarray(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))


def haze(img: np.ndarray, strength: float, veil=(70, 120, 110)) -> np.ndarray:
    """Fade toward the veil colour: out = img * (1 - strength) + veil * strength."""
    out = img.astype(np.float32) * (1 - strength) + np.asarray(veil, np.float32) * strength
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def make(sample: dict, canvas: int, seed: int, epoch: int, index: int,
         preset: str = DEFAULT_PRESET, train: bool = True) -> dict:
    """One training (or evaluation) example. Returns {"image": canvas x canvas x 3 uint8 RGB,
    "boxes": [[class id, cx, cy, w, h]] in canvas pixels, "ignore": [[cx, cy, w, h]],
    "scale": the resize factor used}."""
    steps = PRESETS[preset] if train else PRESETS["none"]
    rng = random.Random(f"sample-{seed}-{epoch}-{index}")
    blur_rng = random.Random(f"blur-{seed}-{epoch}-{index}")
    haze_rng = random.Random(f"haze-{seed}-{epoch}-{index}")
    full = (sample["width"], sample["height"])
    geometry = train and steps["geometry"]
    s = _scale(sample, canvas, rng, geometry, full)
    img = _load(sample["image"], (max(1, round(full[0] * s)), max(1, round(full[1] * s))))
    th, tw = img.shape[:2]
    sx, sy = tw / full[0], th / full[1]
    boxes = [[b[0], b[1] * sx, b[2] * sy, b[3] * sx, b[4] * sy] for b in sample["boxes"]]
    ignore = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in sample.get("ignore", [])]
    C = canvas
    ox = oy = 0
    if geometry:
        if boxes and rng.random() > 0.15:
            _, bx, by, bw, bh = rng.choice(boxes)
            ox = int(round(bx + bw / 2 + rng.uniform(-C / 3, C / 3) - C / 2))
            oy = int(round(by + bh / 2 + rng.uniform(-C / 3, C / 3) - C / 2))
        else:
            ox = rng.randint(min(0, tw - C), max(0, tw - C))
            oy = rng.randint(min(0, th - C), max(0, th - C))
        ox = max(-C // 2, min(ox, tw - C // 2))
        oy = max(-C // 2, min(oy, th - C // 2))
    out = np.full((C, C, 3), GREY, np.uint8)
    x0, y0, dx, dy = max(0, ox), max(0, oy), max(0, -ox), max(0, -oy)
    cw, ch = min(tw - x0, C - dx), min(th - y0, C - dy)
    if cw > 0 and ch > 0:
        out[dy:dy + ch, dx:dx + cw] = img[y0:y0 + ch, x0:x0 + cw]
    boxes = _clip([[c, x - ox, y - oy, w, h] for c, x, y, w, h in boxes], C, MIN_BOX)
    ignore = _clip([[x - ox, y - oy, w, h] for x, y, w, h in ignore], C, 1)
    if geometry and rng.random() < 0.5:
        out = out[:, ::-1]
        for b in boxes:
            b[1] = C - b[1]
        for b in ignore:
            b[0] = C - b[0]
    if train and steps["colour"]:
        out = _hsv(np.ascontiguousarray(out), rng)
    if train and steps["blur"] and blur_rng.random() < steps["blur"]:
        sigma = blur_rng.uniform(*BLUR_SIGMA)
        out = ndimage.gaussian_filter(out, sigma=(sigma, sigma, 0))
    if train and steps["haze"] and haze_rng.random() < steps["haze"]:
        out = haze(out, haze_rng.uniform(*HAZE_STRENGTH), haze_rng.choice(HAZE_VEILS))
    return {"image": np.ascontiguousarray(out), "boxes": boxes, "ignore": ignore, "scale": s}
