"""Tile layout and drawing helpers for the SAM test. No torch here, so tests stay fast."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# One colour per prompt, in prompt order (RGB). Readable on blue-green water.
COLOURS = [
    (255, 64, 160), (170, 60, 255), (255, 60, 40), (255, 170, 0),
    (255, 235, 60), (0, 230, 255), (120, 255, 90), (255, 255, 255),
]


def tile_grid(width: int, height: int, tile: int = 1008, overlap: float = 0.2):
    """Square tiles of `tile` px covering the frame, about `overlap` shared between neighbours.

    Returns a list of (x0, y0, x1, y1). The last tile in each row/column is pushed back so it
    ends exactly at the frame edge. A side shorter than `tile` gets one tile of that length.
    """
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    step = max(1, int(tile * (1 - overlap)))

    def starts(length):
        if length <= tile:
            return [0]
        out = list(range(0, length - tile, step))
        out.append(length - tile)
        return out

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]


def draw_overlay(image: Image.Image, detections: list[dict], prompts: list[str],
                 max_width: int = 1920) -> Image.Image:
    """Colour each mask and outline its box. A detection is a dict with
    prompt, score, box [x0, y0, x1, y1] in frame pixels and mask (bool array for that box)."""
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    for det in detections:
        colour = np.array(COLOURS[prompts.index(det["prompt"]) % len(COLOURS)], np.float32)
        x0, y0 = int(det["box"][0]), int(det["box"][1])
        mask = det.get("mask")
        if mask is None or mask.size == 0:
            continue
        h, w = mask.shape
        region = base[y0:y0 + h, x0:x0 + w]
        m = mask[: region.shape[0], : region.shape[1]]
        region[m] = region[m] * 0.45 + colour * 0.55
    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    width = max(2, out.width // 640)
    for det in detections:
        colour = COLOURS[prompts.index(det["prompt"]) % len(COLOURS)]
        draw.rectangle([float(v) for v in det["box"]], outline=colour, width=width)
    if out.width > max_width:
        out = out.resize((max_width, round(out.height * max_width / out.width)))
    return out
