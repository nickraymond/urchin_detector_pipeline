"""Tile layout, merging tile masks into frame coordinates, tile-edge duplicates (D23), and the
S1 drawing helper. Pure numpy: no torch here, so tests stay fast."""

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


# --- S3: tile masks -> frame masks, duplicates at tile edges (D23) ---
# A detection is a dict: prompt, score, box [x0, y0, x1, y1] in frame pixels (x1, y1 exclusive),
# mask (bool array the size of the box), region (the tile it came from, frame pixels).

EDGE_PX = 2  # a mask this close to an inner tile edge counts as cut off by it


def from_region(mask: np.ndarray, score: float, prompt: str,
                region: tuple[int, int, int, int]) -> dict | None:
    """One SAM mask over a whole region (tile or frame) -> a detection cropped to its tight
    box in frame pixels; None if the mask is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    rx, ry = region[0], region[1]
    return {"prompt": prompt, "score": float(score), "region": tuple(region),
            "box": [x0 + rx, y0 + ry, x1 + rx, y1 + ry],
            "mask": np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=bool)}


def area(det: dict) -> int:
    return int(det["mask"].sum())


def intersection(a: dict, b: dict) -> int:
    """Pixels two detections share."""
    x0, y0 = max(a["box"][0], b["box"][0]), max(a["box"][1], b["box"][1])
    x1, y1 = min(a["box"][2], b["box"][2]), min(a["box"][3], b["box"][3])
    if x0 >= x1 or y0 >= y1:
        return 0
    ma = a["mask"][y0 - a["box"][1]:y1 - a["box"][1], x0 - a["box"][0]:x1 - a["box"][0]]
    mb = b["mask"][y0 - b["box"][1]:y1 - b["box"][1], x0 - b["box"][0]:x1 - b["box"][0]]
    return int((ma & mb).sum())


def iou(a: dict, b: dict) -> float:
    i = intersection(a, b)
    return i / (area(a) + area(b) - i) if i else 0.0


def touches_inner_edge(det: dict, width: int, height: int) -> bool:
    """True if the mask reaches an edge of its tile that is not an edge of the frame."""
    rx0, ry0, rx1, ry1 = det["region"]
    x0, y0, x1, y1 = det["box"]
    return ((rx0 > 0 and x0 - rx0 <= EDGE_PX) or (ry0 > 0 and y0 - ry0 <= EDGE_PX)
            or (rx1 < width and rx1 - x1 <= EDGE_PX) or (ry1 < height and ry1 - y1 <= EDGE_PX))


def remove_duplicates(dets: list[dict], width: int, height: int,
                      iou_above: float = 0.5, cover: float = 0.5) -> list[dict]:
    """D23, per prompt: (1) a mask cut off by an inner tile edge is dropped when a mask of the
    same prompt from another tile, at least as big, covers at least `cover` of it; smallest
    first, so of two cut-off halves the bigger stays. (2) Then of masks with IoU above
    `iou_above` the higher score stays. Returns the kept detections, highest score first."""
    kept: list[dict] = []
    for prompt in dict.fromkeys(d["prompt"] for d in dets):
        mine = [d for d in dets if d["prompt"] == prompt]
        dropped: set[int] = set()
        order = sorted(range(len(mine)), key=lambda i: area(mine[i]))
        for i in order:
            a = mine[i]
            if not touches_inner_edge(a, width, height):
                continue
            need = cover * area(a)
            for j, b in enumerate(mine):
                if (j != i and j not in dropped and b["region"] != a["region"]
                        and area(b) >= area(a) and intersection(a, b) >= need):
                    dropped.add(i)
                    break
        left = sorted((d for i, d in enumerate(mine) if i not in dropped),
                      key=lambda d: -d["score"])
        out: list[dict] = []
        for d in left:
            if all(iou(d, k) <= iou_above for k in out):
                out.append(d)
        kept += out
    return sorted(kept, key=lambda d: -d["score"])


def union_by_prompt(dets: list[dict]) -> list[dict]:
    """Area classes (rock, kelp): an area cut into pieces by tiles is one area, so all masks
    of one prompt become one mask with the best score. Prompts stay apart in sam_raw; the
    class map unions them later (D22 rule 2)."""
    out = []
    for prompt in dict.fromkeys(d["prompt"] for d in dets):
        mine = [d for d in dets if d["prompt"] == prompt]
        x0 = min(d["box"][0] for d in mine)
        y0 = min(d["box"][1] for d in mine)
        x1 = max(d["box"][2] for d in mine)
        y1 = max(d["box"][3] for d in mine)
        mask = np.zeros((y1 - y0, x1 - x0), bool)
        for d in mine:
            bx0, by0, bx1, by1 = d["box"]
            mask[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] |= d["mask"]
        out.append({"prompt": prompt, "score": max(d["score"] for d in mine),
                    "box": [x0, y0, x1, y1], "mask": mask, "region": mine[0]["region"]})
    return out
