"""The evaluation scale and tiles (DESIGN D90), used by the trainer's validation (TRACKER S8 Fixes 2) and,
in S9, by the evaluate job. numpy only, so it is tested without torch.

Training resizes each frame by a rule that needs its labels (samples.BAND). A frame the model has never seen
gets one factor per model instead: EVAL_BOX (44 px, the band's middle) over the median shorter box side of
the *training* split at native pixels, worked out when training first starts and kept in recipe.json
(`train.eval_scale`). The resized frame is cut into canvas-sized tiles that overlap by the largest
training box at that scale, so every object lies whole in at least one tile. A detection that touches an edge
the tile shares with a neighbour is the cut-off copy of an object some other tile holds whole: it is dropped
(`cut_by_inner_edge`). The rest are put back in frame coordinates and NMS runs again across the seams (the
trainer does that part).
"""

from __future__ import annotations

import numpy as np

from app.train import samples

EVAL_BOX = (samples.BAND[0] + samples.BAND[1]) / 2       # 44 px
SCALE_LIMITS = (0.03, 1.5)                                # the bounds training itself uses (samples._scale)
EDGE_PX = 2                                               # a box this close to a tile's edge is taken as cut by it


def eval_scale(train: list[dict]) -> dict:
    """{"raw": 44 / median shorter side, "factor": the same clamped to SCALE_LIMITS, "median_side_px",
    "boxes"} over every box of the training samples; factor None when the training split has no box."""
    sides = sorted(min(b[3], b[4]) for s in train for b in s["boxes"])
    if not sides:
        return {"raw": None, "factor": None, "median_side_px": None, "boxes": 0}
    mid = len(sides) // 2
    median = sides[mid] if len(sides) % 2 else (sides[mid - 1] + sides[mid]) / 2
    raw = EVAL_BOX / max(median, 1e-6)
    return {"raw": round(raw, 6), "factor": round(min(max(raw, SCALE_LIMITS[0]), SCALE_LIMITS[1]), 6),
            "median_side_px": round(median, 3), "boxes": len(sides)}


def wanted_overlap(train: list[dict], factor: float) -> int:
    """The largest training box's longer side at the eval factor, plus the margin at each side that
    `cut_by_inner_edge` treats as touching: with this overlap that box lies whole and clear of the edges in some tile."""
    longest = max((max(b[3], b[4]) for s in train for b in s["boxes"]), default=0.0)
    return int(round(longest * factor) + 2 * EDGE_PX + 1)


def overlap(train: list[dict], factor: float, canvas: int) -> int:
    """The tiles' overlap: `wanted_overlap`, at most half a canvas. When it is capped, objects larger than
    `guaranteed(overlap)` may lie whole in no tile clear of the edges, so `cut_by_inner_edge` keeps their
    edge-touching copies (QE PR 111)."""
    return min(canvas // 2, wanted_overlap(train, factor))


def guaranteed(overlap_px: int) -> int:
    """The longest box side that some tile is sure to hold whole and clear of its inner edges."""
    return max(0, overlap_px - 2 * EDGE_PX - 1)


def origins(length: int, canvas: int, overlap_px: int) -> list[int]:
    """Tile starts along one side of `length` px: the first at 0, the last flush with the far edge,
    neighbours `canvas - overlap_px` apart or closer. A side shorter than the canvas is one tile at 0
    (the rest of the canvas is grey)."""
    if length <= canvas:
        return [0]
    step = max(1, canvas - overlap_px)
    starts = list(range(0, length - canvas, step))
    return starts + [length - canvas]


def plan(width: int, height: int, canvas: int, overlap_px: int) -> list[tuple[int, int]]:
    """Every tile's top-left corner (x, y) in the resized frame, row by row."""
    return [(x, y) for y in origins(height, canvas, overlap_px) for x in origins(width, canvas, overlap_px)]



def cut_by_inner_edge(det: np.ndarray, ox: int, oy: int, canvas: int, width: int, height: int,
                      overlap_px: int) -> np.ndarray:
    """For detections [cx, cy, w, h, ...] in tile coordinates: True where the box touches an edge of the tile
    that lies inside the frame (a neighbouring tile overlaps it) and is no longer than `guaranteed(overlap_px)`,
    so another tile holds it whole. Edges on the frame's own border never count; a larger box is always kept."""
    if not len(det):
        return np.zeros(0, bool)
    x1, y1 = det[:, 0] - det[:, 2] / 2, det[:, 1] - det[:, 3] / 2
    x2, y2 = det[:, 0] + det[:, 2] / 2, det[:, 1] + det[:, 3] / 2
    touches = (((x1 <= EDGE_PX) & (ox > 0)) | ((y1 <= EDGE_PX) & (oy > 0))
               | ((x2 >= canvas - EDGE_PX) & (ox + canvas < width)) | ((y2 >= canvas - EDGE_PX) & (oy + canvas < height)))
    return touches & (np.maximum(det[:, 2], det[:, 3]) <= guaranteed(overlap_px))


def nms(det: np.ndarray, iou: float) -> np.ndarray:
    """Non-maximum suppression per class on [cx, cy, w, h, score, class, ...] rows (numpy, no torch): the rows
    kept, best score first. The seam merge of the evaluate job (D90); the trainer uses torchvision's."""
    if not len(det):
        return det
    det = det[np.argsort(-det[:, 4], kind="stable")]
    x1, y1 = det[:, 0] - det[:, 2] / 2, det[:, 1] - det[:, 3] / 2
    x2, y2 = det[:, 0] + det[:, 2] / 2, det[:, 1] + det[:, 3] / 2
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    keep, gone = [], np.zeros(len(det), bool)
    for i in range(len(det)):
        if gone[i]:
            continue
        keep.append(i)
        same = (det[:, 5] == det[i, 5]) & ~gone
        iw = np.clip(np.minimum(x2[i], x2) - np.maximum(x1[i], x1), 0, None)
        ih = np.clip(np.minimum(y2[i], y2) - np.maximum(y1[i], y1), 0, None)
        inter = iw * ih
        over = inter / (area[i] + area - inter + 1e-9) > iou
        gone |= same & over
    return det[keep]
