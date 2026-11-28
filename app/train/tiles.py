"""The evaluation scale and tiles (DESIGN D90), used by the trainer's validation (TRACKER S8 Fixes 2) and,
in S9, by the evaluate job. numpy only, so it is tested without torch.

Training resizes each frame by a rule that needs its labels (samples.BAND). A frame the model has never seen
gets one factor per model instead: EVAL_BOX (44 px, the band's middle) over the median shorter box side of
the *training* split at native pixels, worked out when training first starts and kept in recipe.json
(`train.eval_scale`). The resized frame is cut into canvas-sized tiles that overlap by the largest
training box at that scale, so every object lies whole in at least one tile; detections from all tiles are
put back in frame coordinates and NMS runs again across the seams (the trainer does that part).
"""

from __future__ import annotations

from app.train import samples

EVAL_BOX = (samples.BAND[0] + samples.BAND[1]) / 2       # 44 px
SCALE_LIMITS = (0.03, 1.5)                                # the bounds training itself uses (samples._scale)


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


def overlap(train: list[dict], factor: float, canvas: int) -> int:
    """The tiles' overlap: the largest training box's longer side at the eval factor, at most half a canvas
    (a larger object cannot fit one tile whole anyway)."""
    longest = max((max(b[3], b[4]) for s in train for b in s["boxes"]), default=0.0)
    return int(min(canvas // 2, round(longest * factor)))


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
