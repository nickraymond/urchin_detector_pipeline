"""Add a missed object (S5 bite 5, spec §6, D46): SAM 3's click or box prompt on one frame.

SAM looks at a square crop around the spot (at least one tile, 1008 px, or twice the box), so a
small far urchin is seen at full resolution, as in a Run (D23). The new mask goes into
sam_raw/<frame>.json under the unit `added` (D40) with `added_by_hand: true` and `reviewed:
true`, so a re-run never replaces it and it wins rule 1 against unreviewed masks (D31). Each add
has a request id, so a resumed job never adds the same object twice (D33).
"""

from __future__ import annotations

from pathlib import Path

from app.sam import overlap, results

REGION_PX = 1008


def region_for(width: int, height: int, point: list[int] | None = None,
               box: list[int] | None = None) -> tuple[int, int, int, int]:
    """The square crop SAM looks at: centred on the click or the box, REGION_PX or twice the
    box's longer side, inside the frame (smaller frames: the whole frame)."""
    if box is not None:
        x0, y0, x1, y1 = box
        cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(REGION_PX, 2 * max(x1 - x0, y1 - y0))
    else:
        (cx, cy), side = point, REGION_PX
    w, h = min(side, width), min(side, height)
    left = int(min(max(cx - w / 2, 0), width - w))
    top = int(min(max(cy - h / 2, 0), height - h))
    return left, top, int(left + w), int(top + h)


def check_request(width: int, height: int, point=None, box=None) -> None:
    """Raise ValueError unless exactly one of a point inside the frame or a box of at least
    4 px inside the frame is given."""
    if (point is None) == (box is None):
        raise ValueError("Give a click or a box, not both and not neither.")
    if point is not None:
        x, y = point
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("The click is outside the frame.")
    else:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or x1 - x0 < 4 or y1 - y0 < 4:
            raise ValueError("Drag a box of at least 4 pixels inside the frame.")


def done_requests(folder: Path, frame: str) -> set[str]:
    raw = results.read(folder, frame) or {}
    return {m.get("request") for m in raw.get("units", {}).get(overlap.ADDED, {}).get("masks", [])}


def save(folder: Path, frame: str, det: dict, class_id: int, request: str, width: int,
         height: int) -> int:
    """Append one hand-added mask (a detection from app.sam.tiles) to the frame's sam_raw file;
    returns its mask id. A request already saved returns that mask's id again."""
    results.raw_dir(folder).mkdir(parents=True, exist_ok=True)
    data = results.read(folder, frame) or results.assign_ids(
        {"format": results.FORMAT, "frame": frame, "width": width, "height": height, "units": {}})
    unit = data["units"].setdefault(overlap.ADDED, {"kind": "object", "added_by_hand": True,
                                                    "masks": []})
    old = next((m for m in unit["masks"] if m.get("request") == request), None)
    if old is not None:
        return old["mask_id"]
    mid = data["next_mask_id"]
    unit["masks"].append({
        "mask_id": mid, "prompt": det["prompt"], "score": round(det["score"], 4),
        "box": det["box"], "area": int(det["mask"].sum()), "rle": results.encode(det, width, height),
        "class_id": class_id, "reviewed": True, "from_older_settings": False,
        "added_by_hand": True, "request": request})
    data["next_mask_id"] = mid + 1
    results._atomic(results.raw_path(folder, frame), data)
    return mid
