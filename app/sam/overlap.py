"""Overlap rules (Lead call D22), applied when results are shown or exported; sam_raw/ stays raw.

1. Two object masks with IoU > 0.5, from any prompts or classes: the higher score is kept and
   flagged for review; the other is left out. D31: a reviewed mask beats an unreviewed one
   whatever the scores (spec: reviewed masks are never silently replaced).
2. Several prompts of one area class: their pixels are unioned.
3. Two area classes on one pixel: the class listed first in classes.yaml wins.
4. Objects sit on top of areas; where two kept objects overlap, the higher score takes the pixels.
5. A class with `seg_as` (kelp shoot) paints the id of that area class (kelp) in the class map.
6. Pixels no class claims are unknown, id 0.
7. An urchin not yet tagged purple or red paints UNTAGGED; it is never exported (D18).

Units saved in sam_raw/ but no longer in classes.yaml are left out (DESIGN, Planned for S3).
S5 (D40): masks added by hand sit in the unit `added` (objects with a class). A mask may carry
review state laid over it by app/review/state.py: `class_id` (changed class), `reviewed` and
`deleted`. A deleted mask is never shown, but it still wins rule 1 against unreviewed masks of
its own run unit, so a re-run's fresh copy of a mask the reviewer deleted stays hidden; another
class's detection of the same spot is not hidden by it. Among reviewed masks a kept one beats a
deleted one. An area mask paints its own `class_id` (a changed class) in that
class's place in the file order (rule 3).
Pure numpy + pycocotools: no torch, so the tests stay fast.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from app.sam import classes as classes_mod
from app.sam import results

UNKNOWN = 0
UNTAGGED = 255   # class-map value of an untagged urchin; never a class id (ids stay small)
IOU_SAME = 0.5
ADDED = "added"  # the sam_raw unit of masks added by hand in label review (D40, bite 5)


def _rle(m: dict) -> dict:
    return {"size": m["rle"]["size"], "counts": m["rle"]["counts"].encode("ascii")}


def objects(raw: dict, class_list: list[dict]) -> list[dict]:
    """The object masks of the current classes after rule 1 (and D31), lowest score first. Each is
    {"unit", "mask", "paint" (class-map value), "flag" (True: won an IoU > 0.5 clash)}."""
    by_id = {c["id"]: c for c in class_list}
    found = []
    keys = [u["key"] for u in classes_mod.run_units(class_list) if u["kind"] == "object"]
    for key in keys + [ADDED]:
        if key not in raw.get("units", {}):
            continue
        for m in raw["units"][key]["masks"]:
            cid = m.get("class_id")
            if cid is None:
                paint = UNTAGGED
            elif cid not in by_id or by_id[cid]["kind"] != "object":
                continue                # a class no longer in classes.yaml, or now an area
            elif by_id[cid].get("seg_as"):
                paint = next(x["id"] for x in class_list if x["name"] == by_id[cid]["seg_as"])
            else:
                paint = cid
            found.append({"unit": key, "mask": m, "paint": paint, "flag": False})
    # D31: a reviewed mask beats an unreviewed one whatever the scores, a kept one beats a
    # deleted one; else the higher score
    found.sort(key=lambda o: (not o["mask"].get("reviewed"), bool(o["mask"].get("deleted")),
                              -o["mask"]["score"]))
    if len(found) > 1:
        rles = [_rle(o["mask"]) for o in found]
        iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
        kept: list[int] = []
        for i in range(len(found)):            # reviewed first, then highest score
            # a deleted mask only hides masks of its own run unit (a re-run's fresh copy of it),
            # never another class's detection of the same spot (QE PR 44)
            clash = [k for k in kept if iou[i, k] > IOU_SAME
                     and not (found[k]["mask"].get("deleted") and found[k]["unit"] != found[i]["unit"])]
            if clash:
                found[clash[0]]["flag"] = True
            else:
                kept.append(i)
        found = [found[k] for k in kept]
    found = [o for o in found if not o["mask"].get("deleted")]
    return sorted(found, key=lambda o: o["mask"]["score"])   # rule 4: the higher score paints last


def class_map(raw: dict, class_list: list[dict]) -> np.ndarray:
    """One class id per pixel (uint8, upright frame size) by rules 2-7."""
    out = np.full((raw["height"], raw["width"]), UNKNOWN, np.uint8)
    order = {c["id"]: i for i, c in enumerate(class_list) if c["kind"] == "area"}
    areas = []
    for u in classes_mod.run_units(class_list):
        if u["kind"] != "area" or u["key"] not in raw.get("units", {}):
            continue
        for m in raw["units"][u["key"]]["masks"]:
            cid = m.get("class_id") or next(iter(u["classes"]))
            if cid in order and not m.get("deleted"):
                areas.append((order[cid], cid, m))
    for _, cid, m in sorted(areas, key=lambda a: -a[0]):   # the first listed is painted last
        x0, y0, x1, y1 = m["box"]
        out[y0:y1, x0:x1][results.decode_box(m["rle"], m["box"])] = cid
    for o in objects(raw, class_list):            # lowest score first: the higher score wins
        x0, y0, x1, y1 = o["mask"]["box"]
        crop = results.decode_box(o["mask"]["rle"], o["mask"]["box"])
        out[y0:y1, x0:x1][crop] = o["paint"]
    return out
