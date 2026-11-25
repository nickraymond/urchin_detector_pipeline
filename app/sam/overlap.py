"""Overlap rules (Lead call D22), applied when results are shown or exported; sam_raw/ stays raw.

1. Two object masks with IoU > 0.5, from any prompts or classes: the higher score is kept and
   flagged for review; the other is left out.
2. Several prompts of one area class: their pixels are unioned.
3. Two area classes on one pixel: the class listed first in classes.yaml wins.
4. Objects sit on top of areas; where two kept objects overlap, the higher score takes the pixels.
5. A class with `seg_as` (kelp shoot) paints the id of that area class (kelp) in the class map.
6. Pixels no class claims are unknown, id 0.
7. An urchin not yet tagged purple or red paints UNTAGGED; it is never exported (D18).

Units saved in sam_raw/ but no longer in classes.yaml are left out (DESIGN, Planned for S3).
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


def _rle(m: dict) -> dict:
    return {"size": m["rle"]["size"], "counts": m["rle"]["counts"].encode("ascii")}


def objects(raw: dict, class_list: list[dict]) -> list[dict]:
    """The object masks of the current classes after rule 1, lowest score first. Each is
    {"unit", "mask", "paint" (class-map value), "flag" (True: won an IoU > 0.5 clash)}."""
    by_id = {c["id"]: c for c in class_list}
    found = []
    for u in classes_mod.run_units(class_list):
        if u["kind"] != "object" or u["key"] not in raw.get("units", {}):
            continue
        for m in raw["units"][u["key"]]["masks"]:
            cid = m.get("class_id")
            if cid is None:
                paint = UNTAGGED
            elif cid not in by_id:      # a class tag no longer in classes.yaml
                continue
            elif by_id[cid].get("seg_as"):
                paint = next(x["id"] for x in class_list if x["name"] == by_id[cid]["seg_as"])
            else:
                paint = cid
            found.append({"unit": u["key"], "mask": m, "paint": paint, "flag": False})
    found.sort(key=lambda o: -o["mask"]["score"])
    if len(found) > 1:
        rles = [_rle(o["mask"]) for o in found]
        iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
        kept: list[int] = []
        for i in range(len(found)):            # highest score first
            clash = [k for k in kept if iou[i, k] > IOU_SAME]
            if clash:
                found[clash[0]]["flag"] = True
            else:
                kept.append(i)
        found = [found[k] for k in kept]
    return found[::-1]


def class_map(raw: dict, class_list: list[dict]) -> np.ndarray:
    """One class id per pixel (uint8, upright frame size) by rules 2-7."""
    out = np.full((raw["height"], raw["width"]), UNKNOWN, np.uint8)
    areas = [u for u in classes_mod.run_units(class_list)
             if u["kind"] == "area" and u["key"] in raw.get("units", {})]
    for u in reversed(areas):                    # the first listed is painted last: it wins
        cid = next(iter(u["classes"]))
        for m in raw["units"][u["key"]]["masks"]:
            out[results.decode(m["rle"])] = cid
    for o in objects(raw, class_list):            # lowest score first: the higher score wins
        x0, y0, x1, y1 = o["mask"]["box"]
        crop = results.decode(o["mask"]["rle"])[y0:y1, x0:x1]
        out[y0:y1, x0:x1][crop] = o["paint"]
    return out
