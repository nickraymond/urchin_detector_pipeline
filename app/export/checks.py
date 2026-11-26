"""Export checks (spec acceptance criteria; TRACKER S5 bite 2). They read only the export folder,
not the project, so they test the files a trainer will get. Each check returns
{"name", "ok", "detail"}; the detail names up to 5 failing files.

- COCO loads with pycocotools, and every annotation has an outline, a known class and a box
  inside its image.
- every image named in COCO, seg_masks and the metrics is in images/
- every seg_masks PNG is 8-bit, the size of its image, and uses only ids in classes.json
- frame_metrics.csv matches: % cover per class against the PNG, counts against COCO
- every exported image is in exactly one split

macOS writes a `._<name>` file next to every file on an exFAT disk such as the SSD (it holds the
file's extended attributes, e.g. com.apple.provenance); those are not ours and are ignored.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

COVER_TOL = 0.0002   # percent; the CSV rounds to 4 decimals


def _result(name: str, bad: list[str], good: str) -> dict:
    return {"name": name, "ok": not bad,
            "detail": good if not bad else f"{len(bad)} problem(s): " + "; ".join(bad[:5])}


def coco_loads(out: Path) -> dict:
    name = "COCO loads with pycocotools"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(out / "detector_coco.json"))
    except Exception as e:        # pycocotools raises plain AssertionError / KeyError / ValueError
        return _result(name, [f"{type(e).__name__}: {e}"], "")
    bad, cats = [], set(coco.getCatIds())
    for a in coco.dataset["annotations"]:
        img = coco.imgs.get(a["image_id"])
        x, y, w, h = a["bbox"]
        if img is None:
            bad.append(f"annotation {a['id']}: no image {a['image_id']}")
        elif a["category_id"] not in cats:
            bad.append(f"annotation {a['id']}: unknown class {a['category_id']}")
        elif not a["segmentation"] or any(len(p) < 6 for p in a["segmentation"]):
            bad.append(f"annotation {a['id']} on {img['file_name']}: no outline")
        elif x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > img["width"] or y + h > img["height"]:
            bad.append(f"annotation {a['id']} on {img['file_name']}: box outside the image")
    return _result(name, bad, f"{len(coco.imgs)} images, {len(coco.anns)} objects, "
                              f"{len(cats)} classes")


def _metrics(out: Path) -> list[dict]:
    with (out / "frame_metrics.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def images_present(out: Path) -> dict:
    coco = json.loads((out / "detector_coco.json").read_text())
    names = {i["file_name"] for i in coco["images"]} | {r["frame"] for r in _metrics(out)}
    bad = sorted(n for n in names if not (out / "images" / n).is_file())
    return _result("images present", bad, f"{len(names)} images")


def masks_match(out: Path) -> dict:
    ids = {c["id"] for c in json.loads((out / "seg_masks" / "classes.json").read_text())["classes"]}
    bad = []
    rows = _metrics(out)
    for r in rows:
        p = out / "seg_masks" / r["seg_mask"]
        if not p.is_file():
            bad.append(f"{r['seg_mask']}: missing")
            continue
        with Image.open(p) as m, Image.open(out / "images" / r["frame"]) as im:
            if m.mode != "L":
                bad.append(f"{p.name}: mode {m.mode}, not 8-bit")
            elif m.size != im.size:
                bad.append(f"{p.name}: {m.size[0]}x{m.size[1]}, image {im.size[0]}x{im.size[1]}")
            else:
                extra = set(np.unique(np.asarray(m)).tolist()) - ids
                if extra:
                    bad.append(f"{p.name}: ids {sorted(extra)} not in classes.json")
    return _result("mask sizes and ids", bad, f"{len(rows)} masks, sizes match, ids in "
                                              "classes.json")


def metrics_match(out: Path) -> dict:
    classes = json.loads((out / "seg_masks" / "classes.json").read_text())["classes"]
    coco = json.loads((out / "detector_coco.json").read_text())
    img_of = {i["file_name"]: i["id"] for i in coco["images"]}
    count: dict[tuple[int, int], int] = {}
    for a in coco["annotations"]:
        count[(a["image_id"], a["category_id"])] = count.get((a["image_id"], a["category_id"]),
                                                             0) + 1
    bad, rows = [], _metrics(out)
    for r in rows:
        p = out / "seg_masks" / r["seg_mask"]
        if not p.is_file():
            continue                                   # masks_match reports it
        with Image.open(p) as m:
            a = np.asarray(m)
        cover = dict(zip(*[x.tolist() for x in np.unique(a, return_counts=True)]))
        for c in classes:
            col = "unknown_cover_pct" if c["id"] == 0 else f"{c['name']} cover_pct"
            want = 100 * cover.get(c["id"], 0) / a.size
            if abs(float(r[col]) - want) > COVER_TOL:
                bad.append(f"{r['frame']}: {col} {r[col]} but the mask gives {want:.4f}")
            if c["kind"] == "object" and r["frame"] in img_of:
                n = count.get((img_of[r["frame"]], c["id"]), 0)
                if int(r[f"{c['name']} count"]) != n:
                    bad.append(f"{r['frame']}: {c['name']} count {r[c['name'] + ' count']} but "
                               f"COCO has {n}")
    return _result("metrics match masks and COCO", bad, f"{len(rows)} frames")


def splits_complete(out: Path) -> dict:
    seen: dict[str, list[str]] = {}
    for s in ("train", "valid", "test"):
        for line in (out / "splits" / f"{s}.txt").read_text().splitlines():
            seen.setdefault(line, []).append(s)
    names = {p.name for p in (out / "images").iterdir() if not p.name.startswith("._")}
    bad = [f"{n}: in no split" for n in sorted(names - set(seen))]
    bad += [f"{n}: in {', '.join(v)}" for n, v in sorted(seen.items()) if len(v) > 1]
    bad += [f"{n}: in a split but not in images/" for n in sorted(set(seen) - names)]
    return _result("every image in one split", bad, f"{len(names)} images")


CHECKS = (coco_loads, images_present, masks_match, metrics_match, splits_complete)


def run_all(out: Path) -> list[dict]:
    found = []
    for check in CHECKS:
        try:
            found.append(check(Path(out)))
        except Exception as e:     # a missing or broken file: report it as that check's failure
            found.append({"name": check.__name__.replace("_", " "), "ok": False,
                          "detail": f"{type(e).__name__}: {e}"})
    return found
