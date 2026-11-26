"""Export (spec §Outputs and folder layout; DESIGN §Planned for S5, D41, D43, D44).

`run()` rebuilds `<project>/export/` from scratch from sam_raw + the review state, so
re-exporting is always safe: everything is written into `export.new/`, checked, and only then
swapped in for the old `export/` (an export that fails half way leaves the old one in place).

export/images/             accepted frames, upright (D44: copied byte for byte without an EXIF
                           rotation, else re-saved upright, JPEG quality 95)
export/detector_coco.json  object classes only: per object a box, a polygon outline and the class;
                           frames with split check "no" left out; each image lists the classes
                           fully checked on it (`checked_classes`, ids) (D41, D42)
export/seg_masks/          one 8-bit PNG per accepted frame, pixel = class id (D22 class map);
                           seg_masks/classes.json maps ids to names and colours
export/splits/             train.txt, valid.txt, test.txt (image file names) and splits.json
export/frame_metrics.csv   per frame: source, timestamp, split, count and % cover per class,
                           largest object %, classes fully checked
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import contourpy
import numpy as np
from PIL import Image

from app import images
from app.export import splits as splits_mod
from app.review import state
from app.sam import classes as classes_mod
from app.sam import results

JPEG_QUALITY = 95          # as for extraction (D17, D44)
SIMPLIFY_PX = 0.5          # polygon points may move at most this far (Douglas-Peucker)


def accepted_frames(conn) -> list[dict]:
    """Accepted frames still kept in frames/ (a frame rejected after it was accepted is left
    out): [{"file", "source", "timestamp_ms"}] in name order."""
    rows = conn.execute(
        "SELECT f.file, f.video, f.timestamp_ms FROM frames f JOIN frame_review r "
        "ON r.frame = f.file WHERE r.accepted = 1 AND f.status IN ('kept', 'kept_hand') "
        "ORDER BY f.file").fetchall()
    return [{"file": r[0], "source": r[1] or "", "timestamp_ms": r[2]} for r in rows]


# ---- images ----

def export_image(src: Path, dst: Path) -> tuple[int, int]:
    """D44: upright pixels. Returns the exported (width, height)."""
    with Image.open(src) as im:
        orientation = im.getexif().get(0x0112, 1)
        if orientation in (None, 1):
            shutil.copyfile(src, dst)
            return im.size
        im.load()
        up = images.upright(im)
        if (im.format or "").upper() == "PNG":
            up.save(dst, format="PNG")
        else:
            up.convert("RGB").save(dst, format="JPEG", quality=JPEG_QUALITY)
        return up.size


# ---- polygons ----

def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on an open line of points (iterative)."""
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        rel = pts[a + 1:b] - pts[a]
        n = np.hypot(*seg)
        d = (np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / n if n
             else np.hypot(rel[:, 0], rel[:, 1]))
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def polygons(crop: np.ndarray, x0: int, y0: int, tol: float = SIMPLIFY_PX) -> list[list[float]]:
    """COCO polygon outlines ([x1, y1, x2, y2, ...] in frame pixels) of a bool mask cropped
    at (x0, y0). Marching squares at 0.5 (contourpy); holes are left out (COCO polygons are
    filled), so a ring becomes a disc. Pixel (i, j) covers [i, i + 1), hence the + 0.5."""
    z = np.pad(crop.astype(np.float64), 1)
    lines = contourpy.contour_generator(z=z, line_type=contourpy.LineType.Separate).lines(0.5)
    loops = []
    for ln in lines:
        if len(ln) < 4:
            continue
        x, y = ln[:, 0], ln[:, 1]
        signed = 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        loops.append((signed, ln))
    if not loops:
        return []
    outer = np.sign(max(loops, key=lambda s: abs(s[0]))[0])   # the largest loop is an outline
    out = []
    for signed, ln in loops:
        if np.sign(signed) != outer:
            continue                                              # a hole
        pts = _simplify(ln, tol)
        if len(pts) < 4:                                          # a speck: keep every point
            pts = ln
        if len(pts) >= 4:                                         # closed: first == last
            pts = pts[:-1] + [x0 - 0.5, y0 - 0.5]                 # pad 1, then + 0.5
            out.append([round(float(v), 1) for v in pts.reshape(-1)])
    return out


# ---- the export ----

def build(folder: Path, conn, class_list: list[dict], out: Path,
          progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Write every output into `out` (an empty folder). Returns a summary for the page."""
    by_id = {c["id"]: c for c in class_list}
    frames = accepted_frames(conn)
    for sub in ("images", "seg_masks", "splits"):
        (out / sub).mkdir(parents=True)
    sp = splits_mod.split(frames)
    stems = [Path(f["file"]).stem for f in frames]
    coco = {"info": {"description": "urchin_detector_pipeline export"}, "images": [],
            "annotations": [],
            "categories": [{"id": c["id"], "name": c["name"], "supercategory": c["kind"]}
                           for c in class_list if c["kind"] == "object"]}
    metrics, left_out = [], 0
    for n, f in enumerate(frames, 1):
        if progress:
            progress(n, len(frames), f["file"])
        fin = state.final(folder, conn, f["file"], class_list)
        if fin["untagged"]:       # accepted before a re-run brought new untagged urchins
            raise ExportError(f"{f['file']} is accepted but has {len(fin['untagged'])} untagged "
                              "urchin(s) (a re-run after accepting?): tag them on the Review page "
                              "and export again.")
        w, h = export_image(folder / "frames" / f["file"], out / "images" / f["file"])
        if (w, h) != (fin["width"], fin["height"]):
            raise ExportError(f"{f['file']}: the image is {w}x{h} but its SAM results are "
                              f"{fin['width']}x{fin['height']}. Label it again (Run).")
        cmap = state.class_map(fin, class_list)
        name = (Path(f["file"]).stem if stems.count(Path(f["file"]).stem) == 1
                else f["file"]) + ".png"
        Image.fromarray(cmap, mode="L").save(out / "seg_masks" / name)
        fr = state.frame_state(conn, f["file"])
        objs = [o for o in fin["objects"] if o["class_id"] in by_id]
        if fr["split_ok"] is False:
            left_out += 1
        else:
            img_id = len(coco["images"]) + 1
            coco["images"].append({"id": img_id, "file_name": f["file"], "width": w, "height": h,
                                   "seg_mask": name, "split": sp["by_frame"][f["file"]],
                                   "checked_classes": fr["checked"]})
            raw = fin["raw"]
            rles = {m["mask_id"]: m for u in raw["units"].values() for m in u["masks"]}
            for o in objs:
                x0, y0, x1, y1 = o["box"]
                crop = results.decode_box(rles[o["mask_id"]]["rle"], o["box"])
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": img_id,
                    "category_id": o["class_id"], "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": int(o["area"]), "segmentation": polygons(crop, x0, y0),
                    "iscrowd": 0, "score": o["score"], "added_by_hand": o["added"]})
        metrics.append(_metrics_row(f, fr, sp["by_frame"][f["file"]], name, cmap, objs,
                                    class_list, w * h))
    _write_json(out / "detector_coco.json", coco)
    _write_json(out / "seg_masks" / "classes.json", {"classes": [
        {"id": 0, "name": "unknown", "kind": "area", "colour": "#000000"}] + [
        {"id": c["id"], "name": c["name"], "kind": c["kind"], "colour": classes_mod.colour(c["id"]),
         **({"seg_as": c["seg_as"]} if c.get("seg_as") else {})} for c in class_list]})
    for s in splits_mod.SPLITS:
        (out / "splits" / f"{s}.txt").write_text(
            "".join(f"{f['file']}\n" for f in frames if sp["by_frame"][f["file"]] == s))
    _write_json(out / "splits" / "splits.json", {k: sp[k] for k in ("grouped_by", "warning",
                                                                     "groups")})
    _write_metrics(out / "frame_metrics.csv", metrics, class_list)
    return {"frames": len(frames), "coco_images": len(coco["images"]),
            "objects": len(coco["annotations"]), "split_no": left_out,
            "per_split": {s: sum(1 for v in sp["by_frame"].values() if v == s)
                          for s in splits_mod.SPLITS},
            "per_class": {c["name"]: sum(1 for a in coco["annotations"]
                                         if a["category_id"] == c["id"])
                          for c in class_list if c["kind"] == "object"},
            "grouped_by": sp["grouped_by"], "split_warning": sp["warning"]}


class ExportError(RuntimeError):
    """The export stopped; the message names the frame and what to do. The old export stays."""


def _metrics_row(f, fr, split, mask_name, cmap, objs, class_list, pixels) -> dict:
    ids, n = np.unique(cmap, return_counts=True)
    cover = dict(zip(ids.tolist(), n.tolist()))
    row = {"frame": f["file"], "source": f["source"], "timestamp_ms": f["timestamp_ms"],
           "split": split, "split_ok": {None: "", True: "yes", False: "no"}[fr["split_ok"]],
           "seg_mask": mask_name,
           "checked_classes": " ".join(str(i) for i in fr["checked"]),
           "largest_object_pct": round(100 * max((o["area"] for o in objs), default=0) / pixels, 4),
           "unknown_cover_pct": round(100 * cover.get(0, 0) / pixels, 4)}
    for c in class_list:
        if c["kind"] == "object":
            row[f"{c['name']} count"] = sum(1 for o in objs if o["class_id"] == c["id"])
        row[f"{c['name']} cover_pct"] = round(100 * cover.get(c["id"], 0) / pixels, 4)
    return row


def _write_metrics(p: Path, rows: list[dict], class_list: list[dict]) -> None:
    head = ["frame", "source", "timestamp_ms", "split", "split_ok", "seg_mask",
            "checked_classes", "largest_object_pct", "unknown_cover_pct"]
    for c in class_list:
        head += ([f"{c['name']} count"] if c["kind"] == "object" else []) + \
                [f"{c['name']} cover_pct"]
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(rows)


def _write_json(p: Path, data: dict) -> None:
    p.write_text(json.dumps(data, separators=(",", ":")))


def run(folder: Path, conn, progress=None) -> dict:
    """Rebuild export/ (see the module doc). Returns {"ok", "summary", "checks", "folder"}.
    When a check fails the new export is not swapped in: it stays in export.failed/ to look at
    and the previous export/ is unchanged. A build error raises ExportError (old export kept)."""
    from app.export import checks
    folder = Path(folder)
    cfg = classes_mod.read(folder)
    if cfg["data"] is None:
        raise ExportError("classes.yaml has problems: fix them on the Labeling page first.")
    class_list = cfg["data"]["classes"]
    new, old, dest = folder / "export.new", folder / "export.old", folder / "export"
    if old.exists() and not dest.exists():   # a run died between its two renames: put it back
        os.replace(old, dest)
    for p in (new, old):
        if p.exists():
            shutil.rmtree(p)
    try:
        summary = build(folder, conn, class_list, new, progress)
        found = checks.run_all(new)
    except Exception:
        shutil.rmtree(new, ignore_errors=True)
        raise
    (new / "checks.json").write_text(json.dumps(found, indent=1))
    (new / "summary.json").write_text(json.dumps(
        {**summary, "ok": all(c["ok"] for c in found),
         "at": datetime.now().isoformat(timespec="seconds")}, indent=1))
    ok = all(c["ok"] for c in found)
    if not ok:
        failed = folder / "export.failed"
        if failed.exists():
            shutil.rmtree(failed)
        os.replace(new, failed)
        return {"ok": False, "summary": summary, "checks": found, "folder": str(failed)}
    if dest.exists():
        os.replace(dest, old)
    os.replace(new, dest)
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(folder / "export.failed", ignore_errors=True)
    return {"ok": True, "summary": summary, "checks": found, "folder": str(dest)}
