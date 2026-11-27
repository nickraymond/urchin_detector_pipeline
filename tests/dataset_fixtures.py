"""A small synthetic export folder in the format app/export/writers.py writes (images,
detector_coco.json, seg_masks/ + classes.json, splits/, frame_metrics.csv, summary.json), built
without SAM so dataset tests are fast. The splits come from the export's own split function.
test_datasets_page.py also freezes a real export, so a drift between this and the writers shows."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from app.export import splits as export_splits

CLASSES = [{"id": 0, "name": "unknown", "kind": "area", "colour": "#000000"},
           {"id": 1, "name": "purple urchin", "kind": "object", "colour": "#aa00ff"},
           {"id": 3, "name": "sea star", "kind": "object", "colour": "#ff8800"},
           {"id": 9, "name": "kelp", "kind": "area", "colour": "#00aa00"}]
HEAD = ["frame", "source", "timestamp_ms", "split", "split_ok", "seg_mask", "checked_classes",
        "largest_object_pct", "unknown_cover_pct"]


def frame(source: str, i: int, urchins: int = 2, stars: int = 0, split_ok: str = "yes",
          checked=(1, 3)) -> dict:
    return {"file": f"{Path(source).stem}_abc123_{i * 2000}.png", "source": source,
            "urchins": urchins, "stars": stars, "split_ok": split_ok, "checked": list(checked)}


def dive(site: str, day: int, n: int, **kw) -> list[dict]:
    """n frames of one video of one dive: site + date group "<site> 2025-10-<day>"."""
    return [frame(f"/videos/{site}-25-10-{day:02d}-0900-xx-GOPR.MP4", i, **kw) for i in range(n)]


def fake_export(folder: Path, frames: list[dict], at: str = "2026-11-27T10:00:00") -> Path:
    """(Re)build <folder>/export/ from scratch, like a real Export does."""
    out = Path(folder) / "export"
    if out.exists():
        shutil.rmtree(out)
    for sub in ("images", "seg_masks", "splits"):
        (out / sub).mkdir(parents=True)
    sp = export_splits.split(frames)
    coco = {"info": {}, "images": [], "annotations": [],
            "categories": [{"id": c["id"], "name": c["name"], "supercategory": "object"}
                           for c in CLASSES if c["kind"] == "object"]}
    rows = []
    for n, f in enumerate(frames):
        rng = np.random.default_rng(abs(hash(f["file"])) % (2 ** 32))
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(
            out / "images" / f["file"])
        mask = Path(f["file"]).stem + ".png"
        Image.fromarray(np.zeros((48, 64), np.uint8), mode="L").save(out / "seg_masks" / mask)
        split = sp["by_frame"][f["file"]]
        if f["split_ok"] != "no":
            img_id = len(coco["images"]) + 1
            coco["images"].append({"id": img_id, "file_name": f["file"], "width": 64, "height": 48,
                                   "seg_mask": mask, "split": split,
                                   "checked_classes": f["checked"]})
            for cid, k in ((1, f["urchins"]), (3, f["stars"])):
                for j in range(k):
                    coco["annotations"].append({
                        "id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": cid,
                        "bbox": [4 + 6 * j, 4, 5, 5], "area": 25, "iscrowd": 0, "score": 0.9,
                        "segmentation": [[4, 4, 9, 4, 9, 9, 4, 9]], "added_by_hand": False})
        rows.append({"frame": f["file"], "source": f["source"], "timestamp_ms": n * 2000,
                     "split": split, "split_ok": f["split_ok"], "seg_mask": mask,
                     "checked_classes": " ".join(map(str, f["checked"])),
                     "largest_object_pct": 0.8, "unknown_cover_pct": 100.0})
    (out / "detector_coco.json").write_text(json.dumps(coco))
    (out / "seg_masks" / "classes.json").write_text(json.dumps({"classes": CLASSES}))
    for s in export_splits.SPLITS:
        (out / "splits" / f"{s}.txt").write_text(
            "".join(f"{f['file']}\n" for f in frames if sp["by_frame"][f["file"]] == s))
    (out / "splits" / "splits.json").write_text(json.dumps(
        {k: sp[k] for k in ("grouped_by", "warning", "groups")}))
    with (out / "frame_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HEAD)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(
        {"frames": len(frames), "objects": len(coco["annotations"]), "ok": True, "at": at,
         "split_no": sum(1 for f in frames if f["split_ok"] == "no")}))
    return out
