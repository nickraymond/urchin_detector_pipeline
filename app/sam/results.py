"""sam_raw/<frame file>.json: SAM's raw masks per frame and run unit (DESIGN, Planned for S3).

Written atomically after each frame, also when nothing is found. Per run unit: its settings
key (a hash of model, prompts and the SAM settings), those settings and prompts, the time
taken, and the masks as COCO RLE of the full upright frame. Re-run rule (SPEC Q7): a unit
runs on a frame only if its saved settings key differs or is missing; on a re-run its
unreviewed masks are replaced and reviewed ones stay, marked from_older_settings. Units not in
this run (e.g. a class removed from classes.yaml) keep their saved results.

sam_raw/class_ids.json lists {id: name} of every class with results, so classes.yaml can keep
those ids and names fixed (D6; app/sam/classes.py).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from app.sam.classes import USED_FILE

FORMAT = 1
KEY_SETTINGS = ("model", "confidence", "tile_px", "tile_overlap", "tile_above_px",
                "min_mask_area_px")


def raw_dir(folder: Path) -> Path:
    return Path(folder) / "sam_raw"


def raw_path(folder: Path, frame: str) -> Path:
    """One file per frame, named after the whole frame file name (a still's .jpg and .png of
    the same stem never share one)."""
    return raw_dir(folder) / f"{frame}.json"


def unit_settings(unit: dict, settings: dict) -> dict:
    return {"prompts": list(unit["prompts"]), "kind": unit["kind"],  # kind: areas are unioned
            **{k: settings.get(k) for k in KEY_SETTINGS}}


def settings_key(unit: dict, settings: dict) -> str:
    """Changes whenever anything that changes this unit's masks changes."""
    blob = json.dumps(unit_settings(unit, settings), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def encode(det: dict, width: int, height: int) -> dict:
    """COCO RLE of the detection's mask on the full frame."""
    full = np.zeros((height, width), dtype=np.uint8, order="F")
    x0, y0, x1, y1 = det["box"]
    full[y0:y1, x0:x1] = det["mask"]
    rle = mask_utils.encode(full)
    return {"size": [int(height), int(width)], "counts": rle["counts"].decode("ascii")}


def decode(rle: dict) -> np.ndarray:
    """Full-frame bool mask from a saved RLE."""
    return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}
                             ).astype(bool)


def read(folder: Path, frame: str) -> dict | None:
    p = raw_path(folder, frame)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except ValueError as e:
        raise ValueError(f"{p} cannot be read ({e}). Move it aside; the frame will be "
                         "labeled again on the next Run.") from e


def to_run(folder: Path, frame: str, units: list[dict], settings: dict) -> list[dict]:
    """The units whose saved settings key for this frame differs or is missing."""
    saved = (read(folder, frame) or {}).get("units", {})
    return [u for u in units
            if saved.get(u["key"], {}).get("settings_key") != settings_key(u, settings)]


def _atomic(p: Path, data: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")))
    os.replace(tmp, p)


def write_frame(folder: Path, frame: str, width: int, height: int, settings: dict,
                done: dict[str, dict]) -> dict:
    """Save one frame's results. `done` maps unit key -> {"unit", "seconds", "dets"}, dets
    being detections as made by app.sam.tiles (box + cropped mask)."""
    raw_dir(folder).mkdir(parents=True, exist_ok=True)
    data = read(folder, frame) or {"format": FORMAT, "frame": frame, "units": {}}
    data.update(width=width, height=height, model=settings.get("model"))
    for key, res in done.items():
        unit = res["unit"]
        old = data["units"].get(key, {}).get("masks", [])
        single = next(iter(unit["classes"])) if not unit["group"] else None
        masks = [{**m, "from_older_settings": True} for m in old if m.get("reviewed")]
        masks += [{"prompt": d["prompt"], "score": round(d["score"], 4), "box": d["box"],
                   "area": int(d["mask"].sum()), "rle": encode(d, width, height),
                   "class_id": single, "reviewed": False, "from_older_settings": False}
                  for d in res["dets"]]
        data["units"][key] = {
            "settings_key": settings_key(unit, settings),
            "settings": unit_settings(unit, settings), "kind": unit["kind"],
            "classes": {str(k): v for k, v in unit["classes"].items()},
            "seconds": round(res["seconds"], 2),
            "run_at": datetime.now().isoformat(timespec="seconds"), "masks": masks}
    _atomic(raw_path(folder, frame), data)
    note_class_ids(folder, done)
    return data


def note_class_ids(folder: Path, done: dict[str, dict]) -> None:
    """Add the classes of these units to sam_raw/class_ids.json (their ids now keep their
    names, D6)."""
    p = raw_dir(folder) / USED_FILE
    used = json.loads(p.read_text()) if p.is_file() else {}
    new = {str(k): v for res in done.values() for k, v in res["unit"]["classes"].items()}
    if any(used.get(k) != v for k, v in new.items()):
        _atomic(p, {**used, **new})
