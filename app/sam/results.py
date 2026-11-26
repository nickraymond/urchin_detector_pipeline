"""sam_raw/<frame file>.json: SAM's raw masks per frame and run unit (DESIGN, Planned for S3).

Written atomically after each frame, also when nothing is found. Per run unit: its settings
key (a hash of model, prompts and the SAM settings), those settings and prompts, the time
taken, and the masks as COCO RLE of the full upright frame. Re-run rule (SPEC Q7): a unit
runs on a frame only if its saved settings key differs or is missing; on a re-run its
unreviewed masks are replaced and reviewed ones stay, marked from_older_settings. Units not in
this run (e.g. a class removed from classes.yaml) keep their saved results.

Every mask has a stable `mask_id` (S5, D40): an integer unique within its frame file, never
reused (`next_mask_id` counts on). Files written before S5 get ids in file order when read, so
the same file always gives the same ids, and the next write saves them. Review state lives in
state.sqlite under (frame, mask_id) (app/review/state.py).

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


def classes_key(units: list[dict], settings: dict) -> str:
    """One key for all units: changes whenever any unit's results would change (an estimate
    made with other classes or settings is out of date, S4 bite 3)."""
    keys = sorted(settings_key(u, settings) for u in units)
    return hashlib.sha1(",".join(keys).encode()).hexdigest()[:12]


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


def run_lengths(counts: str) -> np.ndarray:
    """COCO's compressed RLE string -> run lengths (zeros first, then ones, alternating),
    vectorised from pycocotools' rleFrString: 5 bits per character after subtracting 48,
    bit 0x20 = more characters follow, bit 0x10 on the last one = negative; from the third
    run on, each value is a difference to the run two places before."""
    c = np.frombuffer(counts.encode("ascii"), np.uint8).astype(np.int64) - 48
    if not len(c):
        return np.zeros(0, np.int64)
    ends = np.flatnonzero((c & 0x20) == 0)          # the last character of each number
    starts = np.r_[0, ends[:-1] + 1]
    k = np.arange(len(c)) - np.repeat(starts, ends - starts + 1)
    x = np.add.reduceat((c & 0x1f) << (5 * k), starts)
    neg = (c[ends] & 0x10) != 0
    x[neg] -= np.left_shift(1, 5 * (k[ends][neg] + 1))   # sign-extend
    x[1::2] = np.cumsum(x[1::2])   # run i adds run i-2 (C: if m > 2), so runs 3, 5, ... add up from 1
    x[2::2] = np.cumsum(x[2::2])   # and runs 4, 6, ... add up from 2
    return x


def decode_box(rle: dict, box: list[int]) -> np.ndarray:
    """The bool mask inside box (x0, y0, x1, y1) only, without making the full frame:
    COCO RLE runs down columns, so only the box's columns are filled (bite 5: an object's
    mask on a 5.3K frame is 16 MB when decoded whole, about 8 ms each)."""
    h, w = rle["size"]
    x0, y0, x1, y1 = box
    if (x1 - x0) * (y1 - y0) * 4 > h * w:   # a large area: pycocotools' C decode is faster
        return decode(rle)[y0:y1, x0:x1]
    runs = run_lengths(rle["counts"])
    end = np.cumsum(runs)
    s, e = (end - runs)[1::2], end[1::2]            # the runs of ones
    lo, hi = x0 * h, x1 * h
    s, e = np.clip(s, lo, hi) - lo, np.clip(e, lo, hi) - lo
    keep = e > s
    edge = np.zeros(hi - lo + 1, np.int32)
    np.add.at(edge, s[keep], 1)
    np.add.at(edge, e[keep], -1)
    cols = np.cumsum(edge[:-1]).astype(bool).reshape(x1 - x0, h)
    return cols.T[y0:y1]


def assign_ids(data: dict) -> dict:
    """Give every mask without a `mask_id` the next free one, in file order (units, then
    masks); sets `next_mask_id`. Deterministic: the same file always gets the same ids."""
    masks = [m for u in data.get("units", {}).values() for m in u.get("masks", [])]
    top = max((m["mask_id"] for m in masks if "mask_id" in m), default=0)
    nxt = max(int(data.get("next_mask_id", 1)), top + 1)
    for m in masks:
        if "mask_id" not in m:
            m["mask_id"], nxt = nxt, nxt + 1
    data["next_mask_id"] = nxt
    return data


def read(folder: Path, frame: str) -> dict | None:
    p = raw_path(folder, frame)
    if not p.is_file():
        return None
    try:
        return assign_ids(json.loads(p.read_text()))
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
                done: dict[str, dict], reviewed_ids: set[int] = frozenset()) -> dict:
    """Save one frame's results. `done` maps unit key -> {"unit", "seconds", "dets"}, dets
    being detections as made by app.sam.tiles (box + cropped mask). `reviewed_ids`: the mask
    ids state.sqlite marks reviewed on this frame (S5); they are kept with their ids."""
    raw_dir(folder).mkdir(parents=True, exist_ok=True)
    data = read(folder, frame) or assign_ids({"format": FORMAT, "frame": frame, "units": {}})
    data.update(width=width, height=height, model=settings.get("model"))
    for key, res in done.items():
        unit = res["unit"]
        old = data["units"].get(key, {}).get("masks", [])
        single = next(iter(unit["classes"])) if not unit["group"] else None
        masks = [{**m, "from_older_settings": True} for m in old
                 if m.get("reviewed") or m["mask_id"] in reviewed_ids]
        masks += [{"mask_id": data["next_mask_id"] + i, "prompt": d["prompt"], "score": round(d["score"], 4), "box": d["box"],
                   "area": int(d["mask"].sum()), "rle": encode(d, width, height),
                   "class_id": single, "reviewed": False, "from_older_settings": False}
                  for i, d in enumerate(res["dets"])]
        data["next_mask_id"] += len(res["dets"])
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


def follow_renames(folder: Path, class_list: list[dict]) -> int:
    """After classes.yaml is saved (D39): a renamed class (or group) keeps its id, but its run
    unit's key is its name, so saved units are re-keyed to the current key and their class
    names updated; class_ids.json gets the new names. Units whose ids are not all in one
    current unit are left alone (not shown, as for a removed class). Idempotent; returns the
    number of sam_raw files rewritten. Settings keys do not include names: no re-run."""
    from app.sam.classes import run_units
    units = run_units(class_list)
    names = {c["id"]: c["name"] for c in class_list}
    changed = 0
    d = raw_dir(folder)
    if not d.is_dir():
        return 0
    for p in sorted(d.glob("*.json")):
        if p.name == USED_FILE or p.name.startswith("._"):
            continue
        try:
            data = json.loads(p.read_text())
        except ValueError:
            continue                       # reported when the frame is read
        new, dirty = {}, False
        for key, u in data.get("units", {}).items():
            ids = {int(k) for k in u.get("classes", {})}
            cur = next((c for c in units if ids and ids <= set(c["classes"])), None)
            if cur is not None:
                fresh = {str(i): names[i] for i in sorted(ids)}
                if u["classes"] != fresh:
                    u, dirty = {**u, "classes": fresh}, True
                if cur["key"] != key and cur["key"] not in data["units"]:
                    key, dirty = cur["key"], True
            new[key] = u
        if dirty:
            data["units"] = new
            _atomic(p, data)
            changed += 1
    used_p = d / USED_FILE
    if used_p.is_file():
        used = json.loads(used_p.read_text())
        fixed = {k: names.get(int(k), v) for k, v in used.items()}
        if fixed != used:
            _atomic(used_p, fixed)
    return changed
