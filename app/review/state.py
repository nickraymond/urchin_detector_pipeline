"""Label review state (S5 bite 1, DESIGN D40): what a reviewer did, kept in state.sqlite.

sam_raw/<frame>.json stays what SAM returned (plus a stable mask_id per mask, and masks added by
hand in the unit `added`). Review lives in three tables (app/db.py):

- mask_review (frame, mask_id): class_id (a changed class, else NULL), deleted, reviewed.
  Any action on a mask marks it reviewed; accepting a frame marks every mask shown on it
  reviewed. A re-run keeps reviewed masks with their ids (results.write_frame, SPEC Q7).
- frame_review (frame): accepted, split_ok (NULL = not marked, 1 yes, 0 no: only "no" leaves
  the frame out of the detector file, D41), checked (JSON list of class ids fully checked).
- label_review_log: one row per action with the rows it changed as they were before, so `undo`
  puts them back. Undo is a stack, newest first, per frame or for the whole project.

A frame's final masks = sam_raw with the review laid over it, then the overlap rules (D22, D31;
app/sam/overlap.py). A frame cannot be accepted while it shows an untagged urchin (D18).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from app.sam import overlap, results


class ReviewError(ValueError):
    """A review action that cannot be done; the message says why and what to do."""


def reviewed_ids(folder: Path, frame: str) -> set[int]:
    """Mask ids marked reviewed on this frame. Read-only, for the SAM worker: a re-run keeps
    them. A project without state.sqlite (tests) has none."""
    p = Path(folder).absolute() / "state.sqlite"
    if not p.is_file():
        return set()
    conn = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'mask_review'").fetchone():
            return set()                       # a database from before S5, not opened since
        return {r[0] for r in conn.execute(
            "SELECT mask_id FROM mask_review WHERE frame = ? AND reviewed = 1", (frame,))}
    finally:
        conn.close()


def mask_rows(conn: sqlite3.Connection, frame: str) -> dict[int, dict]:
    return {r["mask_id"]: dict(r) for r in conn.execute(
        "SELECT mask_id, class_id, deleted, reviewed FROM mask_review WHERE frame = ?", (frame,))}


def frame_state(conn: sqlite3.Connection, frame: str) -> dict:
    """{"accepted": bool, "split_ok": True / False / None, "checked": [class ids]}."""
    r = conn.execute("SELECT * FROM frame_review WHERE frame = ?", (frame,)).fetchone()
    if r is None:
        return {"accepted": False, "split_ok": None, "checked": []}
    return {"accepted": bool(r["accepted"]),
            "split_ok": None if r["split_ok"] is None else bool(r["split_ok"]),
            "checked": json.loads(r["checked"])}


def overlay(raw: dict, rows: dict[int, dict]) -> dict:
    """A copy of a sam_raw record with the review laid over each mask (sam_raw is not changed):
    a changed class_id, reviewed, deleted."""
    out = {**raw, "units": {}}
    for key, unit in raw.get("units", {}).items():
        masks = []
        for m in unit.get("masks", []):
            r = rows.get(m["mask_id"])
            if r:
                m = {**m, "reviewed": bool(m.get("reviewed") or r["reviewed"]),
                     "deleted": bool(r["deleted"]),
                     "class_id": m.get("class_id") if r["class_id"] is None else r["class_id"]}
            masks.append(m)
        out["units"][key] = {**unit, "masks": masks}
    return out


def _read(folder: Path, frame: str) -> dict:
    raw = results.read(folder, frame)
    if raw is None:
        raise ReviewError(f"{frame} has no SAM results yet: run SAM labeling on it first.")
    return raw


def final(folder: Path, conn: sqlite3.Connection, frame: str, class_list: list[dict]) -> dict:
    """The frame as review shows and export writes it: {"frame", "width", "height", "objects",
    "areas", "untagged" (mask ids), "raw" (the overlaid record, for class_map)}. Each object is
    {mask_id, unit, class_id (None = untagged urchin), paint, score, box, area, flag, reviewed,
    from_older_settings, added}; each area {mask_id, unit, class_id, box, area, reviewed}."""
    raw = overlay(_read(folder, frame), mask_rows(conn, frame))
    objs = [{"mask_id": o["mask"]["mask_id"], "unit": o["unit"],
             "class_id": o["mask"].get("class_id"), "paint": o["paint"],
             "score": o["mask"]["score"], "box": o["mask"]["box"], "area": o["mask"]["area"],
             "flag": o["flag"], "reviewed": bool(o["mask"].get("reviewed")),
             "from_older_settings": bool(o["mask"].get("from_older_settings")),
             "added": o["unit"] == overlap.ADDED}
            for o in overlap.objects(raw, class_list)]
    area_ids = {c["id"] for c in class_list if c["kind"] == "area"}
    area_units = {u["key"] for u in overlap.classes_mod.run_units(class_list)
                  if u["kind"] == "area"}
    areas = [{"mask_id": m["mask_id"], "unit": k, "class_id": m.get("class_id"),
              "box": m["box"], "area": m["area"], "reviewed": bool(m.get("reviewed"))}
             for k, u in raw["units"].items() if k in area_units
             for m in u["masks"] if not m.get("deleted") and m.get("class_id") in area_ids]
    return {"frame": frame, "width": raw["width"], "height": raw["height"], "objects": objs,
            "areas": areas, "untagged": sorted(o["mask_id"] for o in objs if o["class_id"] is None),
            "raw": raw}


def class_map(fin: dict, class_list: list[dict]) -> np.ndarray:
    """The final class map of a frame from `final()` (D22 rules 2-7, after review)."""
    return overlap.class_map(fin["raw"], class_list)


# ---- actions: each one logs the rows it changes, then commits ----

def _unit_of(raw: dict, mask_ids: list[int]) -> dict[int, str]:
    if not mask_ids:
        raise ReviewError("No mask selected: click a mask first.")
    where = {m["mask_id"]: k for k, u in raw.get("units", {}).items() for m in u["masks"]}
    missing = [i for i in mask_ids if i not in where]
    if missing:
        raise ReviewError(f"No mask {missing} on {raw.get('frame')}: reload the page "
                          "(the frame may have been labeled again).")
    return {i: where[i] for i in mask_ids}


def _frame_row(conn, frame) -> dict | None:
    r = conn.execute("SELECT accepted, split_ok, checked FROM frame_review WHERE frame = ?",
                     (frame,)).fetchone()
    return dict(r) if r else None


def _log(conn, frame: str, action: str, mask_ids=(), with_frame: bool = False) -> None:
    rows = mask_rows(conn, frame)
    before = {"masks": {str(i): rows.get(i) for i in mask_ids},
              "frame": _frame_row(conn, frame) if with_frame else "same"}
    conn.execute("INSERT INTO label_review_log (frame, action, before) VALUES (?, ?, ?)",
                 (frame, action, json.dumps(before)))


def _put_mask(conn, frame: str, mask_id: int, **change) -> None:
    row = {"class_id": None, "deleted": 0, "reviewed": 0,
           **{k: v for k, v in (mask_rows(conn, frame).get(mask_id) or {}).items()
              if k != "mask_id"}, **change, "reviewed": 1}
    conn.execute("INSERT INTO mask_review (frame, mask_id, class_id, deleted, reviewed) "
                 "VALUES (?, ?, ?, ?, ?) ON CONFLICT (frame, mask_id) DO UPDATE SET "
                 "class_id = excluded.class_id, deleted = excluded.deleted, "
                 "reviewed = excluded.reviewed, at = datetime('now', 'localtime')",
                 (frame, mask_id, row["class_id"], int(row["deleted"]), 1))


def _put_frame(conn, frame: str, **change) -> None:
    row = {**(_frame_row(conn, frame) or {"accepted": 0, "split_ok": None, "checked": "[]"}),
           **change}
    conn.execute("INSERT INTO frame_review (frame, accepted, split_ok, checked) "
                 "VALUES (?, ?, ?, ?) ON CONFLICT (frame) DO UPDATE SET "
                 "accepted = excluded.accepted, split_ok = excluded.split_ok, "
                 "checked = excluded.checked, at = datetime('now', 'localtime')",
                 (frame, int(row["accepted"]), row["split_ok"], row["checked"]))


def set_class(conn, folder: Path, frame: str, mask_ids: list[int], class_id: int,
              class_list: list[dict]) -> None:
    """Change the class of masks (also tags an urchin: its type is its class, D18, D39). The
    new class must be in classes.yaml and of the same kind (object or area) as the mask."""
    by_id = {c["id"]: c for c in class_list}
    if class_id not in by_id:
        raise ReviewError(f"No class with id {class_id} in classes.yaml.")
    kinds = {u["key"]: u["kind"] for u in overlap.classes_mod.run_units(class_list)}
    for mid, unit in _unit_of(_read(folder, frame), mask_ids).items():
        kind = "object" if unit == overlap.ADDED else kinds.get(unit)
        if kind != by_id[class_id]["kind"]:
            raise ReviewError(f"Mask {mid} is {'an ' + kind if kind else 'of a removed class'}"
                              f"; {by_id[class_id]['name']} is an {by_id[class_id]['kind']} "
                              "class. Pick a class of the same kind.")
    _log(conn, frame, f"class {by_id[class_id]['name']}", mask_ids)
    for mid in mask_ids:
        _put_mask(conn, frame, mid, class_id=class_id)
    conn.commit()


def delete(conn, folder: Path, frame: str, mask_ids: list[int]) -> None:
    """`X`: the masks are no longer shown, counted or exported (sam_raw keeps them)."""
    _unit_of(_read(folder, frame), mask_ids)
    _log(conn, frame, "delete", mask_ids)
    for mid in mask_ids:
        _put_mask(conn, frame, mid, deleted=1)
    conn.commit()


def accept(conn, folder: Path, frame: str, class_list: list[dict]) -> None:
    """`Enter`: accept the frame for export (D41); refused while an urchin is untagged (D18).
    Every mask shown on it becomes reviewed, so a re-run keeps it."""
    fin = final(folder, conn, frame, class_list)
    if fin["untagged"]:
        n = len(fin["untagged"])
        raise ReviewError(f"{n} urchin{'s' if n > 1 else ''} on this frame "
                          f"{'have' if n > 1 else 'has'} no type yet: tag "
                          f"{'each' if n > 1 else 'it'} purple, red or black first (D18).")
    ids = [o["mask_id"] for o in fin["objects"]] + [a["mask_id"] for a in fin["areas"]]
    _log(conn, frame, "accept", ids, with_frame=True)
    for mid in ids:
        _put_mask(conn, frame, mid)
    _put_frame(conn, frame, accepted=1)
    conn.commit()


def reopen(conn, frame: str) -> None:
    """Take a frame back out of the accepted set (its masks stay reviewed)."""
    _log(conn, frame, "reopen", with_frame=True)
    _put_frame(conn, frame, accepted=0)
    conn.commit()


def set_split(conn, frame: str, ok: bool | None) -> None:
    """Split check: were touching urchins separated correctly? None = not marked."""
    _log(conn, frame, f"split check {'not marked' if ok is None else 'yes' if ok else 'no'}",
         with_frame=True)
    _put_frame(conn, frame, split_ok=None if ok is None else int(ok))
    conn.commit()


def set_checked(conn, frame: str, class_id: int, checked: bool,
                class_list: list[dict]) -> None:
    """Fully checked: the reviewer looked for this class on this frame (spec §6)."""
    by_id = {c["id"]: c for c in class_list}
    if class_id not in by_id:
        raise ReviewError(f"No class with id {class_id} in classes.yaml.")
    now = set(frame_state(conn, frame)["checked"])
    now = (now | {class_id}) if checked else (now - {class_id})
    _log(conn, frame, f"{by_id[class_id]['name']} {'checked' if checked else 'not checked'}",
         with_frame=True)
    _put_frame(conn, frame, checked=json.dumps(sorted(now)))
    conn.commit()


def undo(conn, frame: str | None = None) -> str | None:
    """`Z`: undo the newest action (on this frame, or anywhere if frame is None). Returns what
    was undone, or None when there is nothing left to undo."""
    q = "SELECT * FROM label_review_log" + (" WHERE frame = ?" if frame else "") + \
        " ORDER BY id DESC LIMIT 1"
    r = conn.execute(q, (frame,) if frame else ()).fetchone()
    if r is None:
        return None
    before = json.loads(r["before"])
    for mid, row in before["masks"].items():
        conn.execute("DELETE FROM mask_review WHERE frame = ? AND mask_id = ?", (r["frame"], mid))
        if row:
            conn.execute("INSERT INTO mask_review (frame, mask_id, class_id, deleted, reviewed) "
                         "VALUES (?, ?, ?, ?, ?)", (r["frame"], int(mid), row["class_id"],
                                                    row["deleted"], row["reviewed"]))
    if before["frame"] != "same":
        conn.execute("DELETE FROM frame_review WHERE frame = ?", (r["frame"],))
        if before["frame"]:
            f = before["frame"]
            conn.execute("INSERT INTO frame_review (frame, accepted, split_ok, checked) "
                         "VALUES (?, ?, ?, ?)", (r["frame"], f["accepted"], f["split_ok"],
                                                 f["checked"]))
    conn.execute("DELETE FROM label_review_log WHERE id = ?", (r["id"],))
    conn.commit()
    return f"{r['action']} on {r['frame']}"
