"""Stage 4, review frames (bite 4): `D` moves a frame to rejected/ (never erased), `K` keeps it,
`Z` undoes the last action. Spec §Stage details 4; DESIGN D20.

Kept frames are shown by default; "Show dropped" adds the blurry, duplicate and rejected ones.
`K` on any of those gives the status 'kept_hand' (home frames/), which the filters leave alone.
Every action is a row in review_actions (the undo history): file first, then its row, as in
extract.apply_filters, so a stop in between is repaired by the next Apply or Extract.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from PIL import Image

from app import db
from app.stages import extract

SHOWN = ("kept", "kept_hand")                                   # the default list
SHOWN_ALL = SHOWN + ("dropped_blur", "dropped_duplicate", "rejected")  # with "Show dropped"
LABELS = {"kept": "kept", "kept_hand": "kept by hand", "dropped_blur": "dropped: blurry",
          "dropped_duplicate": "dropped: duplicate", "rejected": "rejected"}


class ReviewError(ValueError):
    """An action that cannot be done; the message says why."""


def frames(conn: sqlite3.Connection, show_dropped: bool = False) -> list[dict]:
    """The frames to review, by video then time: {file, video (name), t, status, label}."""
    shown = SHOWN_ALL if show_dropped else SHOWN
    marks = ",".join("?" * len(shown))
    return [{"file": f, "video": Path(v or "").name, "t": (ms or 0) / 1000, "status": st,
             "label": LABELS[st]}
            for f, v, ms, st in conn.execute(
                f"SELECT file, video, timestamp_ms, status FROM frames WHERE status IN ({marks}) "
                "ORDER BY video, timestamp_ms, file", shown)]


def _set(conn: sqlite3.Connection, folder: Path, file: str, status: str) -> None:
    if not extract.move_home(folder, file, status):
        raise ReviewError(f"The file of {file} is missing from the project folder; "
                          "nothing changed.")
    conn.execute("UPDATE frames SET status = ? WHERE file = ?", (status, file))


def act(conn: sqlite3.Connection, folder: Path, file: str, key: str) -> dict:
    """Apply `D` (reject) or `K` (keep) to one frame. Returns {file, status, label, changed}.
    D on a rejected frame and K on a kept one change nothing and add no undo step."""
    row = conn.execute("SELECT status FROM frames WHERE file = ?", (file,)).fetchone()
    if row is None or row[0] not in SHOWN_ALL:
        raise ReviewError(f"{file} is not a frame of this review.")
    old = row[0]
    if key == "D":
        new = "rejected"
    elif key == "K":
        new = old if old in SHOWN else "kept_hand"
    else:
        raise ReviewError(f"Unknown action {key!r}: use D or K.")
    if new != old:
        _set(conn, folder, file, new)
        conn.execute("INSERT INTO review_actions (frame, old_status, new_status) VALUES (?, ?, ?)",
                     (file, old, new))
        conn.commit()
        mark_progress(conn)
    return {"file": file, "status": new, "label": LABELS[new], "changed": new != old}


def undo(conn: sqlite3.Connection, folder: Path) -> dict:
    """Undo the last review action: the frame goes back to its old status and folder, and the
    step leaves the history. If Extract or the filters changed the frame since, nothing moves;
    the step is dropped and the message says so. {} if there is nothing to undo."""
    last = conn.execute("SELECT id, frame, old_status, new_status FROM review_actions "
                        "ORDER BY id DESC LIMIT 1").fetchone()
    if last is None:
        return {}
    aid, file, old, new = last
    row = conn.execute("SELECT status FROM frames WHERE file = ?", (file,)).fetchone()
    conn.execute("DELETE FROM review_actions WHERE id = ?", (aid,))
    if row is None or row[0] != new:
        conn.commit()
        raise ReviewError(f"{file} changed since that step (now {row[0] if row else 'gone'}); "
                          "nothing undone.")
    _set(conn, folder, file, old)
    conn.commit()
    mark_progress(conn)
    return {"file": file, "status": old, "label": LABELS.get(old, old), "changed": True}


def summary(conn: sqlite3.Connection) -> str:
    c = dict(conn.execute("SELECT status, COUNT(*) FROM frames GROUP BY status").fetchall())
    kept = c.get("kept", 0) + c.get("kept_hand", 0)
    return f"{kept} kept, {c.get('rejected', 0)} rejected"


def mark_progress(conn: sqlite3.Connection) -> None:
    """After an action: 'in progress' with the counts, unless the stage is already done or
    skipped, in which case only the counts change."""
    state = dict(conn.execute("SELECT stage, state FROM stage_status"))["review"]
    db.set_stage(conn, "review", state if state in ("done", "skipped") else "in progress",
                 summary(conn))


def finish(conn: sqlite3.Connection, state: str) -> None:
    """The page's Done or Skip button: 'done' or 'skipped' (spec: the stage can be skipped)."""
    if state not in ("done", "skipped"):
        raise ReviewError(f"Unknown state {state!r}.")
    db.set_stage(conn, "review", state, summary(conn) if state == "done" else "")


THUMB_WIDTH = 640  # one per row on a wide screen still looks sharp enough; the viewer has full size


def thumbnail(folder: Path, file: str) -> Path | None:
    """A 640 px wide copy of a frame for the grid, cached in <project>/framethumbs/ (a GET may
    write a cache, D14a). The whole frame, scaled down, never cropped (spec §4). A frame's file
    never changes under the same name, so the cache is never stale. None if the file is gone."""
    src = extract.find_file(folder, file)
    out = Path(folder) / "framethumbs" / file
    if out.is_file():
        return out
    if src is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 4))
        tmp = out.with_name(out.name + ".tmp")
        im.convert("RGB").save(tmp, format="JPEG", quality=85)
    tmp.replace(out)
    return out
