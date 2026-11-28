"""The rig log (D81): <data_dir>/bench/rig_log.jsonl, one JSON object per line, append-only.

Every change on the rig (a file copied, a model area written, a package installed) writes a "before" line
with what it will do and what undoes it, then an "after" line with how it ended, under the same change id.
A "before" with no "after" means the change was cut off: the page shows it as "not finished", and the rig
must be checked against the undo. The file is never rewritten; a line that cannot be read is shown, not dropped.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def path(data_dir) -> Path:
    return Path(data_dir) / "bench" / "rig_log.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _append(data_dir, line: dict) -> None:
    p = path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())     # the "before" line must be on disk before the change starts


@contextmanager
def change(data_dir, unit: str, board: str, what: str, undo: str, fake: bool = False, **details):
    """Wrap one change on the rig. Writes "before" first; "after" with ok or the error, even when the change
    raises (the error is passed on). `undo` is required: a change nobody can undo is not made."""
    if not what.strip() or not undo.strip():
        raise ValueError("a rig change needs both what it does and what undoes it")
    cid = uuid.uuid4().hex[:12]
    base = {"change": cid, "unit": unit, "board": board, "fake": fake}
    _append(data_dir, {**base, "at": _now(), "phase": "before", "what": what, "undo": undo, **details})
    try:
        yield cid
    except BaseException as e:
        _append(data_dir, {**base, "at": _now(), "phase": "after", "ok": False,
                           "error": f"{type(e).__name__}: {e}"})
        raise
    _append(data_dir, {**base, "at": _now(), "phase": "after", "ok": True})


def read(data_dir) -> list[dict]:
    """One row per change, newest first: before's fields plus `state` (done / failed / not finished)."""
    p = path(data_dir)
    if not p.is_file():
        return []
    rows: dict[str, dict] = {}
    order: list[str] = []
    for n, text in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            line = json.loads(text)
            cid, phase = line["change"], line["phase"]
        except (ValueError, KeyError, TypeError):
            key = f"bad-{n}"
            rows[key] = {"change": key, "state": "unreadable", "what": f"line {n} cannot be read: {text[:200]}"}
            order.append(key)
            continue
        if phase == "before":
            rows[cid] = {**line, "state": "not finished"}
            order.append(cid)
        elif cid in rows:
            row = rows[cid]
            row["ended"] = line.get("at", "")
            row["state"] = "done" if line.get("ok") else "failed"
            row["error"] = line.get("error", "")
    return [rows[k] for k in reversed(order)]


def undo_steps(rows: list[dict]) -> list[dict]:
    """What puts the rig back, in the order to do it: newest change first, real units only. A failed or
    unfinished change is listed too (it may have changed the rig part-way). This list is what keeps the
    backup's restore steps true to the rig (QE PR 94 carried nit 1)."""
    return [r for r in rows if not r.get("fake") and r["state"] in ("done", "failed", "not finished")]
