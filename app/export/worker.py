"""Export as a job (S5 bite 6, D32): `python -m app.export.worker <job.json>`, the "export" kind.

Rebuilds `<project>/export/` with app/export/writers.run (from scratch, checked, swapped in only
when every check passes) and reports one `item` per accepted frame. Not pausable: a stop leaves the
previous export in place, and Export again starts over (the export is rebuilt each time anyway).
"""

from __future__ import annotations

import time
from pathlib import Path

from app import db
from app.export import writers
from app.jobctl import worker as jw


def main() -> None:
    job = jw.load_job()
    folder = Path(job["folder"])
    jw.emit("loaded", seconds=0.0)
    last = [time.perf_counter()]

    def progress(i: int, n: int, name: str) -> None:
        now = time.perf_counter()
        jw.emit("item", i=i, n=n, name=name, seconds=round(now - last[0], 2))
        last[0] = now

    conn = db.connect(folder / "state.sqlite")
    try:
        res = writers.run(folder, conn, progress)
    finally:
        conn.close()
    s = res["summary"]
    if res["ok"]:
        jw.emit("done", message=f"export PASS: {s['frames']} frames, {s['objects']} objects in "
                                f"COCO; every check passed")
    else:
        bad = ", ".join(c["name"] for c in res["checks"] if not c["ok"])
        jw.emit("done", message=f"export checks FAILED ({bad}); the previous export is unchanged; "
                                f"the failed one is in {res['folder']}")


def export_kind():
    from app.jobctl import Kind
    return Kind("export", module="app.export.worker", item="frame", pausable=False)


if __name__ == "__main__":
    jw.run(main)
