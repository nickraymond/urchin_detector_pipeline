"""Frame extraction as a job-module worker (S4 bite 2, D32, D33): `python -m
app.stages.extract_worker <job.json>`. Runs extract.run on the job's plan; a "pause" on stdin
(or the app quitting) stops it before the next frame. Frames already on disk are skipped, so
Resume carries on with the frames still missing.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

from app.jobctl import worker as jw
from app.stages import extract


def main() -> None:
    job = jw.load_job()
    ctl = jw.listen()
    if job.get("grab_hook"):                     # tests only
        module, fn = job["grab_hook"].split(":")
        extract.grab = getattr(importlib.import_module(module), fn)(extract.grab, job)
    jw.emit("loaded", seconds=0.0)

    last = [time.perf_counter()]

    def item(i: int, n: int, name: str, made: bool) -> None:
        now = time.perf_counter()
        extra = {"seconds": round(now - last[0], 2)} if made else {"skipped": True}
        last[0] = now
        jw.emit("item", i=i, n=n, name=name, new=made, counts={"new": int(made)}, **extra)

    blur, dup = extract.saved_filters(Path(job["folder"]))   # as saved now (QE PR 34)
    new = extract.run(Path(job["folder"]), job["samples"],
                      blur_percent=blur, duplicate_distance=dup,
                      should_pause=ctl.pause_requested, on_item=item)
    if new is None:
        jw.emit("paused")
        return
    jw.emit("done", message=f"{new} new frames extracted")


if __name__ == "__main__":
    jw.run(main)
