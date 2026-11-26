"""Helpers for job worker processes (D33). A worker is `python -m <module> <job.json>`:

    job = worker.load_job()
    ctl = worker.listen()
    for item in items:
        if ctl.pause_requested():
            worker.emit("paused"); return
        ... skip items whose output is on disk; do one item; write its output ...
        worker.emit("item", i=i, n=n, name=name, seconds=s, counts={...})
    worker.emit("done")

Events go to stdout as lines "@@ {json}"; anything else a worker prints goes to the job's log.
A line "pause" on stdin, or end of file (the app quit), asks the worker to finish the item in
progress, emit "paused" and exit 0. Progress is truth from the artifacts: the worker skips items
whose output already exists, so a new worker on the same params repeats and skips nothing.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

OUT = sys.stdout


def emit(event: str, **fields) -> None:
    try:
        OUT.write("@@ " + json.dumps({"event": event, **fields}) + "\n")
        OUT.flush()
    except BrokenPipeError:   # the app is gone; finished items are already on disk
        os.dup2(os.open(os.devnull, os.O_WRONLY), OUT.fileno())
        sys.exit(0)


def run(main) -> None:
    """Run main(); on an exception, emit "error" with its message (the page shows it), print
    the traceback to the job's log and exit 1."""
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit("error", message=str(e) or type(e).__name__)
        sys.exit(1)


def load_job(argv: list[str] | None = None) -> dict:
    argv = sys.argv if argv is None else argv
    return json.loads(Path(argv[1]).read_text())


class Control:
    """Listens on stdin in a thread; `pause` or end of file sets the pause flag."""

    def __init__(self, stream=None):
        self._pause = threading.Event()
        self._stream = sys.stdin if stream is None else stream
        threading.Thread(target=self._listen, name="pause-listener", daemon=True).start()

    def _listen(self) -> None:
        try:
            for line in self._stream:
                if line.strip() == "pause":
                    break
        except (OSError, ValueError):
            pass
        self._pause.set()   # "pause", or end of file: the app is gone

    def pause_requested(self) -> bool:
        return self._pause.is_set()

    def rest(self, seconds: float) -> bool:
        """Sleep between items (bite 4's limit), waking early on pause. True if pause came."""
        return self._pause.wait(seconds) if seconds > 0 else self._pause.is_set()


def listen(stream=None) -> Control:
    return Control(stream)


def rest(ctl: Control, job: dict) -> bool:
    """The pause between items from the job's limits (D36). True if a pause came meanwhile."""
    return ctl.rest(float((job.get("limits") or {}).get("pause_s") or 0))


def threads(job: dict) -> int | None:
    """The job's CPU thread cap (D36), for a worker that sets it in its own library."""
    return (job.get("limits") or {}).get("threads")
