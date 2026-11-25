"""Background jobs: one worker thread runs queued jobs one at a time, and each job reports
progress. The small interface (submit, status) is what the S4 job-control module will replace
(DESIGN, Planned for S2; D4). Nothing here knows about labeling.

A job is fn(report), where report(percent, message) updates its status. Status lives in memory
only: if the app stops mid-job, the job's output was written to a temporary name, so the
next start simply queues it again.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Callable

Job = Callable[[Callable[[float, str], None]], None]


class Jobs:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[str, Job]] = queue.Queue()
        self._status: dict[str, dict] = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, name="jobs", daemon=True).start()

    def submit(self, key: str, fn: Job) -> None:
        """Queue fn under key unless that key is already queued or running."""
        with self._lock:
            if self._status.get(key, {}).get("state") in ("queued", "running"):
                return
            self._status[key] = {"state": "queued", "percent": 0.0, "message": "waiting",
                                 "started": None}
        self._queue.put((key, fn))

    def status(self, key: str) -> dict | None:
        with self._lock:
            st = self._status.get(key)
            return dict(st) if st else None

    def _set(self, key: str, **fields) -> None:
        with self._lock:
            self._status[key].update(fields)

    def _worker(self) -> None:
        while True:
            key, fn = self._queue.get()
            started = time.monotonic()
            self._set(key, state="running", started=started, message="starting")

            def report(percent: float, message: str, key=key, started=started) -> None:
                percent = max(0.0, min(100.0, percent))
                elapsed = time.monotonic() - started
                eta = elapsed * (100 - percent) / percent if percent > 1 else None
                self._set(key, percent=round(percent, 1), message=message,
                          eta_s=round(eta) if eta is not None else None)

            try:
                fn(report)
                self._set(key, state="done", percent=100.0, message="done", eta_s=0)
            except Exception as e:  # a failed job must never stop the worker
                traceback.print_exc()
                self._set(key, state="error", message=f"{type(e).__name__}: {e}")
            finally:
                self._queue.task_done()
