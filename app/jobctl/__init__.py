"""Job control (D4, D32): one queue for the whole app, one job at a time, records that survive
a restart, Pause / Resume / Cancel. The module's whole interface is JobControl below.

Standalone: nothing here imports app.stages, app.sam or app.web (a test checks), and nothing
knows what a frame is. The app registers job kinds: a kind names its worker module, turns a
job's params into the worker's job file, and may count items done / total from its own files.

    jc = JobControl(data_dir / "jobs")
    jc.register(Kind("extract", module="app.stages.extract_worker", progress=...))
    jid = jc.submit("extract", {"project": "..."}, project="reef1")
    jc.pause(jid); jc.resume(jid); jc.cancel(jid); jc.status(jid); jc.list()
"""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.jobctl.runner import Runner
from app.jobctl.store import FINISHED, Store, now


class JobError(ValueError):
    """A request that does not fit the job's state; the message says why."""


@dataclass
class Kind:
    name: str
    module: str                                  # the worker: python -m <module> <job.json>
    prepare: Callable[[dict], dict] = lambda params: dict(params)   # params -> job file
    progress: Callable[[dict], tuple[int, int]] | None = None       # params -> (done, total)
    cwd: Path | None = None                      # default: the repo folder
    item: str = "item"                           # what the page calls one item ("frame")
    pausable: bool = True                        # False: Pause refused, Cancel ends the worker
    env: dict = field(default_factory=dict)      # added to the worker's environment


class JobControl:
    def __init__(self, root: Path, start: bool = True):
        """root = <data_dir>/jobs, created on first use; its parent must exist (the app never
        creates data_dir, D13)."""
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        # One app per data folder: a second one would mark the first one's job interrupted
        # and run a second queue (QE PR 32). The lock ends with the process, even on a crash.
        self._lock_fd = os.open(self.root / "app.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lock_fd)
            raise JobError(f"Another copy of the app is already using {self.root}. Use that "
                           "one (its address is in the terminal where it was started), or "
                           "quit it first.") from None
        self.store = Store(self.root / "jobs.sqlite")
        self.interrupted = self.store.mark_interrupted()
        self.kinds: dict[str, Kind] = {}
        self.runner = Runner(self.store, self.root, self.kinds)
        for jid in self.interrupted:
            self.runner.log(jid, "interrupted: the app quit or crashed during the job")
        if start:
            self.runner.start()

    def register(self, kind: Kind) -> None:
        """Also recounts this kind's interrupted jobs from their artifacts: the app may have
        died after a worker finished an item but before its event was read (D33)."""
        self.kinds[kind.name] = kind
        if not kind.progress:
            return
        for job in self.store.all():
            if job["kind"] == kind.name and job["state"] == "interrupted":
                try:
                    done, total = kind.progress(job["params"])
                except Exception as e:   # a missing project folder must not stop the app
                    self.runner.log(job["id"], f"cannot count progress: {e}")
                    continue
                self.store.update(job["id"], done=done, total=total)

    def submit(self, kind: str, params: dict, project: str | None = None) -> int:
        if kind not in self.kinds:
            raise JobError(f"No job kind {kind!r}. Known: {', '.join(sorted(self.kinds))}.")
        jid = self.store.add(kind, params, project=project, message="waiting")
        self.runner.log(jid, f"queued: kind {kind}, project {project}")
        self.runner.wake()
        return jid

    def status(self, job_id: int) -> dict:
        job = self.store.get(job_id)
        if job is None:
            raise JobError(f"No job {job_id}.")
        return job

    def list(self, limit: int = 200) -> list[dict]:
        return self.store.all(limit)

    def latest(self, kind: str | None = None, project: str | None = None) -> dict | None:
        return self.store.latest(kind, project)

    def pause(self, job_id: int) -> dict:
        """Running: the worker finishes the item in progress and exits (model unloaded).
        Queued: it simply waits as paused."""
        if self.store.set_state(job_id, "paused", ("queued",),
                                stop_reason="paused", message="paused"):
            self.runner.log(job_id, "paused while queued")
            return self.status(job_id)
        job = self.status(job_id)   # read after the try: the runner may just have started it
        if job["state"] not in ("running", "pausing"):
            raise JobError(f"Job {job_id} is {job['state']}; only a queued or running job "
                           "can be paused.")
        kind = self.kinds.get(job["kind"])
        if kind is not None and not kind.pausable:
            raise JobError(f"A {job['kind']} job cannot be paused; Cancel ends it (its "
                           "unfinished output is thrown away).")
        if not self.runner.request(job_id, "pause"):
            raise JobError(f"Job {job_id} is between states; try again in a moment.")
        return self.status(job_id)

    def resume(self, job_id: int) -> dict:
        """A paused or interrupted job is queued again; its new worker skips finished items."""
        job = self.status(job_id)
        if not self.store.set_state(job_id, "queued", ("paused", "interrupted"),
                                    message="waiting", stop_reason=None):
            raise JobError(f"Job {job_id} is {job['state']}; only a paused or interrupted "
                           "job can be resumed.")
        self.runner.log(job_id, f"resume (was {job['state']})")
        self.runner.wake()
        return self.status(job_id)

    def cancel(self, job_id: int) -> dict:
        """Ends the job for good; finished items stay on disk."""
        job = self.status(job_id)
        if self.store.set_state(job_id, "cancelled",
                                ("scheduled", "queued", "paused", "interrupted"),
                                ended=now(), stop_reason="cancelled", message="cancelled"):
            self.runner.log(job_id, f"cancelled (was {job['state']})")
            return self.status(job_id)
        job = self.status(job_id)
        if job["state"] in FINISHED:
            raise JobError(f"Job {job_id} is already {job['state']}.")
        kind = self.kinds.get(job["kind"])
        if not self.runner.request(job_id, "cancel",
                                   kill=kind is not None and not kind.pausable):
            raise JobError(f"Job {job_id} is between states; try again in a moment.")
        return self.status(job_id)

    def close(self) -> None:
        """Tests: stop the queue thread and close the database."""
        self.runner.stop(join_s=30)
        self.store.close()
        os.close(self._lock_fd)   # releases the lock
