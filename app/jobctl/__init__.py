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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.jobctl import schedule
from app.jobctl.awake import Awake
from app.jobctl.runner import GUARD_REASON, Runner
from app.jobctl.store import FINISHED, Store, now

RECOVER_S = 60.0   # D36c: free memory must stay at 2x the guard limit this long


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
    def __init__(self, root: Path, start: bool = True, awake_cmd: list[str] | None = None):
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
        # D38: one keep-awake process while any keep-awake job waits or runs (tests: a fake cmd)
        self.awake = Awake(awake_cmd)
        self._awake_stop = threading.Event()
        self.recover_s = RECOVER_S
        self._recovered: dict[int, float] = {}   # job id -> since when memory is free enough
        self.auto_resume_error = ""
        self._started = False
        if start:
            self.start()

    def start(self) -> None:
        """Begin taking jobs. An app that registers its job kinds after creating the module
        passes start=False, registers them, then calls this: a job that is queued or due when
        the app starts must not be picked up before its kind is known (it ended "no job kind
        ... is registered"; QE 8, 2026-11-27). Calling it twice does nothing."""
        if self._started:
            return
        self._started = True
        self.runner.start()
        threading.Thread(target=self._keep_awake, name="keep-awake", daemon=True).start()

    def _keep_awake(self) -> None:
        while not self._awake_stop.wait(2.0):
            try:
                self.awake.update(self.store.keep_awake_needed())
            except Exception as e:   # never stop the app over this; the page shows the error
                self.awake.error = f"keep awake: {e}"
            try:
                self.auto_resume()
            except Exception as e:   # never stop the app over this
                self.auto_resume_error = f"auto-resume: {e}"

    def auto_resume(self) -> list[int]:
        """D36c: a job the memory guard paused resumes by itself once free memory has stayed
        at or above twice its limit for RECOVER_S. Jobs paused by Nick, by Stop by or by
        Cancel never resume by themselves. Returns the ids resumed (tests)."""
        waiting = [j for j in self.list() if j["state"] == "paused"
                   and (j.get("stop_reason") or "").startswith(GUARD_REASON)]
        for gone in set(self._recovered) - {j["id"] for j in waiting}:
            self._recovered.pop(gone)
        if not waiting:
            return []
        free, resumed = self.runner.free_gb(), []
        for j in waiting:
            stop = schedule.parse(j.get("stop_by"))
            if stop is not None and stop <= datetime.now():
                # QE PR 42: its Stop by passed while it waited; resuming would clear the Stop
                # by and run on unattended. It stays paused, now as stopped by the Stop by.
                why = (f"stop by {schedule.show(j['stop_by'])} (it passed while the job was "
                       "paused by the memory guard)")
                self.store.set_state(j["id"], "paused", ("paused",), stop_reason=why,
                                     message=why)
                self._recovered.pop(j["id"], None)
                self.runner.log(j["id"], f"no auto-resume: {why}")
                continue
            need = 2 * float((j.get("limits") or {}).get("memory_guard_gb") or 0)
            if free < need:
                self._recovered.pop(j["id"], None)
                continue
            since = self._recovered.setdefault(j["id"], time.monotonic())
            if time.monotonic() - since < self.recover_s:
                continue
            self._recovered.pop(j["id"])
            summary = j.get("summary") or {}
            summary["auto_resumes"] = summary.get("auto_resumes", 0) + 1
            self.store.update(j["id"], summary=summary)
            self.runner.log(j["id"], f"memory recovered: {free:.1f} GB free for "
                                     f"{self.recover_s:g} s (needs {need:g} GB); resuming")
            try:
                self.resume(j["id"])
                resumed.append(j["id"])
            except JobError as e:
                self.runner.log(j["id"], f"auto-resume refused: {e}")
        return resumed

    def last_summary(self) -> dict | None:
        return self.store.last_summary()

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

    def submit(self, kind: str, params: dict, project: str | None = None,
               start_at: datetime | None = None, stop_by: datetime | None = None,
               keep_awake: bool = False) -> int:
        """Queue a job; with start_at it waits as "scheduled" until then (D37). stop_by pauses
        it at that time with the reason "stop by"."""
        if kind not in self.kinds:
            raise JobError(f"No job kind {kind!r}. Known: {', '.join(sorted(self.kinds))}.")
        if start_at and stop_by and stop_by <= start_at:
            raise JobError("Stop by must come after Start at.")
        state = "scheduled" if start_at else "queued"
        msg = f"starts at {schedule.show(schedule.iso(start_at))}" if start_at else "waiting"
        jid = self.store.add(kind, params, project=project, state=state, message=msg,
                             start_at=schedule.iso(start_at), stop_by=schedule.iso(stop_by),
                             keep_awake=int(keep_awake))
        self.runner.log(jid, f"{state}: kind {kind}, project {project}"
                        + (f", start at {schedule.iso(start_at)}" if start_at else "")
                        + (f", stop by {schedule.iso(stop_by)}" if stop_by else ""))
        self.runner.wake()
        if keep_awake:
            self.awake.update(True)          # at once, not 2 s later
        return jid

    def start_now(self, job_id: int) -> dict:
        """A scheduled job joins the queue at once (its Stop by stays)."""
        if not self.store.set_state(job_id, "queued", ("scheduled",), message="waiting",
                                    start_at=schedule.iso(datetime.now())):
            raise JobError(f"Job {job_id} is {self.status(job_id)['state']}; only a scheduled "
                           "job can be started now.")
        self.runner.log(job_id, "start now (was scheduled)")
        self.runner.wake()
        return self.status(job_id)

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
        # D37 "Resume later": once its Stop by has passed it no longer applies. Before that
        # time it still holds (QE PR 37: a 22:00-07:00 run paused and resumed at 23:00 must
        # still stop at 07:00).
        stop = schedule.parse(job.get("stop_by"))
        passed = {"stop_by": None} if stop is not None and stop <= datetime.now() else {}
        if not self.store.set_state(job_id, "queued", ("paused", "interrupted"),
                                    message="waiting", stop_reason=None, **passed):
            raise JobError(f"Job {job_id} is {job['state']}; only a paused or interrupted "
                           "job can be resumed.")
        self.runner.log(job_id, f"resume (was {job['state']})")
        self.runner.wake()
        return self.status(job_id)

    def cancel(self, job_id: int) -> dict:
        """Ends the job for good; finished items stay on disk. A running or pausing job's
        worker is ended at once (D34a): the item in progress is dropped, never half-written,
        since workers write each item under a temporary name and rename it."""
        job = self.status(job_id)
        if self.store.set_state(job_id, "cancelled",
                                ("scheduled", "queued", "paused", "interrupted"),
                                ended=now(), stop_reason="cancelled", message="cancelled"):
            self.runner.log(job_id, f"cancelled (was {job['state']})")
            return self.status(job_id)
        job = self.status(job_id)
        if job["state"] in FINISHED:
            raise JobError(f"Job {job_id} is already {job['state']}.")
        if not self.runner.request(job_id, "cancel", kill=True):
            raise JobError(f"Job {job_id} is between states; try again in a moment.")
        return self.status(job_id)

    def close(self) -> None:
        """Tests: stop the queue thread and close the database."""
        self.runner.stop(join_s=30)
        self._awake_stop.set()
        self.awake.stop()
        self.store.close()
        os.close(self._lock_fd)   # releases the lock
