"""The one queue thread (D32): takes the oldest queued job, starts its worker process, reads
its "@@ {json}" lines into the job's record and log, and sends "pause" on stdin when asked
(D33, D34). One job at a time for the whole app.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from app.jobctl.store import Store, now

PAUSE_TIMEOUT_S = 600          # D34: a worker that has not paused within 10 min is ended
REPO = Path(__file__).resolve().parents[2]


class Runner:
    def __init__(self, store: Store, root: Path, kinds: dict):
        self.store, self.root, self.kinds = store, Path(root), kinds
        self.pause_timeout_s = PAUSE_TIMEOUT_S
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job_id: int | None = None
        self._want: dict[int, str] = {}      # job id -> "pause" | "cancel"
        self._thread = threading.Thread(target=self._loop, name="jobctl", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def stop(self, join_s: float = 0) -> None:
        """Stop taking new jobs (tests); wait up to join_s for the job in progress to end."""
        self._stop.set()
        self._wake.set()
        if join_s and self._thread.is_alive():
            self._thread.join(join_s)

    def running_id(self) -> int | None:
        with self._lock:
            return self._job_id

    def request(self, job_id: int, what: str) -> bool:
        """Ask the running job's worker to pause (what = "pause" or "cancel"). False if that
        job has no worker right now."""
        with self._lock:
            if self._job_id != job_id or self._proc is None:
                return False
            self._want[job_id] = "cancel" if what == "cancel" else \
                self._want.get(job_id, "pause")
            proc = self._proc
        self.store.set_state(job_id, "pausing", ("running",))
        self.log(job_id, f"{what} asked; finishing the item in progress")
        try:
            proc.stdin.write("pause\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        timer = threading.Timer(self.pause_timeout_s, self._end_if_alive, (job_id, proc))
        timer.daemon = True
        timer.start()
        return True

    def log(self, job_id: int, text: str) -> None:
        with open(self.root / f"{job_id}.log", "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")

    # --- the thread ---

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            if not self.store.set_state(job["id"], "running", ("queued",),
                                        message="starting", ended=None, stop_reason=None,
                                        **({} if job["started"] else {"started": now()})):
                continue   # paused or cancelled between the read and here
            try:
                self._run(job)
            except Exception as e:   # one bad job must never stop the queue
                self.store.update(job["id"], state="error", ended=now(),
                                  stop_reason=f"{type(e).__name__}: {e}")
                self.log(job["id"], f"error: {type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._proc, self._job_id = None, None
                    self._want.pop(job["id"], None)

    def _run(self, job: dict) -> None:
        jid, kind = job["id"], self.kinds.get(job["kind"])
        if kind is None:
            raise RuntimeError(f"no job kind {job['kind']!r} is registered in this app")
        if kind.progress:
            done, total = kind.progress(job["params"])
            self.store.update(jid, done=done, total=total)
        job_file = self.root / f"{jid}.json"
        job_file.write_text(json.dumps({"job_id": jid, **kind.prepare(job["params"])},
                                       indent=1))
        self.log(jid, f"start: kind {job['kind']}, worker {kind.module}")
        with open(self.root / f"{jid}.log", "a") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", kind.module, str(job_file)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                bufsize=1, cwd=kind.cwd or REPO, env={**os.environ, **kind.env},
                start_new_session=True)   # Ctrl-C on the app must not cut an item in half
        with self._lock:
            self._proc, self._job_id = proc, jid
        last, started = None, time.monotonic()
        for line in proc.stdout:
            if not line.startswith("@@ "):
                with open(self.root / f"{jid}.log", "a") as log:
                    log.write(line)
                continue
            try:
                ev = json.loads(line[3:])
            except ValueError:
                self.log(jid, f"unreadable event: {line.strip()}")
                continue
            last = ev.get("event")
            self._on_event(jid, ev)
        code = proc.wait()
        with self._lock:
            want = self._want.get(jid)
        took = round(time.monotonic() - started)
        if last == "done" and code == 0:
            self.store.update(jid, state="done", ended=now(), message="done",
                              stop_reason="finished")
            self.log(jid, f"done in {took} s")
        elif want or last == "paused":
            state = "cancelled" if want == "cancel" else "paused"
            reason = {"cancel": "cancelled", "pause": "paused"}.get(want, "worker paused")
            if last != "paused":
                reason += f" (worker ended with code {code} before saying it paused)"
            self.store.update(jid, state=state, stop_reason=reason, message=reason,
                              **({"ended": now()} if state == "cancelled" else {}))
            self.log(jid, f"{state}: {reason}")
        else:
            reason = f"worker stopped with exit code {code}; see the log {jid}.log"
            self.store.update(jid, state="error", ended=now(), stop_reason=reason,
                              message=reason)
            self.log(jid, f"error: {reason}")

    def _on_event(self, jid: int, ev: dict) -> None:
        e = ev.get("event")
        if e == "loaded":
            self.store.update(jid, message=f"loaded in {ev.get('seconds')} s")
            self.log(jid, f"loaded in {ev.get('seconds')} s")
        elif e == "item":
            i, n = ev.get("i"), ev.get("n")
            self.store.update(jid, done=i, total=n, message=f"item {i} of {n}")
            self.log(jid, "item " + json.dumps({k: v for k, v in ev.items() if k != "event"}))
        elif e in ("paused", "done"):
            self.log(jid, f"worker: {e}")
        else:
            self.log(jid, "event " + json.dumps(ev))

    def _end_if_alive(self, job_id: int, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            self.log(job_id, f"worker did not pause within {self.pause_timeout_s} s; ending it "
                             "(at most the item in progress is lost)")
            proc.kill()
