"""The one queue thread (D32): takes the oldest queued job, starts its worker process, reads
its "@@ {json}" lines into the job's record and log, and sends "pause" on stdin when asked
(D33, D34). One job at a time for the whole app.

A worker runs in its own session, so after an app quit or crash it finishes its item and
exits on its own. Its PID is saved in the job's row: no job starts while a worker recorded
there is still alive (QE PR 32), so a quick restart plus Resume never runs an item twice
and never puts two workers on the GPU.
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


def worker_alive(pid: int, job_file: Path) -> bool:
    """True if process `pid` is still the worker of `job_file` (its command line names the
    file, so a PID reused by another program does not count)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    r = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
    return str(job_file) in r.stdout


class Runner:
    def __init__(self, store: Store, root: Path, kinds: dict):
        self.store, self.root, self.kinds = store, Path(root), kinds
        self.pause_timeout_s = PAUSE_TIMEOUT_S
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._job_id: int | None = None      # set from "running" until the job ends
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
        """Ask the running job to pause (what = "pause" or "cancel"). While its worker is
        still being started the request is kept and sent once it runs. False if this runner
        is not running that job."""
        with self._lock:
            if self._job_id != job_id:
                return False
            self._want[job_id] = "cancel" if what == "cancel" else \
                self._want.get(job_id, "pause")
            proc = self._proc
        self.store.set_state(job_id, "pausing", ("running",))
        self.log(job_id, f"{what} asked; finishing the item in progress")
        if proc is not None:
            self._send_pause(job_id, proc)
        return True

    def _send_pause(self, job_id: int, proc: subprocess.Popen) -> None:
        try:
            proc.stdin.write("pause\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        timer = threading.Timer(self.pause_timeout_s, self._end_if_alive, (job_id, proc))
        timer.daemon = True
        timer.start()

    def log(self, job_id: int, text: str) -> None:
        try:
            with open(self.root / f"{job_id}.log", "a") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")
        except OSError as e:   # e.g. the data SSD was unplugged: say so, keep going
            print(f"jobctl: cannot write the log of job {job_id}: {e}", file=sys.stderr)

    # --- the thread ---

    def lingering(self) -> list[tuple[int, int]]:
        """(job id, pid) of workers recorded in the store that are still alive, other than
        this runner's own. Rows whose worker has ended are cleared."""
        out = []
        for job in self.store.with_worker():
            if job["id"] == self.running_id():
                continue
            pid = job["worker_pid"]
            if worker_alive(pid, self.root / f"{job['id']}.json"):
                out.append((job["id"], pid))
            else:
                self.store.update(job["id"], worker_pid=None)
        return out

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is not None:
                old = self.lingering()
                if old:
                    self.store.update(job["id"], message=(
                        f"waiting for the worker of job {old[0][0]} (pid {old[0][1]}) to "
                        "finish its item"))
                    job = None
            if job is None:
                self._wake.wait(1.0)
                self._wake.clear()
                continue
            with self._lock:
                self._job_id = job["id"]
            if not self.store.set_state(job["id"], "running", ("queued",),
                                        message="starting", ended=None, stop_reason=None,
                                        **({} if job["started"] else {"started": now()})):
                with self._lock:
                    self._job_id = None
                continue   # paused or cancelled between the read and here
            try:
                self._run(job)
            except Exception as e:   # one bad job must never stop the queue
                self.store.update(job["id"], state="error", ended=now(), message=str(e),
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
        try:
            self.store.update(jid, worker_pid=proc.pid)
            with self._lock:
                self._proc = proc
                want = self._want.get(jid)
            if want:   # Pause or Cancel came while the worker was being started
                self._send_pause(jid, proc)
            last, final, took = self._follow(jid, proc)
        except BaseException as e:
            # Never leave a worker running while the queue moves on (D32, QE PR 32).
            if proc.poll() is None:
                proc.kill()
                proc.wait()
                self.log(jid, f"worker ended after an error in the app: {e}")
            self.store.update(jid, worker_pid=None)
            raise
        code = proc.wait()
        self.store.update(jid, worker_pid=None)
        with self._lock:
            want = self._want.get(jid)
        self._finish(jid, last, code, want, final, took)

    def _follow(self, jid: int, proc: subprocess.Popen) -> tuple[str | None, str, int]:
        """Read the worker's lines until it closes stdout."""
        last, final, started = None, "done", time.monotonic()
        for line in proc.stdout:
            if not line.startswith("@@ "):
                with open(self.root / f"{jid}.log", "a") as log:
                    log.write(line)
                continue
            try:
                ev = json.loads(line[3:])
                if not isinstance(ev, dict):
                    raise ValueError("not an object")
            except ValueError:
                self.log(jid, f"unreadable event: {line.strip()}")
                continue
            last = ev.get("event")
            if last == "done" and ev.get("message"):
                final = str(ev["message"])
            self._on_event(jid, ev)
        return last, final, round(time.monotonic() - started)

    def _finish(self, jid: int, last: str | None, code: int, want: str | None,
                final: str, took: int) -> None:
        if last == "done" and code == 0:
            self.store.update(jid, state="done", ended=now(), message=final,
                              stop_reason="finished")
            self.log(jid, f"done in {took} s: {final}")
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
            i, n = _count(ev.get("i")), _count(ev.get("n"))
            self.store.update(jid, done=i or 0, total=n, message=f"item {i} of {n}")
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


def _count(v) -> int | None:
    """A worker's i / n as an int, or None if it sent something else."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None
