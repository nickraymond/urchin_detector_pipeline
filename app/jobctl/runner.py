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
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from app.jobctl import limits as limits_mod
from app.jobctl import schedule
from app.jobctl.store import Store, now

PAUSE_TIMEOUT_S = 600          # D34: a worker that has not paused within 10 min is ended
ROLLING_ITEMS = 20             # D35: remaining time = mean of the last 20 timed items × items left
REPO = Path(__file__).resolve().parents[2]


def end_worker(proc: subprocess.Popen) -> None:
    """Kill a worker and everything it started (its own session, e.g. ffmpeg), then reap it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()


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
        self._waiting_since: dict[int, float] = {}   # left-over worker's job id -> first seen
        self._item = "item"
        self._times: deque[float] = deque(maxlen=ROLLING_ITEMS)   # the running job's items
        self._reason: dict[int, str] = {}    # job id -> why the app paused it (memory guard)
        self.free_gb = limits_mod.free_gb    # tests replace it
        self.guard_every_s = limits_mod.GUARD_EVERY_S
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

    def request(self, job_id: int, what: str, kill: bool = False, reason: str = "") -> bool:
        """Ask the running job to pause (what = "pause" or "cancel"). While its worker is
        still being started the request is kept and sent once it runs. kill: end the worker
        at once instead (a kind that cannot pause; its output is written under temporary
        names). False if this runner is not running that job."""
        with self._lock:
            if self._job_id != job_id:
                return False
            self._want[job_id] = "cancel" if what == "cancel" else \
                self._want.get(job_id, "pause")
            if kill:
                self._want[f"kill-{job_id}"] = "kill"
            if reason:
                self._reason[job_id] = reason
            proc = self._proc
        self.store.set_state(job_id, "pausing", ("running",))
        self.log(job_id, f"{what} asked{' (' + reason + ')' if reason else ''}; " +
                 ("ending the worker" if kill else "finishing the item in progress"))
        if proc is not None:
            self._send_pause(job_id, proc)
        return True

    def _send_pause(self, job_id: int, proc: subprocess.Popen) -> None:
        with self._lock:
            kill = f"kill-{job_id}" in self._want
        if kill:
            end_worker(proc)
            return
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
            if not worker_alive(pid, self.root / f"{job['id']}.json"):
                self.store.update(job["id"], worker_pid=None)
                self._waiting_since.pop(job["id"], None)
                continue
            since = self._waiting_since.setdefault(job["id"], time.monotonic())
            if time.monotonic() - since > self.pause_timeout_s:   # D34: at most 10 min
                self.log(job["id"], f"left-over worker (pid {pid}) still running after "
                                    f"{self.pause_timeout_s} s; ending it (at most its item "
                                    "in progress is lost)")
                try:
                    os.killpg(pid, signal.SIGKILL)   # its own session: pgid = pid
                except (ProcessLookupError, PermissionError):
                    pass
                continue
            out.append((job["id"], pid))
        return out

    def fire_scheduled(self, at: datetime | None = None) -> None:
        """D37: a scheduled job whose start time has come joins the queue; also one whose start
        passed while the app was closed, unless its Stop by has passed too ("missed")."""
        at = at or datetime.now()
        for job in self.store.scheduled():
            start, stop = schedule.parse(job["start_at"]), schedule.parse(job["stop_by"])
            if start is None or start > at:
                continue
            if stop is not None and stop <= at:
                if self.store.set_state(job["id"], "missed", ("scheduled",), ended=now(),
                                        stop_reason="missed: its start time and its Stop by "
                                                    "passed while the app was not running",
                                        message="missed"):
                    self.log(job["id"], "missed: start and Stop by both passed")
            elif self.store.set_state(job["id"], "queued", ("scheduled",), message="waiting"):
                self.log(job["id"], f"start time {job['start_at']} came: queued")

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.fire_scheduled()
            job = self.store.next_queued()
            stop = schedule.parse(job["stop_by"]) if job else None
            if job is not None and stop is not None and stop <= datetime.now():
                # its Stop by came while it waited for another job: it waits as paused
                if self.store.set_state(job["id"], "paused", ("queued",),
                                        stop_reason="stop by: the time passed before it could "
                                                    "start", message="paused (stop by)"):
                    self.log(job["id"], "paused: Stop by passed while it was queued")
                continue
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
                    self._want.pop(f"kill-{job['id']}", None)
                    self._reason.pop(job["id"], None)

    def _run(self, job: dict) -> None:
        jid, kind = job["id"], self.kinds.get(job["kind"])
        if kind is None:
            raise RuntimeError(f"no job kind {job['kind']!r} is registered in this app")
        self._item = kind.item
        self._times.clear()
        self.store.update(jid, eta_s=None)
        if kind.progress:
            done, total = kind.progress(job["params"])
            self.store.update(jid, done=done, total=total)
        lim = limits_mod.load(self.root)          # D36: copied into the job when it starts
        self.store.update(jid, limits=lim, warnings=None)
        job_file = self.root / f"{jid}.json"
        job_file.write_text(json.dumps(
            {"job_id": jid, "limits": lim, **kind.prepare({**job["params"], "limits": lim})},
            indent=1))
        self.log(jid, f"start: kind {job['kind']}, worker {kind.module}, limits "
                      + json.dumps(lim))
        threads = str(lim["threads"])
        cmd = ((["nice", "-n", "10"] if lim["low_priority"] else []) +   # nice execs: same pid
               [sys.executable, "-m", kind.module, str(job_file)])
        with open(self.root / f"{jid}.log", "a") as log:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                bufsize=1, cwd=kind.cwd or REPO,
                env={**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                     **kind.env},
                start_new_session=True)   # Ctrl-C on the app must not cut an item in half
        try:
            self.store.update(jid, worker_pid=proc.pid)
            with self._lock:
                self._proc = proc
                want = self._want.get(jid)
            if want:   # Pause or Cancel came while the worker was being started
                self._send_pause(jid, proc)
            stop_by = schedule.parse(job.get("stop_by"))
            if stop_by is not None:
                threading.Thread(target=self._stop_at, args=(jid, proc, stop_by),
                                 name="stop-by", daemon=True).start()
            if lim["memory_guard"]:
                threading.Thread(target=self._guard, args=(jid, proc, lim["memory_guard_gb"]),
                                 name="memory-guard", daemon=True).start()
            last, final, took, err = self._follow(jid, proc)
        except BaseException as e:
            # Never leave a worker running while the queue moves on (D32, QE PR 32).
            if proc.poll() is None:
                end_worker(proc)
                self.log(jid, f"worker ended after an error in the app: {e}")
            self.store.update(jid, worker_pid=None, eta_s=None)
            raise
        code = proc.wait()
        self.store.update(jid, worker_pid=None, eta_s=None)
        with self._lock:
            want = self._want.get(jid)
            self._want.pop(f"kill-{jid}", None)
            reason = self._reason.pop(jid, None)
        self._finish(jid, last, code, want, final, took, err, reason)

    def _follow(self, jid: int, proc: subprocess.Popen):
        """Read the worker's lines until it closes stdout. Returns (last event, the done
        message, seconds taken, the message of an "error" event or None)."""
        last, final, err, started = None, "done", None, time.monotonic()
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
            if last == "error":
                err = str(ev.get("message") or "error")
            self._on_event(jid, ev)
        return last, final, round(time.monotonic() - started), err

    def _stop_at(self, jid: int, proc: subprocess.Popen, stop_by: datetime) -> None:
        """D37: pause the job when its Stop by comes; Resume carries on later."""
        while proc.poll() is None:
            if datetime.now() >= stop_by:
                self.request(jid, "pause", reason=f"stop by {schedule.show(schedule.iso(stop_by))}")
                return
            time.sleep(1.0)

    def _guard(self, jid: int, proc: subprocess.Popen, threshold_gb: float) -> None:
        """D36: every 2 s while the worker runs, pause the job if free memory is below the
        threshold. The worker finishes its item first, as with any Pause."""
        while proc.poll() is None:
            time.sleep(self.guard_every_s)
            if proc.poll() is not None:
                return
            try:
                free = self.free_gb()
            except Exception as e:   # never stop a job because the figure cannot be read
                self.log(jid, f"memory guard cannot read free memory: {e}")
                return
            if free < threshold_gb:
                self.request(jid, "pause", reason=f"memory guard: {free:.1f} GB free, "
                                                  f"below {threshold_gb:g} GB")
                return

    def _finish(self, jid: int, last: str | None, code: int, want: str | None,
                final: str, took: int, err: str | None = None,
                why: str | None = None) -> None:
        if last == "done" and code == 0:
            self.store.update(jid, state="done", ended=now(), message=final,
                              stop_reason="finished")
            self.log(jid, f"done in {took} s: {final}")
        elif want or last == "paused":
            state = "cancelled" if want == "cancel" else "paused"
            reason = {"cancel": "cancelled", "pause": why or "paused"}.get(want, "worker paused")
            if last != "paused":
                reason += f" (worker ended with code {code} before saying it paused)"
            self.store.update(jid, state=state, stop_reason=reason, message=reason,
                              **({"ended": now()} if state == "cancelled" else {}))
            self.log(jid, f"{state}: {reason}")
        else:
            reason = f"worker stopped with exit code {code}; see the log {jid}.log"
            self.store.update(jid, state="error", ended=now(), stop_reason=reason,
                              message=err or reason)
            self.log(jid, f"error: {reason}")

    def _on_event(self, jid: int, ev: dict) -> None:
        e = ev.get("event")
        if e == "loaded":
            self.store.update(jid, message=f"loaded in {ev.get('seconds')} s")
            self.log(jid, f"loaded in {ev.get('seconds')} s")
        elif e == "item":
            i, n = _count(ev.get("i")), _count(ev.get("n"))
            secs = ev.get("seconds")
            if isinstance(secs, (int, float)) and not isinstance(secs, bool) \
                    and not ev.get("skipped"):
                self._times.append(float(secs))
            eta = (round(sum(self._times) / len(self._times) * (n - i))
                   if self._times and i is not None and n is not None and n >= i else None)
            self.store.update(jid, done=i or 0, total=n, eta_s=eta,
                              message=f"{self._item} {i} of {n}")
            self.log(jid, "item " + json.dumps({k: v for k, v in ev.items() if k != "event"}))
        elif e == "progress":   # a kind without items (playback copy): percent + message
            pct = ev.get("percent")
            fields = {"message": str(ev.get("message") or "")}
            if isinstance(pct, (int, float)) and not isinstance(pct, bool):
                fields.update(done=round(max(0, min(100, pct))), total=100)
            self.store.update(jid, **fields)
        elif e == "error":
            self.log(jid, f"worker error: {ev.get('message')}")
        elif e == "warning":   # shown on the job (e.g. the Apple GPU fell back to the CPU)
            text = str(ev.get("message") or "")
            have = (self.store.get(jid) or {}).get("warnings") or []
            if text and text not in have:
                self.store.update(jid, warnings=have + [text])
            self.log(jid, f"warning: {text}")
        elif e in ("paused", "done"):
            self.log(jid, f"worker: {e}")
        else:
            self.log(jid, "event " + json.dumps(ev))

    def _end_if_alive(self, job_id: int, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            self.log(job_id, f"worker did not pause within {self.pause_timeout_s} s; ending it "
                             "(at most the item in progress is lost)")
            end_worker(proc)


def _count(v) -> int | None:
    """A worker's i / n as an int, or None if it sent something else."""
    return v if isinstance(v, int) and not isinstance(v, bool) else None
