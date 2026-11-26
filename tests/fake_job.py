"""A fake job kind for app/jobctl tests: `python -m tests.fake_job <job.json>` writes one file
per item into params["out"] (item_<i>.txt, written under a temporary name, then renamed) and
skips items already on disk, like the real workers. Every write is also appended to
writes.log, so a test can prove no item was written twice or left out.

Params: out, n, delay (s per item), ignore_pause, crash_at (item number), first_delay,
hang_at (item number: the first worker to reach it hangs there for 120 s, a stuck item).
"""

import os
import sys
import time
from pathlib import Path

from app.jobctl import worker


def main() -> None:
    job = worker.load_job()
    out, n = Path(job["out"]), job["n"]
    ctl = worker.listen()
    worker.emit("loaded", seconds=0.0)
    for i in range(1, n + 1):
        if ctl.pause_requested() and not job.get("ignore_pause"):
            worker.emit("paused")
            return
        target = out / f"item_{i}.txt"
        if target.exists():
            worker.emit("item", i=i, n=n, name=target.name, skipped=True)
            continue
        if job.get("crash_at") == i:
            print(f"fake crash at item {i}", file=sys.stderr)
            sys.exit(3)
        t = time.perf_counter()
        hung = out / ".hung"
        if job.get("hang_at") == i and not hung.exists():
            hung.write_text("x")
            time.sleep(120)
        time.sleep(job.get("first_delay", 0) if i == 1 and "first_delay" in job
                   else job.get("delay", 0))
        tmp = out / f".item_{i}.tmp"
        tmp.write_text(f"{os.getpid()}\n")
        os.replace(tmp, target)
        with open(out / "writes.log", "a") as f:
            f.write(f"{i} {os.getpid()} {time.time():.3f}\n")
        worker.emit("item", i=i, n=n, name=target.name,
                    seconds=round(time.perf_counter() - t, 3), counts={"thing": 1})
    worker.emit("done")


if __name__ == "__main__":
    main()
