"""Freeze a dataset version as a job (S8 bite 2a, D32): `python -m app.datasets.worker <job.json>`,
the "freeze" kind. One item per frame. Not pausable: a stopped freeze leaves only `v<N>.new/`,
which the next freeze removes, and images already stored under their hash are not copied again.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.datasets import versions
from app.jobctl import worker as jw


def main() -> None:
    job = jw.load_job()
    jw.emit("loaded", seconds=0.0)
    last = [time.perf_counter()]

    def progress(i: int, n: int, name: str) -> None:
        now = time.perf_counter()
        jw.emit("item", i=i, n=n, name=name, seconds=round(now - last[0], 2))
        last[0] = now

    res = versions.freeze(Path(job["folder"]), job.get("name") or "", progress)
    m = res["manifest"]
    if not res["new"]:
        jw.emit("done", message=f"nothing changed since version {res['number']}: no new version")
        return
    f = m["counts"]["frames"]
    jw.emit("done", message=f"version {m['number']} frozen: {len(m['frames'])} frames "
                            f"({f['train']} train / {f['valid']} valid / {f['test']} test), "
                            f"hash {m['hash'][:12]}")


def freeze_kind():
    from app.jobctl import Kind
    return Kind("freeze", module="app.datasets.worker", item="frame", pausable=False)


if __name__ == "__main__":
    jw.run(main)
