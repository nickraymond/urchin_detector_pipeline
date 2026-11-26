"""Wrappers for extract.grab inside the extract worker process (extract.GRAB_HOOK): tests
cannot monkeypatch a function in another process. Each call is counted in
<project>/grab_calls.txt so the test can read how many frames were decoded."""

from pathlib import Path


def _counted(real, job, fail_at=None, delay=0.0, never=False):
    log = Path(job["folder"]) / "grab_calls.txt"

    def grab(path, t):
        n = (len(log.read_text().splitlines()) if log.exists() else 0) + 1
        with open(log, "a") as f:
            f.write(f"{t}\n")
        if never:
            raise AssertionError("a frame was decoded again")
        if n == fail_at:
            raise RuntimeError("stopped")
        if delay:
            import time
            time.sleep(delay)
        return real(path, t)
    return grab


def counted(real, job):
    return _counted(real, job)


def fail_third(real, job):
    return _counted(real, job, fail_at=3)


def slow(real, job):
    return _counted(real, job, delay=0.4)


def never(real, job):
    return _counted(real, job, never=True)


def calls(folder) -> int:
    log = Path(folder) / "grab_calls.txt"
    return len(log.read_text().splitlines()) if log.exists() else 0
