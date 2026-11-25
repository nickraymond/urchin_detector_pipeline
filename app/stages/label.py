"""Stage 5, SAM labeling: pick the frames, start the SAM worker process, follow its progress
(DESIGN, Planned for S3). Bite 2 has the driver; the Preview and Run pages come in bites 3-4.

Each call starts one worker process that loads the model once (D9, D26). Callers queue it
through app/jobs.py, whose single worker thread runs one job at a time, so two SAM jobs never
share the GPU (S1: two at once triple the time). The worker's own lines go to a log file in
<project>/sam_logs/, kept for debugging.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from app.sam import classes
from app.stages import extract

MODELS = {"sam3": ("facebook/sam3", "sam3.pt"),           # D10
          "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt")}
LABELED = ("kept", "kept_hand")


class LabelError(RuntimeError):
    """The SAM run failed; the message says why and points at the log."""


def frames_to_label(conn: sqlite3.Connection) -> list[str]:
    """Kept frames (also kept by hand), in name order."""
    marks = ",".join("?" * len(LABELED))
    return [r[0] for r in conn.execute(
        f"SELECT file FROM frames WHERE status IN ({marks}) ORDER BY file", LABELED)]


def device() -> str:
    """The Apple GPU when there is one (S1), else CPU. Asked in a child process, so the app
    itself never imports torch."""
    r = subprocess.run([sys.executable, "-c", "import torch; print('mps' if "
                        "torch.backends.mps.is_available() else 'cpu')"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "cpu"


def run(folder: Path, files: list[str], report, stop: threading.Event | None = None,
        predictor: str | None = None, dev: str | None = None) -> dict:
    """Label `files` (frame names) with the project's classes.yaml. report(percent, message)
    as app/jobs.py expects. Returns {"frames": [per-frame events], "load_seconds"}.
    Raises LabelError on a failure, classes.ClassesError if classes.yaml has problems."""
    folder = Path(folder)
    cfg = classes.ensure(folder)
    settings = cfg["settings"]
    units = [{**u, "classes": {str(k): v for k, v in u["classes"].items()}}
             for u in classes.run_units(cfg["classes"])]
    frames = []
    for f in files:
        p = extract.find_file(folder, f)
        if p is None:
            raise LabelError(f"The file of frame {f} is missing from the project folder.")
        frames.append({"file": f, "path": str(p)})
    repo_id, ckpt = MODELS[settings["model"]]
    logs = folder / "sam_logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_path, log_path = logs / f"job-{stamp}.json", logs / f"run-{stamp}.log"
    job_path.write_text(json.dumps({
        "folder": str(folder), "frames": frames, "units": units, "settings": settings,
        "repo_id": repo_id, "checkpoint": ckpt, "predictor": predictor,
        "device": dev or ("cpu" if predictor else device())}, indent=1))
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")  # S1: CPU fallback on (D8)
    out = {"frames": [], "load_seconds": None, "log": str(log_path)}
    report(0, "loading SAM")
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "app.sam.worker", str(job_path)],
                                stdout=subprocess.PIPE, stderr=log, text=True, env=env,
                                cwd=Path(__file__).resolve().parents[2])
        watcher = None
        if stop is not None:
            def watch():
                while not stop.wait(0.5):
                    if proc.poll() is not None:
                        return
                if proc.poll() is None:
                    proc.terminate()
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
        finished = False
        for line in proc.stdout:
            if not line.startswith("@@ "):
                log.write(line)
                continue
            ev = json.loads(line[3:])
            if ev["event"] == "loaded":
                out["load_seconds"] = ev["seconds"]
                report(0, f"SAM loaded in {ev['seconds']} s")
            elif ev["event"] == "frame":
                out["frames"].append(ev)
                report(100 * ev["i"] / ev["n"], f"frame {ev['i']} of {ev['n']}")
            elif ev["event"] == "done":
                finished = True
        code = proc.wait()
    if stop is not None and stop.is_set():
        out["stopped"] = True
        return out
    if code != 0 or not finished:
        tail = "\n".join(log_path.read_text().splitlines()[-15:])
        raise LabelError(f"SAM stopped with exit code {code} after {len(out['frames'])} of "
                         f"{len(frames)} frames (those are saved). Log: {log_path}\n{tail}")
    job_path.unlink(missing_ok=True)
    return out
