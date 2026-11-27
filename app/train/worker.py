"""Training as a job (S8 bite 4a; D32, D33): `python -m app.train.worker <job.json>`, the
"train" kind. One item per epoch. Nothing here imports torch: the trainer named in the job
file does the learning; this file owns the loop, the files and the pause rules.

A trainer is a class (see TRAINERS in app/train/job.py) with:

    __init__(model_dir, recipe, job)
    describe() -> dict                 library versions, device: recorded with the run
    load() -> (epoch, step)            restore checkpoints/last.* if there is one; where to go on
    run_epoch(epoch, start_step, stop) -> {"done": True, "metrics": {...}} when the epoch
                                       finished, or {"done": False, "step": s} when stop()
                                       became true: it finishes the step in progress first
    save(epoch, step)                  write checkpoints/last.* so that load() returns this
                                       position (under a temporary name, then renamed)
    unload()                           free the model

Rules (TRACKER S8 bite 4a):
- a checkpoint and a metrics line per epoch. The checkpoint is the truth about where the run
  is; on every start train.jsonl is cut back to the epochs before the checkpoint's, so a
  crash between the two writes repeats that epoch's line instead of doubling or losing it;
- Pause finishes the step, saves (also mid-epoch) and unloads; Resume starts a new worker,
  which continues from the last checkpoint: no step is repeated or skipped;
- Cancel ends the worker at once (D34a); checkpoints and metrics stay.
"""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

from app.jobctl import worker as jw
from app.train import models

REPO = Path(__file__).resolve().parents[2]


def code_version() -> dict:
    """The repo's commit and whether files differ from it (a run on changed code says so)."""
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"],
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": sha, "changed_files": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "changed_files": None}


def write_metrics(mdir: Path, rows: list[dict]) -> None:
    tmp = mdir / ".train.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tmp.replace(mdir / "train.jsonl")


def make_trainer(job: dict, mdir: Path, recipe: dict):
    module, _, cls = job["trainer"].partition(":")
    return getattr(importlib.import_module(module), cls)(mdir, recipe, job)


def main() -> None:
    job = jw.load_job()
    mdir = Path(job["model_dir"])
    recipe = models.read_recipe(mdir)
    epochs = int(recipe["train"]["epochs"])
    ctl = jw.listen()
    t0 = time.perf_counter()
    trainer = make_trainer(job, mdir, recipe)
    epoch, step = trainer.load()
    rows = [r for r in models.read_metrics(mdir) if r["epoch"] < epoch]
    write_metrics(mdir, rows)
    recipe["train"].setdefault("recorded", []).append({
        "at": datetime.now().isoformat(timespec="seconds"), "from_epoch": epoch,
        "from_step": step, "code": code_version(), "python": platform.python_version(),
        "platform": platform.platform(), "limits": job.get("limits"), **trainer.describe()})
    models.write_json(mdir / "recipe.json", recipe)
    jw.emit("loaded", seconds=round(time.perf_counter() - t0, 2))
    if recipe["train"]["recorded"][-1].get("warning"):     # e.g. a trainer that fell back to the CPU
        jw.emit("warning", message=recipe["train"]["recorded"][-1]["warning"])
    for done in range(1, epoch + 1):           # epochs an earlier worker finished
        jw.emit("item", i=done, n=epochs, name=f"epoch {done}", skipped=True)
    try:
        while epoch < epochs:
            if ctl.pause_requested():
                jw.emit("paused")
                return
            t = time.perf_counter()
            res = trainer.run_epoch(epoch, step, ctl.pause_requested)
            if not res["done"]:                # paused mid-epoch: the step in progress is done
                trainer.save(epoch, res["step"])
                jw.emit("paused")
                return
            rows.append({"epoch": epoch, "seconds": round(time.perf_counter() - t, 2),
                         **res["metrics"]})
            write_metrics(mdir, rows)          # the line first, then the checkpoint (see above)
            trainer.save(epoch + 1, 0)
            epoch, step = epoch + 1, 0
            shown = {k: v for k, v in res["metrics"].items() if isinstance(v, (int, float))}
            jw.emit("item", i=epoch, n=epochs, name=f"epoch {epoch}",
                    seconds=rows[-1]["seconds"], metrics=shown)
            if jw.rest(ctl, job) and epoch < epochs:
                jw.emit("paused")
                return
    finally:
        trainer.unload()
    jw.emit("done", message=f"trained {epochs} epochs; model in {mdir}")


if __name__ == "__main__":
    jw.run(main)
