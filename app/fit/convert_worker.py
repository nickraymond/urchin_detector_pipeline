"""Worker of the `convert` job (D33): `python -m app.fit.convert_worker <job.json>`, in the app's
environment. Two steps, each skipped when its file is on disk:

1. 8-bit model: starts `app.fit.to_int8` in the model's tool environment (D51) and saves its output as
   int8/convert.log. Pause or Cancel ends that child; nothing half-made counts (the 8-bit file is
   written last). If the step fails, int8/failed.json keeps the reason, so the other cameras' jobs do
   not repeat a conversion that cannot work ("redo" clears it).
2. Compile: the camera's compiler module, `compile(tflite, out_dir, job) -> dict`, and fit.json.

A failed conversion or compile is a result (fit.json, `converts: false`); the job still ends "done".

"redo" makes this camera's cell again from scratch, 8-bit model included. The 8-bit model is shared by the
cameras, so the other cameras' fit.json files then describe the file before it was made again. With seeded
weights and calibration the new file is the same bytes; if it ever is not, it shows: every fit.json keeps
`int8.tflite_sha256` of the file it was made from (QE PR 76).
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from app.fit import convert
from app.jobctl import worker

REPO = Path(__file__).resolve().parents[2]
TAIL_LINES = 25


class ToolsMissing(RuntimeError):
    """The tools cannot run at all (not a property of the model): the job fails and says how to fix it."""


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def tail(path: Path, lines: int = TAIL_LINES) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_int8(job: dict, folder: Path, ctl) -> str:
    """Returns "ok", "failed" or "paused"."""
    python = Path(job["tool_python"])
    if not python.is_file():
        raise ToolsMissing(f"Tool environment not found: {python}. Fix: set the tools up (DESIGN D51).")
    out = folder / "int8"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_int8.tflite.part").unlink(missing_ok=True)     # left by a step that was killed
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
           "URCHIN_TOOLS_DIR": job["tools_dir"], "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    with open(out / "convert.log", "w") as log:
        child = subprocess.Popen([str(python), "-m", job["int8_module"], str(folder / "recipe.json"), str(out)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        while child.poll() is None:
            if ctl.rest(0.5):                       # pause, cancel, or the app quit
                child.terminate()
                try:
                    child.wait(10)
                except subprocess.TimeoutExpired:
                    child.kill()
                convert.int8_path(folder).unlink(missing_ok=True)
                return "paused"
    if child.returncode == 0 and convert.int8_path(folder).is_file():
        return "ok"
    convert.int8_path(folder).unlink(missing_ok=True)
    write_json(convert.int8_failed_path(folder), {
        "stage": "8-bit model", "exit_code": child.returncode, "error": tail(out / "convert.log"),
        "when": datetime.now().isoformat(timespec="seconds")})
    return "failed"


def report(job: dict, folder: Path, body: dict) -> dict:
    arch = job["arch"]
    try:
        int8 = json.loads((folder / "int8" / "int8.json").read_text())
    except (OSError, ValueError):
        int8 = None
    return {"arch_key": job["arch_key"], "camera": job["camera"], "kind": arch["kind"],
            "family": arch["family"], "size": arch["size"], "input": arch["input"],
            "licence": arch["licence"], "licence_mark": arch.get("licence_mark"),
            "when": datetime.now().isoformat(timespec="seconds"), "int8": int8, **body}


def main() -> None:
    job = worker.load_job()
    ctl = worker.listen()
    folder, camera = Path(job["folder"]), job["camera"]
    fit_file = convert.fit_path(folder, camera)
    n = len(convert.STEPS)
    if job.get("redo"):
        shutil.rmtree(fit_file.parent, ignore_errors=True)
        shutil.rmtree(folder / "int8", ignore_errors=True)
    write_json(folder / "recipe.json", {"arch": job["arch"]})
    worker.emit("loaded", seconds=0.0)
    if fit_file.is_file():
        for i, name in enumerate(convert.STEPS, 1):
            worker.emit("item", i=i, n=n, name=name, skipped=True)
        worker.emit("done")
        return

    t = time.perf_counter()
    if convert.int8_path(folder).is_file() or convert.int8_failed_path(folder).is_file():
        worker.emit("item", i=1, n=n, name=convert.STEPS[0], skipped=True)
    else:
        state = run_int8(job, folder, ctl)
        if state == "paused":
            worker.emit("paused")
            return
        worker.emit("item", i=1, n=n, name=convert.STEPS[0], seconds=round(time.perf_counter() - t, 1),
                    counts={"8-bit ok": int(state == "ok")})
    if ctl.pause_requested():
        worker.emit("paused")
        return

    t = time.perf_counter()
    if convert.int8_failed_path(folder).is_file():
        failed = json.loads(convert.int8_failed_path(folder).read_text())
        body = {"converts": False, "stage": failed["stage"], "error": failed["error"], "failing_layers": []}
    else:
        out = fit_file.parent
        out.mkdir(parents=True, exist_ok=True)
        compiler = importlib.import_module(job["compiler"])
        body = compiler.compile(convert.int8_path(folder), out, job)
        body.setdefault("stage", "compile")
    write_json(fit_file, report(job, folder, body))
    worker.emit("item", i=2, n=n, name=convert.STEPS[1], seconds=round(time.perf_counter() - t, 1),
                counts={"converts": int(bool(body.get("converts")))})
    worker.emit("done")


if __name__ == "__main__":
    worker.run(main)
