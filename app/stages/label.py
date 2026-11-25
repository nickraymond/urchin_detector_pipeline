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
PREDICTOR: str | None = None  # tests set a fake ("module:Class"); None = real SAM


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
    predictor = predictor or PREDICTOR
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


# --- Run on every kept frame (bite 4, D26) ---

def run_plan(folder: Path, files: list[str], units: list[dict], settings: dict,
             preview: dict | None = None) -> dict:
    """What Run will do (SPEC Q7): per run unit, on how many frames it runs (its settings key
    differs or is missing); frames with any work; unreadable sam_raw files; an estimate from
    the preview's seconds per frame, scaled by the share of units that run (D26)."""
    from app.sam import results
    todo = {u["key"]: 0 for u in units}
    frames, unreadable = 0, []
    for f in files:
        try:
            run_units = results.to_run(folder, f, units, settings)
        except ValueError:
            unreadable.append(f)
            continue
        for u in run_units:
            todo[u["key"]] += 1
        frames += bool(run_units)
    spf = (preview or {}).get("seconds_per_frame")
    work = sum(todo.values()) / len(units) if units else 0
    return {"units": todo, "frames": frames, "total": len(files), "unreadable": unreadable,
            "estimate_s": round(spf * work) if spf else None}


def run_all(folder: Path, files: list[str], report, stop: threading.Event | None = None,
            **kw) -> str:
    """Run: label every kept frame, skipping units whose results are current, so a stopped
    or crashed Run carries on where it ended. Returns the job's final message."""
    out = run(folder, files, report, stop=stop, **kw)
    done = sum(1 for e in out["frames"] if not e.get("skipped"))
    seen = len(out["frames"])
    if out.get("stopped"):
        return f"stopped after {seen} of {len(files)} frames; Run carries on from there"
    return f"labeled {done} frames, {seen - done} already current"


# --- Prompt preview (bite 3, D26) ---

PREVIEW_FILE = "sam_preview.json"
PREVIEW_WIDTH = 1600  # overlay width on the page; the frame itself is never changed


def preview_frames(files: list[str], other: bool = False, rng=None) -> list[str]:
    """D26: first, middle and last kept frame by name; "Other frames" picks 3 at random."""
    if other:
        import random
        return sorted((rng or random).sample(files, min(3, len(files))))
    if len(files) <= 3:
        return list(files)
    return [files[0], files[len(files) // 2], files[-1]]


def run_preview(folder: Path, files: list[str], report, **kw) -> dict:
    """Label the preview frames (results go to sam_raw/ like any run, so Run later skips
    them) and save the numbers the page shows in sam_preview.json."""
    folder = Path(folder)
    out = run(folder, files, report, **kw)
    done = [e for e in out["frames"] if not e.get("skipped")]
    summary = {"files": files, "at": datetime.now().isoformat(timespec="seconds"),
               "load_seconds": out["load_seconds"], "frames": out["frames"],
               "seconds_per_frame": round(sum(e["seconds"] for e in done) / len(done), 1)
               if done else None,
               "peak_process_gb": max((e["peak_process_gb"] for e in done), default=None),
               "peak_gpu_gb": max((e["peak_gpu_gb"] for e in done), default=None)}
    tmp = folder / (PREVIEW_FILE + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, folder / PREVIEW_FILE)
    return summary


def read_preview(folder: Path) -> dict | None:
    p = Path(folder) / PREVIEW_FILE
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except ValueError:
        return None


def frame_counts(folder: Path, frame: str, units: list[dict], settings: dict) -> dict:
    """{unit key: {"masks": n or None (not run), "older": bool}} for the current classes;
    older = saved with other settings than the current ones."""
    from app.sam import results
    saved = (results.read(folder, frame) or {}).get("units", {})
    out = {}
    for u in units:
        s = saved.get(u["key"])
        out[u["key"]] = {"masks": None if s is None else len(s["masks"]),
                         "older": s is not None
                         and s["settings_key"] != results.settings_key(u, settings)}
    return out


def unit_colour(unit: dict) -> str:
    """A group's masks have no class until tagged: grey (D22 rule 7); else the class colour."""
    return classes.UNTAGGED_COLOUR if unit["group"] else classes.colour(next(iter(unit["classes"])))


def overlay(folder: Path, frame: str, units: list[dict], hidden: set[str],
            class_list: list[dict] | None = None) -> bytes | None:
    """The frame (upright, D27) scaled to PREVIEW_WIDTH with the saved masks of the shown
    units drawn on it: areas first and fainter, objects on top. With class_list: the final
    class map instead (D22, app/sam/overlap.py), one colour per pixel. JPEG bytes; None if
    missing."""
    import io

    import numpy as np
    from PIL import Image, ImageColor, ImageDraw

    from app import images
    from app.sam import results
    path = extract.find_file(folder, frame)
    raw = results.read(folder, frame)
    if path is None or raw is None:
        return None
    im = images.open_upright(path).convert("RGB")
    s = min(1.0, PREVIEW_WIDTH / im.width)
    im = im.resize((round(im.width * s), round(im.height * s)))
    if class_list is not None:
        layer = class_map_layer(raw, class_list, im.size)
        out = io.BytesIO()
        Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB").save(out, "JPEG",
                                                                             quality=85)
        return out.getvalue()
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    order = sorted((u for u in units if u["key"] not in hidden and u["key"] in raw["units"]),
                   key=lambda u: u["kind"] != "area")
    for u in order:
        rgb = ImageColor.getrgb(unit_colour(u))
        alpha = 90 if u["kind"] == "area" else 150
        for m in raw["units"][u["key"]]["masks"]:
            x0, y0, x1, y1 = m["box"]
            crop = results.decode(m["rle"])[y0:y1, x0:x1]
            bx0, by0 = round(x0 * s), round(y0 * s)
            size = (max(1, round(x1 * s) - bx0), max(1, round(y1 * s) - by0))
            mask = Image.fromarray(crop.astype(np.uint8) * 255).resize(size)
            layer.paste(Image.new("RGBA", size, (*rgb, alpha)), (bx0, by0), mask)
            if u["kind"] == "object":
                draw.rectangle([bx0, by0, bx0 + size[0], by0 + size[1]], outline=(*rgb, 255))
    out = io.BytesIO()
    Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB").save(out, "JPEG", quality=85)
    return out.getvalue()


def class_map_layer(raw: dict, class_list: list[dict], size: tuple[int, int]):
    """The class map as a see-through colour layer of `size`: one colour per class id,
    untagged urchins grey, unknown clear."""
    import numpy as np
    from PIL import Image, ImageColor

    from app.sam import overlap
    small = np.array(Image.fromarray(overlap.class_map(raw, class_list)).resize(
        size, Image.NEAREST))
    lut = np.zeros((256, 4), np.uint8)
    for c in class_list:
        lut[c["id"]] = (*ImageColor.getrgb(classes.colour(c["id"])), 140)
    lut[overlap.UNTAGGED] = (*ImageColor.getrgb(classes.UNTAGGED_COLOUR), 140)
    return Image.fromarray(lut[small], "RGBA")
