"""SAM in its own process (D9): `python -m app.sam.worker <job.json>`. Loads the model once,
then labels the job's frames in order and writes sam_raw/<frame>.json after each one
(app/sam/results.py). A job-module worker (D33): started by app/jobctl for the "sam" kind
that app/stages/label.py registers.

Progress goes to stdout as "@@ {json}" lines through app/jobctl/worker.py: `loaded`, one `item`
per frame, then `done` or, after "pause" on stdin, `paused` (Meta's code prints its own lines
too; those go to the job's log). A pause or crash loses at most the frame in progress: every
finished frame is on disk, and the next worker skips units whose settings key is saved.
A preview job (job["preview_file"]) also writes the numbers the Labeling page shows; an estimate
job (job["estimate_file"], D35) writes the time per frame of its frames after the first (warm-up).

The predictor is SAM 3 through Meta's Sam3Processor with the S1 run-time fixes
(app/sam_test/device_patch.py, D8). Tests pass job["predictor"] = "module:Class" for a fake
with the same two methods: set_image(PIL image) -> state, find(state, prompt) -> [(mask, score)].
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

from app import images
from app.jobctl import worker as jw
from app.sam import results, tiles


class SamPredictor:
    """SAM 3 image model, loaded once per job."""

    def __init__(self, job: dict):
        from app.sam_test import device_patch
        device = job["device"]
        device_patch.apply(device)
        import torch
        from huggingface_hub import hf_hub_download
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch, self.device = torch, device
        if jw.threads(job):
            torch.set_num_threads(jw.threads(job))   # D36: the CPU thread cap
        try:
            ckpt = hf_hub_download(job["repo_id"], job["checkpoint"], local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"{job['checkpoint']} is not in the Hugging Face cache. Fix, "
                               f"once, in a terminal: uvx hf download {job['repo_id']} "
                               f"{job['checkpoint']} config.json") from e
        with contextlib.redirect_stdout(sys.stderr):
            model = build_sam3_image_model(device=device, checkpoint_path=ckpt,
                                           load_from_HF=False)
            model = model.to(device)
        self.proc = Sam3Processor(model, device=device,
                                  confidence_threshold=job["settings"]["confidence"])
        self.peak_gpu = 0

    def set_image(self, image):
        with contextlib.redirect_stdout(sys.stderr):
            return self.proc.set_image(image)

    def find(self, state, prompt: str):
        with contextlib.redirect_stdout(sys.stderr):
            out = self.proc.set_text_prompt(prompt, state)
        if self.device == "mps":
            self.torch.mps.synchronize()
            self.peak_gpu = max(self.peak_gpu, self.torch.mps.driver_allocated_memory())
        masks, scores = out["masks"], out["scores"]
        return [(masks[i, 0].cpu().numpy().astype(bool), float(scores[i]))
                for i in range(len(scores))]


def make_predictor(job: dict):
    if job.get("predictor"):
        module, cls = job["predictor"].split(":")
        return getattr(importlib.import_module(module), cls)(job)
    return SamPredictor(job)


def label_frame(pred, image, units: list[dict], settings: dict) -> dict[str, dict]:
    """All units on one upright image: whole if its longer side is at most tile_above_px,
    else in tiles (the image encoder runs once per tile, every prompt reuses it, D23).
    Returns {unit key: {"unit", "seconds", "dets"}}: objects after tile-edge duplicates and
    the minimum area, areas as one mask per prompt."""
    W, H = image.size
    tiled = max(W, H) > settings["tile_above_px"]
    regions = (tiles.tile_grid(W, H, settings["tile_px"], settings["tile_overlap"])
               if tiled else [(0, 0, W, H)])
    found = {u["key"]: [] for u in units}
    secs = {u["key"]: 0.0 for u in units}
    for region in regions:
        state = pred.set_image(image.crop(region) if tiled else image)
        for u in units:
            t = time.perf_counter()
            for prompt in u["prompts"]:
                for mask, score in pred.find(state, prompt):
                    det = tiles.from_region(mask, score, prompt, region)
                    if det is not None:
                        found[u["key"]].append(det)
            secs[u["key"]] += time.perf_counter() - t
    smallest = settings.get("min_mask_area_px") or 0
    out = {}
    for u in units:
        dets = (tiles.union_by_prompt(found[u["key"]]) if u["kind"] == "area"
                else [d for d in tiles.remove_duplicates(found[u["key"]], W, H)
                      if tiles.area(d) >= smallest])
        out[u["key"]] = {"unit": u, "seconds": secs[u["key"]], "dets": dets}
    return out


def peak_process_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)  # bytes on macOS


def item(i: int, n: int, rec: dict) -> None:
    """One frame's `item` event (D33): masks per run unit become the generic counts."""
    jw.emit("item", i=i, n=n, name=rec["file"], **{k: v for k, v in rec.items() if k != "file"},
            counts=rec.get("masks", {}))


FALLBACK = "fall back to run on the CPU"   # PyTorch's warning when an op is missing on MPS


def report_cpu_fallback() -> None:
    """D36: a CPU fallback on the Apple GPU becomes a warning on the job (once per op), and
    still goes to the log as before."""
    import warnings
    shown = warnings.showwarning
    seen: set[str] = set()

    def show(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if FALLBACK in text:
            op = text.split("'")[1] if text.count("'") >= 2 else text[:60]
            if op not in seen:
                seen.add(op)
                jw.emit("warning", message=f"Apple GPU: {op} runs on the CPU (slower)")
        shown(message, category, filename, lineno, file, line)
    warnings.showwarning = show


def main(job: dict) -> None:
    folder, settings = Path(job["folder"]), job["settings"]
    if job.get("device") == "mps":
        report_cpu_fallback()
    units = [{**u, "classes": {int(k): v for k, v in u["classes"].items()}} for u in job["units"]]
    ctl = jw.listen()
    t = time.perf_counter()
    pred = make_predictor(job)
    load_s = round(time.perf_counter() - t, 1)
    jw.emit("loaded", seconds=load_s)
    n, recs = len(job["frames"]), []
    for i, fr in enumerate(job["frames"], 1):
        if ctl.pause_requested():
            jw.emit("paused")
            return
        rec = {"file": fr["file"]}
        try:
            todo = results.to_run(folder, fr["file"], units, settings)
        except ValueError as e:  # an unreadable sam_raw file: leave the frame, go on (QE PR 24)
            todo, rec = [], {**rec, "skipped": True, "unreadable": str(e)}
        if not todo:
            recs.append({"skipped": True, **rec})
            item(i, n, recs[-1])
            continue
        t = time.perf_counter()
        image = images.open_upright(Path(fr["path"])).convert("RGB")
        done = label_frame(pred, image, todo, settings)
        results.write_frame(folder, fr["file"], image.width, image.height, settings, done)
        recs.append({**rec, "seconds": round(time.perf_counter() - t, 2),
                     "masks": {k: len(v["dets"]) for k, v in done.items()},
                     "peak_process_gb": peak_process_gb(),
                     "peak_gpu_gb": round(getattr(pred, "peak_gpu", 0) / 1e9, 2)})
        item(i, n, recs[-1])
        jw.rest(ctl, job)                       # D36: pause between frames (wakes on pause)
    if job.get("preview_file"):
        write_preview(Path(job["preview_file"]), [f["file"] for f in job["frames"]], load_s, recs)
    if job.get("estimate_file"):
        write_estimate(Path(job["estimate_file"]), load_s, recs, units, settings)
    what = ("preview: labeled" if job.get("preview_file") else
            "estimate: labeled" if job.get("estimate_file") else "labeled")
    jw.emit("done", message=final_message(recs, n, what))


def final_message(recs: list[dict], n: int, what: str = "labeled") -> str:
    done = sum(1 for r in recs if not r.get("skipped"))
    bad = [r["file"] for r in recs if r.get("unreadable")]
    note = (f"; {len(bad)} not labeled, their sam_raw file cannot be read (move it aside and "
            f"Run again): {', '.join(bad)}" if bad else "")
    return f"{what} {done} frames, {n - done - len(bad)} already current" + note


def write_estimate(path: Path, load_s: float, recs: list[dict], units: list[dict],
                   settings: dict) -> None:
    """D35: the first frame this worker labels warms up and is not counted; the rest give seconds
    per frame. After a Pause and Resume the frames done before are skipped, so the warm-up is
    the first one labeled after the model loaded again (QE PR 35)."""
    labeled = [r for r in recs if not r.get("skipped")]
    warmup, timed = (labeled[0], labeled[1:]) if labeled else (None, [])
    summary = {"at": datetime.now().isoformat(timespec="seconds"), "load_seconds": load_s,
               "warmup": warmup, "frames": timed,
               "seconds_per_frame": round(sum(r["seconds"] for r in timed) / len(timed), 1)
               if timed else None,
               "key": results.classes_key(units, settings)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, path)


def write_preview(path: Path, files: list[str], load_s: float, recs: list[dict]) -> None:
    """The preview numbers the Labeling page shows (S3 bite 3), written atomically."""
    done = [r for r in recs if not r.get("skipped")]
    summary = {"files": files, "at": datetime.now().isoformat(timespec="seconds"),
               "load_seconds": load_s, "frames": recs,
               "seconds_per_frame": round(sum(r["seconds"] for r in done) / len(done), 1)
               if done else None,
               "peak_process_gb": max((r["peak_process_gb"] for r in done), default=None),
               "peak_gpu_gb": max((r["peak_gpu_gb"] for r in done), default=None)}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, path)


if __name__ == "__main__":
    main(jw.load_job())
