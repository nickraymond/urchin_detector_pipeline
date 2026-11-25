"""SAM in its own process (D9): `python -m app.sam.worker <job.json>`. Loads the model once,
then labels the job's frames in order and writes sam_raw/<frame>.json after each one
(app/sam/results.py). Started and watched by app/stages/label.py.

Progress goes to stdout as lines "@@ {json}" (Meta's code prints its own lines too; those are
ignored). A stop or crash loses at most the frame in progress: every finished frame is on
disk, and the next run skips units whose settings key is already saved.

The predictor is SAM 3 through Meta's Sam3Processor with the S1 run-time fixes
(app/sam_test/device_patch.py, D8). Tests pass job["predictor"] = "module:Class" for a fake
with the same two methods: set_image(PIL image) -> state, find(state, prompt) -> [(mask, score)].
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from app import images
from app.sam import results, tiles

OUT = sys.stdout


def emit(**event) -> None:
    OUT.write("@@ " + json.dumps(event) + "\n")
    OUT.flush()


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


def main(job: dict) -> None:
    folder, settings = Path(job["folder"]), job["settings"]
    units = [{**u, "classes": {int(k): v for k, v in u["classes"].items()}} for u in job["units"]]
    t = time.perf_counter()
    pred = make_predictor(job)
    emit(event="loaded", seconds=round(time.perf_counter() - t, 1))
    n = len(job["frames"])
    for i, fr in enumerate(job["frames"], 1):
        todo = results.to_run(folder, fr["file"], units, settings)
        if not todo:
            emit(event="frame", i=i, n=n, file=fr["file"], skipped=True)
            continue
        t = time.perf_counter()
        image = images.open_upright(Path(fr["path"])).convert("RGB")
        done = label_frame(pred, image, todo, settings)
        results.write_frame(folder, fr["file"], image.width, image.height, settings, done)
        emit(event="frame", i=i, n=n, file=fr["file"], seconds=round(time.perf_counter() - t, 2),
             masks={k: len(v["dets"]) for k, v in done.items()},
             peak_process_gb=peak_process_gb(),
             peak_gpu_gb=round(getattr(pred, "peak_gpu", 0) / 1e9, 2))
    emit(event="done")


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
