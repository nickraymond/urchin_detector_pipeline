"""Add a missed object (S5 bite 5, D46) in its own process: `python -m app.sam.add_worker <job>`.
A job-module worker (D33) of the "sam_add" kind that app/stages/label.py registers: it loads SAM 3
with its click / box head (S1: SAM 3 has those weights, SAM 3.1 does not, D10), runs the one
click or box on a crop around it (app/sam/add.py) and saves the mask under the unit `added`.
Events: `loaded`, one `item`, `done` (its message says what was added, or that SAM found
nothing there). Tests pass job["predictor"] = "module:Class" for a fake with
click(PIL crop, point=(x, y) | box=[x0, y0, x1, y1]) -> (bool mask of the crop, score) or None.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import time
from pathlib import Path

import numpy as np

from app import images
from app.jobctl import worker as jw
from app.sam import add, tiles


class SamClickPredictor:
    """SAM 3 with the interactive (click / box) head, loaded once per job (S1 click_test)."""

    def __init__(self, job: dict):
        from app.sam_test import device_patch
        device = job["device"]
        device_patch.apply(device)
        import torch
        from huggingface_hub import hf_hub_download
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        if jw.threads(job):
            torch.set_num_threads(jw.threads(job))
        try:
            ckpt = hf_hub_download(job["repo_id"], job["checkpoint"], local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"{job['checkpoint']} is not in the Hugging Face cache. Fix, "
                               f"once, in a terminal: uvx hf download {job['repo_id']} "
                               f"{job['checkpoint']} config.json") from e
        with contextlib.redirect_stdout(sys.stderr):
            self.model = build_sam3_image_model(device=device, checkpoint_path=ckpt,
                                                load_from_HF=False,
                                                enable_inst_interactivity=True).to(device)
        self.proc = Sam3Processor(self.model, device=device)

    def click(self, crop, point=None, box=None):
        with contextlib.redirect_stdout(sys.stderr):
            state = self.proc.set_image(crop)
            kw = (dict(point_coords=np.array([point]), point_labels=np.array([1]))
                  if point is not None else dict(box=np.array(box)))
            masks, scores, _ = self.model.predict_inst(state, multimask_output=False, **kw)
        m = np.asarray(masks[0]) > 0
        return (m, float(scores[0])) if m.any() else None


def make_predictor(job: dict):
    if job.get("predictor"):
        module, cls = job["predictor"].split(":")
        return getattr(importlib.import_module(module), cls)(job)
    return SamClickPredictor(job)


def main(job: dict) -> None:
    folder, frame = Path(job["folder"]), job["frame"]
    if job["request"] in add.done_requests(folder, frame):   # resumed after it was saved
        jw.emit("done", message="already added")
        return
    t = time.perf_counter()
    pred = make_predictor(job)
    jw.emit("loaded", seconds=round(time.perf_counter() - t, 1))
    t = time.perf_counter()
    image = images.open_upright(Path(job["path"])).convert("RGB")
    W, H = image.size
    point, box = job.get("point"), job.get("box")
    region = add.region_for(W, H, point=point, box=box)
    x0, y0 = region[0], region[1]
    found = pred.click(image.crop(region),
                       point=None if point is None else [point[0] - x0, point[1] - y0],
                       box=None if box is None else [box[0] - x0, box[1] - y0,
                                                     box[2] - x0, box[3] - y0])
    det = None if found is None else tiles.from_region(found[0], found[1],
                                                       "click" if point else "box", region)
    if det is None:
        jw.emit("item", i=1, n=1, name=frame, seconds=round(time.perf_counter() - t, 2),
                found=False)
        jw.emit("done", message="SAM found nothing there: try a box around it")
        return
    mid = add.save(folder, frame, det, job["class_id"], job["request"], W, H)
    jw.emit("item", i=1, n=1, name=frame, seconds=round(time.perf_counter() - t, 2),
            found=True, mask_id=mid, area=int(det["mask"].sum()))
    jw.emit("done", message=f"added mask {mid} ({int(det['mask'].sum())} px)")


if __name__ == "__main__":
    main(jw.load_job())
