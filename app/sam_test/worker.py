"""One SAM test attempt, run as its own process by run.py. Writes result JSON after every step,
so a crash part-way still leaves the numbers gathered so far."""

from __future__ import annotations

import contextlib
import io
import json
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from app.sam_test import device_patch
from app.sam_test.tiles import draw_overlay, tile_grid

R: dict = {}
FALLBACK_OPS: set[str] = set()


def save():
    R["fallback_ops"] = sorted(FALLBACK_OPS)
    R["patches"] = list(device_patch.CHANGES)
    R["peak_process_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    Path(R["result_path"]).write_text(json.dumps(R, indent=1))


def _note_warning(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    if "fall back to run on the CPU" in text:
        FALLBACK_OPS.add(text.split("'")[1] if "'" in text else text[:120])


def detections(state, prompt, dx=0, dy=0):
    """Processor state -> list of dicts with the mask cropped to its box (frame pixels)."""
    out = []
    masks, boxes, scores = state["masks"], state["boxes"], state["scores"]
    for i in range(len(scores)):
        x0, y0, x1, y1 = [int(v) for v in boxes[i].round().tolist()]
        x0, y0 = max(0, x0), max(0, y0)
        mask = masks[i, 0, y0:y1, x0:x1].cpu().numpy()
        out.append({"prompt": prompt, "score": round(float(scores[i]), 3),
                    "box": [x0 + dx, y0 + dy, x1 + dx, y1 + dy], "mask": mask})
    return out


def run_mode(proc, image, prompts, tiles, sync, tile=1008):
    """All prompts on one frame, whole (resized by SAM to 1008 px) or as tiles of `tile` px."""
    regions = tile_grid(*image.size, tile=tile) if tiles else [(0, 0, *image.size)]
    found = []
    t = time.perf_counter()
    for x0, y0, x1, y1 in regions:
        state = proc.set_image(image.crop((x0, y0, x1, y1)) if tiles else image)
        for prompt in prompts:
            found += detections(proc.set_text_prompt(prompt, state), prompt, x0, y0)
        sync()
    return found, round(time.perf_counter() - t, 2), len(regions)


def click_test(model, proc, image, frame1_found, click):
    """One click and one box on frame 1: as a single object (SAM 1 style) and as an example."""
    best = sorted([d for d in frame1_found if "urchin" in d["prompt"]],
                  key=lambda d: (d["prompt"] != "purple sea urchin", -d["score"]))
    if click:
        cx, cy = [int(v) for v in click.split(",")]
        source = "given with --click"
        box = [cx - 200, cy - 200, cx + 200, cy + 200]
    elif best:
        x0, y0, x1, y1 = best[0]["box"]
        cx, cy, source = (x0 + x1) // 2, (y0 + y1) // 2, f"centre of best '{best[0]['prompt']}'"
        pad = max(10, (x1 - x0) // 10)
        box = [x0 - pad, y0 - pad, x1 + pad, y1 + pad]
    else:
        cx, cy, source = image.width // 2, image.height // 2, "frame centre (no urchin found)"
        box = [cx - 200, cy - 200, cx + 200, cy + 200]
    W, H = image.size
    box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
    res = {"click_point": [cx, cy], "box_prompt": box, "click_source": source}
    state = proc.set_image(image)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for name, kw in (("click", dict(point_coords=np.array([[cx, cy]]),
                                        point_labels=np.array([1]))),
                         ("box", dict(box=np.array(box)))):
            t = time.perf_counter()
            masks, scores, _ = model.predict_inst(state, multimask_output=False, **kw)
            m = masks[0] > 0
            ys, xs = np.nonzero(m)
            b = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) \
                else [0, 0, 0, 0]
            res[name] = {"seconds": round(time.perf_counter() - t, 2), "score": round(float(scores[0]), 3),
                         "mask_pixels": int(m.sum()),
                         "det": {"prompt": name, "score": float(scores[0]), "box": b,
                                 "mask": m[b[1]:b[3], b[0]:b[2]]}}
        proc.reset_all_prompts(state)
        t = time.perf_counter()
        norm = [(box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H,
                (box[2] - box[0]) / W, (box[3] - box[1]) / H]
        similar = detections(proc.add_geometric_prompt(norm, True, state), "similar to box")
        res["box_as_example"] = {"seconds": round(time.perf_counter() - t, 2),
                                 "objects_found": len(similar)}
    res["messages"] = log.getvalue().strip()[-2000:]
    return res, [res["click"]["det"], res["box"]["det"]], similar


def main(job: dict) -> None:
    R.update(job, status="running", frames_done=[])
    warnings.showwarning = _note_warning
    warnings.simplefilter("always")
    device = job["device"]
    try:
        device_patch.apply(device)
        import torch
        from huggingface_hub import hf_hub_download
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        sync = torch.mps.synchronize if device == "mps" else (lambda: None)
        gpu_peak = [0]

        def sync_and_measure():
            sync()
            if device == "mps":
                gpu_peak[0] = max(gpu_peak[0], torch.mps.driver_allocated_memory())
                R["peak_gpu_gb"] = round(gpu_peak[0] / 1e9, 2)

        R["torch"] = torch.__version__
        try:
            ckpt = hf_hub_download(job["repo_id"], job["checkpoint"], local_files_only=True)
        except Exception as e:
            raise RuntimeError(f"{job['checkpoint']} is not in the Hugging Face cache. "
                               f"Fix: uvx hf download {job['repo_id']}") from e
        t, log = time.perf_counter(), io.StringIO()
        with contextlib.redirect_stdout(log):
            model = build_sam3_image_model(device=device, checkpoint_path=ckpt,
                                           load_from_HF=False, enable_inst_interactivity=True)
            model = model.to(device)
        R["load_seconds"] = round(time.perf_counter() - t, 1)
        R["load_messages"] = log.getvalue().strip()[-3000:]
        # the click/box model's weights: count how many the checkpoint did not supply
        R["click_weights_missing"] = log.getvalue().count("'inst_interactive_predictor.")
        R["click_weights_total"] = sum(1 for k in model.state_dict()
                                       if k.startswith("inst_interactive_predictor."))
        proc = Sam3Processor(model, device=device, confidence_threshold=job["cutoff"])
        names = [p.lower() for p in job["prompts"]]
        palette = names + ["similar to box", "click", "box"]
        out = Path(job["out_dir"])
        frames = [Image.open(p).convert("RGB") for p in job["frames"]]

        t = time.perf_counter()
        run_mode(proc, frames[0], names, False, sync_and_measure)  # warm-up: not counted
        R["warmup_seconds"] = round(time.perf_counter() - t, 2)
        save()

        frame1_found = []
        for i, (path, image) in enumerate(zip(job["frames"], frames)):
            entry = {"name": Path(path).name}
            for mode in ("whole", "tiles"):
                found, secs, n = run_mode(proc, image, names, mode == "tiles", sync_and_measure,
                                          job["tile"])
                overlay = f"overlay_{job['attempt']}_{i + 1}_{mode}.jpg"
                draw_overlay(image, found, palette).save(out / overlay, quality=90)
                counts = {p: sum(d["prompt"] == p for d in found) for p in names}
                entry[mode] = {"seconds": secs, "tiles": n, "masks": counts, "overlay": overlay}
                if i == 0:  # click target: best urchin on frame 1, whole or tiles
                    frame1_found += found
                print(f"{entry['name']} {mode}: {secs} s, {sum(counts.values())} masks",
                      file=sys.stderr)
            R["frames_done"].append(entry)
            save()

        try:
            res, dets, similar = click_test(model, proc, frames[0], frame1_found, job["click"])
            for name in ("click", "box"):
                res[name].pop("det")
            draw_overlay(frames[0], dets + similar, palette).save(
                out / f"overlay_{job['attempt']}_clickbox.jpg", quality=90)
            res["overlay"] = f"overlay_{job['attempt']}_clickbox.jpg"
            R["click_test"] = res
        except Exception as e:  # the text-prompt results still count
            R["click_test"] = {"error": f"{type(e).__name__}: {e}",
                               "traceback": traceback.format_exc()}
        R["status"] = "ok"
    except Exception as e:
        R["status"] = "error"
        R["error"] = f"{type(e).__name__}: {e}"
        R["traceback"] = traceback.format_exc()
    save()


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
