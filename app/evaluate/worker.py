"""The `evaluate` job's worker (S9 bite 1, D90): `python -m app.evaluate.worker <job.json>`.

Each test frame is one item: the whole frame at the model's eval factor, its not-fully-checked regions grey,
cut into tiles of the trained canvas; a predictor returns each tile's detections; copies cut by an inner tile
edge are dropped (unless larger than the overlap guarantees), the rest go back to frame coordinates, NMS runs
again across the seams, and the frame's boxes are written in native pixels. Pause is taken between frames; a
new worker skips the frames already in predictions.jsonl. eval.json is written when every frame is done."""

from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.datasets import versions
from app.evaluate import metrics
from app.evaluate import run as runs
from app.jobctl import worker as jw
from app.train import models, samples, tiles
from app.train.worker import code_version

GREY = samples.GREY


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_predictor(spec: str, mdir: Path, recipe: dict, job: dict):
    module, cls = spec.split(":")
    return getattr(importlib.import_module(module), cls)(mdir, recipe, job)


def frame(sample: dict, factor, canvas: int, overlap: int, predictor) -> tuple[list, int]:
    """One test frame -> ([class index, x, y, w, h, score] in native pixels, tiles used)."""
    got = samples.frame_at(sample, factor, canvas)
    img = got["image"].copy()
    for cx, cy, w, h in got["ignore"]:
        img[max(0, int(cy - h / 2)):int(cy + h / 2) + 1, max(0, int(cx - w / 2)):int(cx + w / 2) + 1] = GREY
    th, tw = img.shape[:2]
    cuts = []
    for ox, oy in tiles.plan(tw, th, canvas, overlap):
        tile = np.full((canvas, canvas, 3), GREY, np.uint8)
        part = img[oy:oy + canvas, ox:ox + canvas]
        tile[:part.shape[0], :part.shape[1]] = part
        cuts.append((ox, oy, tile))
    found = []
    for k in range(0, len(cuts), predictor.batch):
        part = cuts[k:k + predictor.batch]
        for (ox, oy, _), det in zip(part, predictor.predict(np.stack([t for *_, t in part]))):
            det = np.asarray(det, np.float32).reshape(-1, 6)
            det = det[~tiles.cut_by_inner_edge(det, ox, oy, canvas, tw, th, overlap)]
            det[:, 0] += ox
            det[:, 1] += oy
            found.append(det)
    det = tiles.nms(np.concatenate(found) if found else np.zeros((0, 6), np.float32), runs.NMS_IOU)
    s, W, H, out = got["scale"], sample["width"], sample["height"], []
    for cx, cy, w, h, sc, c in det:            # QE PR 121 nit 2: clipped to the frame, in native pixels
        x0, y0 = max(0.0, float(cx - w / 2) / s), max(0.0, float(cy - h / 2) / s)
        x1, y1 = min(float(W), float(cx + w / 2) / s), min(float(H), float(cy + h / 2) / s)
        if x1 > x0 and y1 > y0:
            out.append([int(c), round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2), round(float(sc), 4)])
    return out, len(cuts)


def main() -> None:
    job = jw.load_job()
    ctl = jw.listen()
    folder, mid = Path(job["folder"]), job["model_id"]
    mdir = models.root(folder) / mid
    recipe = models.read_recipe(mdir)
    ids, version = list(job["class_ids"]), int(job["version"])
    frames = runs.test_frames(folder, version, ids)
    if f"v{version}-{runs.test_hash(frames, ids)}" != job["key"]:
        raise RuntimeError(f"The test set of version {version} is no longer the one this job was made for "
                           "(the version was changed by hand). Start the evaluation again.")
    out = runs.eval_dir(folder, mid, job["key"])
    out.mkdir(parents=True, exist_ok=True)
    best = mdir / "checkpoints" / "best.pt"
    started, t0 = _now(), time.perf_counter()
    predictor = load_predictor(job["predictor"], mdir, recipe, job)
    jw.emit("loaded", seconds=round(time.perf_counter() - t0, 2))
    factor, canvas = job["eval_scale"]["factor"], int(recipe["train"]["tile"])
    dropped = runs.trim(out)                   # QE PR 121: a cut last line never gets the next one glued on
    if dropped:
        jw.emit("warning", message=f"predictions.jsonl: {dropped} cut or repeated line(s) dropped; those frames "
                                   "are scored again")
    sha = versions.sha256_file(best)           # QE PR 121 nit 1: one set of weights per eval, also across Resume
    mark = out / "weights.sha256"
    if mark.is_file() and mark.read_text().strip() != sha and runs.done_files(out):
        raise RuntimeError(f"best.pt of {mid} changed since this evaluation started (the run was resumed?): its "
                           f"frames would mix two sets of weights. Delete {out} and evaluate again.")
    mark.write_text(sha + "\n")
    done, n, used = set(runs.done_files(out)), len(frames), 0
    for i, s in enumerate(frames, 1):
        if s["file"] in done:
            jw.emit("item", i=i, n=n, name=s["file"], skipped=True)
            continue
        if ctl.pause_requested():
            jw.emit("paused")
            return
        t = time.perf_counter()
        boxes, k = frame(s, factor, canvas, int(job["eval_overlap"]), predictor)
        used += k
        line = {"file": s["file"], "boxes": [[ids[b[0]], *b[1:]] for b in boxes], "tiles": k,
                "seconds": round(time.perf_counter() - t, 3)}
        with open(out / "predictions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        jw.emit("item", i=i, n=n, name=s["file"], seconds=line["seconds"])
    preds = runs.read_predictions(out, [f["file"] for f in frames])    # every test frame once, or no eval.json
    figs = metrics.figures(frames, preds, ids, factor, runs.THRESHOLD, runs.test_groups(folder, version))  # D91, D92
    tmp = out / ".metrics.json.tmp"
    tmp.write_text(json.dumps(figs, indent=1, sort_keys=True))
    os.replace(tmp, out / "metrics.json")
    body = {"model": mid, "weights": str(best.relative_to(mdir)), "weights_sha256": sha,
            "version": version, "test_set": job["key"], "class_ids": ids, "frames": n,
            "eval_scale": job["eval_scale"], "eval_scale_source": job["source"], "eval_overlap": job["eval_overlap"],
            "eval_overlap_capped": job["eval_overlap_capped"], "canvas": canvas, "threshold": runs.THRESHOLD,
            "score_floor": runs.SCORE_FLOOR, "nms_iou": runs.NMS_IOU, "predictor": job["predictor"],
            "device": getattr(predictor, "device", ""), "code": code_version(), "started": started, "ended": _now()}
    tmp = out / ".eval.json.tmp"
    tmp.write_text(json.dumps(body, indent=1, sort_keys=True))
    os.replace(tmp, out / "eval.json")
    jw.emit("done", message=f"evaluated {n} test frames of version {version}; results in {out}")


if __name__ == "__main__":
    jw.run(main)
