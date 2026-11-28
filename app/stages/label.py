"""Stage 5, SAM labeling: pick the frames and describe the SAM job (DESIGN, Planned for S3/S4).

A SAM Preview or Run is a job of the "sam" kind on app/jobctl (D32, D33): params are
{folder, files, preview, predictor}; prepare() turns them into the worker's job file each time
the job starts or resumes, and the worker (app/sam/worker.py) skips units whose results are
current, so a resumed job does only the frames still missing. The job module runs one job at a
time for the whole app, so two SAM jobs never share the GPU (S1: two at once triple the time).
"""

from __future__ import annotations

import functools
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from app import sidecars
from app.jobctl import Kind
from app.sam import classes
from app.stages import extract

MODELS = {"sam3": ("facebook/sam3", "sam3.pt"),           # D10
          "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt")}
LABELED = ("kept", "kept_hand")
PREDICTOR: str | None = None  # tests set a fake ("module:Class"); None = real SAM
ADD_PREDICTOR: str | None = None  # the same for "add a missed object" (bite 5)


class LabelError(RuntimeError):
    """The SAM run failed; the message says why and points at the log."""


def frames_to_label(conn: sqlite3.Connection) -> list[str]:
    """Kept frames (also kept by hand), in name order."""
    marks = ",".join("?" * len(LABELED))
    return [r[0] for r in conn.execute(
        f"SELECT file FROM frames WHERE status IN ({marks}) ORDER BY file", LABELED)]


def device() -> str:
    """D36 "auto": an NVIDIA GPU, else the Apple GPU (S1), else CPU. Asked in a child process,
    so the app itself never imports torch."""
    r = subprocess.run([sys.executable, "-c", "import torch; print('cuda' if "
                        "torch.cuda.is_available() else 'mps' if "
                        "torch.backends.mps.is_available() else 'cpu')"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "cpu"


def build_job(params: dict) -> dict:
    """The SAM worker's job file for params {folder, files, preview, predictor}. Raises
    LabelError if a frame's file is missing, classes.ClassesError if classes.yaml has problems."""
    folder = Path(params["folder"])
    predictor = params.get("predictor")
    cfg = classes.ensure(folder)
    settings = cfg["settings"]
    units = [{**u, "classes": {str(k): v for k, v in u["classes"].items()}}
             for u in classes.run_units(cfg["classes"])]
    frames = []
    for f in params["files"]:
        p = extract.find_file(folder, f)
        if p is None:
            raise LabelError(f"The file of frame {f} is missing from the project folder.")
        frames.append({"file": f, "path": str(p)})
    repo_id, ckpt = MODELS[settings["model"]]
    return {"folder": str(folder), "frames": frames, "units": units, "settings": settings,
            "repo_id": repo_id, "checkpoint": ckpt, "predictor": predictor,
            "device": params.get("device") or ("cpu" if predictor else
                                               pick_device(params.get("limits") or {})),
            "preview_file": str(folder / PREVIEW_FILE) if params.get("preview") else None,
            "estimate_file": str(folder / ESTIMATE_FILE) if params.get("estimate") else None}


def pick_device(lim: dict) -> str:
    """The device from the job's limits (D36); "auto" asks torch in a child process."""
    chosen = lim.get("device") or "auto"
    return device() if chosen == "auto" else chosen


def progress(params: dict) -> tuple[int, int]:
    """Frames done / total, from sam_raw/ (D33): a frame is done when no unit needs running."""
    from app.sam import results
    folder, files = Path(params["folder"]), params["files"]
    try:
        cfg = classes.ensure(folder)
    except classes.ClassesError:
        return 0, len(files)
    units = classes.run_units(cfg["classes"])
    done = 0
    for f in files:
        try:
            done += not results.to_run(folder, f, units, cfg["settings"])
        except ValueError:
            pass
    return done, len(files)


def sam_kind() -> Kind:
    # S1: the CPU fallback on (D8)
    return Kind("sam", module="app.sam.worker", prepare=build_job, progress=progress,
                env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"}, item="frame")


# --- Add a missed object (S5 bite 5, D46): one click or box per job of the "sam_add" kind ---

def add_params(folder: Path, frame: str, class_id: int, point: list[int] | None = None,
               box: list[int] | None = None) -> dict:
    import uuid
    return {"folder": str(folder), "frame": frame, "class_id": class_id, "point": point,
            "box": box, "request": uuid.uuid4().hex[:12], "predictor": ADD_PREDICTOR}


def build_add_job(params: dict) -> dict:
    """The add worker's job file. SAM 3 always: SAM 3.1 has no click / box weights (S1, D10)."""
    folder = Path(params["folder"])
    p = extract.find_file(folder, params["frame"])
    if p is None:
        raise LabelError(f"The file of frame {params['frame']} is missing from the project folder.")
    repo_id, ckpt = MODELS["sam3"]
    predictor = params.get("predictor")
    return {**params, "path": str(p), "repo_id": repo_id, "checkpoint": ckpt,
            "device": params.get("device") or ("cpu" if predictor else
                                               pick_device(params.get("limits") or {}))}


def add_progress(params: dict) -> tuple[int, int]:
    from app.sam import add
    try:
        done = params["request"] in add.done_requests(Path(params["folder"]), params["frame"])
    except ValueError:
        done = False
    return int(done), 1


def add_kind() -> Kind:
    return Kind("sam_add", module="app.sam.add_worker", prepare=build_add_job,
                progress=add_progress, env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
                item="object", pausable=False)


def job_params(folder: Path, files: list[str], preview: bool = False,
               estimate: bool = False) -> dict:
    return {"folder": str(folder), "files": list(files), "preview": preview,
            "estimate": estimate, "predictor": PREDICTOR}


# --- Estimate (S4 bite 3, D35) ---

ESTIMATE_FILE = "sam_estimate.json"
ESTIMATE_FRAMES = 5          # timed frames, after 1 warm-up frame that is not counted (spec)


def frames_with_work(folder: Path, files: list[str], units: list[dict],
                     settings: dict) -> list[str]:
    """The frames Run would label: some unit has no current results. A frame whose results
    file cannot be read is left out, as Run leaves it alone (QE PR 24)."""
    from app.sam import results
    out = []
    for f in files:
        try:
            if results.to_run(folder, f, units, settings):
                out.append(f)
        except ValueError:
            pass
    return out


def estimate_frames(work: list[str]) -> list[str]:
    """1 warm-up + 5 frames spread evenly over the frames Run would label (D35), in order."""
    k = ESTIMATE_FRAMES + 1
    if len(work) <= k:
        return list(work)
    picks = sorted({round(j * (len(work) - 1) / (k - 1)) for j in range(k)})
    return [work[i] for i in picks]


def read_estimate(folder: Path) -> dict | None:
    p = Path(folder) / ESTIMATE_FILE
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except ValueError:
        return None


# --- Run on every kept frame (bite 4, D26) ---

def run_plan(folder: Path, files: list[str], units: list[dict], settings: dict,
             preview: dict | None = None, estimate: dict | None = None) -> dict:
    """What Run will do (SPEC Q7): per run unit, on how many frames it runs (its settings key
    differs or is missing); frames with any work; unreadable sam_raw files; a time estimate.
    The estimate comes from Estimate (D35: seconds per frame × frames with work) when one was
    made with the current classes and settings, else from the preview's timing: a fixed cost
    per frame with work plus a cost per unit (D26)."""
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
    current = bool(estimate and estimate.get("seconds_per_frame")
                   and estimate.get("key") == results.classes_key(units, settings))
    if current:
        est, source = round(frames * estimate["seconds_per_frame"]), "estimate"
    else:
        cost = frame_costs(folder, preview)
        est = (round(frames * cost[0] + sum(todo.values()) * cost[1]) if cost and frames
               else None)
        source = "preview" if est is not None else None
    return {"units": todo, "frames": frames, "total": len(files), "unreadable": unreadable,
            "estimate_s": est, "estimate_from": source,
            "per_frame": round(est / frames, 1) if est and frames else None,
            "estimate_old": bool(estimate) and not current}


def frame_costs(folder: Path, preview: dict | None) -> tuple[float, float] | None:
    """(seconds per frame with any work, seconds per unit) from the preview's frames. The
    fixed part (read, tile, image encoding, which the first unit pays) does not shrink when
    fewer units run: measured 2026-11-25, one unit on 8 4K frames took 28 s per frame, not
    62 s / 8 units."""
    import statistics

    from app.sam import results
    rows = []
    for e in (preview or {}).get("frames", []):
        if e.get("skipped") or not e.get("seconds"):
            continue
        try:
            raw = results.read(folder, e["file"]) or {}
        except ValueError:
            continue
        secs = [u["seconds"] for u in raw.get("units", {}).values()
                if u.get("run_at", "") <= (preview.get("at") or "~")]
        if secs:
            rows.append((e["seconds"], secs))
    if not rows:
        return None
    unit = statistics.median(s for _, secs in rows for s in secs)
    fixed = statistics.mean(total - sum(secs) + max(secs) - unit for total, secs in rows)
    return max(fixed, 0.0), unit


# --- Prompt preview (bite 3, D26) ---

PREVIEW_FILE = "sam_preview.json"
PREVIEW_WIDTH = 1600  # overlay width on the page; the frame itself is never changed
LARGE_WIDTH = 3200    # overlay width in the large viewer (bite 5); smaller frames stay as they are


def preview_frames(files: list[str], other: bool = False, rng=None) -> list[str]:
    """D26: first, middle and last kept frame by name; "Other frames" picks 3 at random."""
    if other:
        import random
        return sorted((rng or random).sample(files, min(3, len(files))))
    if len(files) <= 3:
        return list(files)
    return [files[0], files[len(files) // 2], files[-1]]


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


def labeled_frames(folder: Path, files: list[str]) -> list[tuple[str, int]]:
    """S4 bite 6c ("Labeled so far"): the kept frames that have a sam_raw results file, newest
    first, with the file's mtime (ns). Only stats the folder, so a page can ask often."""
    import os

    from app.sam import results
    raw = results.raw_dir(folder)
    try:
        times = {e.name: e.stat().st_mtime_ns for e in os.scandir(raw)
                 if e.name.endswith(".json") and not sidecars.is_sidecar(e.name)}
    except FileNotFoundError:
        return []
    have = [(f, times[f + ".json"]) for f in files if f + ".json" in times]
    return sorted(have, key=lambda ft: (-ft[1], ft[0]))


def unit_colour(unit: dict) -> str:
    """A group's masks have no class until tagged: grey (D22 rule 7); else the class colour."""
    return classes.UNTAGGED_COLOUR if unit["group"] else classes.colour(next(iter(unit["classes"])))


def overlay(folder: Path, frame: str, units: list[dict], hidden: set[str],
            class_list: list[dict] | None = None, width: int = PREVIEW_WIDTH) -> bytes | None:
    """The frame (upright, D27) scaled to `width` with the saved masks of the shown units
    drawn on it: areas first and fainter, objects on top with a box. With class_list: the
    final class map instead (D22, app/sam/overlap.py), one colour per pixel. JPEG bytes;
    None if missing. The scaled frame and the scaled masks are cached per file version, so
    a class toggle only pastes and encodes (bite 5)."""
    import io

    from PIL import Image, ImageColor, ImageDraw

    from app.sam import results
    path = extract.find_file(folder, frame)
    rp = results.raw_path(folder, frame)
    if path is None or not rp.is_file():
        return None
    st, rst = path.stat(), rp.stat()   # a changed file (time or size) is read again
    im = _scaled_frame(str(path), (st.st_mtime_ns, st.st_size), width)
    rkey = (str(folder), frame, (rst.st_mtime_ns, rst.st_size))
    if class_list is not None:
        layer = _map_layer(*rkey, json.dumps(class_list, sort_keys=True), im.size)
    else:
        pieces = _mask_pieces(*rkey, im.size)
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        order = sorted((u for u in units if u["key"] not in hidden and u["key"] in pieces),
                       key=lambda u: u["kind"] != "area")
        for u in order:
            rgb = ImageColor.getrgb(unit_colour(u))
            alpha = 90 if u["kind"] == "area" else 150
            for (bx0, by0), size, mask in pieces[u["key"]]:
                layer.paste(Image.new("RGBA", size, (*rgb, alpha)), (bx0, by0), mask)
                if u["kind"] == "object":
                    draw.rectangle([bx0, by0, bx0 + size[0], by0 + size[1]],
                                   outline=(*rgb, 255), width=max(1, round(im.width / 1600)))
    out = io.BytesIO()
    Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB").save(out, "JPEG", quality=85)
    return out.getvalue()


@functools.lru_cache(maxsize=6)
def _scaled_frame(path: str, version: tuple, width: int):
    """The upright frame scaled down to `width` (never up). Cached: a 5.3K JPEG takes about
    0.3 s to read and scale."""
    from PIL import Image

    from app import images
    im = images.open_upright(Path(path)).convert("RGB")
    s = min(1.0, width / im.width)
    return im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)


@functools.lru_cache(maxsize=24)
def _mask_pieces(folder: str, frame: str, version: tuple, size: tuple[int, int]) -> dict:
    """{unit key: [((x, y), (w, h), mask image)]}: each saved mask decoded inside its box only
    (results.decode_box) and scaled to `size`. Cached per sam_raw file version."""
    import numpy as np
    from PIL import Image

    from app.sam import results
    raw = results.read(Path(folder), frame)
    sx, sy = size[0] / raw["width"], size[1] / raw["height"]
    out = {}
    for key, u in raw["units"].items():
        out[key] = []
        for m in u["masks"]:
            x0, y0, x1, y1 = m["box"]
            if x1 <= x0 or y1 <= y0:
                continue
            crop = results.decode_box(m["rle"], m["box"])
            bx0, by0 = round(x0 * sx), round(y0 * sy)
            wh = (max(1, round(x1 * sx) - bx0), max(1, round(y1 * sy) - by0))
            out[key].append(((bx0, by0), wh,
                             Image.fromarray(crop.astype(np.uint8) * 255).resize(wh)))
    return out


@functools.lru_cache(maxsize=6)
def _map_layer(folder: str, frame: str, version: tuple, class_json: str,
               size: tuple[int, int]):
    from app.sam import results
    return class_map_layer(results.read(Path(folder), frame), json.loads(class_json), size)


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
