"""Stage 3, extract frames. Bite 3a: the sampling plan (every N seconds inside the clips, or the
whole video), the cap spread evenly, and pulling full-resolution frames from the original video
(read only) with a blur score and a hash for the filters of bite 3b. Spec §Stage details 3;
DESIGN D17.

Measured on the SSD (DEV_LOG 2026-11-24): 0.5 s (4K) to 0.8 s (5.3K) per frame by seek; JPEG
quality 95 save and both scores ~0.06 s.
"""

from __future__ import annotations

import io
import os
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage

from app import db

JPEG_QUALITY = 95          # spec: quality >= 95
BLUR_WIDTH = 1024          # blur is scored on a 1024 px wide grey copy, so 4K and 5.3K compare
FRAME_NAME = re.compile(r"^[A-Za-z0-9_-]+_[0-9a-f]{6}_\d+\.jpg$")


def frames_dir(project_folder: Path) -> Path:
    return Path(project_folder) / "frames"


def unplanned_dir(project_folder: Path) -> Path:
    """Frames from an earlier plan that the current settings no longer ask for. Moved here,
    never erased; they come back if the settings ask for them again (QE PR 7)."""
    return Path(project_folder) / "dropped" / "unplanned"


def frame_name(video_path: str, code: str, t: float) -> str:
    """<video-stem>_<path-code>_<timestamp-ms>.jpg; characters other than letters, digits, - and _
    in the stem become _ (spaces in ReefCheck site names)."""
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(video_path).stem)
    return f"{stem}_{code}_{round(t * 1000)}.jpg"


def _times(start: float, end: float, every: float) -> list[float]:
    n = int(np.ceil(round((end - start) / every, 6))) if end > start else 0
    return [round(start + k * every, 3) for k in range(n)]


def sample_plan(videos: list[dict], all_clips: list[dict], every_s: float,
                cap: int) -> tuple[list[dict], int]:
    """Sample times for every video, in video then time order: inside its clips, or the whole
    video if it has none. Returns (samples after the cap, live count before the cap). Over the
    cap, samples are taken evenly across the whole list (as ml/yolox_urchin/hil_stills.py
    sample_indices does), so every video keeps its share."""
    samples = []
    for v in videos:
        mine = sorted((c for c in all_clips if c["video"] == v["path"]),
                      key=lambda c: c["start_s"])
        spans = [(c["start_s"], c["end_s"]) for c in mine] or [(0.0, v["duration_s"] or 0.0)]
        for a, b in spans:
            samples += [{"video": v, "t": t} for t in _times(a, b, every_s)]
    live = len(samples)
    if live > cap > 0:
        keep = sorted({int(round(x)) for x in np.linspace(0, live - 1, cap)})
        samples = [samples[i] for i in keep]
    return samples, live


def grab(video_path: str, t: float) -> Image.Image:
    """One full-resolution frame at t seconds, decoded by ffmpeg to a PPM image on a pipe
    (nothing written). The PPM header carries the size, so a video with rotation metadata (a
    phone, an upright GoPro) comes out upright with its width and height swapped (QE PR 7)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", video_path,
                        "-frames:v", "1", "-f", "image2pipe", "-c:v", "ppm", "-"],
                       capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout.startswith(b"P6"):
        raise RuntimeError(f"ffmpeg could not read a frame at {t:.2f} s of {video_path}: "
                           f"{r.stderr.decode(errors='replace').strip()[-300:] or 'no data'}")
    with Image.open(io.BytesIO(r.stdout)) as im:
        return im.convert("RGB")


def blur_score(im: Image.Image) -> float:
    """Variance of the Laplacian on a 1024 px wide grey copy: higher = sharper."""
    h = max(1, round(BLUR_WIDTH * im.height / im.width))
    grey = np.asarray(im.convert("L").resize((BLUR_WIDTH, h)), dtype=np.float32)
    return round(float(ndimage.laplace(grey).var()), 2)


def dhash(im: Image.Image) -> str:
    """64-bit difference hash as 16 hex characters (near-duplicate filter, bite 3b)."""
    g = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def counts(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT status, COUNT(*) FROM frames GROUP BY status").fetchall()
    out = {"kept": 0, "dropped_blur": 0, "dropped_duplicate": 0, "rejected": 0, "unplanned": 0}
    out.update({r[0]: r[1] for r in rows})
    out["total"] = sum(v for k, v in out.items() if k != "unplanned")  # frames in the plan
    return out


def run(project_folder: Path, samples: list[dict],
        report: Callable[[float, str], None] = lambda p, m: None) -> int:
    """Extract every sample not yet on disk; returns how many were new. Each frame is written
    to a temporary name, then its database row, so a stop mid-run loses at most that frame and
    the next run carries on (resume). Frames kept from an earlier plan that this plan does not
    ask for move to dropped/unplanned/ with status 'unplanned'; a planned frame found there
    moves back. Frames a later stage dropped or rejected are left as they are."""
    folder = frames_dir(project_folder)
    aside = unplanned_dir(project_folder)
    folder.mkdir(parents=True, exist_ok=True)
    conn = db.connect(Path(project_folder) / "state.sqlite")  # own connection: runs in a thread
    try:
        db.set_stage(conn, "extract", "in progress", f"0 of {len(samples)} frames")
        status = dict(conn.execute("SELECT file, status FROM frames"))
        planned, new = set(), 0
        for i, s in enumerate(samples):
            v = s["video"]
            name = frame_name(v["path"], v["path_code"], s["t"])
            planned.add(name)
            if status.get(name) == "unplanned" and (aside / name).is_file():
                os.replace(aside / name, folder / name)
                conn.execute("UPDATE frames SET status='kept' WHERE file=?", (name,))
                conn.commit()
            elif name not in status or not (folder / name).is_file():
                if not Path(v["path"]).is_file():
                    raise FileNotFoundError(f"Video not found: {v['path']}. Plug in the SSD "
                                            "and press Extract again; finished frames are kept.")
                im = grab(v["path"], s["t"])
                tmp = folder / (name + ".tmp")
                im.save(tmp, format="JPEG", quality=JPEG_QUALITY)
                os.replace(tmp, folder / name)
                conn.execute("INSERT OR REPLACE INTO frames (file, video, timestamp_ms, "
                             "blur_score, hash, status) VALUES (?, ?, ?, ?, ?, 'kept')",
                             (name, v["path"], round(s["t"] * 1000), blur_score(im), dhash(im)))
                conn.commit()
                new += 1
            report(100 * (i + 1) / len(samples), f"frame {i + 1} of {len(samples)}")
        for name, st in status.items():
            if st == "kept" and name not in planned:
                if (folder / name).is_file():
                    aside.mkdir(parents=True, exist_ok=True)
                    os.replace(folder / name, aside / name)
                conn.execute("UPDATE frames SET status='unplanned' WHERE file=?", (name,))
                conn.commit()
        c = counts(conn)
        db.set_stage(conn, "extract", "done", f"{c['kept']} frames")
        return new
    except Exception:
        db.set_stage(conn, "extract", "in progress", "stopped with an error; press Extract again")
        raise
    finally:
        conn.close()
