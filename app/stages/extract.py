"""Stage 3, extract frames. Bite 3a: the sampling plan (every N seconds inside the clips, or the
whole video), the cap spread evenly, and pulling full-resolution frames from the original video
(read only) with a blur score and a hash. Bite 3b: the blur and near-duplicate filters, which
move frames (never erase them) between frames/, dropped/blurry/ and dropped/duplicate/.
Spec §Stage details 3; DESIGN D17, D19.

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


# Each frame status and the folder its file lives in. 'rejected' (bite 4) is left alone here.
HOMES = {"kept": "frames", "dropped_blur": "dropped/blurry",
         "dropped_duplicate": "dropped/duplicate", "unplanned": "dropped/unplanned"}
FILTERED = ("kept", "dropped_blur", "dropped_duplicate")  # the statuses the filters choose between


def home(project_folder: Path, status: str) -> Path:
    return Path(project_folder) / HOMES[status]


def find_file(project_folder: Path, name: str) -> Path | None:
    """Where a frame's file is now, looking in every status folder (a stop between a move and
    its database update can leave it in the folder of its old or its new status)."""
    for status in HOMES:
        if (home(project_folder, status) / name).is_file():
            return home(project_folder, status) / name
    return None


def move_home(project_folder: Path, name: str, status: str) -> bool:
    """Move a frame's file to the folder of `status` (never erasing); False if it is nowhere."""
    found = find_file(project_folder, name)
    target = home(project_folder, status) / name
    if found is None:
        return False
    if found != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(found, target)
    return True


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


def hash_distance(a: str | None, b: str | None) -> int:
    """Number of differing bits between two dHashes (0-64); 64 if one is missing."""
    if not a or not b:
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def filter_frames(rows: list[dict], blur_percent: float, duplicate_distance: int) -> dict:
    """The filter rule, pure (no files, no decoding). rows: frames with file, video,
    timestamp_ms, blur_score, hash, status. Only frames whose status is in FILTERED take part;
    'rejected' and 'unplanned' frames are left alone. Per video:
      1. blur: the blurriest blur_percent % of its frames (rounded down) -> 'dropped_blur';
      2. then, in time order, a frame whose hash differs from the previous *kept* frame by
         fewer than duplicate_distance bits -> 'dropped_duplicate' (0 = off). Blur runs first,
         so a blurry frame never anchors a duplicate check.
    Returns {"status": {file: new status}, "ranked": {video: (blur-dropped, others)}, each
    list nearest the cut-off first, "pairs": [(file, previous kept file, distance)]}; the last
    two feed the examples on the page."""
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        if r["status"] in FILTERED:
            by_video.setdefault(r["video"], []).append(r)
    status, pairs, ranked = {}, [], {}
    for video, frames in by_video.items():
        frames.sort(key=lambda r: r["timestamp_ms"])
        by_blur = sorted(frames, key=lambda r: (r["blur_score"] or 0.0, r["timestamp_ms"]))
        n_blur = int(len(frames) * blur_percent / 100 + 1e-9)
        blurry = {r["file"] for r in by_blur[:n_blur]}
        ranked[video] = (by_blur[:n_blur][::-1], by_blur[n_blur:])  # nearest the cut-off first
        anchor = None
        for r in frames:
            if r["file"] in blurry:
                status[r["file"]] = "dropped_blur"
                continue
            if anchor is not None:
                d = hash_distance(r["hash"], anchor["hash"])
                pairs.append((r["file"], anchor["file"], d))
                if d < duplicate_distance:
                    status[r["file"]] = "dropped_duplicate"
                    continue
            status[r["file"]] = "kept"
            anchor = r
    return {"status": status, "ranked": ranked, "pairs": pairs}


def frame_rows(conn: sqlite3.Connection) -> list[dict]:
    return [dict(zip(("file", "video", "timestamp_ms", "blur_score", "hash", "status"), r))
            for r in conn.execute("SELECT file, video, timestamp_ms, blur_score, hash, status "
                                  "FROM frames")]


def _round_robin(lists: list[list], n: int) -> list:
    """Up to n items, taking the first of each list, then the second of each, and so on."""
    out = []
    for i in range(max((len(x) for x in lists), default=0)):
        out += [x[i] for x in lists if i < len(x)]
    return out[:n]


def preview(conn: sqlite3.Connection, blur_percent: float, duplicate_distance: int,
            n: int = 3) -> dict:
    """What Apply would do, read only: the counts, plus examples at each cut-off: the frames
    just dropped and just kept by blur (nearest each video's cut-off) and the duplicate pairs
    nearest the distance cut-off on both sides, each with the kept frame it was compared to."""
    rows = frame_rows(conn)
    info = {r["file"]: r for r in rows}
    result = filter_frames(rows, blur_percent, duplicate_distance)
    c = {k: 0 for k in ("kept", "dropped_blur", "dropped_duplicate", "rejected")}
    for r in rows:
        st = result["status"].get(r["file"], r["status"])
        if st in c:
            c[st] += 1
    c["total"] = sum(c.values())

    def ex(file, **more):
        r = info[file]
        return {"file": file, "video": Path(r["video"]).name, "t": r["timestamp_ms"] / 1000,
                "blur": r["blur_score"], "status": r["status"], **more}

    ranked = list(result["ranked"].values())
    pairs = result["pairs"]
    dup_drop = sorted((p for p in pairs if p[2] < duplicate_distance), key=lambda p: -p[2])
    dup_keep = sorted((p for p in pairs if p[2] >= duplicate_distance), key=lambda p: p[2])
    return {
        "counts": c,
        "blur_dropped": [ex(r["file"]) for r in _round_robin([d for d, _ in ranked], n)],
        "blur_kept": [ex(r["file"]) for r in _round_robin([k for _, k in ranked], n)],
        "dup_dropped": [ex(f, match=ex(a), distance=d) for f, a, d in dup_drop[:n]],
        "dup_kept": [ex(f, match=ex(a), distance=d) for f, a, d in dup_keep[:n]],
    }


def apply_filters(conn: sqlite3.Connection, project_folder: Path, blur_percent: float,
                  duplicate_distance: int) -> dict:
    """Give every filtered frame its new status and move its file to that status's folder,
    file first, then its row. A stop in between is repaired by the next Apply or Extract:
    the file is found wherever it is and moved home. Nothing is erased."""
    result = filter_frames(frame_rows(conn), blur_percent, duplicate_distance)
    for file, st in result["status"].items():
        move_home(project_folder, file, st)
        conn.execute("UPDATE frames SET status = ? WHERE file = ?", (st, file))
    conn.commit()
    return counts(conn)


def counts(conn: sqlite3.Connection) -> dict:
    rows = conn.execute("SELECT status, COUNT(*) FROM frames GROUP BY status").fetchall()
    out = {"kept": 0, "dropped_blur": 0, "dropped_duplicate": 0, "rejected": 0, "unplanned": 0}
    out.update({r[0]: r[1] for r in rows})
    out["total"] = sum(v for k, v in out.items() if k != "unplanned")  # frames in the plan
    return out


def run(project_folder: Path, samples: list[dict],
        report: Callable[[float, str], None] = lambda p, m: None,
        blur_percent: float = 0, duplicate_distance: int = 0) -> int:
    """Extract every sample not yet on disk; returns how many were new. Each frame is written
    to a temporary name, then its database row, so a stop mid-run loses at most that frame and
    the next run carries on (resume). Frames kept from an earlier plan that this plan does not
    ask for move to dropped/unplanned/ with status 'unplanned'; a planned frame found there
    moves back. A planned frame whose file sits in its status folder (kept or dropped) is never
    decoded again. Rejected frames (bite 4) are left as they are. At the end the saved blur and
    duplicate filters are applied, so kept + blurry + duplicate add up to the plan (bite 3b)."""
    folder = frames_dir(project_folder)
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
            st = status.get(name)
            if st == "rejected":
                pass                                     # bite 4's decision stands
            elif st is not None and find_file(project_folder, name) is not None:
                # already extracted: bring it home (back from dropped/unplanned/, or repair a
                # stop between a move and its database update), never decode it again
                if st == "unplanned":
                    st = "kept"
                move_home(project_folder, name, st)
                conn.execute("UPDATE frames SET status = ? WHERE file = ?", (st, name))
                conn.commit()
            else:
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
            if st in FILTERED and name not in planned:
                move_home(project_folder, name, "unplanned")
                conn.execute("UPDATE frames SET status='unplanned' WHERE file=?", (name,))
                conn.commit()
        c = apply_filters(conn, project_folder, blur_percent, duplicate_distance)
        db.set_stage(conn, "extract", "done", f"{c['kept']} frames")
        return new
    except Exception:
        db.set_stage(conn, "extract", "in progress", "stopped with an error; press Extract again")
        raise
    finally:
        conn.close()
