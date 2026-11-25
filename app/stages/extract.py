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


# Each frame status and the folder its file lives in. 'kept_hand' (kept by hand in frame review,
# D20) and 'rejected' (deleted in frame review) are set only by review; the filters leave them alone.
HOMES = {"kept": "frames", "kept_hand": "frames", "dropped_blur": "dropped/blurry",
         "dropped_duplicate": "dropped/duplicate", "unplanned": "dropped/unplanned",
         "rejected": "rejected"}
FILTERED = ("kept", "dropped_blur", "dropped_duplicate")  # the statuses the filters choose between
KEPT = ("kept", "kept_hand")                               # both count as kept (D20)


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
    'rejected' and 'unplanned' frames are left alone, and so are 'kept_hand' frames (kept by
    hand in review, D20), which still anchor the duplicate check like any kept frame. Per video:
      1. blur: the blurriest blur_percent % of its frames (rounded down) -> 'dropped_blur';
      2. then, in time order, a frame whose hash differs from the previous *kept* frame by
         fewer than duplicate_distance bits -> 'dropped_duplicate' (0 = off). Blur runs first,
         so a blurry frame never anchors a duplicate check.
    Returns {"status": {file: new status}, "distance": {file: bits differing from the
    previous kept frame}} (no distance for a video's first kept frame or a blurry one),
    "drop_at": {file: the lowest blur_percent at which that frame is dropped as blurry}}."""
    by_video: dict[str, list[dict]] = {}
    for r in rows:
        if r["status"] in FILTERED or r["status"] == "kept_hand":
            by_video.setdefault(r["video"], []).append(r)
    status, distance, drop_at = {}, {}, {}
    for video, frames in by_video.items():
        frames.sort(key=lambda r: r["timestamp_ms"])
        by_blur = sorted((r for r in frames if r["status"] != "kept_hand"), key=lambda r: (r["blur_score"] or 0.0, r["timestamp_ms"]))
        n_blur = int(len(by_blur) * blur_percent / 100 + 1e-9)
        blurry = {r["file"] for r in by_blur[:n_blur]}
        for k, r in enumerate(by_blur):  # the k+1 blurriest go once len * p / 100 >= k + 1
            drop_at[r["file"]] = 100 * (k + 1) / len(by_blur)
        anchor = None
        for r in frames:
            if r["status"] == "kept_hand":
                anchor = r                      # kept by hand: never changed, still an anchor
                continue
            if r["file"] in blurry:
                status[r["file"]] = "dropped_blur"
                continue
            if anchor is not None:
                d = distance[r["file"]] = hash_distance(r["hash"], anchor["hash"])
                if d < duplicate_distance:
                    status[r["file"]] = "dropped_duplicate"
                    continue
            status[r["file"]] = "kept"
            anchor = r
    return {"status": status, "distance": distance, "drop_at": drop_at}


def frame_rows(conn: sqlite3.Connection) -> list[dict]:
    return [dict(zip(("file", "video", "timestamp_ms", "blur_score", "hash", "status"), r))
            for r in conn.execute("SELECT file, video, timestamp_ms, blur_score, hash, status "
                                  "FROM frames")]


def preview(conn: sqlite3.Connection, blur_percent: float, duplicate_distance: int,
            planned: set[str] | None = None) -> dict:
    """What Apply would do, read only: the counts, and every frame each filter would drop, in
    one row per filter (Nick 2026-11-25). Blurry frames in the order the blur slider drops them
    (the cut is per video, so by the setting at which each one goes, then most blurry first): a
    stronger setting only adds frames on the right. Duplicates from least to most unique (fewest
    bits differing from the last kept frame first); the frame each one is compared with changes
    with the setting, so a stronger setting can also add frames mid-row."""
    rows = frame_rows(conn)
    result = filter_frames(rows, blur_percent, duplicate_distance)
    new = result["status"]
    c = {k: 0 for k in ("kept", "kept_hand", "dropped_blur", "dropped_duplicate", "rejected")}
    for r in rows:
        st = new.get(r["file"], r["status"])
        if st == "rejected" and planned is not None and r["file"] not in planned:
            continue                      # rejected outside the plan: counted apart (D20)
        if st in c:
            c[st] += 1
    c["total"] = sum(c.values())
    c["kept"] += c["kept_hand"]           # the kept count includes frames kept by hand (D20)

    def ex(r):
        return {"file": r["file"], "video": Path(r["video"]).name, "t": r["timestamp_ms"] / 1000,
                "blur": r["blur_score"], "distance": result["distance"].get(r["file"])}

    blurry = [r for r in rows if new.get(r["file"]) == "dropped_blur"]
    dups = [r for r in rows if new.get(r["file"]) == "dropped_duplicate"]
    return {
        "counts": c,
        "blur_dropped": [ex(r) for r in sorted(blurry, key=lambda r: (
            result["drop_at"][r["file"]], r["blur_score"] or 0))],
        "dup_dropped": [ex(r) for r in sorted(dups, key=lambda r: result["distance"][r["file"]])],
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


def counts(conn: sqlite3.Connection, planned: set[str] | None = None) -> dict:
    """Frames per status. 'kept' includes 'kept_hand' (also given on its own); 'total' =
    kept + dropped blurry + dropped duplicate + rejected = the plan (D20). Given the planned
    file names, rejected frames outside the plan are counted apart, as 'rejected_unplanned',
    and left out of the total."""
    out = {"kept": 0, "kept_hand": 0, "dropped_blur": 0, "dropped_duplicate": 0,
           "rejected": 0, "unplanned": 0}
    for file, st in conn.execute("SELECT file, status FROM frames"):
        if st == "rejected" and planned is not None and file not in planned:
            st = "rejected_unplanned"
        out[st] = out.get(st, 0) + 1
    out["kept"] += out["kept_hand"]
    out["total"] = (out["kept"] + out["dropped_blur"] + out["dropped_duplicate"]
                    + out["rejected"])
    return out


def last_review_status(conn: sqlite3.Connection, name: str) -> str | None:
    """The status frame review last gave this frame, or None."""
    row = conn.execute("SELECT new_status FROM review_actions WHERE frame = ? "
                       "ORDER BY id DESC LIMIT 1", (name,)).fetchone()
    return row[0] if row else None


def run(project_folder: Path, samples: list[dict],
        report: Callable[[float, str], None] = lambda p, m: None,
        blur_percent: float = 0, duplicate_distance: int = 0) -> int:
    """Extract every sample not yet on disk; returns how many were new. Each frame is written
    to a temporary name, then its database row, so a stop mid-run loses at most that frame and
    the next run carries on (resume). Frames kept from an earlier plan that this plan does not
    ask for move to dropped/unplanned/ with status 'unplanned'; a planned frame found there
    moves back. A planned frame whose file sits in its status folder (kept or dropped) is never
    decoded again. Rejected frames (bite 4) are left as they are, also outside the plan; a frame
    kept by hand that leaves the plan is set aside like a kept one, and comes back kept by hand. At the end the saved blur and
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
                if st == "unplanned":   # back in the plan; a frame kept by hand stays so (D20)
                    st = "kept_hand" if last_review_status(conn, name) == "kept_hand" else "kept"
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
            if st in (*FILTERED, "kept_hand") and name not in planned:
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
