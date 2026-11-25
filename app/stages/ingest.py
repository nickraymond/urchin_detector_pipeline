"""Stage 1, ingest: find videos in a folder (and its subfolders), read their details with
ffprobe, check one frame decodes, group GoPro chapters, and add picked videos to a project.
Bite 5: folders of still images (JPEG / PNG) too; they skip clips and extraction and join at
frame review. Spec §Stage details 1. Sources are only ever read (SPEC constraint 4).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app import db
from app.stages.extract import blur_score, dhash

VIDEO_EXT = {".mp4", ".mov"}
STILL_EXT = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png"}  # still image -> its copy's suffix
MAX_STILLS = 2000  # per add: copying is done at once, while the page waits
SIDECAR_EXT = {".lrv", ".thm"}  # GoPro low-res proxy and thumbnail: never listed as videos
# GoPro names: G + H/X/L (encoding) + 2-digit chapter + 4-digit recording number, e.g. GX010123.
GOPRO = re.compile(r"^G([HXL])(\d{2})(\d{4})$", re.IGNORECASE)
TIMEOUT_S = 60
MAX_VIDEOS = 200  # per page load; a bigger folder would hang the page (bite 3 adds background jobs)
_probe_cache: dict[tuple, dict] = {}  # (path, size, mtime) -> probe result; saves re-reading


class IngestError(ValueError):
    """A problem the user can fix; the message says how."""


@dataclass
class Video:
    path: Path
    size: int
    recording: str = ""
    chapter: int = 1
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    codec: str = ""
    has_lrv: bool = False
    warning: str = ""
    also_in: list[str] = field(default_factory=list)  # folders holding a same-name, same-size copy

    @property
    def code(self) -> str:
        return path_code(self.path)


def path_code(path: Path) -> str:
    """6 hex characters from the full path; the same hash the S1 test uses (DESIGN)."""
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()[:6]


def recording_of(path: Path) -> tuple[str, int]:
    """GoPro chapters of one recording share a key; any other name is its own recording."""
    m = GOPRO.match(path.stem)
    if m:
        return f"G{m.group(1).upper()}{m.group(3)}", int(m.group(2))
    return path.stem, 1


def find_files(folder: Path, suffixes) -> list[Path]:
    """Files with these suffixes under `folder`, skipping macOS `._` files and hidden folders."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            p = Path(root) / f
            if not f.startswith(".") and p.suffix.lower() in suffixes:
                found.append(p)
    return found


def find_videos(folder: Path) -> list[Path]:
    """Every .MP4 / .MOV under `folder` (sidecars have other suffixes, so never listed)."""
    return find_files(folder, VIDEO_EXT)


def find_stills(folder: Path) -> list[Path]:
    """Every .JPG / .JPEG / .PNG under `folder`."""
    return find_files(folder, STILL_EXT)


def lrv_path(path: Path) -> Path | None:
    """GoPro writes GX010123.MP4 next to GL010123.LRV; some cameras keep the same name."""
    stems = [path.stem]
    if GOPRO.match(path.stem):
        stems.append("GL" + path.stem[2:])
    for s in stems:
        for ext in (".LRV", ".lrv"):
            if (path.parent / (s + ext)).is_file():
                return path.parent / (s + ext)
    return None


def has_lrv(path: Path) -> bool:
    return lrv_path(path) is not None


def probe(path: Path) -> dict:
    """Duration, size, fps and codec from ffprobe, then decode one frame with ffmpeg.
    Never raises: a problem becomes a `warning` text."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _probe_cache:
        return _probe_cache[key]
    out: dict = {"warning": ""}
    r, tool = None, "ffprobe"
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-print_format", "json",
             "-show_entries", "stream=codec_name,width,height,avg_frame_rate:format=duration",
             str(path)], capture_output=True, text=True, timeout=TIMEOUT_S)
        info = json.loads(r.stdout or "{}")
        stream = (info.get("streams") or [{}])[0]
        num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
        out.update(codec=stream.get("codec_name", ""), width=stream.get("width"),
                   height=stream.get("height"),
                   fps=round(float(num) / float(den), 3) if float(den or 0) else None,
                   duration_s=float(info["format"]["duration"]))
        tool = "ffmpeg"
        d = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=TIMEOUT_S)
        if d.returncode != 0:
            out["warning"] = ("ffmpeg cannot decode a frame of this video: "
                              f"{(d.stderr.strip().splitlines() or ['no message'])[-1]}")
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError) as e:
        msg = (r.stderr.strip() if r is not None and r.stderr else "") or str(e)
        out["warning"] = f"{tool} cannot read this video: {msg[:200]}"
    _probe_cache[key] = out
    return out


THUMB_CODE = re.compile(r"^[0-9a-f]{6}$")


def thumbnail(v: Video, thumbs_dir: Path) -> Path | None:
    """A 240 px wide JPEG of one frame 10 % into the video, like Finder shows. Written to
    thumbs_dir (in data_dir, never beside the video); made once, then reused until the video
    at that path changes (size or modified time, kept in <code>.key)."""
    out = Path(thumbs_dir) / f"{v.code}.jpg"
    key_file = out.with_suffix(".key")
    try:
        key = f"{v.size} {v.path.stat().st_mtime_ns}"
    except OSError:
        return out if out.is_file() else None
    if out.is_file() and key_file.is_file() and key_file.read_text() == key:
        return out
    if v.warning or not v.duration_s:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.jpg")
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{v.duration_s * 0.1:.2f}",
                            "-i", str(v.path), "-frames:v", "1", "-vf", "scale=240:-2",
                            "-q:v", "4", "-y", str(tmp)], capture_output=True, timeout=TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or not tmp.is_file():
        return None
    os.replace(tmp, out)
    key_file.write_text(key)
    return out


def scan(folder: Path) -> list[Video]:
    """Videos under `folder`, one row per distinct video, in recording and chapter order.
    Same file name and same size = the same video (DESIGN D13); the copies go in `also_in`."""
    found = find_videos(folder)
    if len(found) > MAX_VIDEOS:
        raise IngestError(f"{folder} holds {len(found)} videos, more than {MAX_VIDEOS} at once. "
                          "Open one of its subfolders instead.")
    by_key: dict[tuple[str, int], Video] = {}
    for p in found:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        key = (p.name, size)
        if key in by_key:
            by_key[key].also_in.append(str(p.parent))
            continue
        rec, chapter = recording_of(p)
        by_key[key] = Video(path=p, size=size, recording=str(p.parent / rec), chapter=chapter,
                            has_lrv=has_lrv(p), **probe(p))
    return sorted(by_key.values(), key=lambda v: (v.recording, v.chapter))


def added(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM videos ORDER BY recording, chapter")]


def added_keys(conn: sqlite3.Connection) -> dict[tuple[str, int], str]:
    """(file name, size) -> path of every video in the project: the D13 identity."""
    return {(Path(r["path"]).name, r["size_bytes"]): r["path"]
            for r in conn.execute("SELECT path, size_bytes FROM videos")}


def update_status(conn: sqlite3.Connection) -> None:
    n, recs = conn.execute("SELECT COUNT(*), COUNT(DISTINCT recording) FROM videos").fetchone()
    stills = conn.execute("SELECT COUNT(*) FROM stills").fetchone()[0]
    parts = ([f"{n} video{'s' * (n != 1)} ({recs} recording{'s' * (recs != 1)})"] if n else []) \
        + ([f"{stills} still image{'s' * (stills != 1)}"] if stills else [])
    if parts:
        db.set_stage(conn, "ingest", "done", ", ".join(parts))
    else:
        db.set_stage(conn, "ingest", "not started")


def add(conn: sqlite3.Connection, videos: list[Video]) -> int:
    """Add videos to the project (only the database; the files are not touched). A video
    whose file name and size are already in the project, from any folder, is skipped (D13):
    one dive must never end up in both the training and the test split."""
    keys = added_keys(conn)
    rows = []
    for v in videos:
        if v.warning or (v.path.name, v.size) in keys:
            continue
        keys[(v.path.name, v.size)] = str(v.path)
        rows.append((str(v.path), v.code, v.recording, v.chapter, v.duration_s, v.size, v.width,
                     v.height, v.fps, v.codec, v.warning))
    conn.executemany("INSERT OR IGNORE INTO videos (path, path_code, recording, chapter, "
                     "duration_s, size_bytes, width, height, fps, codec, warning) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    update_status(conn)
    return len(rows)


def remove(conn: sqlite3.Connection, path: str) -> None:
    """Take a video out of the project. The video file itself is never touched."""
    conn.execute("DELETE FROM videos WHERE path = ?", (path,))
    update_status(conn)


# --- Still images (bite 5) ---

def still_name(path: Path) -> str:
    """The copy's name in frames/: <stem>_<path-code>_0<.jpg | .png>, the frame naming rule with
    timestamp 0, so two stills of the same name in different folders never collide."""
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(path).stem)
    return f"{stem}_{path_code(path)}_0{STILL_EXT[Path(path).suffix.lower()]}"


def added_stills(conn: sqlite3.Connection) -> dict[tuple[str, int], str]:
    """(file name, size) -> source path of every still in the project (the D13 identity)."""
    return {(r[0], r[1]): r[2] for r in conn.execute("SELECT name, size_bytes, path FROM stills")}


def add_stills(conn: sqlite3.Connection, folder: Path, frames_dir: Path) -> dict:
    """Copy every still image under `folder` into frames/ (byte for byte; the source is only
    read) and add it as a kept frame with no timestamp. A still whose name and size are already
    in the project is skipped; one that Pillow cannot read is skipped and counted. File first,
    then its rows, so a stop in between leaves at most a copy that the next add replaces.
    Returns {added, skipped, unreadable}."""
    found = find_stills(folder)
    if len(found) > MAX_STILLS:
        raise IngestError(f"{folder} holds {len(found)} still images, more than {MAX_STILLS} "
                          "at once. Open one of its subfolders instead.")
    have = added_stills(conn)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out = {"added": 0, "skipped": 0, "unreadable": 0}
    for p in found:
        size = p.stat().st_size
        if (p.name, size) in have:
            out["skipped"] += 1
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:  # Pillow raises many types for a damaged file
            out["unreadable"] += 1
            continue
        name = still_name(p)
        tmp = frames_dir / (name + ".tmp")
        shutil.copyfile(p, tmp)
        os.replace(tmp, frames_dir / name)
        with Image.open(frames_dir / name) as im:
            blur, h = blur_score(im.convert("RGB")), dhash(im)
        conn.execute("INSERT OR REPLACE INTO stills (path, name, size_bytes, file) "
                     "VALUES (?, ?, ?, ?)", (str(p), p.name, size, name))
        conn.execute("INSERT OR REPLACE INTO frames (file, video, timestamp_ms, blur_score, hash, "
                     "status) VALUES (?, ?, NULL, ?, ?, 'kept')", (name, str(p), blur, h))
        conn.commit()
        have[(p.name, size)] = str(p)
        out["added"] += 1
    update_status(conn)
    return out
