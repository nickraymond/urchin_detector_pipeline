"""Stage 1, ingest: find videos in a folder (and its subfolders), read their details with
ffprobe, check one frame decodes, group GoPro chapters, and add picked videos to a project.
Spec §Stage details 1. Source videos are only ever read (SPEC constraint 4).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app import db

VIDEO_EXT = {".mp4", ".mov"}
SIDECAR_EXT = {".lrv", ".thm"}  # GoPro low-res proxy and thumbnail: never listed as videos
# GoPro names: G + H/X/L (encoding) + 2-digit chapter + 4-digit recording number, e.g. GX010123.
GOPRO = re.compile(r"^G([HXL])(\d{2})(\d{4})$", re.IGNORECASE)
TIMEOUT_S = 60
_probe_cache: dict[tuple, dict] = {}  # (path, size, mtime) -> probe result; saves re-reading


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


def find_videos(folder: Path) -> list[Path]:
    """Every .MP4 / .MOV under `folder`, skipping macOS `._` files, sidecars and hidden folders."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            p = Path(root) / f
            if not f.startswith(".") and p.suffix.lower() in VIDEO_EXT:
                found.append(p)
    return found


def has_lrv(path: Path) -> bool:
    """GoPro writes GX010123.MP4 next to GL010123.LRV; some cameras keep the same name."""
    stems = {path.stem}
    if GOPRO.match(path.stem):
        stems.add("GL" + path.stem[2:])
    return any((path.parent / (s + ext)).exists() for s in stems for ext in (".LRV", ".lrv"))


def probe(path: Path) -> dict:
    """Duration, size, fps and codec from ffprobe, then decode one frame with ffmpeg.
    Never raises: a problem becomes a `warning` text."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _probe_cache:
        return _probe_cache[key]
    out: dict = {"warning": ""}
    r = None
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
        d = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=TIMEOUT_S)
        if d.returncode != 0:
            out["warning"] = ("ffmpeg cannot decode a frame of this video: "
                              f"{(d.stderr.strip().splitlines() or ['no message'])[-1]}")
    except (subprocess.TimeoutExpired, OSError, ValueError, KeyError) as e:
        msg = (r.stderr.strip() if r is not None and r.stderr else "") or str(e)
        out["warning"] = f"ffprobe cannot read this video: {msg[:200]}"
    _probe_cache[key] = out
    return out


def scan(folder: Path) -> list[Video]:
    """Videos under `folder`, one row per distinct video, in recording and chapter order.
    Same file name and same size = the same video (DESIGN D13); the copies go in `also_in`."""
    by_key: dict[tuple[str, int], Video] = {}
    for p in find_videos(folder):
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


def update_status(conn: sqlite3.Connection) -> None:
    n, recs = conn.execute("SELECT COUNT(*), COUNT(DISTINCT recording) FROM videos").fetchone()
    if n:
        db.set_stage(conn, "ingest", "done", f"{n} video{'s' * (n != 1)} "
                     f"({recs} recording{'s' * (recs != 1)})")
    else:
        db.set_stage(conn, "ingest", "not started")


def add(conn: sqlite3.Connection, videos: list[Video]) -> int:
    """Add videos to the project (only the database; the files are not touched)."""
    rows = [(str(v.path), v.code, v.recording, v.chapter, v.duration_s, v.size, v.width,
             v.height, v.fps, v.codec, v.warning) for v in videos if not v.warning]
    conn.executemany("INSERT OR IGNORE INTO videos (path, path_code, recording, chapter, "
                     "duration_s, size_bytes, width, height, fps, codec, warning) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    update_status(conn)
    return len(rows)


def remove(conn: sqlite3.Connection, path: str) -> None:
    """Take a video out of the project. The video file itself is never touched."""
    conn.execute("DELETE FROM videos WHERE path = ?", (path,))
    update_status(conn)
