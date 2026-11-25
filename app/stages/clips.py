"""Stage 2, pick clips. Bite 2a: a low-res H.264 playback copy per video and a thumbnail strip,
both in the project folder; the source video is only read. Spec §Stage details 2; DESIGN D12.

Measured on the SSD (DEV_LOG 2026-11-24): a 960 px copy with the Mac's H.264 encoder takes
about 0.2x (4K) to 0.5x (5.3K 60 fps) the video's length; a strip from the copy takes ~2 s.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from app.stages import ingest

COPY_WIDTH = 960      # Nick chose 960 px, 2026-11-24
COPY_BITRATE = "2M"
STRIP_EVERY_S = 5
STRIP_WIDTH = 160


def playback_path(project_folder: Path, code: str) -> Path:
    return Path(project_folder) / "playback" / f"{code}.mp4"


def strip_dir(project_folder: Path, code: str) -> Path:
    return Path(project_folder) / "strips" / code


def strip_frames(project_folder: Path, code: str) -> list[Path]:
    d = strip_dir(project_folder, code)
    return sorted(d.glob("*.jpg")) if d.is_dir() else []


def ready(project_folder: Path, code: str) -> bool:
    return playback_path(project_folder, code).is_file() and bool(strip_frames(project_folder, code))


def _run_ffmpeg(cmd: list[str], duration_s: float, report: Callable[[float, str], None],
                lo: float, hi: float, what: str) -> None:
    """Run ffmpeg with -progress on stdout; map its position onto lo..hi percent."""
    proc = subprocess.Popen(cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("out_time_us=") and duration_s:
            try:
                done = int(line.split("=", 1)[1]) / 1e6 / duration_s
            except ValueError:
                continue
            report(lo + (hi - lo) * min(done, 1.0), what)
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while {what}: {err.strip()[-300:]}")


def make_playback(video: dict, project_folder: Path,
                  report: Callable[[float, str], None] = lambda p, m: None) -> Path:
    """Make the playback copy, then the strip. `video` is a row of the project's videos table.
    Output goes to temporary names first, so a stopped job never leaves a half file."""
    src, code, duration = Path(video["path"]), video["path_code"], video["duration_s"] or 0
    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {src}. Plug in the SSD and reload.")
    out = playback_path(project_folder, code)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.is_file():
        tmp = out.with_name(f"{code}.tmp.mp4")
        lrv = ingest.lrv_path(src)
        if lrv:  # GoPro already made a low-res H.264 proxy: just repackage it, no re-encode
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(lrv), "-map", "0:v:0", "-c", "copy",
                   "-movflags", "+faststart", str(tmp)]
            what = "repackaging the GoPro .LRV file"
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-an",
                   "-vf", f"scale={COPY_WIDTH}:-2", "-c:v", "h264_videotoolbox",
                   "-b:v", COPY_BITRATE, "-movflags", "+faststart", str(tmp)]
            what = "making the playback copy"
        report(0, what)
        _run_ffmpeg(cmd, duration, report, 0, 90, what)
        os.replace(tmp, out)

    strip = strip_dir(project_folder, code)
    if not strip_frames(project_folder, code):
        tmp_dir = strip.with_name(code + ".tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        # Frame n is at n * STRIP_EVERY_S seconds (fps=1/5 samples the middle of each 5 s
        # window, so use select on time instead, which starts at 0 s).
        _run_ffmpeg(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-vf",
                     f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{STRIP_EVERY_S})',"
                     f"scale={STRIP_WIDTH}:-2", "-fps_mode", "vfr", "-q:v", "5",
                     str(tmp_dir / "%04d.jpg")],
                    duration, report, 90, 100, "making the thumbnail strip")
        shutil.rmtree(strip, ignore_errors=True)
        os.replace(tmp_dir, strip)
    return out
