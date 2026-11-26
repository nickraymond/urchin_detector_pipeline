"""Stage 2, pick clips. Bite 2a: a low-res H.264 playback copy per video and a thumbnail strip,
both in the project folder; the source video is only read. Spec §Stage details 2; DESIGN D12.

Measured on the SSD (DEV_LOG 2026-11-24): a 960 px copy with the Mac's H.264 encoder takes
about 0.2x (4K) to 0.5x (5.3K 60 fps) the video's length; a strip from the copy takes ~2 s.
"""

from __future__ import annotations

import json
import math
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
    """The strip's NNNN.jpg files only: on the SSD macOS also writes a `._NNNN.jpg` shadow
    file next to each one, which is not an image (seen in Nick's test, 2026-11-24)."""
    d = strip_dir(project_folder, code)
    return sorted(d.glob("[0-9][0-9][0-9][0-9].jpg")) if d.is_dir() else []


def ready(project_folder: Path, code: str) -> bool:
    return playback_path(project_folder, code).is_file() and bool(strip_frames(project_folder, code))


def strip_filter() -> str:
    """Thumbnail n is the first frame at or after n * STRIP_EVERY_S seconds (absolute time, so
    it never drifts from its label; at 29.97 fps it is at most one frame, ~33 ms, late)."""
    return f"select='gte(t\\,selected_n*{STRIP_EVERY_S})',scale={STRIP_WIDTH}:-2"


def codec_of(path: Path) -> str:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


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
        if lrv and codec_of(lrv) == "h264":
            # GoPro already made a low-res H.264 proxy: just repackage it, no re-encode
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(lrv), "-map", "0:v:0", "-c", "copy",
                   "-movflags", "+faststart", str(tmp)]
            what = "repackaging the GoPro .LRV file"
        else:
            # A proxy in another codec is still smaller to decode than the original.
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(lrv or src), "-map", "0:v:0", "-an",
                   "-vf", f"scale='min({COPY_WIDTH},iw)':-2", "-c:v", "h264_videotoolbox",
                   "-b:v", COPY_BITRATE, "-movflags", "+faststart", str(tmp)]
            what = "making the playback copy" + (" from the GoPro .LRV file" if lrv else "")
        report(0, what)
        _run_ffmpeg(cmd, duration, report, 0, 90, what)
        os.replace(tmp, out)

    strip = strip_dir(project_folder, code)
    if not strip_frames(project_folder, code):
        tmp_dir = strip.with_name(code + ".tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        _run_ffmpeg(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-vf", strip_filter(),
                     "-fps_mode", "vfr", "-q:v", "5",
                     str(tmp_dir / "%04d.jpg")],
                    duration, report, 90, 100, "making the thumbnail strip")
        shutil.rmtree(strip, ignore_errors=True)
        os.replace(tmp_dir, strip)
    return out


def frame_at(project_folder: Path, code: str, t: float, duration_s: float) -> Path | None:
    """A 160 px picture of the playback copy at time t (to 0.1 s), for a clip's row: made once
    with ffmpeg into <project>/clipthumbs/<code>/, then reused (a cache, D14a)."""
    copy = playback_path(project_folder, code)
    if not copy.is_file() or not math.isfinite(t) or not 0 <= t <= duration_s + 0.05:
        return None
    t = min(round(t, 1), max(0.0, duration_s - 0.05))
    out = Path(project_folder) / "clipthumbs" / code / f"{round(t * 1000):08d}.jpg"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.jpg")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(copy),
                        "-frames:v", "1", "-vf", f"scale={STRIP_WIDTH}:-2", "-q:v", "5", str(tmp)],
                       capture_output=True, timeout=60)
    if r.returncode != 0 or not tmp.is_file():
        return None
    os.replace(tmp, out)
    return out


# --- clips.json: the stretches of each video to extract from (bite 2b) ---

MIN_CLIP_S = 0.5
MAX_NOTE = 200


class ClipError(ValueError):
    """A clip the user must correct; the message says what is wrong."""


def clips_path(project_folder: Path) -> Path:
    return Path(project_folder) / "clips.json"


def load_clips(project_folder: Path) -> list[dict]:
    """All clips of the project: [{video, start_s, end_s, note}], or [] if none saved yet."""
    path = clips_path(project_folder)
    if not path.is_file():
        return []
    fix = "Restore it from a backup, or move it aside to start the clip list again."
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ClipError(f"{path} cannot be read ({e}). {fix}") from e
    items = data.get("clips") if isinstance(data, dict) else None
    ok = isinstance(items, list) and all(
        isinstance(c, dict) and isinstance(c.get("video"), str)
        and all(isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool)
                and math.isfinite(c[k]) for k in ("start_s", "end_s"))
        for c in items)
    if not ok:
        raise ClipError(f"{path} cannot be read (not a list of clips with video, start_s and "
                        f"end_s). {fix}")
    return items


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def check_clips(video_clips: list[dict], duration_s: float) -> list[dict]:
    """Clean and check one video's clips; sorted by start. Raises ClipError on any problem."""
    out = []
    for c in video_clips:
        try:
            start, end = round(float(c["start_s"]), 2), round(float(c["end_s"]), 2)
        except (KeyError, TypeError, ValueError) as e:
            raise ClipError(f"A clip needs a start and an end time in seconds ({e}).") from e
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ClipError("A clip's start and end must be real numbers of seconds.")
        note = str(c.get("note") or "").strip()[:MAX_NOTE]
        if start < 0 or end > duration_s + 0.05:
            raise ClipError(f"Clip {_fmt(start)}–{_fmt(end)} is outside the video "
                            f"(0:00–{_fmt(duration_s)}).")
        # Messages say what to do next (Nick's test, 2026-11-24: "end must come after start"
        # was not clear when start and end were the same moment).
        if end == start:
            raise ClipError(f"This clip would be 0 s long: start and end are both {_fmt(start)}. "
                            "Play or step the video forward, then press O (or Set end here).")
        if end < start:
            raise ClipError(f"This clip would end ({_fmt(end)}) before it starts "
                            f"({_fmt(start)}). Set the end to a later time, or the start to an "
                            "earlier one.")
        if end - start < MIN_CLIP_S:
            raise ClipError(f"This clip is only {end - start:.1f} s long; a clip must be at "
                            f"least {MIN_CLIP_S} s. Move the video further before pressing O.")
        out.append({"start_s": start, "end_s": min(end, round(duration_s, 2)), "note": note})
    out.sort(key=lambda c: c["start_s"])
    for a, b in zip(out, out[1:]):
        if b["start_s"] < a["end_s"]:
            raise ClipError(f"Clips can't share time: {_fmt(b['start_s'])}–{_fmt(b['end_s'])} "
                            f"overlaps the clip {_fmt(a['start_s'])}–{_fmt(a['end_s'])}. Mark "
                            "outside it, or change that clip with Set start here / Set end "
                            "here, or delete it.")
    return out


def save_video_clips(project_folder: Path, video_path: str, duration_s: float,
                     video_clips: list[dict]) -> list[dict]:
    """Replace one video's clips in clips.json (atomic write); return them, checked."""
    checked = check_clips(video_clips, duration_s)
    others = [c for c in load_clips(project_folder) if c["video"] != video_path]
    data = {"clips": others + [{"video": video_path, **c} for c in checked]}
    path = clips_path(project_folder)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)
    return checked


def frames_in(seconds: float, every_s: float) -> int:
    """Samples at 0, N, 2N ... inside a stretch: how many extraction will take (before the cap
    and the blur / duplicate filters)."""
    return math.ceil(round(seconds / every_s, 6)) if seconds > 0 else 0


def summary(videos: list[dict], all_clips: list[dict], every_s: float) -> dict:
    """Per video code: clips, marked seconds, frames; plus totals. A video with no clips counts
    as the whole video (spec §Stage details 3)."""
    per = {}
    for v in videos:
        mine = [c for c in all_clips if c["video"] == v["path"]]
        marked = sum(c["end_s"] - c["start_s"] for c in mine)
        frames = (sum(frames_in(c["end_s"] - c["start_s"], every_s) for c in mine) if mine
                  else frames_in(v["duration_s"] or 0, every_s))
        per[v["path_code"]] = {"clips": len(mine), "marked_s": round(marked, 1), "frames": frames}
    return {"videos": per, "clips": sum(p["clips"] for p in per.values()),
            "marked_s": round(sum(p["marked_s"] for p in per.values()), 1),
            "frames": sum(p["frames"] for p in per.values())}


def playback_params(project_folder: Path, video: dict) -> dict:
    """Params of a "playback" job: the video's row (path, code, duration)."""
    return {"folder": str(project_folder), "code": video["path_code"],
            "video": {"path": video["path"], "path_code": video["path_code"],
                      "duration_s": video["duration_s"]}}


def playback_kind():
    from app.jobctl import Kind
    return Kind("playback", module="app.stages.playback_worker", pausable=False)
