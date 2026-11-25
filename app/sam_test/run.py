"""S1 SAM test: does SAM 3 / 3.1 run on this laptop, how fast, with how much memory.

    uv run sam-test --video "/path/to/video.MP4"

Pulls 3 frames from the video (read-only), then runs SAM in a separate worker process per
attempt (Apple GPU, Apple GPU with CPU fallback, CPU), stopping at the first that works.
A crash in the worker cannot take the report down with it. Each setting combination gets
<data_dir>/sam_test/<model>_tile<N>_cutoff<C>/report.html; <data_dir>/sam_test/report.html
lists all runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
import webbrowser
from pathlib import Path

from app.sam_test import report

# "urchin" instead of "sea urchin": far more urchins kept in S1 (DESIGN D10).
PROMPTS = ["urchin", "purple sea urchin", "red sea urchin", "sea star",
           "sunflower sea star", "sea anemone", "fish"]
MODELS = {"sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
          "sam3": ("facebook/sam3", "sam3.pt")}
ATTEMPTS = {"mps": ("mps", False), "mps-fallback": ("mps", True), "cpu": ("cpu", False)}


def load_data_dir(config_path: Path) -> Path:
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}\n"
                 "Fix: cp config.example.toml config.local.toml, then set data_dir in it.")
    with open(config_path, "rb") as f:
        data_dir = tomllib.load(f).get("data_dir")
    if not data_dir:
        sys.exit(f"{config_path} has no data_dir. Set it to a folder outside the repo.")
    return Path(data_dir)


def extract_frames(video: Path, out_dir: Path, count: int = 3) -> list[Path]:
    """Grab `count` frames at even fractions of the video (25/50/75 % for 3). Reads only."""
    if not video.is_file():
        sys.exit(f"Video not found: {video}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(video)], capture_output=True, text=True)
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        sys.exit(f"ffprobe could not read the duration of {video}:\n{probe.stderr}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(1, count + 1):
        t = duration * i / (count + 1)
        out = out_dir / f"frame{i}_{t:07.2f}s.png"
        if not out.exists():
            r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
                                "-frames:v", "1", "-y", str(out)], capture_output=True, text=True)
            if r.returncode != 0 or not out.exists() or out.stat().st_size < 10_000:
                sys.exit(f"ffmpeg failed to extract a frame at {t:.1f} s:\n{r.stderr}")
        frames.append(out)
    print(f"frames: {', '.join(p.name for p in frames)} in {out_dir}")
    return frames


def run_attempt(name: str, job: dict, out_dir: Path) -> dict:
    """Run one worker process; always return a result dict, even if the worker crashed."""
    device, fallback = ATTEMPTS[name]
    result_path = out_dir / f"result_{name}.json"
    result_path.unlink(missing_ok=True)
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1" if fallback else "0")
    job = dict(job, attempt=name, device=device, mps_fallback=fallback,
               result_path=str(result_path))
    print(f"\n=== attempt: {name} (device {device}, CPU fallback {'on' if fallback else 'off'})")
    proc = subprocess.run([sys.executable, "-m", "app.sam_test.worker", json.dumps(job)],
                          env=env, stderr=subprocess.PIPE, text=True)
    stderr_tail = "\n".join(proc.stderr.splitlines()[-40:])
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.setdefault("attempt", name)
    result.setdefault("device", device)
    result.setdefault("mps_fallback", fallback)
    if proc.returncode != 0 and result.get("status") != "error":
        result["status"] = "error"
        result["error"] = f"worker process exited with code {proc.returncode} (hard crash)"
    if result.get("status") != "ok":
        result["stderr_tail"] = stderr_tail
        print(f"attempt {name} FAILED: {result.get('error')}")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="S1: test SAM 3 / 3.1 on this laptop")
    ap.add_argument("--video", required=True, type=Path, help="one ReefCheck video (read-only)")
    # SAM 3 by default: same text results as 3.1, and its click model loads (DESIGN D10)
    ap.add_argument("--model", choices=list(MODELS), default="sam3")
    ap.add_argument("--attempts", default="mps,mps-fallback,cpu",
                    help="tried in order; stops at the first that works")
    ap.add_argument("--all", action="store_true", help="run every attempt, even after a success")
    ap.add_argument("--tile", type=int, default=1008,
                    help="tile size in frame pixels for the 'tiles' mode (SAM resizes each to 1008)")
    ap.add_argument("--cutoff", type=float, default=0.5,
                    help="keep objects with a score at or above this (0-1)")
    ap.add_argument("--prompts", help="comma-separated text prompts (default: the S1 list)")
    ap.add_argument("--click", help="x,y in frame-1 pixels for the click test (default: auto)")
    ap.add_argument("--config", type=Path, default=Path("config.local.toml"))
    ap.add_argument("--no-open", action="store_true", help="do not open the report")
    args = ap.parse_args(argv)

    if not 100 <= args.tile <= 4000 or not 0 < args.cutoff < 1:
        sys.exit("--tile must be 100-4000 px and --cutoff between 0 and 1")
    top = load_data_dir(args.config) / "sam_test"
    frames = extract_frames(args.video, top / "frames")
    # each setting combination gets its own folder, so runs can be compared side by side
    prompts = [p.strip().lower() for p in args.prompts.split(",") if p.strip()] if args.prompts \
        else PROMPTS
    out_dir = top / f"{args.model}_tile{args.tile}_cutoff{args.cutoff:g}" \
        f"{'' if prompts == PROMPTS else '_own-prompts'}"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_id, ckpt = MODELS[args.model]
    job = {"model": args.model, "repo_id": repo_id, "checkpoint": ckpt, "prompts": prompts,
           "frames": [str(p) for p in frames], "out_dir": str(out_dir), "click": args.click,
           "tile": args.tile, "cutoff": args.cutoff}

    results = []
    for name in [a.strip() for a in args.attempts.split(",") if a.strip()]:
        if name not in ATTEMPTS:
            sys.exit(f"Unknown attempt {name!r}; choose from {', '.join(ATTEMPTS)}")
        results.append(run_attempt(name, job, out_dir))
        if results[-1].get("status") == "ok" and not args.all:
            break
        if "not in the Hugging Face cache" in str(results[-1].get("error")):
            break  # a missing download is not a device problem; other attempts would fail too

    page = report.write_report(out_dir, args.video.name, args.model, prompts, results,
                               settings={"tile": args.tile, "cutoff": args.cutoff})
    report.write_index(top)
    print(f"\nreport: {page}\nall runs: {top / 'report.html'}")
    if not args.no_open:
        webbrowser.open(page.as_uri())


if __name__ == "__main__":
    main()
