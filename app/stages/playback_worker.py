"""Playback copy and thumbnail strip of one video as a job-module worker (S4 bite 2, D32):
`python -m app.stages.playback_worker <job.json>`. It cannot pause (one ffmpeg run); Cancel
ends it, and its output, written under temporary names, is thrown away (D15).
"""

from __future__ import annotations

from pathlib import Path

from app.jobctl import worker as jw
from app.stages import clips


def main() -> None:
    job = jw.load_job()
    jw.emit("loaded", seconds=0.0)
    clips.make_playback(job["video"], Path(job["folder"]),
                        lambda pct, msg: jw.emit("progress", percent=round(pct, 1), message=msg))
    jw.emit("done", message="ready")


if __name__ == "__main__":
    jw.run(main)
