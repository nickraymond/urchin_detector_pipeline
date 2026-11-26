"""Resource limits for jobs (D36): one settings file for the laptop, <data_dir>/jobs/settings.json,
edited on the Jobs page and copied into each job when it starts (the job's row keeps them).

- device: "auto" (NVIDIA GPU, else Apple GPU, else CPU; the kind resolves it), "mps", "cuda", "cpu"
- threads: CPU threads for the worker (OMP_NUM_THREADS, MKL_NUM_THREADS; SAM also calls
  torch.set_num_threads); default half of the cores
- low_priority: start the worker with `nice -n 10`
- pause_s: seconds the worker rests between items (still listening for "pause")
- memory_guard / memory_guard_gb: while a job runs, the runner reads free memory every 2 s and
  pauses the job below the threshold
- batch: 1, shown and fixed

Free memory is psutil.virtual_memory().available: on macOS that is free + inactive pages
(psutil 7.2.2 source, _psosx.py), i.e. what can be handed to a process without swapping out
someone's active memory. psutil is a direct dependency (BSD-3-Clause, its PyPI metadata).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEVICES = ("auto", "mps", "cuda", "cpu")
GUARD_EVERY_S = 2.0


def default_threads() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def defaults() -> dict:
    return {"device": "auto", "threads": default_threads(), "low_priority": True,
            "pause_s": 0.0, "memory_guard": True, "memory_guard_gb": MEMORY_GUARD_GB,
            "batch": 1}


# Proposed from S4 bite 4's measurement (DESIGN, Bench results): while SAM runs, the figure sits
# at 5.5-8 GB with short dips to 3.5 GB (tiled 4K) and 2.8 GB (whole-frame 5.3K); 2 GB pauses
# neither run. One run of 3 frames per mode: re-check on longer runs.
MEMORY_GUARD_GB = 2.0


class LimitsError(ValueError):
    """A value on the settings form that cannot be used; the message says which and why."""


def path(root: Path) -> Path:
    return Path(root) / "settings.json"


def load(root: Path) -> dict:
    """The saved limits merged over the defaults; a missing or unreadable file gives the
    defaults (the page says so when it cannot read it)."""
    try:
        saved = json.loads(path(root).read_text())
    except (OSError, ValueError):
        saved = {}
    out = defaults()
    try:
        out.update(check({**out, **{k: v for k, v in saved.items() if k in out}}))
    except LimitsError:
        pass
    return out


def check(values: dict) -> dict:
    """Checked and typed limits; raises LimitsError naming the first bad value."""
    out = {}
    if values.get("device") not in DEVICES:
        raise LimitsError(f"Device must be one of {', '.join(DEVICES)}.")
    out["device"] = values["device"]
    try:
        threads = int(values.get("threads"))
    except (TypeError, ValueError):
        raise LimitsError("CPU threads must be a whole number.") from None
    cores = os.cpu_count() or 1
    if not 1 <= threads <= cores:
        raise LimitsError(f"CPU threads must be from 1 to {cores} (this computer's cores).")
    out["threads"] = threads
    out["low_priority"] = _bool(values.get("low_priority"))
    out["memory_guard"] = _bool(values.get("memory_guard"))
    try:
        pause_s = float(values.get("pause_s"))
        guard = float(values.get("memory_guard_gb"))
    except (TypeError, ValueError):
        raise LimitsError("Pause and the memory threshold must be numbers.") from None
    if not 0 <= pause_s <= 600:
        raise LimitsError("Pause between frames must be from 0 to 600 s.")
    if not 0.5 <= guard <= 256:
        raise LimitsError("The memory threshold must be from 0.5 to 256 GB.")
    out["pause_s"], out["memory_guard_gb"] = pause_s, guard
    out["batch"] = 1
    return out


def save(root: Path, values: dict) -> dict:
    checked = check(values)
    p = path(root)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(checked, indent=1))
    os.replace(tmp, p)
    return checked


def free_gb() -> float:
    """The memory guard's one figure: psutil's available memory, in GB (10^9 bytes)."""
    import psutil
    return psutil.virtual_memory().available / 1e9


def _bool(v) -> bool:
    return v in (True, "1", "on", "true", "yes", 1)
