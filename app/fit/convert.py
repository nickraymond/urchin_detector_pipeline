"""The `convert` job kind (S6 bite 3a; spec stage 11): one model × one camera → 8-bit model → the camera's
compiler → `fit.json`. This file is the app's side: where things live (D53), the job's params, progress
from the files on disk (D33), and reading a fit report back. No torch here.

    <folder>/recipe.json
    <folder>/int8/model_int8.tflite, int8.json, convert.log      (failed.json when the 8-bit step failed)
    <folder>/targets/<camera>/fit.json, compile.log, the compiler's files

<folder> is <data_dir>/fit/<arch key>/ for an untrained candidate.

A model that does not convert is a result: the job ends "done" and fit.json says `converts: false` with
the stage, the reason and the failing layers. A job error means the tools themselves could not run.
"""

from __future__ import annotations

import json
from pathlib import Path

from app import modeldefs
from app.jobctl import JobError, Kind

CAMERAS = {"n6": "OpenMV N6", "pi": "Pi Zero 2W", "ae3": "OpenMV AE3"}
# camera -> the module with compile(tflite, out_dir, job) -> dict (the Pi: bite 3c)
COMPILERS: dict[str, str] = {"n6": "app.fit.n6", "ae3": "app.fit.ae3"}
# [fit] in config.local.toml (D52): where Nick's earlier work left the N6 toolchain
FIT_DEFAULTS = {"openmv_dir": "~/openmv-dev/openmv", "sdk_dir": "~/openmv-sdk-1.6.0-linux-x86_64",
                "docker_image": "firmware-builder:latest"}


def fit_settings(raw: dict | None = None) -> dict:
    """The [fit] table with defaults filled in and ~ expanded; unknown keys are ignored."""
    raw = raw if isinstance(raw, dict) else {}
    out = {k: str(raw.get(k) or v) for k, v in FIT_DEFAULTS.items()}
    return {k: str(Path(v).expanduser()) if k.endswith("_dir") else v for k, v in out.items()}
STEPS = ("8-bit model", "compile")


def model_dir(data_dir: Path, arch: dict) -> Path:
    return Path(data_dir) / "fit" / modeldefs.arch_key(arch)


def int8_path(folder: Path) -> Path:
    return Path(folder) / "int8" / "model_int8.tflite"


def int8_failed_path(folder: Path) -> Path:
    return Path(folder) / "int8" / "failed.json"


def fit_path(folder: Path, camera: str) -> Path:
    return Path(folder) / "targets" / camera / "fit.json"


def read_fit(folder: Path, camera: str) -> dict | None:
    """The fit report, or None when this cell has not been tried (or its file cannot be read)."""
    try:
        return json.loads(fit_path(folder, camera).read_text())
    except (OSError, ValueError):
        return None


def params(arch: dict, camera: str, data_dir: Path, tools_dir: Path, redo: bool = False,
           fit: dict | None = None) -> dict:
    """The params a page submits for one cell."""
    return {"arch": arch, "camera": camera, "folder": str(model_dir(data_dir, arch)),
            "tools_dir": str(tools_dir), "redo": redo, "fit": fit_settings(fit)}


def build_job(p: dict) -> dict:
    """params -> the worker's job file; refuses what cannot run, in plain words."""
    problems = modeldefs.check(p["arch"])
    if problems:
        raise JobError("Cannot convert this model: " + "; ".join(problems) + ".")
    compiler = p.get("compiler") or COMPILERS.get(p["camera"])
    if not compiler:
        raise JobError(f"No compiler is set up in the app for camera {p['camera']!r} "
                       f"(known: {', '.join(COMPILERS) or 'none yet'}).")
    settings = modeldefs.export_settings(p["arch"])
    tool_python = p.get("tool_python") or str(Path(p["tools_dir"]) / settings["env"] / "bin" / "python")
    return {**p, "fit": fit_settings(p.get("fit")), "compiler": compiler, "tool_python": tool_python,
            "int8_module": p.get("int8_module") or "app.fit.to_int8",
            "arch_key": modeldefs.arch_key(p["arch"])}


def progress(p: dict) -> tuple[int, int]:
    folder = Path(p["folder"])
    if p.get("redo"):
        return 0, len(STEPS)
    if fit_path(folder, p["camera"]).is_file():
        return len(STEPS), len(STEPS)
    return int(int8_path(folder).is_file() or int8_failed_path(folder).is_file()), len(STEPS)


def convert_kind() -> Kind:
    return Kind("convert", module="app.fit.convert_worker", prepare=build_job, progress=progress, item="step")
