"""Create, list and open projects: one folder each under <data_dir>/projects/<name>/,
holding project.json (name, settings) and state.sqlite (DESIGN, Planned for S2)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from app import db

FORMAT_VERSION = 1
NAME_RULE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
NAME_HELP = "Use 1-64 letters, digits, - or _, starting with a letter or digit."
# Starting values; the extract stage (bite 3) makes them adjustable. Spec §Stage details 3,
# DESIGN D12. blur_threshold None = start at the 10th-percentile score of the sampled frames.
DEFAULT_SETTINGS = {"seconds_between_samples": 2.0, "frame_cap": 300, "blur_threshold": None,
                    "duplicate_distance": 4, "jpeg_quality": 95}


class ProjectError(ValueError):
    """A problem the user can fix; the message says how."""


def projects_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "projects"


def write_json_atomic(path: Path, data: dict) -> None:
    """Write via a temporary file and os.replace, so a crash never leaves half a file
    (pattern reused from ADIN_SPI_OpenMV/ml/fomo/label_gui.py save_records)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def project_path(data_dir: Path, name: str) -> Path:
    """The folder for `name`, refusing any name that could point outside projects/."""
    if not NAME_RULE.match(name or ""):
        raise ProjectError(f"Bad project name {name!r}. {NAME_HELP}")
    return projects_dir(data_dir) / name


def create(data_dir: Path, name: str) -> Path:
    folder = project_path(data_dir, name)
    if folder.exists():
        raise ProjectError(f"A project named {name!r} already exists. Pick another name "
                           "or open it from the list.")
    folder.mkdir(parents=True)
    write_json_atomic(folder / "project.json", {
        "name": name, "created": datetime.now().isoformat(timespec="seconds"),
        "format_version": FORMAT_VERSION, "settings": dict(DEFAULT_SETTINGS)})
    db.connect(folder / "state.sqlite", create=True).close()
    return folder


def open_project(data_dir: Path, name: str) -> dict:
    """Read project.json; raise ProjectError with a fix if the folder is missing or damaged."""
    folder = project_path(data_dir, name)
    meta = folder / "project.json"
    if not meta.is_file():
        raise ProjectError(f"No project named {name!r} in {projects_dir(data_dir)}.")
    try:
        info = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ProjectError(f"{meta} cannot be read ({e}). Restore it from a backup, or "
                           "move the folder aside and create the project again.") from e
    info["folder"] = folder
    return info


def list_projects(data_dir: Path) -> list[str]:
    """Names of folders under projects/ that hold a project.json, sorted."""
    top = projects_dir(data_dir)
    if not top.is_dir():
        return []
    return sorted(p.name for p in top.iterdir()
                  if NAME_RULE.match(p.name) and (p / "project.json").is_file())
