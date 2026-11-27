"""Read config.local.toml. Never raises: every problem becomes a plain message with a fix,
which the dashboard shows in the browser (SPEC constraint 7: the terminal only starts the app)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("config.local.toml")
DEFAULT_TOOLS_DIR = Path.home() / "NereusData" / "urchin_detector_pipeline" / "tools"


@dataclass
class Config:
    path: Path
    data_dir: Path | None = None
    video_roots: list[Path] = field(default_factory=list)
    outside_data_dir: Path | None = None   # Build 2: public datasets for pre-training (S8)
    # Build 2's tool environments (D51): on the laptop's own disk, never the exFAT SSD
    tools_dir: Path = DEFAULT_TOOLS_DIR
    fit: dict = field(default_factory=dict)   # the [fit] table as written (app/fit/convert.py reads it)
    problems: list[str] = field(default_factory=list)   # the app cannot work until these are fixed
    warnings: list[str] = field(default_factory=list)   # the app works; some folders are missing

    @property
    def ok(self) -> bool:
        return not self.problems


def load(path: Path = DEFAULT_PATH) -> Config:
    path = Path(path).expanduser().absolute()
    cfg = Config(path=path)
    if not path.is_file():
        cfg.problems.append(
            f"Settings file not found: {path}. Fix: in the repo folder run "
            "`cp config.example.toml config.local.toml`, set data_dir and video_roots in it, "
            "then reload this page.")
        return cfg
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        cfg.problems.append(f"Cannot read {path}: {e}. Fix: compare it with config.example.toml.")
        return cfg

    data_dir = raw.get("data_dir")
    if not isinstance(data_dir, str) or not data_dir.strip():
        cfg.problems.append(f"{path.name} has no data_dir. Fix: set data_dir to a folder "
                            "outside the repo, for example on the SSD.")
    else:
        cfg.data_dir = Path(data_dir).expanduser()
        # Never create data_dir: if the SSD is unplugged, /Volumes/<SSD> is missing and creating
        # the folder would write to the laptop's own disk instead.
        if not cfg.data_dir.is_dir():
            cfg.problems.append(
                f"Data folder not found: {cfg.data_dir}. Fix: plug in the SSD, or create the "
                f"folder, or correct data_dir in {path.name}; then reload this page.")

    tools = raw.get("tools_dir")
    if tools is not None:
        if isinstance(tools, str) and tools.strip():
            cfg.tools_dir = Path(tools).expanduser()
        else:
            cfg.warnings.append(f"tools_dir in {path.name} must be a folder path; using {cfg.tools_dir}.")

    if isinstance(raw.get("fit"), dict):
        cfg.fit = raw["fit"]

    roots = raw.get("video_roots", [])
    if isinstance(roots, str):
        roots = [roots]
    if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
        cfg.problems.append(f"video_roots in {path.name} must be a list of folder paths, "
                            'for example ["/Volumes/SSD/videos"].')
        roots = []
    cfg.video_roots = [Path(r).expanduser() for r in roots]
    outside = raw.get("outside_data_dir")
    if isinstance(outside, str) and outside.strip():
        cfg.outside_data_dir = Path(outside).expanduser()
        if not cfg.outside_data_dir.is_dir():
            cfg.warnings.append(f"Outside data folder not found: {cfg.outside_data_dir}. "
                                f"Correct outside_data_dir in {path.name}, or remove the line.")
    elif outside is not None:
        cfg.warnings.append(f"outside_data_dir in {path.name} must be a folder path; ignored.")
    for r in cfg.video_roots:
        if not r.is_dir():
            cfg.warnings.append(f"Video folder not found: {r}. Plug in the SSD or correct "
                                f"video_roots in {path.name}.")
    return cfg
