"""Model definitions for Build 2 (DESIGN D51, D54): what a candidate is and how to build it untrained.

Plain data plus PyTorch. Nothing here imports the rest of `app`, so the same code runs in the app's
environment (training, S8) and in the conversion environment under `tools_dir` (a test checks). This
file itself imports no torch: the "What runs where" page reads CANDIDATES without loading a model.

A candidate is the `arch` part of a recipe (D54):

    {"kind": "detector", "family": "yolox", "size": "nano", "variant": {"stem": "conv"},
     "input": [256, 256], "classes": [...], "outputs": {...}, "code": {"repo", "commit"},
     "licence": "Apache-2.0", "licence_mark": None}

    arch_key(arch)  -> "yolox-nano-conv-256-c5"   (the folder name under <data_dir>/fit/, D53)
    build(arch)     -> the model, untrained, as the module that is exported (raw output maps)

Third-party model code is an unmodified git checkout at a pinned commit under
`<tools_dir>/third_party/<name>` (D51); `third_party(name)` puts it on sys.path and says how to get it
when it is missing.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

KINDS = ("detector", "segmenter", "instance")
DEFAULT_TOOLS_DIR = Path.home() / "NereusData" / "urchin_detector_pipeline" / "tools"

# Build 2 trains on sea stars and urchins only (Nick, 2026-11-27; names as in default_classes.yaml).
BUILD2_CLASSES = ["purple urchin", "red urchin", "black urchin", "sea star", "sunflower sea star"]

# family -> the module in this package that builds it
FAMILIES = {"yolox": "app.modeldefs.yolox"}


class ModelDefError(ValueError):
    """A recipe that cannot be built, or model code that is not on the laptop; the message says why."""


def tools_dir() -> Path:
    """The convert job passes tools_dir from config.local.toml in URCHIN_TOOLS_DIR (D51)."""
    return Path(os.environ.get("URCHIN_TOOLS_DIR") or DEFAULT_TOOLS_DIR).expanduser()


def third_party(name: str, commit: str, repo: str) -> Path:
    """Put <tools_dir>/third_party/<name> on sys.path. The checkout is not verified against
    `commit` here (no git call in a model build); the tools set-up step pins it."""
    path = tools_dir() / "third_party" / name
    if not path.is_dir():
        raise ModelDefError(
            f"Model code not found: {path}. Fix: set the tools up (it checks out {repo} at {commit[:7]}).")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def arch_key(arch: dict) -> str:
    """<family>-<size>[-<variant values>]-<input px>-c<classes>; square inputs show one number."""
    h, w = arch["input"]
    px = str(h) if h == w else f"{h}x{w}"
    variant = [str(v) for _, v in sorted((arch.get("variant") or {}).items())]
    key = "-".join([arch["family"], arch["size"], *variant, px, f"c{len(arch['classes'])}"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9x-]*", key):
        raise ModelDefError(f"Recipe gives a folder name the app will not use: {key!r}.")
    return key


def check(arch: dict) -> list[str]:
    """Problems with a recipe's arch part, in plain words; empty when it can be built."""
    problems = []
    for field in ("kind", "family", "size", "input", "classes", "code", "licence"):
        if field not in arch:
            problems.append(f"the recipe has no {field}")
    if problems:
        return problems
    if arch["kind"] not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}, not {arch['kind']!r}")
    if arch["family"] not in FAMILIES:
        problems.append(f"model family {arch['family']!r} is not in the app (known: {', '.join(FAMILIES)})")
    inp = arch["input"]
    if not (isinstance(inp, list) and len(inp) == 2 and all(isinstance(v, int) and v > 0 for v in inp)):
        problems.append("input must be [height, width] in pixels")
    if not arch["classes"]:
        problems.append("the recipe lists no classes")
    if not (arch["code"].get("repo") and arch["code"].get("commit")):
        problems.append("the recipe does not say which code (repo and commit) it is built from")
    if not problems:
        problems += importlib.import_module(FAMILIES[arch["family"]]).check(arch)
    return problems


def build(arch: dict):
    """The untrained model as it is exported: eval mode, raw outputs as named in arch["outputs"]."""
    problems = check(arch)
    if problems:
        raise ModelDefError("Cannot build this model: " + "; ".join(problems) + ".")
    return importlib.import_module(FAMILIES[arch["family"]]).build(arch)


def candidates(kind: str | None = None) -> list[dict]:
    """Every candidate architecture of "what runs where" (spec §First result), each family's own list."""
    out = []
    for module in dict.fromkeys(FAMILIES.values()):
        out += importlib.import_module(module).candidates()
    return [a for a in out if kind is None or a["kind"] == kind]
