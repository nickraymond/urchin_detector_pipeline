"""Model definitions for Build 2 (DESIGN D51, D54): what a candidate is and how to build it untrained.

Plain data plus PyTorch. Nothing here imports the rest of `app`, so the same code runs in the app's
environment (training, S8) and in the conversion environment under `tools_dir` (a test checks). This
package imports no torch until a model is built: the "What runs where" page calls candidates() without
loading a model.

A candidate is the `arch` part of a recipe (D54):

    {"kind": "detector", "family": "yolox", "size": "nano", "variant": {"stem": "conv"},
     "input": [256, 256], "classes": [...], "outputs": {...}, "code": {"repo", "commit"},
     "export": {"env": "convert-venv", "opset": 13},   # which tool environment builds it (D51)
     "licence": "Apache-2.0", "licence_mark": None}

    arch_key(arch)  -> "yolox-nano-conv-256-c5"   (the folder name under <data_dir>/fit/, D53)
    build(arch)     -> the model, untrained, as the module that is exported (raw output maps)

Third-party model code is an unmodified git checkout at a pinned commit under
`<tools_dir>/third_party/<name>` (D51); `third_party(...)` loads its one package from there and says how
to get it when it is missing.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path

KINDS = ("detector", "segmenter", "instance")
DEFAULT_TOOLS_DIR = Path.home() / "NereusData" / "urchin_detector_pipeline" / "tools"

# Build 2 trains on sea stars and urchins only (Nick, 2026-11-27; names as in default_classes.yaml).
BUILD2_CLASSES = ["purple urchin", "red urchin", "black urchin", "sea star", "sunflower sea star"]

# family -> the module in this package that builds it
FAMILIES = {"yolox": "app.modeldefs.yolox", "rfdetr": "app.modeldefs.rfdetr"}


class ModelDefError(ValueError):
    """A recipe that cannot be built, or model code that is not on the laptop; the message says why."""


def tools_dir() -> Path:
    """The convert job passes tools_dir from config.local.toml in URCHIN_TOOLS_DIR (D51)."""
    return Path(os.environ.get("URCHIN_TOOLS_DIR") or DEFAULT_TOOLS_DIR).expanduser()


def third_party(name: str, commit: str, repo: str, package: str) -> Path:
    """Make `package` importable from <tools_dir>/third_party/<name>/<package>. Only that package is
    loaded: the checkout's root is not put on sys.path, because it holds other top-level packages
    (YOLOX has its own `tests` and `tools`) that would shadow ours (QE PR 68). The checkout is not
    verified against `commit` here; the convert job records and checks it."""
    path = tools_dir() / "third_party" / name
    init = path / package / "__init__.py"
    if not init.is_file():
        raise ModelDefError(
            f"Model code not found: {path}. Fix: set the tools up (it checks out {repo} at {commit[:7]}).")
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, init, submodule_search_locations=[str(init.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[package]
            raise
    return path


DEFAULT_EXPORT = {"env": "convert-venv", "opset": 13}   # the known-good YOLOX route (D52)


def export_settings(arch: dict) -> dict:
    """Which tool environment builds and exports this model, and the ONNX opset."""
    return {**DEFAULT_EXPORT, **(arch.get("export") or {})}


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
    if not isinstance(arch["code"], dict) or not isinstance(arch.get("variant") or {}, dict):
        return problems + ["code and variant must each be a table of names and values"]
    inp = arch["input"]
    if not (isinstance(inp, list) and len(inp) == 2 and all(isinstance(v, int) and v > 0 for v in inp)):
        problems.append("input must be [height, width] in pixels")
    classes = arch["classes"]
    if not (isinstance(classes, list) and classes and all(isinstance(c, str) and c for c in classes)):
        problems.append("the recipe lists no classes (classes must be a list of class names)")
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
