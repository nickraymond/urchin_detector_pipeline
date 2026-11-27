""""What runs where" as data (S6 bite 4; spec §First result): model kind × camera, each cell listing its
candidates with their fit reports and a headline. Reads only `fit.json` files (D53); no torch, no tools.

Headline of one candidate on one camera:
  not tried        no fit.json yet
  does not run     it does not convert or compile, or its file is larger than the camera's model area
  runs, partly on the CPU   some layers are left on the CPU (N6: software steps; AE3: CPU operators)
  runs             everything on the NPU. N6 "hybrid" steps count as running: the measured YOLOX-Nano
                   has 2 of them and ran at 10.6 ms (earlier work). On the Pi "runs" means the file runs
                   on its CPU under the runtime; how fast is S7's.
A cell's headline is its best candidate's (spec). These are compiler results, not measurements.
"""

from __future__ import annotations

from pathlib import Path

from app import modeldefs
from app.fit import convert

CAMERA_ORDER = ("n6", "pi", "ae3")
KIND_LABELS = {"detector": "Detector", "segmenter": "Segmenter", "instance": "Instance segmenter"}
RANK = {"runs": 0, "partly": 1, "no": 2, "not_tried": 3}
HEADLINES = {"runs": "runs", "partly": "runs, partly on the CPU", "no": "does not run", "not_tried": "not tried"}

# Measured on the boards in Nick's earlier work; shown as "earlier work" until re-measured through the app
# (spec). Source: ADIN_SPI_OpenMV/ml/yolox_urchin/STAGE1.md, bench window of 2026-08-24 (300-run windows;
# single class "urchin", 256 px).
EARLIER = [
    {"kind": "detector", "camera": "n6", "model": "YOLOX-Nano", "ms": 10.6, "mj": 11.5, "date": "2026-08-24"},
    {"kind": "detector", "camera": "n6", "model": "YOLOX-Tiny", "ms": 31.2, "mj": 38.25, "date": "2026-08-24"},
    {"kind": "detector", "camera": "ae3", "model": "YOLOX-Nano", "ms": 26.4, "mj": 6.7, "date": "2026-08-24"},
    {"kind": "detector", "camera": "ae3", "model": "YOLOX-Tiny (from the model area only)", "ms": 58.4,
     "mj": 17.61, "date": "2026-08-24"},
]
EARLIER_SOURCE = "ADIN_SPI_OpenMV/ml/yolox_urchin/STAGE1.md"


def headline(fit: dict | None, camera: str) -> tuple[str, str]:
    """(code, one line saying why) for one candidate on one camera."""
    if fit is None:
        return "not_tried", ""
    if not fit.get("converts"):
        return "no", f"stopped at the {fit.get('stage', 'compile')} step"
    size, limit = fit.get("file_bytes"), fit.get("limit_bytes")
    if size and limit and size > limit:
        return "no", f"its file ({mib(size)}) is larger than the model area ({mib(limit)})"
    p = fit.get("placement") or {}
    if camera == "pi":
        odd = len(fit.get("failing_layers") or [])
        return ("partly", f"{odd} operator type(s) not built into the runtime") if odd else (
            "runs", "on the CPU; speed and memory are measured on the bench Pi (S7)")
    if p.get("cpu"):
        unit = "steps" if p["unit"] == "epochs" else p["unit"]     # ST calls its steps "epochs"
        return "partly", f"{p['cpu']} of {p['total']} {unit} on the CPU"
    if p.get("hybrid"):
        return "runs", f"all on the NPU ({p['hybrid']} hybrid steps)"
    return "runs", "all on the NPU"


def mib(n: int | None) -> str:
    if n is None:
        return "–"
    return f"{n / 1048576:.1f} MiB" if n >= 1048576 else f"{n / 1024:.0f} KiB"


def table(data_dir: Path, active: set[tuple[str, str]] = frozenset()) -> list[dict]:
    """One row per kind; row["cells"][camera] = {"code", "headline", "candidates": [...], "earlier": [...]}.
    `active` holds (arch key, camera) pairs with a job waiting or running."""
    rows = []
    for kind in modeldefs.KINDS:
        cells = {}
        for camera in CAMERA_ORDER:
            cands = []
            for arch in modeldefs.candidates(kind):
                key = modeldefs.arch_key(arch)
                fit = convert.read_fit(convert.model_dir(data_dir, arch), camera)
                code, why = headline(fit, camera)
                cands.append({"key": key, "arch": arch, "fit": fit, "code": code, "headline": HEADLINES[code],
                              "why": why, "mark": arch.get("licence_mark"), "active": (key, camera) in active})
            best = min(cands, key=lambda c: RANK[c["code"]]) if cands else None
            cells[camera] = {"code": best["code"] if best else "not_tried",
                             "headline": HEADLINES[best["code"]] if best else "no candidates",
                             "best": best, "candidates": cands,
                             "earlier": [e for e in EARLIER if (e["kind"], e["camera"]) == (kind, camera)]}
        rows.append({"kind": kind, "label": KIND_LABELS[kind], "cells": cells})
    return rows


def missing(data_dir: Path) -> list[tuple[dict, str]]:
    """Every (candidate, camera) with no fit report yet, in table order: what "Run all" queues."""
    return [(arch, camera) for kind in modeldefs.KINDS for arch in modeldefs.candidates(kind)
            for camera in CAMERA_ORDER
            if convert.read_fit(convert.model_dir(data_dir, arch), camera) is None]
