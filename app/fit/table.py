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

from itertools import combinations
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


# What the vendor's own models take in each camera's model area. Loading a model on the N6 means writing the
# whole area, so every image carries them (ml/build_romfs_n6.sh). Not rebuilt here: derived from the earlier
# work's images of 2026-08-25 (~/nereus_ml/romfs_n6_stage1/out/romfs0.img, 19,057,780 bytes, and
# romfs_ae3_stage1/out/romfs0.img, 15,251,036 bytes) minus the two YOLOX files they also hold (6,527,040
# bytes as given to the image builder). Approximate; S7's image build gives the real figure.
VENDOR_BYTES = {"n6": 19_057_780 - 6_527_040, "ae3": 15_251_036 - 6_527_040}
VENDOR_SOURCE = "the vendor's models as in the earlier work's images of 2026-08-25 (approximate)"
PI_MEMORY_BYTES = 512 * 1024 * 1024


def stored_together(rows: list[dict]) -> dict[str, dict]:
    """Per camera: the best candidate of each kind that runs there, and for each pair and all three,
    whether their files fit side by side and how much room is left (spec, "More than one model at
    once", the laptop part). Loaded together and run in one wake-up are measured on the boards (S7)."""
    out = {}
    for camera in CAMERA_ORDER:
        best = {}
        for row in rows:
            cands = [c for c in row["cells"][camera]["candidates"]
                     if c["code"] in ("runs", "partly") and c["fit"].get("file_bytes")]
            if cands:
                pick = min(cands, key=lambda c: (RANK[c["code"]], c["fit"]["file_bytes"]))
                best[row["label"]] = {"key": pick["key"], "bytes": pick["fit"]["file_bytes"], "mark": pick["mark"]}
        if camera == "pi":
            room, taken, note = PI_MEMORY_BYTES, 0, ("against the Pi's 512 MB of memory, files only; the SD card "
                                                      "is not the limit, and memory in use is measured in S7")
        else:
            limit = next((c["fit"]["limit_bytes"] for row in rows for c in row["cells"][camera]["candidates"]
                          if c["fit"] and c["fit"].get("limit_bytes")), None)
            room, taken, note = limit, VENDOR_BYTES[camera], f"in the model area, beside {VENDOR_SOURCE}"
        combos = []
        for n in (2, 3):
            for labels in combinations(best, n):
                size = sum(best[l]["bytes"] for l in labels)
                left = None if room is None else room - taken - size
                combos.append({"labels": labels, "bytes": size, "left": left,
                               "fits": None if left is None else left >= 0})
        out[camera] = {"best": best, "combos": combos, "room": room, "taken": taken, "note": note}
    return out

