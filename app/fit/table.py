""""What runs where" as data (S6 bite 4; spec §First result): model kind × camera, each cell listing its
candidates with their fit reports and a headline. Reads only `fit.json` files (D53); no torch, no tools.

Headline of one candidate on one camera:
  not tried        no fit.json yet
  does not run     it does not convert or compile, its file is larger than the camera's model area, or it
                   needs more memory than the camera's memory figure (AE3: the free memory measured in the
                   earlier work, a reference, not re-measured; Pi: 512 MB; N6: no figure recorded yet, so
                   not checked, and the page says so)
  runs, partly on the CPU   some layers are left on the CPU (N6: software steps; AE3: CPU operators)
  runs             everything on the NPU. N6 "hybrid" steps count as running: the measured YOLOX-Nano
                   has 2 of them and ran at 10.6 ms (earlier work). On the Pi "runs" means the file runs
                   on its CPU under the runtime; how fast is S7's.
A cell's headline is its best candidate's (spec), ranked among permissive candidates only: a
Roboflow-licensed one (SPEC hard constraint 6, evaluation only) is never a cell's best; the best of those
is shown on its own line. A model that stops at the 8-bit step "does not convert with today's route":
no compiler saw it, so that is not a compiler verdict. These are compiler results, not measurements.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

from app import modeldefs
from app.fit import convert

CAMERA_ORDER = ("n6", "pi", "ae3")
KIND_LABELS = {"detector": "Detector", "segmenter": "Segmenter", "instance": "Instance segmenter"}
RANK = {"runs": 0, "partly": 1, "no": 2, "not_tried": 3}
NO_ROUTE = "does not convert with today's route (8-bit step)"
HEADLINES = {"runs": "runs", "partly": "runs, partly on the CPU", "no": "does not run", "not_tried": "not tried"}

# Measured on the boards in Nick's earlier work; shown as "earlier work" until re-measured through the app
# (spec). Source: ADIN_SPI_OpenMV/ml/yolox_urchin/STAGE1.md, bench window of 2026-08-24 (300-run windows;
# single class "urchin"). The energy figures are that table's gross ones (it also has net-over-idle).
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
        if fit.get("stage") == "8-bit model":
            return "no", "the 8-bit step failed before any compiler ran: not a compiler verdict"
        if camera == "pi":                          # no compiler there: the runtime itself refused the file
            return "no", "LiteRT could not load or run its 8-bit file"
        kind = "an internal error" if "INTERNAL ERROR" in (fit.get("error") or "") else "an error"
        return "no", f"the compiler stopped with {kind} ({fit.get('tool') or 'compile step'})"
    p = fit.get("placement") or {}
    unit = "steps" if p.get("unit") == "epochs" else p.get("unit")      # ST calls its steps "epochs"
    on_cpu = f"; {p['cpu']} of {p['total']} {unit} on the CPU" if p.get("cpu") and camera != "pi" else ""
    size, limit = fit.get("file_bytes"), fit.get("limit_bytes")
    need, have = fit.get("memory_bytes"), memory_figure(fit)
    memory = (f"needs {mib(need)} of memory against {mib(have['bytes'])} ({have['what']})"
              if need and have and need > have["bytes"] else "")
    if size and limit and size > limit:
        return "no", (f"compiles, too big for the model area: its file is {mib(size)} of {mib(limit)}"
                      + (f"; {memory}" if memory else "") + on_cpu)
    if memory:
        return "no", f"compiles, but {memory}{on_cpu}"
    if camera == "pi":
        odd = len(fit.get("failing_layers") or [])
        return ("partly", f"{odd} operator type(s) not built into the runtime") if odd else (
            "runs", "on the CPU; speed and memory are measured on the bench Pi (S7)")
    if p.get("cpu"):
        return "partly", on_cpu[2:]
    if p.get("hybrid"):
        return "runs", f"all on the NPU ({p['hybrid']} hybrid steps)"
    return "runs", "all on the NPU"


def memory_figure(fit: dict) -> dict | None:
    """The camera's memory figure a model's `memory_bytes` is compared with, or None when none is recorded
    (the N6 today). The AE3's is a reference from the earlier work, not a hard limit."""
    ref = fit.get("memory_reference")
    if ref and ref.get("bytes"):
        return {"bytes": ref["bytes"], "what": "free on the board in the earlier work, not re-measured; S7 measures it on the board"}
    if fit.get("memory_limit_bytes"):
        return {"bytes": fit["memory_limit_bytes"], "what": "the board's memory"}
    return None


def label(code: str, fit: dict | None) -> str:
    """The words for one candidate's result. A model no compiler saw is worded apart from "does not run"."""
    if code == "no" and fit and not fit.get("converts") and fit.get("stage") == "8-bit model":
        return NO_ROUTE
    return HEADLINES[code]


def mib(n: int | None) -> str:
    if n is None:
        return "–"
    return f"{n / 1048576:.1f} MiB" if n >= 1048576 else f"{n / 1024:.0f} KiB"


def table(data_dir: Path, active: set[tuple[str, str]] = frozenset()) -> list[dict]:
    """One row per kind; row["cells"][camera] = {"code", "headline", "best", "best_marked", "candidates": [...],
    "earlier": [...]}. "best" is ranked among permissive candidates only; "best_marked" is the best
    Roboflow-licensed one that has a result, shown on its own line.
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
                cands.append({"key": key, "arch": arch, "fit": fit, "code": code, "headline": label(code, fit),
                              "why": why, "mark": arch.get("licence_mark"), "active": (key, camera) in active,
                              "older": convert.older_recipe(fit, arch),
                              "second": second(data_dir, arch, camera, active)})
            best = min((c for c in cands if c["mark"] is None), key=lambda c: RANK[c["code"]], default=None)
            marked = min((c for c in cands if c["mark"] and c["fit"]), key=lambda c: RANK[c["code"]], default=None)
            cells[camera] = {"code": best["code"] if best else "not_tried",
                             "headline": HEADLINES[best["code"]] if best else "no candidates",
                             "best": best, "best_marked": marked, "candidates": cands,
                             "earlier": [e for e in EARLIER if (e["kind"], e["camera"]) == (kind, camera)]}
        rows.append({"kind": kind, "label": KIND_LABELS[kind], "cells": cells})
    return rows


def second(data_dir: Path, arch: dict, camera: str, active: set[tuple[str, str]]) -> dict | None:
    """A candidate's result from the camera's second compiler (the N6's ST Edge AI Core 4.1.0, D71), shown on its
    own line under the candidate: it never sets a cell's headline and "Run all" does not queue it."""
    if camera not in convert.SECOND:
        return None
    step, words = convert.SECOND[camera]
    key = modeldefs.arch_key(arch)
    fit = convert.read_fit(convert.model_dir(data_dir, arch), step)
    code, why = headline(fit, camera)
    return {"step": step, "words": words, "fit": fit, "code": code, "headline": label(code, fit), "why": why,
            "active": (key, step) in active, "older": convert.older_recipe(fit, arch)}


def missing(data_dir: Path) -> list[tuple[dict, str]]:
    """Every (candidate, camera) with no fit report yet, in table order: what "Run all" queues."""
    return [(arch, camera) for kind in modeldefs.KINDS for arch in modeldefs.candidates(kind)
            for camera in CAMERA_ORDER
            if convert.read_fit(convert.model_dir(data_dir, arch), camera) is None]


# What the vendor's own models take in each camera's model area. Loading a model on the N6 means writing the
# whole area, so every image carries them (ml/build_romfs_n6.sh). Not rebuilt here: derived from the earlier
# work's images of 2026-08-25 (~/nereus_ml/romfs_n6_stage1/out/romfs0.img, 19,057,780 bytes, and
# romfs_ae3_stage1/out/romfs0.img, 15,251,036 bytes) minus the two compiled YOLOX files each image holds
# (their `build/` folders: N6 1,202,776 + 5,277,816 bytes; AE3 1,014,816 + 4,966,720). Approximate; S7's
# image build gives the real figure.
VENDOR_BYTES = {"n6": 19_057_780 - 6_480_592, "ae3": 15_251_036 - 5_981_536}
VENDOR_SOURCE = "the vendor's models, from the earlier image build of 2026-08-25 (approximate)"
PI_MEMORY_BYTES = 512 * 1024 * 1024


def stored_together(rows: list[dict]) -> dict[str, dict]:
    """Per camera: the best permissive candidate of each kind that runs there (what could ship together; a
    Roboflow-licensed one is never picked, SPEC hard constraint 6), and for each pair and all three,
    whether their files fit side by side and how much room is left (spec, "More than one model at
    once", the laptop part). Loaded together and run in one wake-up are measured on the boards (S7)."""
    out = {}
    for camera in CAMERA_ORDER:
        best = {}
        for row in rows:
            cands = [c for c in row["cells"][camera]["candidates"]
                     if c["code"] in ("runs", "partly") and c["fit"].get("file_bytes") and c["mark"] is None]
            if cands:
                pick = min(cands, key=lambda c: (RANK[c["code"]], c["fit"]["file_bytes"]))
                best[row["label"]] = {"key": pick["key"], "bytes": pick["fit"]["file_bytes"], "mark": pick["mark"]}
        if camera == "pi":
            room, taken, note = PI_MEMORY_BYTES, 0, ("files (in memory once loaded) against the Pi's 512 MB of RAM; "
                                                      "storage on the SD card is not the limit, and memory in "
                                                      "use is measured in S7")
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

