"""RF-DETR Atto / Femto / Pico: the detector sizes below Nano, from Roboflow's `rfdetr_plus` package.

LICENCE: Roboflow Platform Model License, PML 1.0 (the LICENSE inside the wheel; THIRD_PARTY_LICENSES).
EVALUATION ONLY (SPEC hard constraint 6; Nick, 2026-11-27): these may be compiled and timed for "what
runs where"; their weights, trained ones included, never go on a field unit or into a product until Nick
has the licence. Every recipe here carries licence_mark "Roboflow licence", which the app shows wherever
the model appears. Lead call D59.

What the package sends out (S6 bite 2d; DESIGN D59 and Bench results): read statically, `rfdetr_plus`
1.3.0 has no telemetry, licence-check or upload code. But a default build downloads: the RF-DETR weights
file from repo.roboflow.com, and, when no weights file is named, the backbone's pretrained weights from
the Hugging Face Hub through `timm`. build() below uses the package's own switch for "no upstream
weights" (`force_no_pretrain`), and the Hub libraries are set offline; measured with every connection
refused: the three sizes build and run with 0 network attempts. No frame goes through this code in S6.
"""

from __future__ import annotations

import os

PINNED = "rfdetr-plus==1.3.0"     # needs rfdetr >= 1.12 (its metadata), so it has its own tool environment
CODE = {"repo": "https://github.com/roboflow/rf-detr_plus", "commit": "pip:" + PINNED}
# size -> (config class in rfdetr_plus.models.detection, stock input px, boxes returned): read from the package
SIZES = {"atto": ("RFDETRAttoConfig", 380, 300), "femto": ("RFDETRFemtoConfig", 384, 200),
         "pico": ("RFDETRPicoConfig", 560, 200)}
WRAPPERS = {"atto": "RFDETRAtto", "femto": "RFDETRFemto", "pico": "RFDETRPico"}   # the classes `rfdetr` exposes
OUTPUT_NAMES = ["dets", "labels"]
LICENCE_MARK = "Roboflow licence"


def arch(size: str, classes: list[str]) -> dict:
    from app.modeldefs import RFDETR_ROUTE_NOTE
    _, px, queries = SIZES[size]
    return {
        "kind": "detector", "family": "rfdetrplus", "size": size, "variant": {},
        "input": [px, px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES,
                    "layout": f"dets: {queries} boxes (cx, cy, w, h; 0..1 of the input); labels: {queries} rows "
                              "of one logit per class plus one"},
        # bite 2e: RF-DETR's own export route; with ours (onnx2tf 1.28.8 on our ONNX file) these did not convert
        "export": {"env": "rfdetr-plus-tflite-venv", "opset": 17, "simplify": False, "route": "rfdetr-own"},
        "route_note": RFDETR_ROUTE_NOTE,
        "code": dict(CODE), "licence": "PML-1.0", "licence_mark": LICENCE_MARK,
    }


def candidates() -> list[dict]:
    from app.modeldefs import BUILD2_CLASSES
    return [arch(size, BUILD2_CLASSES) for size in SIZES]


def check(a: dict) -> list[str]:
    problems = []
    if a["size"] not in SIZES:
        problems.append(f"size must be one of {', '.join(SIZES)}, not {a['size']!r}")
    elif a["input"] != [SIZES[a["size"]][1]] * 2:
        problems.append(f"only the stock input ({SIZES[a['size']][1]} px) is offered for RF-DETR {a['size']}")
    if a.get("licence_mark") != LICENCE_MARK or a.get("licence") != "PML-1.0":
        problems.append('this model is under Roboflow\'s PML 1.0 and must carry the mark "Roboflow licence"')
    return problems


def build_model(a: dict):
    """Roboflow's wrapper object, untrained, on the CPU: what RF-DETR's own export is called on (bite 2e).

    The wrapper has no argument for "no upstream weights": with `pretrain_weights=None` it still asks the Hugging
    Face Hub for the backbone's weights (bite 2d). The switch lives in the package's `MODEL_DEFAULTS`
    (`force_no_pretrain`), which the wrapper reads through the default argument of
    `rfdetr._namespace._namespace_from_configs`; that default is replaced here, in this process only. If a later
    `rfdetr` moves it, the build fails loudly: the 8-bit step refuses every network connection."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import dataclasses
        import rfdetr
        import rfdetr._namespace as namespace
        from rfdetr.models._defaults import MODEL_DEFAULTS
        import rfdetr_plus  # noqa: F401  (the sizes below come from it; a clear message when it is missing)
    except ImportError as e:
        from app.modeldefs import ModelDefError
        raise ModelDefError(f"Roboflow's rfdetr_plus is not in this environment ({e}). Fix: set the tools up "
                            f"(it installs {PINNED} into rfdetr-plus-tflite-venv).") from None
    namespace._namespace_from_configs.__defaults__ = (dataclasses.replace(MODEL_DEFAULTS, force_no_pretrain=True),)
    return getattr(rfdetr, WRAPPERS[a["size"]])(pretrain_weights=None, num_classes=len(a["classes"]),
                                                resolution=a["input"][0], device="cpu")


def build(a: dict):
    """The network, untrained, with no download of any kind, switched to its export form."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import dataclasses
        from rfdetr.models import MODEL_DEFAULTS
        from rfdetr.models.lwdetr import build_model_from_config
        from rfdetr_plus.models import detection
    except ImportError as e:
        from app.modeldefs import ModelDefError
        raise ModelDefError(f"Roboflow's rfdetr_plus is not in this environment ({e}). Fix: set the tools up "
                            f"(it installs {PINNED} into rfdetr-plus-tflite-venv).") from None
    config = getattr(detection, SIZES[a["size"]][0])(pretrain_weights=None, num_classes=len(a["classes"]))
    core = build_model_from_config(config, defaults=dataclasses.replace(MODEL_DEFAULTS, force_no_pretrain=True))
    core.eval()
    core.export()
    return core
