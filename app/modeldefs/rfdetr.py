"""RF-DETR Nano (detector): a transformer detector on a DINOv2 backbone, Roboflow, Apache-2.0
(`roboflow/rf-detr` LICENSE, branch `develop`; the package's own metadata says "Apache License 2.0").
A candidate in "what runs where" (Nick, 2026-11-27); never compiled for the cameras before.

Unlike YOLOX this is the pip package `rfdetr` (PINNED below), installed in its own tool environment,
`<tools_dir>/rfdetr-venv` (D51): it needs `transformers`, which the app's environment does not have.

Built untrained with `pretrain_weights=None`. Read in the package (1.9.4, `detr.py`, `models/backbone/
dinov2.py`): weights are downloaded only when `pretrain_weights` names a file, and Nano's patch size of
16 means no DINOv2 weights are loaded either. Measured 2026-11-27 with every network connection refused:
the model builds and runs (DEV_LOG). build() also sets the Hugging Face libraries to offline.

The sizes below Nano (Atto, Femto, Pico) are NOT here: they are the `rfdetr[plus]` package under Roboflow's
PML 1.0 (bite 2d, SPEC hard constraint 6).
"""

from __future__ import annotations

import os

# 1.12.1 since bite 2e: its export route is the one that converts the whole family (DESIGN §Bench results).
# Bites 2a-ii and 2c-i built and read 1.9.4, the version of Nick's earlier work (~/nereus_ml/venvs/rfdetr);
# the network code of 1.12.1 was read again for bite 2e (D59's note).
PINNED = "rfdetr==1.12.1"
CODE = {"repo": "https://github.com/roboflow/rf-detr", "commit": "pip:" + PINNED}
STOCK_INPUT = 384          # RFDETRNanoConfig.resolution
QUERIES = 300              # boxes per image the model always returns
OUTPUT_NAMES = ["dets", "labels"]
# RF-DETR-Seg Nano, the instance segmenter (Apache-2.0 at every size, SPEC Q26): RFDETRSegNanoConfig has
# resolution 312, patch 12, 100 queries; measured: masks come out at 1/4 of the input (78 × 78).
SEG_INPUT, SEG_QUERIES, SEG_PATCH = 312, 100, 12
SIZES = {"nano": "RFDETRNano", "seg-nano": "RFDETRSegNano"}
def _route_note() -> str:
    from app.modeldefs import RFDETR_ROUTE_NOTE
    return RFDETR_ROUTE_NOTE


def seg_arch(input_px: int, classes: list[str]) -> dict:
    a = arch(input_px, classes)
    a.update(kind="instance", size="seg-nano")
    a["outputs"] = {"names": [*OUTPUT_NAMES, "masks"],
                    "layout": f"dets: {SEG_QUERIES} boxes (cx, cy, w, h; 0..1); labels: {SEG_QUERIES} rows of one "
                              "logit per class plus one; masks: one raw mask per box at 1/4 of the input"}
    return a


def arch(input_px: int, classes: list[str]) -> dict:
    return {
        "kind": "detector", "family": "rfdetr", "size": "nano", "variant": {},
        "input": [input_px, input_px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES,
                    "layout": f"dets: {QUERIES} boxes (cx, cy, w, h; 0..1 of the input); labels: {QUERIES} rows "
                              "of one logit per class plus one (no sigmoid applied)"},
        # bite 2e: RF-DETR's own export, then a full-integer step of ours (app/fit/to_int8_rfdetr.py). Before
        # that: {"env": "rfdetr-venv", "opset": 17, "simplify": True} with onnx2tf 1.28.8, which converted Nano
        # only after an onnxsim pass and left operators plain LiteRT does not have
        "export": {"env": "rfdetr-tflite-venv", "opset": 17, "simplify": False, "route": "rfdetr-own"},
        "route_note": _route_note(),
        "code": dict(CODE), "licence": "Apache-2.0", "licence_mark": None,
    }


def candidates() -> list[dict]:
    """The stock 384 px, and 256 px to compare with YOLOX at the same input."""
    from app.modeldefs import BUILD2_CLASSES
    return [*(arch(px, BUILD2_CLASSES) for px in (STOCK_INPUT, 256)), seg_arch(SEG_INPUT, BUILD2_CLASSES)]


def check(a: dict) -> list[str]:
    problems = []
    if a["size"] not in SIZES or (a["size"] == "seg-nano") != (a["kind"] == "instance"):
        problems.append("only RF-DETR Nano (detector) and RF-DETR-Seg Nano (instance) are offered here; Atto, "
                        "Femto and Pico are under Roboflow's own licence (bite 2d) and the larger sizes are not "
                        "candidates for the cameras")
    step = SEG_PATCH if a["size"] == "seg-nano" else 32
    if isinstance(a["input"], list) and (a["input"][0] != a["input"][1] or a["input"][0] % step):
        problems.append(f"this RF-DETR's input must be square and a multiple of {step} (its backbone refuses "
                        "other sizes)")
    return problems


def build_model(a: dict):
    """Roboflow's wrapper object, untrained; `.model.model` is the network itself."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        import rfdetr
    except ImportError as e:
        from app.modeldefs import ModelDefError
        raise ModelDefError(f"RF-DETR's code is not in this environment ({e}). Fix: set the tools up "
                            f"(it installs {PINNED} into rfdetr-tflite-venv).") from None
    # device="cpu": on this laptop torch 2.15's Apple-GPU path crashes inside RF-DETR's export (measured, bite 2e)
    return getattr(rfdetr, SIZES[a["size"]])(pretrain_weights=None, num_classes=len(a["classes"]),
                                             resolution=a["input"][0], device="cpu")


def build(a: dict):
    """The network switched to its own export form: forward returns (dets, labels[, masks])."""
    core = build_model(a).model.model.eval()
    core.export()
    return core
