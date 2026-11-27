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

PINNED = "rfdetr==1.9.4"   # the version of Nick's earlier work (~/nereus_ml/venvs/rfdetr); PyPI has newer
CODE = {"repo": "https://github.com/roboflow/rf-detr", "commit": "pip:" + PINNED}
STOCK_INPUT = 384          # RFDETRNanoConfig.resolution
QUERIES = 300              # boxes per image the model always returns
OUTPUT_NAMES = ["dets", "labels"]


def arch(input_px: int, classes: list[str]) -> dict:
    return {
        "kind": "detector", "family": "rfdetr", "size": "nano", "variant": {},
        "input": [input_px, input_px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES,
                    "layout": f"dets: {QUERIES} boxes (cx, cy, w, h; 0..1 of the input); labels: {QUERIES} rows "
                              "of one logit per class plus one (no sigmoid applied)"},
        "export": {"env": "rfdetr-venv", "opset": 17},
        "code": dict(CODE), "licence": "Apache-2.0", "licence_mark": None,
    }


def candidates() -> list[dict]:
    """The stock 384 px, and 256 px to compare with YOLOX at the same input."""
    from app.modeldefs import BUILD2_CLASSES
    return [arch(px, BUILD2_CLASSES) for px in (STOCK_INPUT, 256)]


def check(a: dict) -> list[str]:
    problems = []
    if a["size"] != "nano":
        problems.append("only RF-DETR Nano is offered here; Atto, Femto and Pico are under Roboflow's own "
                        "licence (bite 2d) and the larger sizes are not candidates for the cameras")
    if isinstance(a["input"], list) and (a["input"][0] != a["input"][1] or a["input"][0] % 32):
        problems.append("RF-DETR input must be square and a multiple of 32 (its backbone refuses other sizes)")
    return problems


def build_model(a: dict):
    """Roboflow's wrapper object, untrained; `.model.model` is the network itself."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from rfdetr import RFDETRNano
    except ImportError as e:
        from app.modeldefs import ModelDefError
        raise ModelDefError(f"RF-DETR's code is not in this environment ({e}). Fix: set the tools up "
                            f"(it installs {PINNED} into rfdetr-venv).") from None
    return RFDETRNano(pretrain_weights=None, num_classes=len(a["classes"]), resolution=a["input"][0])


def build(a: dict):
    """The network switched to its own export form: forward returns (dets, labels)."""
    core = build_model(a).model.model.eval()
    core.export()
    return core
