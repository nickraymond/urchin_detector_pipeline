"""torchvision's small segmenters: LR-ASPP and DeepLabV3 on a MobileNetV3-Large backbone. Code BSD-3-Clause
(`pytorch/vision` LICENSE at tag v0.28.0, read 2026-11-27; THIRD_PARTY_LICENSES). Candidates for the per-pixel
segmenter; whether they fit an NPU is unproven (build2_spec.md) and is what the table shows.

Built untrained: `weights=None, weights_backbone=None`, so nothing is downloaded and no weights licence
applies. torchvision's published weights are a separate check (they "may have their own licenses").

The exported module returns one map of raw scores, (1, classes, H, W) at the input's size (torchvision
upsamples it itself, bilinear); the class of a pixel is the largest score, taken outside the model.
"""

from __future__ import annotations

CODE = {"repo": "https://github.com/pytorch/vision", "commit": "pip:torchvision"}
SIZES = {"lraspp-mnv3l": "lraspp_mobilenet_v3_large", "deeplabv3-mnv3l": "deeplabv3_mobilenet_v3_large"}
OUTPUT_NAMES = ["scores"]


def arch(size: str, input_px: int, classes: list[str]) -> dict:
    return {
        "kind": "segmenter", "family": "tvseg", "size": size, "variant": {},
        "input": [input_px, input_px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES,
                    "layout": "scores: one raw score per class and pixel, at the input's size; the largest wins"},
        "export": {"env": "convert-venv", "opset": 13, "simplify": False},
        "code": dict(CODE), "licence": "BSD-3-Clause", "licence_mark": None,
    }


def candidates() -> list[dict]:
    from app.modeldefs import SEGMENTER_CLASSES
    return [arch(size, 256, SEGMENTER_CLASSES) for size in SIZES]


def check(a: dict) -> list[str]:
    problems = []
    if a["size"] not in SIZES:
        problems.append(f"torchvision segmenter must be one of {', '.join(SIZES)}, not {a['size']!r}")
    if any(v % 32 for v in a["input"] if isinstance(v, int)):
        problems.append("input height and width must be multiples of 32 (the backbone halves the image five times)")
    return problems


def build_model(a: dict):
    """torchvision's own module, untrained; forward returns {"out": scores}."""
    from torchvision.models import segmentation
    return getattr(segmentation, SIZES[a["size"]])(weights=None, weights_backbone=None,
                                                   num_classes=len(a["classes"]))


def export_module(model):
    import torch

    class Scores(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)["out"]

    return Scores(model).eval()


def build(a: dict):
    return export_module(build_model(a).eval())
