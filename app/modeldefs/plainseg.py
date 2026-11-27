"""A plain conv segmenter, this project's own code (no third-party model code, no weights): the fallback
candidate built only from layer types the earlier work saw both NPU compilers accept in YOLOX-Nano
(conv, depthwise conv, add, nearest-neighbour upsampling; `ml/compile_gate_report.md`), with ReLU6 as the
activation. No attention, no squeeze-and-excite, no hard-swish, no bilinear resize, no transpose.

Shape: a stride-2 stem, then three stride-2 stages of depthwise-separable blocks (features at 1/4, 1/8,
1/16 of the input), then two upsample-and-add steps back to 1/4, and a 1×1 conv to one raw score per
class. The scores map is (1, classes, H/4, W/4): % cover per class needs no finer grid, and the class of
a pixel is the largest score, taken outside the model.

It is a candidate for "what runs where", not a tuned design: widths are a first guess (DESIGN D57).
"""

from __future__ import annotations

CODE = {"repo": "this repository", "commit": "app/modeldefs/plainseg.py"}
SIZES = {"s": (16, 32, 64, 96), "m": (24, 48, 96, 160)}   # channels: stem, 1/4, 1/8, 1/16
OUTPUT_NAMES = ["scores"]
OUT_STRIDE = 4
# shown wherever the app shows this candidate (spec lead, 2026-11-27)
NOTE = ("our own untrained design: it shows whether this kind of model can place on the NPU; it is not a "
        "published model with a known accuracy")


def arch(size: str, input_px: int, classes: list[str]) -> dict:
    return {
        "kind": "segmenter", "family": "plainseg", "size": size, "variant": {},
        "input": [input_px, input_px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES, "stride": OUT_STRIDE,
                    "layout": "scores: one raw score per class and cell, cells of 4×4 pixels; the largest wins"},
        "export": {"env": "convert-venv", "opset": 13, "simplify": False},
        "code": dict(CODE), "licence": "own code", "licence_mark": None,
        "note": NOTE,
    }


def candidates() -> list[dict]:
    from app.modeldefs import SEGMENTER_CLASSES
    return [arch(size, 256, SEGMENTER_CLASSES) for size in SIZES]


def check(a: dict) -> list[str]:
    problems = []
    if a["size"] not in SIZES:
        problems.append(f"plain segmenter size must be one of {', '.join(SIZES)}, not {a['size']!r}")
    if any(v % 16 for v in a["input"] if isinstance(v, int)):
        problems.append("input height and width must be multiples of 16")
    return problems


def build_model(a: dict):
    import torch
    from torch import nn

    def conv(cin, cout, k=3, stride=1, groups=1):
        return nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))

    def block(cin, cout, stride):          # depthwise 3×3, then 1×1
        return nn.Sequential(conv(cin, cin, 3, stride, groups=cin), conv(cin, cout, 1))

    class PlainSeg(nn.Module):
        def __init__(self, widths, classes):
            super().__init__()
            c0, c4, c8, c16 = widths
            self.stem = conv(3, c0, 3, 2)
            self.s4 = nn.Sequential(block(c0, c4, 2), block(c4, c4, 1))
            self.s8 = nn.Sequential(block(c4, c8, 2), block(c8, c8, 1))
            self.s16 = nn.Sequential(block(c8, c16, 2), block(c16, c16, 1), block(c16, c16, 1))
            self.lat16, self.lat8 = conv(c16, c8, 1), conv(c8, c4, 1)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.fuse8, self.fuse4 = block(c8, c8, 1), block(c4, c4, 1)
            self.head = nn.Conv2d(c4, classes, 1)

        def forward(self, x):
            f4 = self.s4(self.stem(x))
            f8 = self.s8(f4)
            f16 = self.s16(f8)
            p8 = self.fuse8(f8 + self.up(self.lat16(f16)))
            p4 = self.fuse4(f4 + self.up(self.lat8(p8)))
            return self.head(p4)

    return PlainSeg(SIZES[a["size"]], len(a["classes"]))


def build(a: dict):
    return build_model(a).eval()
