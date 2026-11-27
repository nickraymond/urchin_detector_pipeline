"""YOLOX Nano / Tiny with a plain first layer: the detector that Nick's earlier work compiled for both
NPUs and measured on both boards (build2_spec.md, "Models that are safe to use").

Adapted from Nick's `ADIN_SPI_OpenMV/ml/yolox_urchin/model.py` (`build_model`, `RawExport`) and the gate's
`export_candidates.py`. Megvii's YOLOX is used unmodified from a checkout at CODE["commit"] (Apache-2.0;
THIRD_PARTY_LICENSES); two things are changed on the built model, as in the earlier work:

1. The first layer ("Focus", a slicing trick) is replaced by a plain stride-2 conv with the same output
   channels: Vela accepts only stride-1 slices, so Focus cannot run on the AE3's NPU
   (`ml/compile_gate_report.md`, "Deviations"). The model is trained from scratch either way.
2. The exported module returns the head's three raw maps, one per stride (8, 16, 32), each with
   4 box values + objectness + one score per class, sigmoids applied; boxes are decoded outside the model.

Importing YOLOX needs `loguru`, `tabulate` and `cv2` besides torch (measured 2026-11-27: the app's
environment lacks them; the conversion environment has them).
"""

from __future__ import annotations

CODE = {"repo": "https://github.com/Megvii-BaseDetection/YOLOX", "commit": "6ddff4824372906469a7fae2dc3206c7aa4bbaee"}
CHECKOUT = "YOLOX"     # folder under <tools_dir>/third_party/; the convert job checks its commit
SIZES = {"nano": "yolox-nano", "tiny": "yolox-tiny"}   # size -> YOLOX's own experiment name
STRIDES = (8, 16, 32)
OUTPUT_NAMES = ["out_s8", "out_s16", "out_s32"]


def arch(size: str, input_px: int, classes: list[str]) -> dict:
    return {
        "kind": "detector", "family": "yolox", "size": size, "variant": {"stem": "conv"},
        "input": [input_px, input_px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES, "strides": list(STRIDES),
                    "layout": "per stride: 4 box values (x, y offsets in cells; log w, log h), "
                              "objectness, one score per class; sigmoids applied"},
        "export": {"env": "convert-venv", "opset": 13},
        "code": dict(CODE), "licence": "Apache-2.0", "licence_mark": None,
    }


def candidates() -> list[dict]:
    """256 px is the size the earlier work compiled and measured; 320 px is one step up (the input
    must be a multiple of 32)."""
    from app.modeldefs import BUILD2_CLASSES
    return [arch(size, px, BUILD2_CLASSES) for size in SIZES for px in (256, 320)]


def check(a: dict) -> list[str]:
    problems = []
    if a["size"] not in SIZES:
        problems.append(f"YOLOX size must be one of {', '.join(SIZES)}, not {a['size']!r}")
    if (a.get("variant") or {}).get("stem") != "conv":
        problems.append('only the plain first layer (variant stem "conv") is offered: the stock one '
                        "cannot run on the AE3's NPU")
    if any(v % 32 for v in a["input"] if isinstance(v, int)):
        problems.append("YOLOX input height and width must be multiples of 32")
    return problems


def build_model(a: dict):
    """The whole YOLOX model (backbone + head), untrained, with the plain first layer. Training (S8)
    uses this; export uses build()."""
    from app.modeldefs import third_party
    third_party(CHECKOUT, CODE["commit"], CODE["repo"], package="yolox")
    from yolox.exp.build import get_exp_by_name
    from yolox.models.network_blocks import BaseConv

    exp = get_exp_by_name(SIZES[a["size"]])
    exp.num_classes = len(a["classes"])
    model = exp.get_model()
    stem_out = model.backbone.backbone.stem.conv.conv.out_channels
    model.backbone.backbone.stem = BaseConv(3, stem_out, ksize=3, stride=2, act="silu")
    return model


def export_module(model):
    """Wrap a YOLOX model so it returns the three raw maps (NCHW: 4 + 1 + classes channels each)."""
    import torch

    class RawExport(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone, self.head = m.backbone, m.head

        def forward(self, x):
            outs = []
            for k, (cls_conv, reg_conv, xk) in enumerate(
                    zip(self.head.cls_convs, self.head.reg_convs, self.backbone(x))):
                xs = self.head.stems[k](xk)
                cls_out = self.head.cls_preds[k](cls_conv(xs))
                reg_feat = reg_conv(xs)
                outs.append(torch.cat([self.head.reg_preds[k](reg_feat),
                                       self.head.obj_preds[k](reg_feat).sigmoid(),
                                       cls_out.sigmoid()], 1))
            return tuple(outs)

    return RawExport(model).eval()


def build(a: dict):
    return export_module(build_model(a).eval())
