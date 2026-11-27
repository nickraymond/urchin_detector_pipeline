"""A small conv instance segmenter: YOLOX Nano / Tiny (plain first layer) plus prototype masks, the idea of
YOLACT (Bolya et al., 2019): the network outputs K prototype masks for the whole image and K coefficients
per box; an instance's mask is the coefficient-weighted sum of the prototypes, cropped to its box, and
that last step runs outside the model.

This is our own code on top of Megvii's unmodified YOLOX (Apache-2.0); no YOLACT code is used (YOLACT's
repository is MIT, but its models use ResNet-50 / -101 or Darknet-53 backbones, its readme, read
2026-11-27, and are not small). It is a candidate for "what
runs where", with only layer types YOLOX-Nano already uses (conv, SiLU, nearest upsampling), so it shows
whether a conv instance segmenter can sit fully on an NPU (DESIGN D58). It is not a published
architecture and has no training code yet (S10 would add the mask loss).

Outputs: the three YOLOX maps, each with K more channels (4 box + objectness + classes + K
coefficients), and `protos` (1, K, H/4, W/4).
"""

from __future__ import annotations

from app.modeldefs import yolox

CODE = dict(yolox.CODE)
CHECKOUT = yolox.CHECKOUT
PROTOS = 16                      # K; YOLACT uses 32 at 550 px
OUTPUT_NAMES = [*yolox.OUTPUT_NAMES, "protos"]
LICENCE = "Apache-2.0 (YOLOX) + own code"
# shown wherever the app shows this candidate (spec lead, 2026-11-27)
NOTE = ("our own untrained design, not trained anywhere yet: it shows whether a conv instance segmenter can "
        "place on the NPU; it is not a published model with a known accuracy")


def arch(size: str, input_px: int, classes: list[str]) -> dict:
    return {
        "kind": "instance", "family": "yoloxproto", "size": size, "variant": {"stem": "conv"},
        "input": [input_px, input_px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES, "strides": list(yolox.STRIDES), "protos": PROTOS,
                    "layout": f"per stride: 4 box values, objectness, one score per class (sigmoids applied), "
                              f"{PROTOS} mask coefficients; protos: {PROTOS} prototype masks at 1/4 of the input"},
        "export": {"env": "convert-venv", "opset": 13, "simplify": False},
        "code": dict(CODE), "licence": LICENCE, "licence_mark": None,
        "note": NOTE,
    }


def candidates() -> list[dict]:
    from app.modeldefs import BUILD2_CLASSES
    return [arch(size, 256, BUILD2_CLASSES) for size in yolox.SIZES]


def check(a: dict) -> list[str]:
    return yolox.check(a)


def build(a: dict):
    import torch
    from torch import nn

    base = yolox.build_model(a)
    from yolox.models.network_blocks import BaseConv

    class YoloxProto(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone, self.head = m.backbone, m.head
            c8 = m.head.stems[0].conv.in_channels            # FPN channels at stride 8
            hidden = m.head.stems[0].conv.out_channels       # the head's width
            self.proto = nn.Sequential(BaseConv(c8, hidden, 3, 1, act="silu"),
                                       nn.Upsample(scale_factor=2, mode="nearest"),
                                       BaseConv(hidden, hidden, 3, 1, act="silu"),
                                       nn.Conv2d(hidden, PROTOS, 1))
            self.coef = nn.ModuleList(nn.Conv2d(hidden, PROTOS, 1) for _ in m.head.stems)

        def forward(self, x):
            fpn = self.backbone(x)
            outs = []
            for k, (cls_conv, reg_conv, xk) in enumerate(zip(self.head.cls_convs, self.head.reg_convs, fpn)):
                xs = self.head.stems[k](xk)
                reg_feat = reg_conv(xs)
                outs.append(torch.cat([self.head.reg_preds[k](reg_feat),
                                       self.head.obj_preds[k](reg_feat).sigmoid(),
                                       self.head.cls_preds[k](cls_conv(xs)).sigmoid(),
                                       self.coef[k](reg_feat)], 1))
            return (*outs, self.proto(fpn[0]))

    return YoloxProto(base).eval()
