"""RTMDet-Ins-tiny: a published small conv instance segmenter (OpenMMLab's mmdetection, Apache-2.0: the LICENSE
files of `open-mmlab/mmdetection`, `mmcv` and `mmengine`, read on GitHub 2026-11-27). S6 bite 2c-ii.

A candidate for "what runs where", next to this project's own `yoloxproto`: its readme gives 5.6 M parameters
and COCO scores (`configs/rtmdet/README.md`), so unlike `yoloxproto` it has a known accuracy and training code.

How it is built here, and what is left out:
- The pip packages `mmdet` (PINNED), `mmengine` and `mmcv-lite`, in their own tool environment `mmdet-venv`
  (D51). mmdetection imports `mmcv.ops`, which loads mmcv's compiled operators (`mmcv._ext`); `mmcv-lite` does
  not have them and PyPI has no macOS build of full mmcv (only a source archive). The raw network uses none of them (they are
  box post-processing such as NMS, and layers of other models), so build() answers those imports with a
  stand-in that fails if anything calls it. mmdet's own code is not changed.
- Untrained: the config's pretrained backbone (`init_cfg`, a download from download.openmmlab.com) is removed,
  and `init_weights()` is never called. Measured with every connection refused: 0 network attempts.
- The exported module is backbone -> neck -> head, raw maps. Left outside the model, as for the other
  candidates: box decoding, NMS, and the mask step. RTMDet-Ins predicts, per position, 169 numbers that are the
  weights of a tiny 3-layer conv ("dynamic conv"); an instance's mask is that conv run on the 8-channel mask
  features plus 2 coordinate channels. On a camera that step would run on the CPU for the boxes kept.
- No image normalisation inside (mmdet's data preprocessor is not part of the export).
"""

from __future__ import annotations

PINNED = "mmdet==3.3.0"
CODE = {"repo": "https://github.com/open-mmlab/mmdetection", "commit": "pip:" + PINNED}
CONFIG = "rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"      # inside the package: mmdet/.mim/configs/
STRIDES = (8, 16, 32)
KERNEL = 169                     # (8 + 2) * 8 + 8, then 8 * 8 + 8, then 8 * 1 + 1: the dynamic conv's weights
MASK_CHANNELS = 8
OUTPUT_NAMES = [f"{what}_s{s}" for what in ("cls", "box", "kernel") for s in STRIDES] + ["mask_feat"]


def arch(input_px: int, classes: list[str]) -> dict:
    return {
        "kind": "instance", "family": "rtmdetins", "size": "tiny", "variant": {},
        "input": [input_px, input_px], "classes": list(classes),
        "outputs": {"names": OUTPUT_NAMES, "strides": list(STRIDES),
                    "layout": f"per stride: class scores (no sigmoid), 4 box distances, {KERNEL} dynamic-conv "
                              f"weights; mask_feat: {MASK_CHANNELS} mask-feature maps at 1/8 of the input"},
        "export": {"env": "mmdet-venv", "opset": 13, "simplify": False},
        "code": dict(CODE), "licence": "Apache-2.0", "licence_mark": None,
    }


def candidates() -> list[dict]:
    from app.modeldefs import BUILD2_CLASSES
    return [arch(256, BUILD2_CLASSES)]


def check(a: dict) -> list[str]:
    problems = []
    if a["size"] != "tiny":
        problems.append("only RTMDet-Ins-tiny is offered (the larger sizes are not candidates for the cameras)")
    if isinstance(a["input"], list) and (a["input"][0] != a["input"][1] or a["input"][0] % 32):
        problems.append("RTMDet-Ins's input must be square and a multiple of 32")
    return problems


def _answer_mmcv_ops() -> None:
    """Let `import mmcv.ops` and `from mmcv.ops import X` succeed without mmcv's compiled operators."""
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import sys
    import types

    if "mmcv.ops" in sys.modules or importlib.util.find_spec("mmcv") is None:
        return
    try:                                              # mmcv-lite ships mmcv/ops/*.py but not what they load
        if importlib.util.find_spec("mmcv._ext") is not None:
            return                                    # full mmcv is installed: nothing to stand in for
    except (ImportError, ValueError):
        pass

    class Missing:
        """One compiled mmcv operator: importable and subclassable, never usable."""

        def __init__(self, *a, **k):
            raise RuntimeError("mmcv's compiled operators are not installed (mmcv-lite); RTMDet-Ins's raw "
                               "network should not need them.")

        def __init_subclass__(cls, **k):
            pass

    class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname == "mmcv.ops" or fullname.startswith("mmcv.ops."):
                return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
            return None

        def create_module(self, spec):
            module = types.ModuleType(spec.name)
            module.__path__ = []

            def stand_in(attr):
                if attr.startswith("__"):
                    raise AttributeError(attr)
                return type(attr, (Missing,), {})

            module.__getattr__ = stand_in
            return module

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, Finder())


def build(a: dict):
    _answer_mmcv_ops()
    try:
        import mmdet
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet.registry import MODELS
    except ImportError as e:
        from app.modeldefs import ModelDefError
        raise ModelDefError(f"mmdetection is not in this environment ({e}). Fix: set the tools up "
                            f"(it installs {PINNED} into mmdet-venv).") from None
    from pathlib import Path

    from torch import nn

    init_default_scope("mmdet")
    cfg = Config.fromfile(str(Path(mmdet.__file__).parent / ".mim" / "configs" / CONFIG))
    cfg.model.backbone.init_cfg = None               # no pretrained backbone: nothing is downloaded
    cfg.model.bbox_head.num_classes = len(a["classes"])
    model = MODELS.build(cfg.model)

    class Raw(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.backbone, self.neck, self.head = m.backbone, m.neck, m.bbox_head

        def forward(self, x):
            cls, box, kernel, mask_feat = self.head(self.neck(self.backbone(x)))
            return (*cls, *box, *kernel, mask_feat)

    return Raw(model).eval()
