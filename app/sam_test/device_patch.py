"""Make Meta's SAM 3 code build on a Mac, without editing Meta's files.

Meta's code assumes an NVIDIA GPU (code read of facebookresearch/sam3 @ 31cb7e6):
  1. `import triton` in sam3/model/edt.py, imported on the image path. Triton (NVIDIA only)
     has no macOS build; edt_triton is only *called* by video-tracking and training code.
  2. `PositionEmbeddingSine.__init__` precomputes a cache with device="cuda"
     (sam3/model/position_encoding.py:55). The precompute only exists for torch.compile;
     the same cache is filled lazily on the input's device if we skip it.
  3. `TransformerDecoder.__init__` builds a coordinate cache with device="cuda"
     (sam3/model/decoder.py:303).

Every change made here is recorded in CHANGES so the report can show it.
"""

from __future__ import annotations

import sys
import types

import torch

CHANGES: list[str] = []
_TARGET = {"device": "cpu"}


def _stub_triton_edt() -> None:
    """Replace sam3.model.edt (the only triton import on the image path) with a stub.

    triton itself is NOT faked: torch checks for it and would break.
    """
    try:
        import triton  # noqa: F401
        return
    except ImportError:
        pass

    def edt_triton(*_args, **_kwargs):
        raise RuntimeError(
            "sam3.model.edt.edt_triton needs triton (NVIDIA only), which is not available "
            "on macOS. This SAM 3 feature (video tracking / training) cannot run here."
        )

    stub = types.ModuleType("sam3.model.edt")
    stub.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = stub
    CHANGES.append("edt: triton-only module replaced by a stub that errors if called")


def apply(device: str) -> None:
    """Call once, before building the model. `device` is "mps", "cpu" or "cuda"."""
    _TARGET["device"] = device
    if CHANGES:  # already applied; only the target device changes
        return
    _stub_triton_edt()
    if torch.cuda.is_available():
        CHANGES.append("CUDA present: no device changes needed")
        return

    from sam3.model import decoder, position_encoding

    pe_init = position_encoding.PositionEmbeddingSine.__init__

    def pe_init_no_precompute(self, *args, **kwargs):
        kwargs["precompute_resolution"] = None
        pe_init(self, *args, **kwargs)

    position_encoding.PositionEmbeddingSine.__init__ = pe_init_no_precompute
    CHANGES.append("position_encoding: skip the CUDA-only precompute; cache fills on first use")

    get_coords = decoder.TransformerDecoder._get_coords  # a staticmethod

    def get_coords_on_target(H, W, device):
        if str(device) == "cuda":
            device = _TARGET["device"]
        return get_coords(H, W, device)

    decoder.TransformerDecoder._get_coords = staticmethod(get_coords_on_target)
    CHANGES.append('decoder: coordinate cache built on the chosen device instead of "cuda"')

    from sam3.model import vitdet

    vitdet.addmm_act = addmm_act_plain
    CHANGES.append(
        "vitdet: fused bfloat16 aten::_addmm_activation replaced by linear + activation in "
        "float32 (its bfloat16 output fed float32 layers: CPU errors, MPS gave NaN scores)"
    )

    from sam3.model import geometry_encoders

    enc = geometry_encoders.SequenceGeometryEncoder
    for name in ("_encode_points", "_encode_boxes"):
        setattr(enc, name, _skip_when_empty(getattr(enc, name)))
    CHANGES.append(
        "geometry_encoders: an empty point/box list returns an empty result directly "
        "(MPS asserts on zero-size tensors; text-only prompts pass an empty list); "
        "pin_memory skipped inside them (on a Mac it pins to the Apple GPU, which breaks "
        "CPU and MPS runs alike; pinning only speeds up CUDA)"
    )


def _skip_when_empty(encode):
    """Wrap _encode_points/_encode_boxes: with zero items, return an empty (0, bs, d) result."""

    def wrapper(self, *args, **kwargs):
        # Meta calls these as (points|boxes, *_mask, *_labels, img_feats), by keyword.
        vals = list(args) + list(kwargs.values())
        items, items_mask, img_feats = vals[0], vals[1], vals[3]
        if items.shape[0] == 0:
            empty = img_feats.new_zeros((0, items.shape[1], self.d_model))
            return empty, items_mask
        # _encode_boxes calls pin_memory(), which fails for MPS ("storage on different
        # device"). Pinning only speeds up CUDA copies, so skip it during this call.
        real_pin = torch.Tensor.pin_memory
        torch.Tensor.pin_memory = lambda t, *a, **k: t
        try:
            return encode(self, *args, **kwargs)
        finally:
            torch.Tensor.pin_memory = real_pin

    return wrapper


def addmm_act_plain(activation, linear, mat1):
    """linear + activation, as sam3.perflib.fused.addmm_act, but kept in the input's dtype.

    Meta's version casts to bfloat16 and relies on CUDA autocast to cast back; without CUDA
    the bfloat16 output reaches float32 layers (CPU: dtype error; MPS: NaN scores, S1 run
    2026-11-24).
    """
    y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        return torch.relu(y)
    if activation in (torch.nn.functional.gelu, torch.nn.GELU):
        return torch.nn.functional.gelu(y)
    raise ValueError(f"Unexpected activation {activation}")
