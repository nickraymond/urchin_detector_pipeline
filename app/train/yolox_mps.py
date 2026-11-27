"""Run-time fixes that let Megvii's YOLOX loss run on the Apple GPU (and on torch 2.10 at all off
CUDA). YOLOX casts with legacy type strings (`tensor.type()` gives 'torch.mps.FloatTensor', which
torch refuses: seen 2026-11-27, "invalid type: 'torch.mps.FloatTensor'"). Three functions are
replaced with device-aware equivalents; the maths is unchanged. Adapted from Nick's earlier work,
`ADIN_SPI_OpenMV/ml/yolox_urchin/model.py` `_patch_head_for_mps` @ 994b8c2. The checkout under
tools_dir stays unmodified (D51). Training only: export never calls these (app/modeldefs).

These fixes were thought not to be enough (PR 72: same loss on both devices, class-branch
gradients 1e6-1e7 times larger on the Apple GPU, NaN within a few steps). The cause was not in
the loss: the image batch was channels-last in memory, and with that layout YOLOX's training
graph gets wrong weight gradients on the Apple GPU (measured 2026-11-27, torch 2.10.0 and 2.13.0,
already with the sum of YOLOX's first layer's output as the loss; a plain conv stack is right in
both layouts, so the operation at fault is not isolated). app/train/yolox.py now builds a
contiguous batch and checks the two devices against each other before every start (D65)."""

from __future__ import annotations

import torch


def apply() -> None:
    import yolox.models.yolo_head as head_mod
    import yolox.utils.boxes as boxes_mod
    from yolox.models.losses import IOUloss
    from yolox.models.yolo_head import YOLOXHead
    from yolox.utils import meshgrid

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch, n_ch = output.shape[0], 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).to(output.device, output.dtype)
            self.grids[k] = grid
        output = output.view(batch, 1, n_ch, hsize, wsize).permute(0, 1, 3, 4, 2).reshape(
            batch, hsize * wsize, -1)
        grid = grid.view(1, -1, 2).to(output.device, output.dtype)
        output = output.clone()
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def bboxes_iou(a, b, xyxy=True):
        if a.shape[1] != 4 or b.shape[1] != 4:
            raise IndexError
        if xyxy:
            tl, br = torch.max(a[:, None, :2], b[:, :2]), torch.min(a[:, None, 2:], b[:, 2:])
            area_a, area_b = torch.prod(a[:, 2:] - a[:, :2], 1), torch.prod(b[:, 2:] - b[:, :2], 1)
        else:
            tl = torch.max(a[:, None, :2] - a[:, None, 2:] / 2, b[:, :2] - b[:, 2:] / 2)
            br = torch.min(a[:, None, :2] + a[:, None, 2:] / 2, b[:, :2] + b[:, 2:] / 2)
            area_a, area_b = torch.prod(a[:, 2:], 1), torch.prod(b[:, 2:], 1)
        inside = (tl < br).to(tl.dtype).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * inside
        return area_i / (area_a[:, None] + area_b - area_i)

    def iou_forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        pred, target = pred.view(-1, 4), target.view(-1, 4)
        tl = torch.max(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
        br = torch.min(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
        area_p, area_g = torch.prod(pred[:, 2:], 1), torch.prod(target[:, 2:], 1)
        inside = (tl < br).to(tl.dtype).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * inside
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)
        if self.loss_type == "giou":
            c_tl = torch.min(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
            c_br = torch.max(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
            area_c = torch.prod(c_br - c_tl, 1)
            loss = 1 - (iou - (area_c - area_u) / area_c.clamp(1e-16)).clamp(min=-1.0, max=1.0)
        else:
            loss = 1 - iou ** 2
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss

    YOLOXHead.get_output_and_grid = get_output_and_grid
    boxes_mod.bboxes_iou = bboxes_iou
    head_mod.bboxes_iou = bboxes_iou
    IOUloss.forward = iou_forward
