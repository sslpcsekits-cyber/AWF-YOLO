"""Losses: Inner-SIoU box regression and class-imbalance alternatives for the classification branch."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _xywh2xyxy(b):
    xy, wh = b[..., :2], b[..., 2:4] / 2
    return torch.cat((xy - wh, xy + wh), -1)


def inner_siou(box1, box2, ratio=0.75, eps=1e-7):
    """Inner-SIoU similarity for xyxy boxes of shape (n, 4); returns (n, 1). Loss = 1 - value.

    value = IoU_inner - (distance_cost + shape_cost) / 2, i.e. the SCYLLA-IoU (SIoU) loss with its IoU
    term replaced by the IoU of auxiliary inner boxes (same centres, sides scaled by `ratio`), which is
    the Inner-IoU formulation L_Inner-SIoU = L_SIoU + IoU - IoU_inner.
    """
    x1, y1, x2, y2 = box1.chunk(4, -1)
    X1, Y1, X2, Y2 = box2.chunk(4, -1)
    w1, h1 = (x2 - x1).clamp(min=eps), (y2 - y1).clamp(min=eps)
    w2, h2 = (X2 - X1).clamp(min=eps), (Y2 - Y1).clamp(min=eps)
    cx1, cy1, cx2, cy2 = (x1 + x2) / 2, (y1 + y2) / 2, (X1 + X2) / 2, (Y1 + Y2) / 2

    # IoU of the inner (auxiliary) boxes
    iw = (torch.min(cx1 + w1 * ratio / 2, cx2 + w2 * ratio / 2) - torch.max(cx1 - w1 * ratio / 2, cx2 - w2 * ratio / 2)).clamp(0)
    ih = (torch.min(cy1 + h1 * ratio / 2, cy2 + h2 * ratio / 2) - torch.max(cy1 - h1 * ratio / 2, cy2 - h2 * ratio / 2)).clamp(0)
    inter = iw * ih
    iou_inner = inter / ((w1 * h1 + w2 * h2) * ratio ** 2 - inter + eps)

    # SIoU angle, distance and shape costs (computed on the original boxes)
    cw = torch.max(x2, X2) - torch.min(x1, X1)
    ch = torch.max(y2, Y2) - torch.min(y1, Y1)
    s_cw, s_ch = cx2 - cx1, cy2 - cy1
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2) + eps
    sin_a1, sin_a2 = s_cw.abs() / sigma, s_ch.abs() / sigma
    sin_a = torch.where(sin_a1 > math.sqrt(2) / 2, sin_a2, sin_a1).clamp(max=1 - eps)
    angle = torch.cos(torch.arcsin(sin_a) * 2 - math.pi / 2)
    gamma = angle - 2
    dist = 2 - torch.exp(gamma * (s_cw / (cw + eps)) ** 2) - torch.exp(gamma * (s_ch / (ch + eps)) ** 2)
    om_w = (w1 - w2).abs() / torch.max(w1, w2)
    om_h = (h1 - h2).abs() / torch.max(h1, h2)
    shape = (1 - torch.exp(-om_w)) ** 4 + (1 - torch.exp(-om_h)) ** 4
    return iou_inner - 0.5 * (dist + shape)


def enable_inner_siou(ratio=0.75):
    """Make Ultralytics' detection box loss use Inner-SIoU instead of CIoU (training loss only).

    BboxLoss calls the module-level name `bbox_iou` in ultralytics.utils.loss; we replace that name only,
    so the label assigner and all evaluation metrics keep using the standard IoU.
    """
    import ultralytics.utils.loss as L
    if not hasattr(L, "bbox_iou"):
        raise RuntimeError("This Ultralytics version has no ultralytics.utils.loss.bbox_iou; Inner-SIoU hook failed.")
    original = getattr(L, "_awf_original_bbox_iou", L.bbox_iou)
    L._awf_original_bbox_iou = original

    def bbox_iou_inner_siou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, **kwargs):
        if xywh:
            box1, box2 = _xywh2xyxy(box1), _xywh2xyxy(box2)
        return inner_siou(box1, box2, ratio=ratio, eps=eps)

    L.bbox_iou = bbox_iou_inner_siou
    return original


class FocalBCE(nn.Module):
    """Element-wise focal binary cross-entropy (reduction='none'), drop-in for v8DetectionLoss.bce."""

    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, true):
        bce = F.binary_cross_entropy_with_logits(pred, true, reduction="none")
        p = pred.sigmoid()
        p_t = true * p + (1 - true) * (1 - p)
        a_t = true * self.alpha + (1 - true) * (1 - self.alpha)
        return bce * a_t * (1 - p_t) ** self.gamma


class WeightedBCE(nn.Module):
    """Class-weighted binary cross-entropy (reduction='none'); weights broadcast over the class axis."""

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, pred, true):
        return F.binary_cross_entropy_with_logits(pred, true, reduction="none") * self.w.to(pred.device, pred.dtype)
