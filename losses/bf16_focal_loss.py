"""BFloat16-safe wrapper around mmdet FocalLoss.

mmcv's sigmoid_focal_loss CUDA kernel only supports float16/float32.
Under bfloat16 AMP the head produces bfloat16 logits which crash the
kernel. This wrapper casts pred (and target if floating) to float32
before calling the parent forward, then returns the float32 loss.
"""

import torch

try:
    from mmdet.models.losses import FocalLoss
    from mmdet.registry import MODELS
except ImportError:  # pragma: no cover - local config-parse only
    from mmengine.model import BaseModel as FocalLoss
    from mmengine.registry import MODELS


@MODELS.register_module()
class BF16SafeFocalLoss(FocalLoss):

    def forward(self, pred, target, weight=None,
                avg_factor=None, reduction_override=None):
        if pred.dtype == torch.bfloat16:
            pred = pred.float()
            if target.is_floating_point():
                target = target.float()
            if weight is not None and weight.dtype == torch.bfloat16:
                weight = weight.float()
        return super().forward(
            pred, target, weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
        )
