"""V2 - Transformer-Neck (AIFI + CCFM, RT-DETR, Zhao et al. 2024).

Inherits ``configs/fpn.py`` wholesale - backbone, head, optimizer,
schedule, augmentations, dataloader and all runtime settings come from
V1. The only difference is the ``model.neck`` dict; ``_delete_=True``
ensures the parent FPN neck is fully replaced (not merged).
"""

_base_ = ['./fpn.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='AifiNeck',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
)
