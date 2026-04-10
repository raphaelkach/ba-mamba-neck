"""V3 - SSM-Neck (MambaFPN, Liang et al. 2026).

Inherits ``configs/fpn.py`` wholesale - only the ``model.neck`` dict is
replaced (via ``_delete_=True``). Backbone, head, optimizer, schedule,
dataloader and runtime settings are identical to V1/V2.
"""

_base_ = ['./fpn.py']

model = dict(
    neck=dict(
        _delete_=True,
        type='MambaNeck',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
)
