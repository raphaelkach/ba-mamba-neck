"""V3 - MambaNeck (SSM-based FPN, Liang et al. 2026)."""

_base_ = [
    './_base_/dataset.py',
    './_base_/schedule.py',
    './_base_/runtime.py',
    './_base_/model.py',
]

model = dict(
    neck=dict(
        type='MambaNeck',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
)
