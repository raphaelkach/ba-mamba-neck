"""V1 - FPN (CNN-Neck Baseline, Lin et al. 2017)."""

_base_ = [
    './_base_/dataset.py',
    './_base_/schedule.py',
    './_base_/runtime.py',
    './_base_/model.py',
]

model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
    ),
)
