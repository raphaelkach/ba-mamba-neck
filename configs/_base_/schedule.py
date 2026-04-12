"""Optimizer + LR-Schedule (geteilt zwischen V1/V2/V3).

24 Epochen, AdamW, CosineAnnealingLR mit 500-iter Linear-Warmup,
grad_clip max_norm=0.1. Ceteris-paribus fuer alle drei Necks.
"""

# Training loop
max_epochs = 24

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

# LR schedule: 500 iter linear warmup -> cosine annealing over 24 epochs.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        eta_min=1e-6,
        convert_to_iter_based=True,
    ),
]

auto_scale_lr = dict(enable=False, base_batch_size=16)
