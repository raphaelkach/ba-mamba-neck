"""Runtime-Config: Hooks, Logger, Checkpoints, Reproducibility.

Seed wird ueber CLI gesetzt:
    python tools/train.py configs/fpn.py --cfg-options randomness.seed=42
"""

default_scope = 'mmdet'

# Register project-local modules (custom hooks, custom necks) with the
# MMDet registry before Runner setup.
custom_imports = dict(
    imports=['hooks', 'necks'],
    allow_failed_imports=False,
)

# Reproducibility
randomness = dict(seed=None, deterministic=True)

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        by_epoch=True,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

# Custom hook: Trainingszeit pro Epoche loggen.
custom_hooks = [
    dict(type='EpochTimerHook', priority='LOW'),
]

# Logging / Visualizer (WandB)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

# Misc
load_from = None
resume = False
work_dir = './work_dirs/'
