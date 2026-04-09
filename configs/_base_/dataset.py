"""VisDrone-DET 2019 dataset config (COCO-Format).

Geteilt zwischen V1 (FPN), V2 (AIFI+CCFM) und V3 (MambaFPN).
NUR Resize + RandomFlip — kein Mosaic, kein MixUp.

Begruendung: Mosaic und MixUp veraendern die Objektgroessen-
verteilung und wuerden den AP_S-Vergleich konfundieren. Die
Standard-Pipeline (Resize+Flip) ist in Lin et al. 2017 (FPN)
und Tan et al. 2020 (EfficientDet) etabliert.
"""

# -----------------------------------------------------------------------------
# Paths & metadata
# -----------------------------------------------------------------------------
dataset_type = 'CocoDataset'
data_root = '/content/visdrone/'

# 10 VisDrone classes (IDs 1..10 nach data/prepare.py).
metainfo = dict(
    classes=(
        'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
        'tricycle', 'awning-tricycle', 'bus', 'motor',
    ),
)

backend_args = None
image_scale = (640, 640)

# -----------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
    ),
]

# -----------------------------------------------------------------------------
# Dataloaders
# -----------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train_sliced.json',
        data_prefix=dict(img='sliced/train_sliced_images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=4),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val_unsliced.json',
        data_prefix=dict(img='raw/VisDrone2019-DET-val/images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

# -----------------------------------------------------------------------------
# Evaluators
# -----------------------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val_unsliced.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator
