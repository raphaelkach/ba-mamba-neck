"""Shared constants for the evaluation pipeline."""

NECKS = ['fpn', 'aifi', 'mamba']
SEEDS = [42, 123, 456, 789, 1024, 2048, 3407, 4096, 5555, 7777]
CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor',
]

COCO_SMALL = 32 * 32
COCO_MEDIUM = 96 * 96

COLORS = {'fpn': '#0072B2', 'aifi': '#56B4E9', 'mamba': '#009E73'}
NECK_LABELS = {'fpn': 'FPN (V1)', 'aifi': 'AIFI (V2)', 'mamba': 'Mamba (V3)'}

DEFAULT_DATA_ROOT = '/content/visdrone'
DEFAULT_CKPT_DIR = '/content/drive/MyDrive/ba'
DEFAULT_RESULTS_DIR = 'results'
