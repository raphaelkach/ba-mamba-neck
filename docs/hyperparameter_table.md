# Hyperparameter Table

_Manually curated - these are design decisions, not derived metrics._
_Identical for V1 (FPN), V2 (AifiNeck) and V3 (MambaNeck) per the_
_ceteris-paribus rule in CLAUDE.md._

| Parameter | Value | Source / justification |
|---|---|---|
| Backbone | ResNet-50 (torchvision ImageNet) | Most commonly used in the cited literature (Lin 2017, Tan 2020, Zhu 2021, Carion 2020, Liang 2026) |
| Frozen stages | 4 (all) | Isolate neck as the only independent variable |
| Norm eval | True | Freeze BN statistics with the backbone |
| Head | FCOS + ATSS | Anchor-free; ATSS raises AP_S by +2.9 pp (Zhang et al. 2020, p. 9760) |
| Num classes | 10 | VisDrone-DET after dropping `ignored` and `others` |
| Optimizer | AdamW | Standard for Transformer- and Mamba-based detectors |
| Learning rate | 1e-4 | MMDetection default for AdamW + ResNet-50 necks |
| Weight decay | 0.05 | Standard for AdamW in detection (RT-DETR, DETR) |
| Schedule | CosineAnnealingLR, 24 epochs | 2x schedule, matches RT-DETR / ATSS baselines |
| Warmup | LinearLR, 500 iterations | Linear warmup from 1e-3 * base_lr |
| Grad clip | max_norm = 0.1 | Stabilizes mixed conv/attention/ssm training |
| Batch size | 8 | A100 40GB VRAM compatible at 640x640 |
| Input size | 640 x 640 | Matches SAHI slice size |
| Augmentation | Resize(keep_ratio) + RandomFlip(0.5) | Minimal - no Mosaic / MixUp to preserve object-size distribution |
| Data preprocessor | ImageNet mean/std, bgr_to_rgb, pad_size_divisor=32 | Standard MMDet `DetDataPreprocessor` |
| Seeds | 42, 123, 456, 789, 1024, 2048, 3407, 4096, 5555, 7777 | 10 seeds - Friedman test requires >= 5 |
| Deterministic | True | `torch.backends.cudnn.deterministic=True` |
| Val interval | every 2 epochs | Balance between compute and curve resolution |
| Checkpoint interval | every 2 epochs, keep best `coco/bbox_mAP`, max 3 | Disk-budget friendly on Colab |
| Primary metric | AP_S (COCO small, area < 32^2) | Research question is about small objects |
| Statistics | Friedman + Nemenyi post-hoc | Non-parametric, multi-sample (Demsar 2006) |
| Framework | MMDetection 3.x + MMEngine | As fixed in CLAUDE.md |
| Hardware | Google Colab A100 | As fixed in CLAUDE.md |
