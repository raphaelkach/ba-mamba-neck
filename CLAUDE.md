# BA Mamba-Neck - Controlled Three-Way Comparison

## Research question
What is the effect of the fusion module (CNN-FPN vs. Transformer-MHSA
vs. Vision-Mamba-SSM) on detection performance for small objects in
high-resolution aerial imagery?

## Experimental design
- Backbone: ResNet-50 (ImageNet-pretrained, frozen_stages=4) - FIXED
  Most commonly used backbone in the cited literature
  (Lin 2017, Tan 2020, Zhu 2021, Carion 2020, Liang 2026).
  Frozen so that only the neck varies as the independent variable.
- Head: FCOS+ATSS (Tian et al. 2019 + Zhang et al. 2020) - FIXED
  Anchor-free (no anchor tuning needed for extreme scale variance
  10-1000px). ATSS adaptive threshold improves AP_S by 2.9 pp
  (Zhang et al. 2020, p. 9760).
- Neck (independent variable, 3 levels):
  V1: FPN (Lin et al. 2017) - CNN baseline
  V2: Efficient Hybrid Encoder (Zhao et al. 2024, RT-DETR) -
      adapted as Transformer neck. AIFI (MHSA on C5) + CCFM
      (CNN cross-scale fusion). Selected because RT-DETR achieves
      the highest AP_S (34.8) in the DETR progression.
  V3: MambaFPN architecture (Liang et al. 2026) as template -
      SSM neck. VSS blocks with cross-scan (4 directions) replace
      conv layers in FPN. Liang report AP_S +2.7 pp on COCO.
      Whether this holds on aerial data is the open question.
- Dataset: VisDrone-DET 2019 (Zhu et al. 2022)
  10 classes, axis-aligned boxes, 50-100 objects per image.
- Seeds: 42, 123, 456, 789, 1024, 2048, 3407, 4096, 5555, 7777
- Statistics: Friedman test + Nemenyi post-hoc (Demsar 2006)
- Primary metric: AP_S (average precision for small objects)
- Framework: MMDetection 3.x
- Hardware: Google Colab A100

## Ceteris-paribus rule
Between V1, V2 and V3, only neck=dict(...) differs.
Backbone, head, optimizer, LR schedule, augmentations,
batch size, number of epochs, and seeds are identical.

## Neck interface (all three necks)
Input:  List[Tensor] - 4 feature maps {C2, C3, C4, C5} from ResNet
Output: Tuple[Tensor] - 5 feature maps {P3, P4, P5, P6, P7} for ATSSHead

## Do not modify
- configs/_base_/ - only change when explicitly requested
- Backbone weights - ImageNet-pretrained, do not fine-tune

## Code style
- Type hints, Google-style docstrings
- Configs in MMDetection Python style (not YAML)
- Logging via mmengine.logging
