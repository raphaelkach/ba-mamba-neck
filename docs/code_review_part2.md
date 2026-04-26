# Code-Review Teil 2/4 — Code-Werte

**Datum:** 2026-04-26
**Commit:** 039c2c5 (Branch: claude/review-mamba-neck-part1-MPXyt)
**Referenz:** `configs/baseline_snapshots/fpn_baseline.json`
(AIFI- und Mamba-Snapshots laut Teil 1 byte-identisch außerhalb `model.neck`)

## Zusammenfassung
Alle 52 in der Aufgabe geforderten Werte stimmen exakt mit den Erwartungen
überein — Optimizer, Schedule, Detektor, Backbone, Daten/Pipeline und
Reproduzierbarkeit sind sauber konfiguriert. Einzige Auffälligkeit:
`env_cfg.cudnn_benchmark=True` koexistiert mit `randomness.deterministic=True`,
was eine widersprüchliche Konfigurationsangabe ist (WICHTIG, dokumentationswert
für die BA-Methodikkapitel). Trainingspipeline ist frei von Mosaic/MixUp/
ColorJitter, Augmentation entspricht der Spezifikation.

## Befund-Übersicht
- KRITISCH: 0
- WICHTIG: 1  (cudnn_benchmark=True neben deterministic=True)
- INFO: 1     (siehe Bereich 4: Detail RandomChoiceResize)

## Bereich 1: Optimizer & Schedule

| Pfad | Erwartet | Gefunden | Status |
|---|---|---|---|
| optim_wrapper.optimizer.type | `'AdamW'` | `'AdamW'` | ✅ |
| optim_wrapper.optimizer.lr | `2e-4` | `0.0002` | ✅ |
| optim_wrapper.optimizer.weight_decay | `0.05` | `0.05` | ✅ |
| optim_wrapper.dtype | `'bfloat16'` | `'bfloat16'` | ✅ |
| optim_wrapper.clip_grad.max_norm | `0.1` | `0.1` | ✅ |
| param_scheduler[0].type | `'LinearLR'` | `'LinearLR'` | ✅ |
| param_scheduler[0].start_factor | `1e-3` | `0.001` | ✅ |
| param_scheduler[0].end | `500` | `500` | ✅ |
| param_scheduler[1].type | `'CosineAnnealingLR'` | `'CosineAnnealingLR'` | ✅ |
| param_scheduler[1].T_max | `24` | `24` | ✅ |
| param_scheduler[1].eta_min | `1e-6` | `1e-06` | ✅ |

Zusätzlich notiert (nicht verlangt, aber relevant): `optim_wrapper.type` =
`AmpOptimWrapper` mit `dtype='bfloat16'` — konsistent mit der Wahl von
`BF16SafeFocalLoss`. `param_scheduler[0].by_epoch=False` und
`param_scheduler[0].begin=0` sowie `param_scheduler[1].convert_to_iter_based=True`
sind korrekt für 500-Iter-Warmup gefolgt von Cosine über 24 Epochen.

## Bereich 2: Detektor

| Pfad | Erwartet | Gefunden | Status |
|---|---|---|---|
| model.type | `'ATSS'` | `'ATSS'` | ✅ |
| model.bbox_head.type | `'ATSSHead'` | `'ATSSHead'` | ✅ |
| model.bbox_head.num_classes | `10` | `10` | ✅ |
| model.bbox_head.stacked_convs | `4` | `4` | ✅ |
| model.bbox_head.feat_channels | `256` | `256` | ✅ |
| model.bbox_head.anchor_generator.ratios | `[1.0]` | `[1.0]` | ✅ |
| model.bbox_head.anchor_generator.scales_per_octave | `1` | `1` | ✅ |
| model.bbox_head.anchor_generator.octave_base_scale | `8` | `8` | ✅ |
| model.bbox_head.anchor_generator.strides | `[8,16,32,64,128]` | `[8,16,32,64,128]` | ✅ |
| model.bbox_head.loss_cls.type | `'BF16SafeFocalLoss'` | `'BF16SafeFocalLoss'` | ✅ |
| model.bbox_head.loss_cls.alpha | `0.25` | `0.25` | ✅ |
| model.bbox_head.loss_cls.gamma | `2.0` | `2.0` | ✅ |
| model.bbox_head.loss_bbox.type | `'GIoULoss'` | `'GIoULoss'` | ✅ |
| model.bbox_head.loss_bbox.loss_weight | `2.0` | `2.0` | ✅ |
| model.train_cfg.assigner.type | `'ATSSAssigner'` | `'ATSSAssigner'` | ✅ |
| model.train_cfg.assigner.topk | `9` | `9` | ✅ |
| model.test_cfg.nms_pre | `1000` | `1000` | ✅ |
| model.test_cfg.score_thr | `0.05` | `0.05` | ✅ |
| model.test_cfg.nms.iou_threshold | `0.6` | `0.6` | ✅ |
| model.test_cfg.max_per_img | `100` | `100` | ✅ |

Zusätzlich notiert: `model.bbox_head.loss_cls.use_sigmoid=True` und
`model.bbox_head.loss_centerness` (CrossEntropy, sigmoid, weight=1.0) sind
gesetzt — kanonische FCOS/ATSS-Konfiguration. `bbox_coder` ist
`DeltaXYWHBBoxCoder` (Standard für ATSS in MMDetection 3.x).

## Bereich 3: Backbone

| Pfad | Erwartet | Gefunden | Status |
|---|---|---|---|
| model.backbone.type | `'ResNet'` | `'ResNet'` | ✅ |
| model.backbone.depth | `50` | `50` | ✅ |
| model.backbone.frozen_stages | `4` | `4` | ✅ |
| model.backbone.norm_eval | `True` | `True` | ✅ |
| model.backbone.out_indices | `[0,1,2,3]` | `[0,1,2,3]` | ✅ |
| model.backbone.init_cfg.checkpoint | `'torchvision://resnet50'` | `'torchvision://resnet50'` | ✅ |

Zusätzlich notiert: `init_cfg.type='Pretrained'`, `norm_cfg={type:'BN', requires_grad:False}`,
`num_stages=4`, `style='pytorch'` — sind alle konsistent mit einem komplett
gefrorenen ImageNet-vortrainierten ResNet-50 (frozen_stages=4 friert alle 4
Stages ein). Dies entspricht exakt der Vorgabe aus CLAUDE.md.

## Bereich 4: Daten + Pipeline

| Pfad | Erwartet | Gefunden | Status |
|---|---|---|---|
| train_dataloader.batch_size | `16` | `16` | ✅ |
| val_dataloader.batch_size | `1` | `1` | ✅ |
| train_dataloader.num_workers | `8` | `8` | ✅ |
| train_dataloader.dataset.ann_file | `'annotations/train_sliced.json'` | `'annotations/train_sliced.json'` | ✅ |
| val_dataloader.dataset.ann_file | `'annotations/val_unsliced.json'` | `'annotations/val_unsliced.json'` | ✅ |
| train_pipeline RandomChoiceResize | 11 Skalen 480²–800² (Δ32) | 11 Skalen, exakt `[480²,512²,544²,576²,608²,640²,672²,704²,736²,768²,800²]`, gleichförmiger Schritt 32 verifiziert | ✅ |
| train_pipeline RandomFlip | `prob=0.5` | `prob=0.5` | ✅ |
| train_pipeline (Mosaic/MixUp/ColorJitter) | nicht vorhanden | Pipeline = `LoadImageFromFile, LoadAnnotations, RandomChoiceResize, RandomFlip, PackDetInputs` — keine der drei verbotenen Augmentations | ✅ |
| train_dataloader.dataset.filter_cfg.filter_empty_gt | `True` | `True` | ✅ |
| train_dataloader.dataset.filter_cfg.min_size | `4` | `4` | ✅ |
| max_epochs | `24` | `24` | ✅ |
| train_cfg.val_interval | `2` | `2` | ✅ |

INFO: `RandomChoiceResize` hat `keep_ratio=True` mit quadratischen Zielen.
Bei nicht-quadratischen Eingängen wirkt nur die größere Achse als Limit
(MMCV-Verhalten); das ist beabsichtigt für 640-zentrierten Multi-Scale-Train,
sollte aber im BA-Methodenkapitel erwähnt werden, damit Reviewer nicht
fälschlich eine harte 640²-Skalierung erwarten.

Zusätzlich notiert: `train_dataloader.batch_sampler.type='AspectRatioBatchSampler'`,
`persistent_workers=True`, `pin_memory=True`, `prefetch_factor=4`. Test- und
Val-Dataloader nutzen `num_workers=2` mit `batch_size=1` (Single-Image-Eval auf
unsliced 640²-Resize) — konsistent. Test-Pipeline ist `Resize 640` ohne Random-Augmentation.

## Bereich 5: Reproduzierbarkeit

| Pfad | Erwartet | Gefunden | Status |
|---|---|---|---|
| randomness.deterministic | `True` | `True` | ✅ |
| env_cfg.cudnn_benchmark | `True` (Konflikt prüfen) | `True` | ⚠️ WICHTIG (siehe unten) |
| custom_hooks (EMAHook).type | `'EMAHook'` | `'EMAHook'` | ✅ |
| custom_hooks (EMAHook).ema_type | `'ExpMomentumEMA'` | `'ExpMomentumEMA'` | ✅ |
| custom_hooks (EMAHook).momentum | `0.0002` | `0.0002` | ✅ |
| default_hooks.checkpoint.save_best | `'coco/bbox_mAP'` | `'coco/bbox_mAP'` | ✅ |
| default_hooks.checkpoint.interval | `2` | `2` | ✅ |
| default_hooks.checkpoint.max_keep_ckpts | `3` | `3` | ✅ |
| auto_scale_lr.enable | `False` | `False` | ✅ |
| auto_scale_lr.base_batch_size | `16` | `16` | ✅ |

### WICHTIG: cudnn_benchmark=True + deterministic=True

Die Konfiguration enthält gleichzeitig:

```json
"randomness": { "deterministic": true, "seed": null },
"env_cfg":   { "cudnn_benchmark": true, ... }
```

Diese beiden Aussagen sind konzeptionell widersprüchlich: cuDNN-Benchmark
wählt zur Laufzeit die schnellste Convolution-Implementation pro Eingangsform
und führt damit a) eine workload-abhängige Algorithmenauswahl und b) potenziell
nichtdeterministische Algorithmen ein. PyTorchs deterministischer Modus
verlangt dagegen explizit `cudnn.deterministic=True` und `cudnn.benchmark=False`.

Im Runtime-Verhalten von MMEngine wird `set_randomness(deterministic=True)`
nach `setup_env(env_cfg)` ausgeführt, sodass die spätere Setzung in der Praxis
gewinnen sollte — d.h. das tatsächliche Verhalten wäre vermutlich
deterministisch, und `cudnn_benchmark=True` wirkt als toter Eintrag. Das ist
aber **nicht im Snapshot beweisbar** und sollte für die Reproduzierbarkeits-
Argumentation der BA entweder
1. durch Bereinigung auf `cudnn_benchmark=False` aufgelöst werden, oder
2. im Methodenkapitel explizit kommentiert werden, mit Verweis auf die
   tatsächliche `set_randomness`-Reihenfolge.

Zur Reproduzierbarkeit zusätzlich notiert (nicht verlangt, aber relevant):
- `randomness.seed=null` im Snapshot ist erwartungskonform — der Seed wird
  pro Run via Notebook-Variable gesetzt (10 Seeds, siehe CLAUDE.md). Der
  Snapshot fixiert daher nur die *deterministische Strategie*, nicht den Seed.
- `resume=true` im Snapshot ermöglicht Wiederaufnahme abgebrochener Trainings —
  konsistent mit der Notebook-Sync-Regel und unproblematisch, solange Seeds
  identisch bleiben.
- `EpochTimerHook` als zusätzlicher `custom_hook` (Priority `LOW`) ist
  konsistent mit `hooks/epoch_timer_hook.py` aus Teil 1.
- `default_hooks.checkpoint.rule='greater'` und `by_epoch=True` sind plausibel
  für `save_best='coco/bbox_mAP'`.
