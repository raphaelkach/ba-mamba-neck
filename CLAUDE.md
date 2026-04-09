# BA Mamba-Neck — Kontrollierter Dreifach-Vergleich

## Forschungsfrage
Welchen Einfluss hat die Wahl des Fusionsmoduls (CNN-FPN vs.
Transformer-MHSA vs. Vision-Mamba-SSM) auf die Detektionsleistung
bei kleinen Objekten in hochaufgelösten Luftbilddaten?

## Experimentaldesign
- Backbone: ResNet-50 (ImageNet-Pretrained, frozen_stages=4) — FIX
  Begründung: Meistverwendetes Backbone in der zitierten Literatur
  (Lin 2017, Tan 2020, Zhu 2021, Carion 2020, Liang 2026).
  Frozen, damit NUR das Neck als unabhängige Variable variiert wird.
- Head: FCOS+ATSS (Tian et al. 2019 + Zhang et al. 2020) — FIX
  Begründung: Anchor-frei (kein Anchor-Tuning nötig bei extremer
  Skalierungsvarianz 10–1000px). ATSS adaptiver Schwellenwert
  steigert AP_S um 2,9 Pp (Zhang et al. 2020, S. 9760).
- Neck (unabhängige Variable, 3 Stufen):
  V1: FPN (Lin et al. 2017) — CNN-Baseline
  V2: Efficient Hybrid Encoder (Zhao et al. 2024, RT-DETR) —
      adaptiert als Transformer-Neck. AIFI (MHSA auf C5) + CCFM
      (CNN Cross-Scale Fusion). Gewählt weil RT-DETR den höchsten
      AP_S (34,8) in der DETR-Progression erreicht.
  V3: MambaFPN-Architektur (Liang et al. 2026) als Vorlage —
      SSM-Neck. VSS-Blocks mit Cross-Scan (4 Richtungen) ersetzen
      Conv-Layers im FPN. Liang berichten AP_S +2,7 Pp auf COCO.
      Ob das auf Luftbilddaten hält, ist die offene Frage.
- Datensatz: VisDrone-DET 2019 (Zhu et al. 2022)
  10 Klassen, achsparallele Boxen, 50–100 Objekte/Bild.
- Seeds: 42, 123, 456, 789, 1024, 2048, 3407, 4096, 5555, 7777
- Statistik: Friedman-Test + Nemenyi Post-hoc (Demšar 2006)
- Primärmetrik: AP_S (Average Precision für kleine Objekte)
- Framework: MMDetection 3.x
- Hardware: Google Colab A100

## Ceteris-Paribus-Regel
Zwischen V1, V2 und V3 unterscheidet sich NUR neck=dict(...).
Backbone, Head, Optimizer, LR-Schedule, Augmentationen,
Batchgröße, Epochenzahl und Seeds sind IDENTISCH.

## Neck-Interface (alle drei Necks)
Input:  List[Tensor] — 4 Feature-Maps {C2, C3, C4, C5} vom ResNet
Output: Tuple[Tensor] — 5 Feature-Maps {P3, P4, P5, P6, P7} für ATSSHead

## Dateien NICHT anfassen
- configs/_base_/ → nur ändern wenn explizit gewünscht
- Backbone-Gewichte → ImageNet-Pretrained, nicht feintunen

## Code-Stil
- Type Hints, Google-Style Docstrings
- Configs in MMDetection-Python-Style (nicht YAML)
- Logging über mmengine.logging
