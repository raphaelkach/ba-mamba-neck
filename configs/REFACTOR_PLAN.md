# Config Inheritance Refactor Plan

## Ist-Analyse (aktuelle Struktur)

```
_base_/dataset.py   (101 Zeilen) - Dataloaders, Pipelines, Evaluators
_base_/schedule.py  ( 49 Zeilen) - Optimizer (AMP), LR-Schedule, max_epochs
_base_/runtime.py   ( 72 Zeilen) - Hooks, EMA, Reproducibility, custom_imports

fpn.py (91 Zeilen)
  _base_ = [dataset, schedule, runtime]
  + data_preprocessor (ImageNet mean/std)
  + model = dict(
      type='ATSS',
      backbone=ResNet-50 (frozen),
      neck=FPN,              <-- EINZIGER variierender Block
      bbox_head=ATSSHead,
      train_cfg, test_cfg
    )

aifi.py (19 Zeilen)
  _base_ = ['./fpn.py']     <-- erbt ALLES von fpn.py
  model.neck = AifiNeck (_delete_=True)

mamba.py (18 Zeilen)
  _base_ = ['./fpn.py']     <-- erbt ALLES von fpn.py
  model.neck = MambaNeck (_delete_=True)
```

Problem: fpn.py ist gleichzeitig V1-Config UND gemeinsame Basis
fuer V2/V3. Das ist semantisch falsch - fpn.py sollte NUR das
FPN-Neck definieren, nicht die gesamte Modell-/Preprocessing-Basis.

## Varianten-Bewertung

### Variante A: _base_/model.py (neckless)
Neue Datei `_base_/model.py` mit data_preprocessor + model-Block
OHNE neck. Alle drei top-level Configs erben von _base_/ und
definieren NUR ihren neck-Block.

- Risiko: **GERING** - mmengine merged dicts rekursiv, ein
  fehlender neck in _base_ + neck in child = saubere Komposition.
- Aufwand: ~95 Zeilen neue Datei, ~85 Zeilen aus fpn.py entfernt,
  aifi.py und mamba.py verlieren _delete_=True (nicht mehr noetig).
- Lesbarkeit: **HOCH** - jede Config hat eine klare Rolle.
  Kein "V1 ist auch die Basis fuer V2/V3".

### Variante B: FPN bleibt Standard in _base_/model.py
model.py enthaelt das komplette model-dict MIT neck=FPN.
fpn.py wird trivial (nur _base_). aifi/mamba ueberschreiben
per _delete_=True.

- Risiko: **GERING** - funktional identisch zu heute.
- Aufwand: ~95 Zeilen neue Datei, fpn.py wird 10 Zeilen.
- Lesbarkeit: **MITTEL** - FPN ist "versteckt" in _base_,
  nicht sofort klar welches Neck V1 verwendet.

### Variante C: model.py mit neck=None Platzhalter
model.py definiert neck=dict() als leeren Platzhalter.
Jeder child-Config ueberschreibt neck komplett.

- Risiko: **MITTEL** - leeres dict koennte Probleme machen
  wenn mmdet versucht es zu instanziieren vor dem Merge.
- Aufwand: aehnlich wie A.
- Lesbarkeit: **MITTEL** - neck=dict() ist verwirrend.

## Empfehlung: Variante A

Variante A ist die sauberste Loesung. Das model-dict in
_base_/model.py enthaelt backbone, head, train_cfg, test_cfg
und data_preprocessor - aber KEIN neck. Jede der drei top-level
Configs erbt von allen vier _base_-Dateien und fuegt NUR ihr
neck hinzu. Kein _delete_=True noetig, keine Verwechslung
zwischen "Basis" und "V1-Experiment". Die gemergte Config
bleibt byte-identisch (verifizierbar via Baseline-Snapshots).

## Umbau-Schrittliste

### Neue Dateien
| Datei | Zeilen | Inhalt |
|---|---|---|
| configs/_base_/model.py | ~80 | data_preprocessor + model (ohne neck) |

### Modifizierte Dateien
| Datei | Aenderung |
|---|---|
| configs/fpn.py | _base_ erweitern um model.py, data_preprocessor + model-Rumpf entfernen, NUR neck=FPN definieren |
| configs/aifi.py | _base_ auf _base_/* (nicht fpn.py), NUR neck=AifiNeck, KEIN _delete_=True |
| configs/mamba.py | _base_ auf _base_/* (nicht fpn.py), NUR neck=MambaNeck, KEIN _delete_=True |

### Geloeschte Dateien
Keine.

### Reihenfolge
1. _base_/model.py anlegen (aus fpn.py extrahiert, ohne neck)
2. fpn.py umschreiben: _base_ = 4 Dateien, nur neck=FPN
3. aifi.py umschreiben: _base_ = 4 Dateien, nur neck=AifiNeck
4. mamba.py umschreiben: _base_ = 4 Dateien, nur neck=MambaNeck
5. Baseline-Diff ausfuehren: scripts/export_merged_configs.py
6. diff gegen configs/baseline_snapshots/ - muss leer sein
7. scripts/check_ceteris_paribus.py ausfuehren
8. Commit

## Risiken

### mmengine Merge-Semantik
- model=dict(neck=...) in child + model=dict(backbone=..., head=...)
  in _base_: mmengine merged diese rekursiv. Neck wird HINZUGEFUEGT,
  nicht ueberschrieben. Das ist genau was wir wollen.
- KEIN _delete_=True noetig, da _base_/model.py gar kein neck hat.

### data_preprocessor
- data_preprocessor MUSS in _base_/model.py als eigene
  Top-Level-Variable definiert werden, GENAU WIE HEUTE in fpn.py:
    data_preprocessor = dict(type='DetDataPreprocessor', ...)
  Die Referenzierung via model=dict(..., data_preprocessor=data_preprocessor)
  ist Python-Syntax im Quellcode, fuehrt aber dazu, dass die gemergte
  Config data_preprocessor als eigenen Top-Level-Key enthaelt.
  Die Baseline-Snapshots bestaetigen dies (34 Top-Level-Keys,
  data_preprocessor und model nebeneinander). Verschachtelung von
  data_preprocessor INNERHALB model=dict() wuerde die gemergte
  Config-Struktur aendern und Phase 5 failen lassen.

### Stolperfalle: _base_ Liste vs. Einzeldatei
- _base_ = ['_base_/dataset.py', '_base_/schedule.py', '_base_/runtime.py', '_base_/model.py']
  funktioniert, aber die Reihenfolge kann bei Konflikten relevant sein.
  Da die vier _base_-Dateien disjunkte Keys definieren, ist die
  Reihenfolge egal.

### Tests nach Umbau
1. scripts/export_merged_configs.py -> diff gegen Baseline (PRIMAER)
2. scripts/check_ceteris_paribus.py (Exit 0)
3. PYTHONPATH=. python -c "Config.fromfile('configs/fpn.py')" etc. (Parse-Check)
4. Manuell: model.data_preprocessor korrekt gemergt?
5. Anzahl Top-Level-Keys vergleichen (muss exakt 34 sein pro Config)
6. Anzahl rekursiver Keys vergleichen (muss 297 bei aifi/mamba
   und 299 bei fpn sein, wie in den Baselines)

## Non-Negotiable Invariants

1. data_preprocessor bleibt ein eigener Top-Level-Key in der
   gemergten Config, neben model. NICHT verschachtelt in
   model=dict(...). In _base_/model.py als Top-Level-Variable.

2. Die Reihenfolge der _base_-Liste in fpn.py / aifi.py / mamba.py
   muss deterministisch und in allen drei Configs identisch sein:
   _base_ = ['_base_/dataset.py', '_base_/schedule.py',
             '_base_/runtime.py', '_base_/model.py']

3. _base_/model.py enthaelt KEINEN neck-Key. Der neck wird
   ausschliesslich in fpn.py / aifi.py / mamba.py definiert.

4. Alle anderen Felder (backbone, bbox_head, train_cfg, test_cfg,
   init_cfg, data_preprocessor) bleiben in _base_/model.py
   byte-identisch zu dem, was heute in fpn.py steht.
   Kein Refactoring der Werte, nur Verschiebung.

5. Keine Top-Level-Variablen (data_preprocessor etc.) verbleiben
   in fpn.py / aifi.py / mamba.py. Diese wandern vollstaendig
   nach _base_/model.py.

6. Nach dem Refactor MUSS in fpn.py / aifi.py / mamba.py
   _base_ = [liste der 4 _base_-Dateien] stehen.
   KEIN Eintrag './fpn.py' mehr in aifi.py oder mamba.py.

7. Die gemergte Config muss nach dem Refactor 34 Top-Level-Keys
   haben (identisch zu den Baseline-Snapshots). Anders = FAIL.
