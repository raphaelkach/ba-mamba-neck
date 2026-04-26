# Code-Review Teil 1/4 — Vollständigkeit + Konsistenz

**Datum:** 2026-04-26
**Commit:** 7767f2ce3a0508baf38e39a25fb0858ef42e8a9d
**Branch:** claude/review-mamba-neck-part1-MPXyt

## Zusammenfassung
Alle in der Aufgabe genannten Dateien sind vorhanden; keine fehlt. Die drei
Baseline-Snapshots (`fpn_baseline.json`, `aifi_baseline.json`,
`mamba_baseline.json`) sind außerhalb von `model.neck` byte-identisch
(SHA-256 nach Entfernen von `model.neck`:
`d49a4fee033e8e8b661f9a543c3fa5f50e9f858d7bc389282d2935185fe365c7` für alle
drei). Ceteris-paribus ist auf Snapshot-Ebene erfüllt.

## Befund-Übersicht
- KRITISCH: 0
- WICHTIG: 0
- INFO: 1

## Aufgabe 1: Strukturelle Vollständigkeit

| Datei | Existiert | Größe (Bytes) | Bemerkung |
|---|---|---|---|
| configs/fpn.py | ✅ | 383 | — |
| configs/aifi.py | ✅ | 350 | — |
| configs/mamba.py | ✅ | 333 | — |
| configs/_base_/dataset.py | ✅ | 2926 | — |
| configs/_base_/schedule.py | ✅ | 1268 | — |
| configs/_base_/runtime.py | ✅ | 1837 | — |
| configs/_base_/model.py | ✅ | 1947 | — |
| configs/baseline_snapshots/fpn_baseline.json | ✅ | 11981 | — |
| configs/baseline_snapshots/aifi_baseline.json | ✅ | 11924 | — |
| configs/baseline_snapshots/mamba_baseline.json | ✅ | 11925 | — |
| necks/aifi_neck.py | ✅ | 9060 | — |
| necks/mamba_neck.py | ✅ | 11224 | — |
| necks/__init__.py | ✅ | 160 | — |
| losses/__init__.py | ✅ | 80 | exportiert `BF16SafeFocalLoss` aus `losses.bf16_focal_loss` |
| losses/bf16_focal_loss.py | ✅ | 1192 | nicht explizit verlangt, aber Implementation der oben gelisteten Klasse |
| hooks/epoch_timer_hook.py | ✅ | 1206 | — |
| hooks/__init__.py | ✅ | 126 | — |
| scripts/test_necks.py | ✅ | 4900 | — |
| scripts/export_merged_configs.py | ✅ | 1677 | — |
| scripts/generate_architecture_report.py | ✅ | 12956 | — |
| scripts/generate_data_report.py | ✅ | 9605 | — |
| scripts/check_ceteris_paribus.py | ✅ | 2587 | — |
| notebooks/01_data.ipynb | ✅ | 13978 | — |
| notebooks/02_train_fpn.ipynb | ✅ | 22037 | — |
| notebooks/02_train_aifi.ipynb | ✅ | 22039 | — |
| notebooks/02_train_mamba.ipynb | ✅ | 22594 | — |
| notebooks/03_eval.ipynb | ✅ | 10006 | — |
| notebooks/04_erf.ipynb | ✅ | 4086 | — |
| data/prepare.py | ✅ | 11892 | — |
| eval/plot_style.py | ✅ | 2237 | — |
| eval/constants.py | ✅ | 591 | — |
| README.md | ✅ | 3550 | — |
| CLAUDE.md | ✅ | 2858 | — |
| requirements.txt | ✅ | 172 | — |

**Befund:** Keine fehlenden Dateien. Keine KRITISCH-Markierungen.

INFO: `losses/__init__.py` ist 80 Bytes; die eigentliche
`BF16SafeFocalLoss`-Implementation liegt erwartungsgemäß in
`losses/bf16_focal_loss.py` (1192 Bytes) und wird im `__init__` re-exportiert.

## Aufgabe 2: Ceteris-Paribus-Konsistenz

Methodik: Die drei JSON-Snapshots wurden geladen, der Schlüssel `model.neck`
entfernt, und anschließend mit `json.dumps(..., sort_keys=True)` deterministisch
serialisiert sowie SHA-256 gehasht. Zusätzlich wurde rekursiv ein Pfad-für-Pfad-
Diff berechnet (Top-Level-Keys plus alle verschachtelten Keys/Listen-Indizes).

### Hashes (SHA-256, ohne `model.neck`)

| Variante | Hash |
|---|---|
| FPN   | `d49a4fee033e8e8b661f9a543c3fa5f50e9f858d7bc389282d2935185fe365c7` |
| AIFI  | `d49a4fee033e8e8b661f9a543c3fa5f50e9f858d7bc389282d2935185fe365c7` |
| MAMBA | `d49a4fee033e8e8b661f9a543c3fa5f50e9f858d7bc389282d2935185fe365c7` |

### Diff-Ergebnis

```
FPN   vs AIFI  (ohne model.neck): 0 Differenzen
FPN   vs MAMBA (ohne model.neck): 0 Differenzen
AIFI  vs MAMBA (ohne model.neck): 0 Differenzen
```

**Resultat: byte-identisch außerhalb `model.neck`.**

### `model.neck` je Variante (zur Dokumentation, nicht Teil des Vergleichs)

```json
FPN:   {"add_extra_convs": "on_output", "in_channels": [256, 512, 1024, 2048], "num_outs": 5, "out_channels": 256, "start_level": 1, "type": "FPN"}
AIFI:  {"in_channels": [256, 512, 1024, 2048], "num_outs": 5, "out_channels": 256, "type": "AifiNeck"}
MAMBA: {"in_channels": [256, 512, 1024, 2048], "num_outs": 5, "out_channels": 256, "type": "MambaNeck"}
```

Die ABI (`in_channels` 4-tupel von ResNet-50, `out_channels=256`, `num_outs=5`)
ist über alle drei Necks gleich; das stimmt mit der in `CLAUDE.md` definierten
Neck-Schnittstelle überein (ResNet-{C2..C5} → {P3..P7}).

**Befund:** Ceteris-Paribus auf Snapshot-Ebene erfüllt. Keine KRITISCH-Markierung.

> Hinweis (Scope): Geprüft wurde der gespeicherte Snapshot-Stand. Ob die
> tatsächlich für jeden Seed-Run verwendete, gemergte Konfiguration mit dem
> jeweiligen Snapshot übereinstimmt, ist nicht Teil dieses Teils und wird in
> einem späteren Review-Teil über `scripts/check_ceteris_paribus.py` bzw.
> `scripts/export_merged_configs.py` zu prüfen sein.
