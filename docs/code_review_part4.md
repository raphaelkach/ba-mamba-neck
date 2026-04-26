# Code-Review Teil 4/4 — Dokumentation + Reproduzierbarkeit

**Datum:** 2026-04-26
**Commit:** 201e153 (Branch: claude/review-mamba-neck-part1-MPXyt)

## Zusammenfassung
Die Trainings-Pipeline selbst (Notebooks → Configs → Hooks → run_meta.json)
ist sauber reproduzierbar — Seeds werden deterministisch gesetzt, Resume
funktioniert per-Epoch-Checkpoint, und pro Seed wird ein Lauf-Metadaten-JSON
geschrieben. Schwächen liegen ausschließlich in **Dokumentation und Tooling
außerhalb des Trainings**: das README enthält einen ungültigen Trainings-CLI-
Befehl, `requirements.txt` ist locker gepinnt während die Notebooks hart
pinnen, und es existiert kein CI-Workflow für `test_necks.py` /
`check_ceteris_paribus.py`. Keine kritischen Befunde, 4 wichtige Punkte und
2 INFO-Punkte.

## Befund-Übersicht
- KRITISCH: 0
- WICHTIG: 4
- INFO: 2

## Aufgabe 1: Dokumentations-Konsistenz

### CLAUDE.md

| Prüfpunkt | Befund | Status |
|---|---|---|
| `"Head: FCOS+ATSS"` | CLAUDE.md:13 — historisch korrekt (ATSS-Paper baut auf RetinaNet/FCOS auf), aber der Code verwendet `model.type='ATSS'` mit `ATSSHead`. Präziser wäre `"Head: ATSS (Zhang et al. 2020) with ATSSAssigner; centerness branch as in FCOS"`. | INFO |
| `"Anchor-free"` | CLAUDE.md:14 — faktisch ist ATSS in MMDetection anchor-basiert mit **1 Anchor pro Position** (`ratios=[1.0]`, `scales_per_octave=1`, `octave_base_scale=8`). Das wird in der Literatur oft als „quasi-anchor-free“ bezeichnet (entspricht Zhang et al. 2020 Sec. 3 Tab. 2). | INFO |
| Optimizer-Config konsistent? | CLAUDE.md nennt keine Optimizer-Werte direkt — Verweis ist nur generisch („optimizer, LR schedule … identical“). `_base_/schedule.py` setzt AdamW lr=2e-4, weight_decay=0.05, bfloat16, max_norm=0.1 — passt zum Snapshot. | ✅ |
| `"FP16"` / `"float16"` erwähnt? | Nicht in CLAUDE.md. `_base_/schedule.py:6` dokumentiert ausdrücklich, warum bfloat16 (8-bit Exponent) gewählt wurde. | ✅ |

**Bewertung:** Zwei kosmetische Präzisierungen (Head-Bezeichnung,
„anchor-free“-Etikett) sind sinnvoll. Beides INFO, kein blockierender Befund.

### README.md

| Prüfpunkt | Befund | Status |
|---|---|---|
| Trainings-Befehl korrekt? | README.md:38–44 nennt `!python -m mmdet.utils.train configs/fpn.py --work-dir ...`. **`mmdet.utils.train` existiert in MMDetection 3.x nicht als ausführbares Modul** — der korrekte CLI-Pfad ist `python tools/train.py …` oder `mim train mmdet configs/…`. Die tatsächliche Trainings-Ausführung erfolgt allerdings nicht über CLI sondern direkt im Notebook (Cell 7: `Runner.from_cfg(cfg); runner.train()`), d.h. das CLI-Beispiel im README wäre für externe Reproduzenten ein Stolperstein, blockiert aber die BA-Workflows nicht. | ⚠️ WICHTIG |
| Notebook-Links | README listet alle 6 Notebooks (01_data, 02_train_{fpn,aifi,mamba}, 03_eval, 04_erf) mit Colab-Badges. Alle 6 Dateien existieren laut Teil 1. | ✅ |
| Setup-Sequenz vollständig? | README:30 sagt nur `!pip install -r requirements.txt`. Tatsächlich benötigt der Stack: ① pinned torch (`2.5.1+cu124`), ② `mmengine==0.10.7`, `mmcv==2.2.0`, `mmdet==3.3.0`, ③ **mamba-ssm + causal-conv1d von Git mit `--no-build-isolation`** und CUDA-Env-Variablen (`CUDA_HOME`, `TORCH_CUDA_ARCH_LIST=8.0`). Diese Schritte sind ausschließlich in `02_train_mamba.ipynb` Cell 2 dokumentiert, nicht im README. Wer dem README folgt, bekommt einen unbrauchbaren Mamba-Build. | ⚠️ WICHTIG |

**Empfehlung README:**
- Zeile 38–44 ersetzen durch Verweis auf Notebooks oder auf den korrekten
  CLI-Pfad.
- Setup-Block (Zeile 27–32) um Hinweis ergänzen: „Use the install cell of
  the respective Colab notebook for the full pinned stack including
  mamba-ssm/causal-conv1d build flags. `requirements.txt` is a minimal
  list and not the canonical reproducibility manifest.“

## Aufgabe 2: Reproduzierbarkeit

### 1. Versionsfixierung

| Quelle | Pinning | Bewertung |
|---|---|---|
| `requirements.txt` | `torch>=2.1.0`, `mmcv>=2.0.0`, `mmdet>=3.0.0`, kein Pin für `mamba-ssm`, `causal-conv1d`, `sahi`, `scipy`, `wandb`, `fvcore`, `pandas` | ❌ unzureichend für Reproduktion |
| Notebook Cell 2 (alle drei) | `torch==2.5.1+cu124`, `mmengine==0.10.7`, `mmcv==2.2.0`, `mmdet==3.3.0` (jeweils mit Assert-Verifikation), `transformers<=4.44.0`, `numpy<2`, mamba-ssm `git@v2.2.2`, causal-conv1d `git@v1.4.0` | ✅ canonical für die Studie |

**Befund WICHTIG:** Die `requirements.txt` ist **nicht** die maßgebliche
Version-Quelle und sollte entweder a) aktualisiert werden, sodass sie die
Notebook-Pins exakt widerspiegelt, oder b) im README explizit als
„Minimal-Liste, nicht Reproduktionsbasis“ ausgewiesen werden.

### 2. Seed-Determinismus

Der Seed wird **nicht** über `--cfg-options randomness.seed=…` gesetzt
(wie der Docstring in `_base_/runtime.py:3–4` suggeriert), sondern direkt
im Notebook Cell 7: `cfg.randomness = dict(seed=seed, deterministic=True)`.
Funktional äquivalent, aber der Docstring ist veraltet.

INFO: `_base_/runtime.py:3–4` Docstring-Hinweis aktualisieren oder
zusätzlich Notebook-Aufruf erwähnen.

### 3. Daten-Hashes

`data/prepare.py` enthält **keine SHA256/MD5-Verifikation** der heruntergeladenen
VisDrone-ZIPs. Stattdessen werden statistische Quasi-Hashes berechnet
(`summary.json` mit `num_images`, `num_annotations`, `size_buckets`,
Klassen-Histogramm pro Split — `prepare.py:285–304`). Das gibt eine schwache
Integritätsprüfung (Zählwerte müssen passen), aber keine kryptografische.

⚠️ WICHTIG: Für eine BA, die `Dataset: VisDrone-DET 2019` als kontrollierte
Vergleichsbasis nennt, sollte mindestens ein Eintrag mit erwarteter
Annotation-Anzahl pro Split (sliced + unsliced, train + val) als „Soll“
in `data/prepare.py` oder einer separaten Datei (`data/expected_stats.json`)
hinterlegt sein, der bei Abweichung wirft.

### 4. Resume-Logik

| Quelle | Befund | Status |
|---|---|---|
| `_base_/runtime.py:71` | `resume = True` | ✅ |
| Notebook Cell 7 | Sucht zusätzlich `glob('{work_dir}/epoch_*.pth')`, setzt `cfg.resume=True; cfg.load_from = epoch_ckpts[-1]` und überspringt Seeds, deren `best_*.pth` bereits existiert. | ✅ |
| Hygiene | Zusätzliche Bereinigung: `run_meta.json` mit `train_time_sec < 10` wird gelöscht (verhindert dass abgebrochene Runs als „fertig“ erkannt werden). | ✅ |

Resume-Logik ist robust für Colab-Disconnects; konsistent zur 75–100h-
Restzeit-Anforderung im Mamba-Lauf.

### 5. Run-Metadaten

Pro Seed wird `run_meta.json` mit folgenden Feldern geschrieben (Cell 7
in allen drei Notebooks):

```
neck, seed, train_time_sec, train_time_h, peak_gpu_mem_gb,
gpu_name, cuda_version, pytorch_version, mmdet_version,
num_epochs, batch_size, lr, amp, tf32, ema, multi_scale,
num_workers, pin_memory, cudnn_benchmark, optimizer, resumed
```

✅ Sehr ausführlich. Das ist Bestpractice und für die BA-Reproduzierbarkeit
ein starkes Argument.

INFO: Das Feld `tf32: True` im `run_meta.json` ist ein Konfigurations-Annotat
ohne entsprechende explizite Runtime-Einstellung in `runtime.py`/Schedule —
in PyTorch 2.x ist TF32 für `matmul` standardmäßig aktiv auf Ampere, aber das
ist nirgends im Code explizit gesetzt. Wert ist daher informativ, nicht
verifiziert. Kein Befund, aber Erwähnen im BA-Methodenkapitel.

### 6. CI-Tests

| Skript | Existiert? | Aufruf | CI-Workflow? |
|---|---|---|---|
| `scripts/test_necks.py` | ✅ (4900 B) | `PYTHONPATH=. python scripts/test_necks.py` (Docstring) | ❌ kein `.github/workflows/` |
| `scripts/check_ceteris_paribus.py` | ✅ (2587 B) | `PYTHONPATH=. python scripts/check_ceteris_paribus.py` (Docstring) | ❌ kein `.github/workflows/` |

Verzeichnis `.github/workflows/` existiert nicht. Beide Skripte sind nur
manuell oder als Notebook-Subprozess aufrufbar. `check_ceteris_paribus.py`
sollte als Pre-Push- oder PR-Gate laufen, da es genau die Vorgabe aus
CLAUDE.md mechanisch durchsetzt.

⚠️ WICHTIG: Fehlender CI-Workflow ist eine Reproduzierbarkeits-Schwäche
zweiter Ordnung. Empfehlung: Minimal-Workflow `ceteris-paribus.yml`
hinzufügen, der bei jedem Push die zwei Skripte ausführt.

## Aufgabe 3: Bekannte Auffälligkeiten — Status-Bestätigung

### 1. `cudnn_benchmark=True` UND `deterministic=True`

Bestätigt in Teil 2 (siehe `docs/code_review_part2.md`, Bereich 5).
`runtime.py:21` setzt `cudnn_benchmark=True`, `runtime.py:17` setzt
`deterministic=True`. MMEngine-Reihenfolge (`setup_env` vor
`set_randomness`) sorgt dafür, dass `deterministic=True` zur Laufzeit
gewinnt; `cudnn_benchmark=True` ist dann effektiv ein toter Eintrag.

**Status:** WICHTIG (bereits in Teil 2 gezählt, nicht doppelt). Empfehlung:
in `runtime.py` auf `cudnn_benchmark=False` ändern oder im BA-Methodikkapitel
mit Verweis auf MMEngine-Reihenfolge dokumentieren.

### 2. `auto_scale_lr.enable=False` mit `base_batch_size=16` und `train_dataloader.batch_size=16`

Konsistent: Wenn die effektive Batchgröße = `base_batch_size` ist, wird kein
LR-Scaling benötigt. `enable=False` mit identischen Werten ist die korrekte
„No-Op-Konfiguration“. ✅ Kein Befund.

### 3. SAHI-Overlap = 20%

`data/prepare.py:66` — `SLICE_OVERLAP = 0.2`. Verwendet in beide Richtungen
(`overlap_height_ratio=SLICE_OVERLAP`, `overlap_width_ratio=SLICE_OVERLAP`)
in `slice_split()` (Zeile 241–242). ✅ Konsistent mit Erwartung.

### 4. `filter_empty_gt=True, min_size=4` im train_dataloader

`configs/_base_/dataset.py:68` — `filter_cfg=dict(filter_empty_gt=True, min_size=4)`.
Im Snapshot bestätigt (Teil 2 Bereich 4). ✅

## Befund-Auflistung

| ID | Schweregrad | Befund | Datei |
|---|---|---|---|
| T4-1 | WICHTIG | Trainings-CLI-Befehl `python -m mmdet.utils.train` ist ungültig | `README.md:38–44` |
| T4-2 | WICHTIG | Setup-Block des README dokumentiert nicht mamba-ssm-Build-Schritte | `README.md:27–32` |
| T4-3 | WICHTIG | `requirements.txt` ist locker gepinnt, Notebooks pinnen hart — Inkonsistenz | `requirements.txt` |
| T4-4 | WICHTIG | Kein CI-Workflow für `check_ceteris_paribus.py` und `test_necks.py` | (fehlt) `.github/workflows/` |
| T4-5 | INFO | „FCOS+ATSS“ und „Anchor-free“ in CLAUDE.md sind ungenau | `CLAUDE.md:13–14` |
| T4-6 | INFO | `runtime.py`-Docstring suggeriert CLI-Seed-Setup; Notebooks setzen es per `cfg.randomness` | `configs/_base_/runtime.py:3–4` |

Optional empfehlenswert (kein eigener Befund):
- `data/prepare.py` um SHA256-Soll-Werte oder erwartete Annotation-Counts erweitern.
- `tf32`-Flag im `run_meta.json` durch explizite Runtime-Einstellung
  (`torch.backends.cuda.matmul.allow_tf32 = True`) untermauern.
