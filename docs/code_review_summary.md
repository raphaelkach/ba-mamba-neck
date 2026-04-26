# Code-Review BA Mamba-Neck — Konsolidierter Audit-Report

**Stand:** 2026-04-26
**Reviewer:** Claude Code (Opus 4.7)
**Repo-Commit-Hash:** 201e153 (Branch: `claude/review-mamba-neck-part1-MPXyt`)
**Geprüfte Dateien:** 34 (Configs, Snapshots, Custom-Module, Notebooks, Scripts, Daten-Pipeline, Doku)
**Trainings-Status zum Audit-Zeitpunkt:** FPN 10/10 ✅, AIFI 10/10 ✅, Mamba in Bearbeitung (≈75–100h Restzeit)

## Executive Summary

Das Repository ist **wissenschaftlich solide**: Ceteris-paribus zwischen
FPN/AIFI/Mamba ist auf Snapshot-Ebene byte-exakt eingehalten (gleicher
SHA-256 außerhalb `model.neck`), alle 52 erwarteten Werte (Optimizer,
Schedule, Detektor, Backbone, Daten-Pipeline, Reproduzierbarkeit) stimmen,
die Custom-Module (BF16SafeFocalLoss, SS2D-Float32-Fix, AifiNeck, MambaNeck)
sind korrekt implementiert und gegen die jeweiligen numerischen Fallstricke
abgesichert, und die Notebooks teilen 8 von 11 Zellen byte-identisch. Es gibt
**keine kritischen Befunde**, die das Fortsetzen der BA blockieren würden;
sieben „WICHTIG“-Punkte (Konfigurations- und Dokumentations-Hygiene) und
sechs INFO-Punkte sollten aber vor Abgabe oder im BA-Methodikkapitel
adressiert werden.

**Befund-Übersicht (alle Teile zusammengefasst):**
- KRITISCH: 0
- WICHTIG: 7
- INFO: 6

## KRITISCH (verzerrt Vergleich, blockiert BA)

**Keine.** Ceteris-paribus auf Snapshot-Ebene ist byte-exakt eingehalten,
keine Werte-Abweichung von der Spezifikation, keine fehlende Datei.

## WICHTIG (Inkonsistenz, sollte vor Abgabe gefixt werden)

| ID | Befund | Datei (+ Zeile) | Erwartet | Gefunden | Empfehlung |
|---|---|---|---|---|---|
| T2-1 | `cudnn_benchmark=True` koexistiert mit `randomness.deterministic=True` (widersprüchliche Konfig; MMEngine-Reihenfolge sorgt zwar zur Laufzeit dafür, dass deterministic gewinnt — aber die Konfig liest sich wie ein Bug) | `configs/_base_/runtime.py:17, 21` | konsistent: entweder beide auf nicht-deterministisch oder `cudnn_benchmark=False` | beide aktiv | Auf `cudnn_benchmark=False` ändern oder im BA-Methodenkapitel mit Verweis auf MMEngine-Reihenfolge dokumentieren |
| T3-1 | Cell 3 (Setup 1b) im Mamba-Notebook weicht durch zwei Verifikations-Zeilen (`import mamba_ssm; print(version)`) von FPN/AIFI ab — strikter Verstoß gegen die Notebook-Sync-Regel in CLAUDE.md | `notebooks/02_train_mamba.ipynb`, Cell 3 | byte-identisch zu FPN/AIFI | +2 Zeilen mamba-ssm-Import-Smoke | Zwei Zeilen in Cell 2 (Install) verschieben (dort sind sie ohnehin durch Install-Ausnahme abgedeckt) ODER CLAUDE.md erweitern |
| T4-1 | Trainings-CLI-Befehl `python -m mmdet.utils.train` ist in MMDetection 3.x kein gültiger Modulpfad | `README.md:38–44` | `python tools/train.py …` oder `mim train mmdet …` | nicht ausführbar | CLI-Beispiel korrigieren oder durch Verweis auf Notebooks ersetzen |
| T4-2 | README-Setup-Block dokumentiert nicht die Mamba-spezifischen Build-Schritte (CUDA-ENV, `--no-build-isolation`, git-Tags) | `README.md:27–32` | Hinweis auf Notebook-Install-Cell als kanonische Reproduktionsbasis | nur `pip install -r requirements.txt` | Setup-Block um Disclaimer erweitern, der auf `02_train_mamba.ipynb` Cell 2 verweist |
| T4-3 | `requirements.txt` ist locker gepinnt (`torch>=2.1.0`, `mmcv>=2.0.0`, kein Pin für mamba-ssm/causal-conv1d), während Notebooks hart pinnen | `requirements.txt:1–17` | identische Pins zu Notebook Cell 2 oder klarer Disclaimer | divergente, lockere Pins | Auf Notebook-Pins synchronisieren oder explizit als „Minimal-Liste, nicht Reproduktionsmanifest“ deklarieren |
| T4-4 | Kein CI-Workflow vorhanden, der `check_ceteris_paribus.py` und `test_necks.py` automatisch ausführt | (fehlt) `.github/workflows/` | mindestens ein PR-Gate für Ceteris-Paribus-Diff | manuell aufzurufen | Minimal-Workflow `ci.yml` hinzufügen, der beide Skripte bei Push auf jede Branch ausführt |
| T4-5 | `data/prepare.py` enthält keine kryptografische Daten-Integritätsverifikation | `data/prepare.py` | mindestens erwartete Annotation-Anzahl pro Split als Soll-Wert | nur statistische Logs | `data/expected_stats.json` mit Soll-Counts (sliced + unsliced, train + val) hinzufügen, der bei Abweichung wirft |

## INFO (kosmetisch, nicht blockierend)

| ID | Befund | Datei | Empfehlung |
|---|---|---|---|
| T1-1 | `losses/__init__.py` ist nur ein 80-Byte-Re-Export-Stub | `losses/__init__.py` | Kein Handlungsbedarf, dokumentationswert |
| T2-2 | `RandomChoiceResize` mit `keep_ratio=True` und quadratischen Zielen wirkt bei nicht-quadratischen Inputs nur als Größtachsen-Limit | `configs/_base_/dataset.py:32–37` | Im BA-Methodenkapitel erwähnen, damit Reviewer nicht harte 640²-Skalierung erwarten |
| T3-2 | SS2D `forward()` deaktiviert `autocast` ohne Inline-Kommentar, der den BF16-Overflow erklärt | `necks/mamba_neck.py:152` | Inline-Kommentar oder Modul-Docstring-Eintrag mit Verweis auf `selective_scan_fn`-Numerik |
| T3-3 | AifiNeck `input_proj` enthält Conv 1×1 + BN + SiLU, MambaNeck `input_proj` ist bare Conv 1×1 (architektonisch motiviert; VSSBlock hat eigenes LayerNorm) | `necks/aifi_neck.py:206`, `necks/mamba_neck.py:255` | Im BA-Methodenkapitel dokumentieren, kein Code-Fix nötig |
| T4-A | „FCOS+ATSS“ und „Anchor-free“ in CLAUDE.md sind technisch unpräzise (Code: `model.type='ATSS'` mit ATSSHead, 1 Anchor/Position) | `CLAUDE.md:13–14` | Auf `"ATSS (Zhang et al. 2020) with ATSSHead and FCOS-style centerness"` und `"single-anchor (quasi anchor-free)"` präzisieren |
| T4-B | `runtime.py`-Docstring schlägt CLI-Seed-Setup vor; Notebooks setzen Seed direkt per `cfg.randomness` | `configs/_base_/runtime.py:3–4` | Docstring ergänzen oder zur Notebook-Variante umschreiben |

---

## Empfohlene Nächste Schritte

### Vor Fortsetzung der BA (sofort)
**Keine sofortigen Aktionen erforderlich.** Der laufende Mamba-Trainingslauf
darf ungestört durchlaufen — keiner der Befunde berührt die Trainings-
Korrektheit. Nach Abschluss können die folgenden Punkte für den Methodikteil
adressiert werden.

### Vor Abgabe der BA (Reihenfolge nach Aufwand)
1. **T3-1** Cell 3 (Setup 1b) im Mamba-Notebook synchronisieren (zwei Zeilen
   in Cell 2 verschieben). 5 Min, eliminiert den einzigen Notebook-Sync-
   Verstoß. Achtung: Notebook-Sync-Regel verlangt, dass jede Änderung an
   einem Notebook simultan auf alle drei angewendet wird — die Verschiebung
   findet aber ohnehin nur im Mamba-Notebook statt (wo sie geboren wurde).
2. **T2-1** `cudnn_benchmark=True → False` in `runtime.py:21` setzen ODER
   im BA-Methodikkapitel als Limitation dokumentieren (mit MMEngine-
   Reihenfolge-Argument). 1 Min vs. 1 Absatz Methodik.
3. **T4-1, T4-2** README aktualisieren: CLI-Befehl korrigieren, Setup-Block
   um Mamba-Build-Hinweise erweitern. 10 Min.
4. **T4-3** `requirements.txt` entweder hart pinnen wie Notebooks oder
   Disclaimer ergänzen. 5 Min.
5. **T4-A** CLAUDE.md L13–14 präzisieren („ATSS“ / „single-anchor“). 2 Min.

### Nice-to-have (optional)
- **T4-4** Minimal-CI-Workflow für `check_ceteris_paribus.py` (würde bei
  zukünftigen PRs jeden Ceteris-Paribus-Verstoß sofort blocken).
- **T4-5** Erwartete Daten-Stats als `expected_stats.json` mit Assert in
  `data/prepare.py`.
- **T3-2** Inline-Kommentar an `mamba_neck.py:152` zum BF16-Overflow.
- **T2-2, T3-3, T4-B** Dokumentationspräzisierungen ohne Code-Fix.

## Verweise auf Detail-Reports
- Teil 1: `docs/code_review_part1.md` — Strukturelle Vollständigkeit + Ceteris-Paribus-Snapshots
- Teil 2: `docs/code_review_part2.md` — Code-Werte (52 Werte gegen Snapshot)
- Teil 3: `docs/code_review_part3.md` — Custom-Module + Notebook-Sync
- Teil 4: `docs/code_review_part4.md` — Dokumentation + Reproduzierbarkeit

## Anhang: Versionsstände

Aus Notebook-Install-Cell (kanonisch, identisch in `02_train_{fpn,aifi,mamba}.ipynb`):

| Komponente | Version | Quelle |
|---|---|---|
| Python | 3.12 (Colab default) | Notebook Cell 2 Kommentar |
| torch | `2.5.1+cu124` (hart gepinnt + Assert-Verifikation) | Notebook Cell 2 |
| torchvision | implizit über torch-Wheel | Notebook Cell 2 |
| mmengine | `0.10.7` (hart gepinnt + Assert) | Notebook Cell 2 |
| mmcv | `2.2.0` (hart gepinnt + Assert; mmdet-3.3.0-Assertion via Runtime-Patch hochgesetzt) | Notebook Cell 2 |
| mmdet | `3.3.0` (hart gepinnt + Assert) | Notebook Cell 2 |
| mamba-ssm | `v2.2.2` von `git+https://github.com/state-spaces/mamba.git` (`--no-build-isolation`, `TORCH_CUDA_ARCH_LIST=8.0`) | Notebook Cell 2 (nur Mamba-Notebook) |
| causal-conv1d | `v1.4.0` von `git+https://github.com/Dao-AILab/causal-conv1d.git` (`--no-build-isolation`) | Notebook Cell 2 (nur Mamba-Notebook) |
| transformers | `<=4.44.0` | Notebook Cell 2 |
| numpy | `<2` | Notebook Cell 2 |
| pycocotools, sahi, scipy, scikit-posthocs, matplotlib, pandas, wandb, fvcore, torchinfo, deepdiff | nicht hart gepinnt | `requirements.txt` (Loose) |
| Hardware | NVIDIA A100 (sm_80) | CLAUDE.md, Notebook Cell 2 |

**Konsistenz-Hinweis:** Die in `requirements.txt` angegebenen lockeren Pins
(`torch>=2.1.0`, `mmcv>=2.0.0`, `mmdet>=3.0.0`, kein Pin für mamba-ssm /
causal-conv1d) sind **nicht** die maßgebliche Reproduktionsbasis — siehe
T4-3.
