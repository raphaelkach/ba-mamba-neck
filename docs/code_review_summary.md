# Code-Review BA Mamba-Neck — Konsolidierter Audit-Report

**Stand:** 2026-04-26
**Reviewer:** Claude Code (Opus 4.7)
**Repo-Commit-Hash:** 9bc8c52 (Branch: `claude/review-mamba-neck-part1-MPXyt`)

## Executive Summary

Code ist vollständig und reproduzierbar. Die Trainings-Pipeline
(Notebooks → Configs → Hooks) entspricht in allen geprüften Werten
den dokumentierten Erwartungen, die drei Architekturvarianten sind
ceteris-paribus auf Snapshot-Ebene SHA-256-identisch außerhalb
des `model.neck`-Eintrags. Sechs WICHTIG-Befunde betreffen
ausschließlich Dokumentation, Tooling und ein Konfigurations-
Spannungsfeld — keiner verzerrt den wissenschaftlichen Vergleich.

**Befund-Übersicht:**
- KRITISCH: 0
- WICHTIG: 6
- INFO: 6

## Konsolidierte WICHTIG-Befunde

| ID | Befund | Datei (+ Zeile) | Empfehlung |
|---|---|---|---|
| T2-1 | `cudnn_benchmark=True` koexistiert mit `randomness.deterministic=True` — widersprüchliche Konfig-Aussage; MMEngine-Reihenfolge sorgt zwar zur Laufzeit dafür, dass deterministic gewinnt, aber die Konfig liest sich wie ein Bug | `configs/_base_/runtime.py:17, 21` | Auf `cudnn_benchmark=False` ändern oder im BA-Methodenkapitel mit Verweis auf MMEngine-Reihenfolge dokumentieren |
| T3-1 | Cell 3 (Setup 1b) im Mamba-Notebook weicht durch zwei Verifikations-Zeilen (`import mamba_ssm; print(version)`) von FPN/AIFI ab — strikter Verstoß gegen die Notebook-Sync-Regel in CLAUDE.md | `notebooks/02_train_mamba.ipynb` Cell 3 | Zwei Zeilen in Cell 2 (Install) verschieben **oder** Notebook-Sync-Regel in CLAUDE.md um „Setup darf einen mamba-ssm-Import-Smoke enthalten“ erweitern |
| T4-1 | Trainings-CLI-Befehl `python -m mmdet.utils.train` ist in MMDetection 3.x kein gültiger Modulpfad | `README.md:38–44` | Auf `python tools/train.py …` korrigieren oder durch Verweis auf Notebooks ersetzen |
| T4-2 | README-Setup-Block dokumentiert nicht die Mamba-spezifischen Build-Schritte (CUDA-ENV, `--no-build-isolation`, git-Tags) | `README.md:27–32` | Setup-Block um Hinweis erweitern, der für die volle Reproduktion auf `02_train_mamba.ipynb` Cell 2 verweist |
| T4-3 | `requirements.txt` ist locker gepinnt (`torch>=2.1.0`, `mmcv>=2.0.0`, kein Pin für mamba-ssm/causal-conv1d), während Notebooks hart pinnen | `requirements.txt:1–17` | Auf Notebook-Pins synchronisieren oder explizit als „Minimal-Liste, nicht Reproduktionsmanifest“ deklarieren |
| T4-4 | Kein CI-Workflow vorhanden, der `check_ceteris_paribus.py` und `test_necks.py` automatisch ausführt | (fehlt) `.github/workflows/` | Minimal-Workflow `ci.yml` hinzufügen, der beide Skripte bei Push auf jede Branch ausführt |

## Konsolidierte INFO-Befunde

| ID | Befund | Datei | Empfehlung |
|---|---|---|---|
| T1-1 | `losses/__init__.py` ist nur ein 80-Byte-Re-Export-Stub (Implementation in `losses/bf16_focal_loss.py`) | `losses/__init__.py` | Kein Handlungsbedarf, dokumentationswert |
| T2-2 | `RandomChoiceResize` mit `keep_ratio=True` und quadratischen Zielen wirkt bei nicht-quadratischen Inputs nur als Größtachsen-Limit | `configs/_base_/dataset.py:32–37` | Im BA-Methodenkapitel erwähnen, damit Reviewer nicht harte 640²-Skalierung erwarten |
| T3-2 | SS2D `forward()` deaktiviert `autocast` ohne Inline-Kommentar, der den BF16-Overflow erklärt | `necks/mamba_neck.py:152` | Inline-Kommentar oder Modul-Docstring-Eintrag mit Verweis auf `selective_scan_fn`-Numerik |
| T3-3 | AifiNeck `input_proj` enthält Conv 1×1 + BN + SiLU, MambaNeck `input_proj` ist bare Conv 1×1 (architektonisch motiviert: VSSBlock hat eigenes LayerNorm) | `necks/aifi_neck.py:206`, `necks/mamba_neck.py:255` | Im BA-Methodenkapitel dokumentieren, kein Code-Fix nötig |
| T4-A | „FCOS+ATSS“ und „Anchor-free“ in CLAUDE.md sind technisch unpräzise (Code: `model.type='ATSS'` mit ATSSHead, 1 Anchor/Position via `ratios=[1.0]`) | `CLAUDE.md:13–14` | Auf `"ATSS (Zhang et al. 2020) with ATSSHead and FCOS-style centerness"` und `"single-anchor (quasi anchor-free)"` präzisieren |
| T4-B | `runtime.py`-Docstring schlägt CLI-Seed-Setup vor; Notebooks setzen Seed direkt per `cfg.randomness` | `configs/_base_/runtime.py:3–4` | Docstring um Notebook-Aufruf-Beispiel ergänzen |

## Empfohlene Nächste Schritte

### Vor Fortsetzung der BA (sofort)
keine

### Während Kap. 4 schreiben (parallel)
1. CLAUDE.md präzisieren (FCOS+ATSS → ATSS-Detektor mit ATSSHead; „anchor-free“ → „single-anchor / quasi anchor-free“)

### Nach Mamba-Training-Abschluss (vor BA-Abgabe)
1. README.md korrigieren (CLI-Befehl + Setup-Doku für mamba-ssm-Build)
2. requirements.txt mit Notebook-Pins synchronisieren
3. Notebook-Sync-Regel in CLAUDE.md erweitern (Setup darf mamba-ssm-Import-Smoke enthalten) **oder** zwei Zeilen aus Cell 3 in Cell 2 des Mamba-Notebooks verschieben
4. CI-Workflow für ceteris-paribus.yml ergänzen (PR-Gate mit `check_ceteris_paribus.py` + `test_necks.py`)
5. SS2D-Inline-Kommentar zum BF16/float32-Fix ergänzen (`necks/mamba_neck.py:152`)
6. runtime.py-Docstring aktualisieren (Notebook-Pfad zur Seed-Setzung erwähnen)

## Verweise auf Detail-Reports
- Teil 1: `docs/code_review_part1.md` (Vollständigkeit + Ceteris-Paribus)
- Teil 2: `docs/code_review_part2.md` (Code-Werte)
- Teil 3: `docs/code_review_part3.md` (Custom-Module + Notebook-Sync)
- Teil 4: `docs/code_review_part4.md` (Dokumentation + Reproduzierbarkeit)

## Anhang: Versionsstände

Quelle: Notebook Cell 2 (kanonisch, identisch in `02_train_{fpn,aifi,mamba}.ipynb`), mit Assert-Verifikation pro Run.

| Komponente | Version |
|---|---|
| Python | 3.12 (Colab default) |
| torch | 2.5.1+cu124 |
| mmdet | 3.3.0 |
| mmengine | 0.10.7 |
| mmcv | 2.2.0 |
| mamba-ssm | 2.2.2 (`git+https://github.com/state-spaces/mamba.git@v2.2.2`, `--no-build-isolation`, `TORCH_CUDA_ARCH_LIST=8.0`) |
| causal-conv1d | 1.4.0 (`git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0`, `--no-build-isolation`) |
| transformers | <=4.44.0 |
| numpy | <2 |
| Hardware | NVIDIA A100 (sm_80) |
