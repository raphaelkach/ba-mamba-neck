# BA Mamba-Neck

Kontrollierter Dreifach-Vergleich von Fusionsmodulen (Necks) für die
Detektion kleiner Objekte in hochaufgelösten Luftbilddaten (VisDrone-DET 2019):

- **V1 — FPN** (CNN-Baseline, Lin et al. 2017)
- **V2 — AIFI + CCFM** (Transformer-Neck, RT-DETR, Zhao et al. 2024)
- **V3 — MambaFPN** (SSM-Neck, Liang et al. 2026)

Backbone (ResNet-50, frozen) und Head (FCOS+ATSS) sind in allen drei
Varianten identisch — einzige unabhängige Variable ist das Neck
(Ceteris-Paribus-Design).

## Notebooks (Colab)

| # | Notebook | Beschreibung | |
|---|---|---|---|
| 1 | `01_data.ipynb` | Datenvorbereitung (Download, COCO-Konvertierung, SAHI-Slicing) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/01_data.ipynb) |
| 2 | `02_train.ipynb` | Training (3x ausfuehren: fpn, aifi, mamba) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/02_train.ipynb) |
| 3 | `03_eval.ipynb` | Evaluation + Figures (Metriken, Statistik, Scaling, Qualitativ) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/03_eval.ipynb) |
| 4 | `04_erf.ipynb` | ERF-Analyse (Effective Receptive Field, 3 Necks x 3 Levels) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/04_erf.ipynb) |

## Setup (Google Colab A100)

```bash
!git clone <repo-url> ba-mamba-neck
%cd ba-mamba-neck
!pip install -r requirements.txt
!python data/prepare.py
```

## Training

```bash
# V1 — CNN-Neck
!python -m mmdet.tools.train configs/fpn.py

# V2 — Transformer-Neck
!python -m mmdet.tools.train configs/aifi.py

# V3 — SSM-Neck
!python -m mmdet.tools.train configs/mamba.py
```

## Struktur

- `configs/` — MMDetection-Configs (V1/V2/V3) + gemeinsame `_base_/`
- `necks/` — Custom Neck-Implementierungen (AIFI, MambaFPN)
- `data/` — VisDrone → COCO-Format Konvertierung
- `eval/` — Metriken, Statistik (Friedman/Nemenyi), Scaling-Analyse, ERF
- `notebooks/` — Colab-Workflow (Daten → Training → Eval → ERF)
- `scripts/check_ceteris_paribus.py` — Diff-Check der drei Configs
- `docs/` — auto-generierte Reports

Details zu Forschungsfrage, Experimentaldesign, Seeds, Statistik und
Ceteris-Paribus-Regel: siehe [CLAUDE.md](CLAUDE.md).
