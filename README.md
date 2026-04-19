# BA Mamba-Neck

Controlled comparison of three neck architectures for small-object
detection in aerial imagery (VisDrone-DET 2019):

- **V1 - FPN** (CNN baseline, Lin et al. 2017)
- **V2 - AIFI + CCFM** (Transformer neck, adapted from RT-DETR, Zhao et al. 2024)
- **V3 - MambaFPN** (SSM neck, Liang et al. 2026)

Backbone (ResNet-50, frozen) and head (FCOS+ATSS) are identical across
all three variants. The neck is the only independent variable
(ceteris-paribus design).

## Notebooks (Colab)

| # | Notebook | Description | |
|---|---|---|---|
| 1 | `01_data.ipynb` | Data preparation (download, COCO conversion, SAHI slicing) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/01_data.ipynb) |
| 2a | `02_train_fpn.ipynb` | Training V1 (CNN/FPN neck) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/02_train_fpn.ipynb) |
| 2b | `02_train_aifi.ipynb` | Training V2 (AIFI+CCFM Transformer neck) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/02_train_aifi.ipynb) |
| 2c | `02_train_mamba.ipynb` | Training V3 (MambaFPN neck) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/02_train_mamba.ipynb) |
| 3 | `03_eval.ipynb` | Evaluation and figures (metrics, statistics, scaling, qualitative) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/03_eval.ipynb) |
| 4 | `04_erf.ipynb` | Effective receptive field analysis (3 necks x 3 pyramid levels) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/raphaelkach/ba-mamba-neck/blob/main/notebooks/04_erf.ipynb) |

## Setup (Google Colab A100)

```bash
!git clone https://github.com/raphaelkach/ba-mamba-neck.git
%cd ba-mamba-neck
!pip install -r requirements.txt
!python data/prepare.py
```

## Training

```bash
# V1 - CNN neck
!python -m mmdet.utils.train configs/fpn.py --work-dir /content/drive/MyDrive/ba/fpn/seed_42

# V2 - Transformer neck
!python -m mmdet.utils.train configs/aifi.py --work-dir /content/drive/MyDrive/ba/aifi/seed_42

# V3 - SSM neck
!python -m mmdet.utils.train configs/mamba.py --work-dir /content/drive/MyDrive/ba/mamba/seed_42
```

See `notebooks/02_train_fpn.ipynb`, `notebooks/02_train_aifi.ipynb` and
`notebooks/02_train_mamba.ipynb` for the full 10-seed loop with resume
logic and version pinning. Each notebook is designed to run in its own
parallel Colab session.

## Repository structure

- `configs/` - MMDetection configs (V1/V2/V3) and shared `_base_/`
- `necks/` - Custom neck implementations (AifiNeck, MambaNeck)
- `data/` - VisDrone to COCO format conversion
- `eval/` - Metrics, statistics (Friedman/Nemenyi), scaling, ERF
- `notebooks/` - Colab workflow (data, training, evaluation, ERF)
- `scripts/` - Ceteris-paribus check, neck tests, report generators
- `docs/` - Auto-generated reports (architecture, data, hyperparameters)

## Experimental design

Details on the research question, experimental design, seeds, statistical
tests, and the ceteris-paribus rule are documented in [CLAUDE.md](CLAUDE.md).
