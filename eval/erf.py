"""Gradient-based Effective Receptive Field analysis (Luo et al. 2016).

For each neck variant, loads the best-seed checkpoint and computes the
ERF heatmap at three pyramid levels (P3, P4, P5) by back-propagating
from the central neuron through 500 validation images.

Quantitative measures per neck per level:
    * ERF area (pixels above 10% threshold)
    * ERF area (pixels above 50% threshold)
    * Gini coefficient (gradient uniformity)
    * Entropy of normalised gradient distribution
    * Max distance from centre above 10% threshold

Outputs:
    * ``results/erf_quantitative.csv``
    * ``results/erf_heatmaps/{neck}_P{level}.png``

Usage:
    PYTHONPATH=. python eval/erf.py \
        [--data-root /content/visdrone] \
        [--ckpt-dir /content/drive/MyDrive/ba] \
        [--results results] [--n-images 500]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import necks as _  # noqa: F401

from eval.constants import DEFAULT_CKPT_DIR, DEFAULT_DATA_ROOT, NECKS
from eval.utils import get_best_checkpoint

LEVELS = ['P3', 'P4', 'P5']
LEVEL_IDX = {'P3': 0, 'P4': 1, 'P5': 2}


def _build_detector(cfg_path: str, ckpt: str, device: torch.device):
    from mmdet.apis import init_detector
    return init_detector(cfg_path, ckpt, device=str(device))


def _load_val_images(data_root: Path, n: int) -> List[Path]:
    img_dir = data_root / 'raw' / 'VisDrone2019-DET-val' / 'images'
    images = sorted(img_dir.glob('*.jpg'))
    return images[:n]


def _compute_erf(det, level_idx: int, images: List[Path],
                 device: torch.device) -> np.ndarray:
    """Accumulate absolute input gradients from central neuron at `level_idx`."""
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
            std=[58.395 / 255, 57.12 / 255, 57.375 / 255]),
    ])

    grad_accum = None
    det.eval()

    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device).requires_grad_(True)

        # Forward through backbone + neck
        backbone_feats = det.backbone(x)
        neck_feats = det.neck(backbone_feats)
        feat = neck_feats[level_idx]  # (1, C, H, W)

        # Central neuron
        _, C, H, W = feat.shape
        cy, cx = H // 2, W // 2
        target = feat[0, :, cy, cx].sum()

        target.backward()
        grad = x.grad.data.abs().squeeze(0).mean(0)  # (640, 640)
        if grad_accum is None:
            grad_accum = grad.cpu().numpy()
        else:
            grad_accum += grad.cpu().numpy()
        x.grad = None
        det.zero_grad()

    return grad_accum / len(images)


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient of a 2D array."""
    flat = arr.flatten()
    flat = np.sort(flat)
    n = len(flat)
    if n == 0 or flat.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * flat) / (n * flat.sum())) - (n + 1) / n)


def _entropy(arr: np.ndarray) -> float:
    """Shannon entropy of normalised 2D array."""
    flat = arr.flatten()
    flat = flat / flat.sum() if flat.sum() > 0 else flat
    flat = flat[flat > 0]
    return float(-np.sum(flat * np.log2(flat)))


def _erf_metrics(heatmap: np.ndarray) -> Dict:
    norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    H, W = norm.shape
    cy, cx = H // 2, W // 2

    mask_10 = norm >= 0.1
    mask_50 = norm >= 0.5
    area_10 = int(mask_10.sum())
    area_50 = int(mask_50.sum())

    # max distance from centre above 10%
    ys, xs = np.where(mask_10)
    if len(ys) > 0:
        dists = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        max_dist = float(dists.max())
    else:
        max_dist = 0.0

    return dict(
        area_10=area_10,
        area_50=area_50,
        gini=round(_gini(norm), 4),
        entropy=round(_entropy(norm), 4),
        max_distance=round(max_dist, 2),
    )


def _save_heatmap(heatmap: np.ndarray, out_path: Path,
                  title: str = '') -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    ax.imshow(norm, cmap='hot', interpolation='bilinear')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def analyse_erf(data_root: Path, ckpt_dir: Path,
                results_dir: Path, n_images: int) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    heatmap_dir = results_dir / 'erf_heatmaps'
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    images = _load_val_images(data_root, n_images)
    print(f'Using {len(images)} validation images')

    quant_rows: List[Dict] = []

    for neck in NECKS:
        seed, ckpt_path = get_best_checkpoint(ckpt_dir, neck)
        print(f'\n=== {neck} (seed={seed}, ckpt={ckpt_path.name}) ===')
        det = _build_detector(f'configs/{neck}.py',
                              str(ckpt_path), device)

        for level_name in LEVELS:
            lidx = LEVEL_IDX[level_name]
            print(f'  {level_name} ...', end=' ', flush=True)
            heatmap = _compute_erf(det, lidx, images, device)

            metrics = _erf_metrics(heatmap)
            metrics.update(neck=neck, level=level_name)
            quant_rows.append(metrics)

            out = heatmap_dir / f'{neck}_{level_name}.png'
            _save_heatmap(heatmap, out,
                          title=f'{neck} – {level_name} ERF')
            print(f'area_10={metrics["area_10"]}  '
                  f'area_50={metrics["area_50"]}  '
                  f'gini={metrics["gini"]}')
        del det
        torch.cuda.empty_cache()

    # Write quantitative CSV
    csv_path = results_dir / 'erf_quantitative.csv'
    cols = ['neck', 'level', 'area_10', 'area_50',
            'gini', 'entropy', 'max_distance']
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(quant_rows)
    print(f'\nwrote {csv_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path,
                    default=Path(DEFAULT_DATA_ROOT))
    ap.add_argument('--ckpt-dir', type=Path,
                    default=Path(DEFAULT_CKPT_DIR))
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--n-images', type=int, default=500)
    args = ap.parse_args()
    analyse_erf(args.data_root, args.ckpt_dir, args.results, args.n_images)


if __name__ == '__main__':
    main()
