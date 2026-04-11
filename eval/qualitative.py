"""Qualitative visualisation of the 10 highest-divergence images.

Reads ``results/per_image_divergence.csv`` (the 10 images where Mamba
and FPN differ the most) and renders each image with predictions from
all three necks side by side. Small-object GTs are highlighted in red,
predictions with confidence >= 0.3 in green.

Outputs:
    * ``results/figures/qualitative/{filename}_{neck}.png`` (30 images)

Usage:
    PYTHONPATH=. python eval/qualitative.py \
        [--data-root /content/visdrone] \
        [--ckpt-dir /content/drive/MyDrive/ba] \
        [--results results] [--conf-thr 0.3]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import necks as _  # noqa: F401

from eval.constants import (COCO_SMALL, COLORS, DEFAULT_CKPT_DIR,
                             DEFAULT_DATA_ROOT, NECKS)
from eval.utils import get_best_checkpoint


def _load_gt(data_root: Path) -> Dict[str, List[Dict]]:
    """Return {filename: [ann_dict, ...]} for val set."""
    coco_path = data_root / 'annotations' / 'val_unsliced.json'
    with coco_path.open() as f:
        coco = json.load(f)
    id_to_name = {img['id']: img['file_name'] for img in coco['images']}
    gt: Dict[str, List[Dict]] = {}
    for ann in coco['annotations']:
        fname = id_to_name.get(ann['image_id'])
        if fname:
            gt.setdefault(fname, []).append(ann)
    return gt


def _draw(img_path: Path, gt_anns: List[Dict], preds: dict,
          neck: str, conf_thr: float, out_path: Path) -> None:
    """Draw GT boxes (red for small, grey otherwise) and preds (green)."""
    img = Image.open(img_path).convert('RGB')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)

    # GT boxes
    for ann in gt_anns:
        x, y, w, h = ann['bbox']
        area = w * h
        color = 'red' if area < COCO_SMALL else '#888888'
        lw = 1.5 if area < COCO_SMALL else 0.8
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=lw,
            edgecolor=color, facecolor='none', linestyle='--'))

    # Predictions
    bboxes = preds['bboxes']
    scores = preds['scores']
    for i in range(len(scores)):
        if scores[i] < conf_thr:
            continue
        x1, y1, x2, y2 = bboxes[i]
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1,
            edgecolor='lime', facecolor='none'))
        ax.text(x1, max(0, y1 - 3), f'{scores[i]:.2f}',
                color='lime', fontsize=5, weight='bold')

    n_preds = int((np.array(scores) >= conf_thr).sum())
    ax.set_title(
        f'{neck.upper()} | GT={len(gt_anns)} | Preds(>={conf_thr})={n_preds}',
        fontsize=11, color=COLORS.get(neck, 'black'))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualise(data_root: Path, ckpt_dir: Path,
              results_dir: Path, conf_thr: float) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = results_dir / 'figures' / 'qualitative'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load divergence list
    div_path = results_dir / 'per_image_divergence.csv'
    if not div_path.exists():
        print(f'{div_path} not found – run eval/per_image.py first')
        return
    with div_path.open() as f:
        filenames = [r['file_name'] for r in csv.DictReader(f)]

    gt = _load_gt(data_root)
    img_dir = data_root / 'raw' / 'VisDrone2019-DET-val' / 'images'

    # Run inference per neck and draw
    from mmdet.apis import init_detector, inference_detector

    for neck in NECKS:
        _, ckpt = get_best_checkpoint(ckpt_dir, neck)
        print(f'{neck}: {ckpt.name}')
        det = init_detector(f'configs/{neck}.py',
                            str(ckpt), device=str(device))

        for fname in filenames:
            img_path = img_dir / fname
            if not img_path.exists():
                continue
            result = inference_detector(det, str(img_path))
            preds = dict(
                bboxes=result.pred_instances.bboxes.cpu().numpy().tolist(),
                scores=result.pred_instances.scores.cpu().numpy().tolist(),
            )
            stem = Path(fname).stem
            out = out_dir / f'{stem}_{neck}.png'
            _draw(img_path, gt.get(fname, []), preds,
                  neck, conf_thr, out)
        del det
        torch.cuda.empty_cache()

    print(f'\nwrote {len(filenames) * len(NECKS)} images to {out_dir}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path,
                    default=Path(DEFAULT_DATA_ROOT))
    ap.add_argument('--ckpt-dir', type=Path,
                    default=Path(DEFAULT_CKPT_DIR))
    ap.add_argument('--results', type=Path, default=Path('results'))
    ap.add_argument('--conf-thr', type=float, default=0.3)
    args = ap.parse_args()
    visualise(args.data_root, args.ckpt_dir, args.results, args.conf_thr)


if __name__ == '__main__':
    main()
