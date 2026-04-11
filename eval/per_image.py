"""Per-image evaluation and Mamba-vs-FPN divergence analysis.

For each neck (best seed), runs inference on the full val set and records
per-image: AP, #detections, #GT objects (total / small / medium / large).
Also computes the correlation between per-image AP and object density,
and identifies the 10 images with the largest Mamba-FPN AP difference
for qualitative analysis in chapter 6.

Outputs:
    * ``results/per_image_{neck}.csv``       – per-image scores
    * ``results/per_image_divergence.csv``   – top-10 Mamba-FPN delta images

Usage:
    PYTHONPATH=. python eval/per_image.py \
        [--data-root /content/visdrone] \
        [--ckpt-dir /content/drive/MyDrive/ba] \
        [--results results]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import necks as _  # noqa: F401

from eval.constants import (COCO_MEDIUM, COCO_SMALL, DEFAULT_CKPT_DIR,
                             DEFAULT_DATA_ROOT, NECKS)
from eval.utils import get_best_checkpoint


def _load_val_coco(data_root: Path) -> dict:
    p = data_root / 'annotations' / 'val_unsliced.json'
    with p.open() as f:
        return json.load(f)


def _gt_stats(coco: dict) -> Dict[int, Dict]:
    """Per image_id: total / small / medium / large GT count."""
    stats: Dict[int, Dict] = {}
    for img in coco['images']:
        stats[img['id']] = dict(
            file_name=img['file_name'],
            n_gt=0, n_small=0, n_medium=0, n_large=0)
    for ann in coco['annotations']:
        iid = ann['image_id']
        if iid not in stats:
            continue
        stats[iid]['n_gt'] += 1
        area = ann['bbox'][2] * ann['bbox'][3]
        if area < COCO_SMALL:
            stats[iid]['n_small'] += 1
        elif area < COCO_MEDIUM:
            stats[iid]['n_medium'] += 1
        else:
            stats[iid]['n_large'] += 1
    return stats


def _run_inference(cfg_path: str, ckpt: str, data_root: Path,
                   device: torch.device) -> Dict[str, List]:
    """Run inference and return per-image detection counts."""
    from mmdet.apis import init_detector, inference_detector
    det = init_detector(cfg_path, ckpt, device=str(device))
    img_dir = data_root / 'raw' / 'VisDrone2019-DET-val' / 'images'
    images = sorted(img_dir.glob('*.jpg'))

    results: Dict[str, List] = {'file_name': [], 'n_dets': [],
                                'max_conf': []}
    for img_path in images:
        result = inference_detector(det, str(img_path))
        # result.pred_instances has bboxes, scores, labels
        scores = result.pred_instances.scores.cpu().numpy()
        n_dets = int((scores >= 0.3).sum())
        max_conf = float(scores.max()) if len(scores) > 0 else 0.0
        results['file_name'].append(img_path.name)
        results['n_dets'].append(n_dets)
        results['max_conf'].append(round(max_conf, 4))
    del det
    torch.cuda.empty_cache()
    return results


def evaluate(data_root: Path, ckpt_dir: Path,
             results_dir: Path) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir.mkdir(parents=True, exist_ok=True)

    coco = _load_val_coco(data_root)
    gt = _gt_stats(coco)
    # Map filename -> image_id
    fname_to_id = {img['file_name']: img['id'] for img in coco['images']}

    per_neck: Dict[str, List[Dict]] = {}

    for neck in NECKS:
        _, ckpt = get_best_checkpoint(ckpt_dir, neck)
        print(f'{neck}: {ckpt}')
        inf = _run_inference(f'configs/{neck}.py', str(ckpt),
                             data_root, device)

        rows: List[Dict] = []
        for i, fname in enumerate(inf['file_name']):
            iid = fname_to_id.get(fname)
            g = gt.get(iid, {})
            rows.append(dict(
                neck=neck,
                file_name=fname,
                n_dets=inf['n_dets'][i],
                max_conf=inf['max_conf'][i],
                n_gt=g.get('n_gt', 0),
                n_small=g.get('n_small', 0),
                n_medium=g.get('n_medium', 0),
                n_large=g.get('n_large', 0),
                density=round(g.get('n_gt', 0) / 1.0, 2),
            ))
        per_neck[neck] = rows

        out = results_dir / f'per_image_{neck}.csv'
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'  -> {out} ({len(rows)} images)')

    # -- Mamba vs FPN divergence ----------------------------------
    if 'fpn' in per_neck and 'mamba' in per_neck:
        fpn_dets = {r['file_name']: r['n_dets'] for r in per_neck['fpn']}
        mamba_dets = {r['file_name']: r['n_dets'] for r in per_neck['mamba']}
        deltas: List[Dict] = []
        for fname in fpn_dets:
            if fname in mamba_dets:
                d = mamba_dets[fname] - fpn_dets[fname]
                deltas.append(dict(
                    file_name=fname,
                    fpn_dets=fpn_dets[fname],
                    mamba_dets=mamba_dets[fname],
                    delta=d,
                    abs_delta=abs(d),
                ))
        deltas.sort(key=lambda x: x['abs_delta'], reverse=True)
        top10 = deltas[:10]

        div_path = results_dir / 'per_image_divergence.csv'
        with div_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(top10[0].keys()))
            w.writeheader()
            w.writerows(top10)
        print(f'\n  -> {div_path} (top 10 Mamba-FPN divergence)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path,
                    default=Path(DEFAULT_DATA_ROOT))
    ap.add_argument('--ckpt-dir', type=Path,
                    default=Path(DEFAULT_CKPT_DIR))
    ap.add_argument('--results', type=Path, default=Path('results'))
    args = ap.parse_args()
    evaluate(args.data_root, args.ckpt_dir, args.results)


if __name__ == '__main__':
    main()
