"""Merge per-seed result CSVs into unified detection tables.

Reads ``results/{neck}_seed_results.csv`` (produced by notebook 02) for
each of the three neck variants, enriches rows with neck parameter
counts, and writes:

    * ``results/detection_all.csv``     – 30 rows (3 necks x 10 seeds)
    * ``results/detection_summary.csv`` – per-neck mean/std/median/min/max
    * ``results/detection_deltas.csv``  – AIFI-vs-FPN and Mamba-vs-FPN

Usage:
    PYTHONPATH=. python eval/metrics.py [--results results]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


NECKS = ['fpn', 'aifi', 'mamba']
SEEDS = [42, 123, 456, 789, 1024, 2048, 3407, 4096, 5555, 7777]

CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor',
]

METRIC_COLS = [
    'mAP', 'mAP_50', 'mAP_75', 'AP_S', 'AP_M', 'AP_L',
    'AR_1', 'AR_10', 'AR_100', 'AR_S', 'AR_M', 'AR_L',
] + [f'AP_{c}' for c in CLASS_NAMES]

META_COLS = ['train_time_h', 'peak_gpu_mem_gb', 'best_epoch', 'num_params_neck']


def _read_seed_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in r:
                if k not in ('neck',):
                    try:
                        r[k] = float(r[k])
                    except (ValueError, TypeError):
                        pass
            rows.append(r)
    return rows


def _neck_param_count(neck_name: str) -> int:
    """Return the number of parameters in a neck (import-safe)."""
    try:
        import torch
        import necks as _  # noqa: F811 – register modules
        if neck_name == 'fpn':
            from mmdet.models.necks import FPN
            m = FPN(in_channels=[256, 512, 1024, 2048],
                    out_channels=256, start_level=1,
                    add_extra_convs='on_output', num_outs=5)
        elif neck_name == 'aifi':
            from necks.aifi_neck import AifiNeck
            m = AifiNeck(in_channels=[256, 512, 1024, 2048],
                         out_channels=256, num_outs=5)
        elif neck_name == 'mamba':
            from necks.mamba_neck import MambaNeck
            m = MambaNeck(in_channels=[256, 512, 1024, 2048],
                          out_channels=256, num_outs=5)
        else:
            return 0
        return sum(p.numel() for p in m.parameters())
    except Exception:
        return 0


def merge(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []

    for neck in NECKS:
        csv_path = results_dir / f'{neck}_seed_results.csv'
        if not csv_path.exists():
            print(f'WARNING: {csv_path} not found – skipping {neck}')
            continue
        neck_params = _neck_param_count(neck)
        for r in _read_seed_csv(csv_path):
            r['num_params_neck'] = neck_params
            all_rows.append(r)

    if not all_rows:
        print('No seed CSVs found in', results_dir)
        return

    # -- detection_all.csv ----------------------------------------
    all_cols = ['neck', 'seed'] + METRIC_COLS + META_COLS
    all_path = results_dir / 'detection_all.csv'
    with all_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f'wrote {all_path} ({len(all_rows)} rows)')

    # -- detection_summary.csv ------------------------------------
    summary_rows: List[Dict] = []
    for neck in NECKS:
        subset = [r for r in all_rows if r['neck'] == neck]
        if not subset:
            continue
        row: Dict = {'neck': neck, 'n_seeds': len(subset)}
        for col in METRIC_COLS + META_COLS:
            vals = np.array([float(r.get(col, 0)) for r in subset])
            row[f'{col}_mean'] = round(float(vals.mean()), 4)
            row[f'{col}_std'] = round(float(vals.std()), 4)
            row[f'{col}_median'] = round(float(np.median(vals)), 4)
            row[f'{col}_min'] = round(float(vals.min()), 4)
            row[f'{col}_max'] = round(float(vals.max()), 4)
        summary_rows.append(row)

    if summary_rows:
        sum_path = results_dir / 'detection_summary.csv'
        cols = list(summary_rows[0].keys())
        with sum_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)
        print(f'wrote {sum_path}')

    # -- detection_deltas.csv -------------------------------------
    fpn_means = {col: np.mean([float(r.get(col, 0))
                               for r in all_rows if r['neck'] == 'fpn'])
                 for col in METRIC_COLS}
    delta_rows: List[Dict] = []
    for neck in ('aifi', 'mamba'):
        subset = [r for r in all_rows if r['neck'] == neck]
        if not subset:
            continue
        row: Dict = {'comparison': f'{neck}_vs_fpn'}
        for col in METRIC_COLS:
            neck_mean = np.mean([float(r.get(col, 0)) for r in subset])
            delta = neck_mean - fpn_means.get(col, 0)
            pct = 100 * delta / fpn_means[col] if fpn_means.get(col) else 0
            row[f'{col}_delta'] = round(float(delta), 4)
            row[f'{col}_pct'] = round(float(pct), 2)
        delta_rows.append(row)

    if delta_rows:
        delta_path = results_dir / 'detection_deltas.csv'
        cols = list(delta_rows[0].keys())
        with delta_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(delta_rows)
        print(f'wrote {delta_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    args = ap.parse_args()
    merge(args.results)


if __name__ == '__main__':
    main()
