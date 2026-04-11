"""Merge per-neck convergence CSVs and compute cross-seed summaries.

Reads ``results/{neck}_convergence.csv`` (produced by notebook 02,
cell 6) and writes:

    * ``results/convergence_all.csv``     – all epochs, all seeds, all necks
    * ``results/convergence_summary.csv`` – per-neck per-epoch mean +/- std

Usage:
    PYTHONPATH=. python eval/convergence.py [--results results]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from eval.constants import NECKS
METRIC_COLS = ['train_loss', 'cls_loss', 'bbox_loss', 'centerness_loss',
               'val_mAP', 'val_AP_S', 'val_AP_M', 'val_AP_L', 'lr']


def merge(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []

    for neck in NECKS:
        path = results_dir / f'{neck}_convergence.csv'
        if not path.exists():
            print(f'WARNING: {path} not found – skipping {neck}')
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                all_rows.append(row)

    if not all_rows:
        print('No convergence CSVs found')
        return

    # -- convergence_all.csv --------------------------------------
    all_cols = ['neck', 'seed', 'epoch'] + METRIC_COLS
    all_path = results_dir / 'convergence_all.csv'
    with all_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f'wrote {all_path} ({len(all_rows)} rows)')

    # -- convergence_summary.csv ----------------------------------
    # Group by (neck, epoch), compute mean/std per metric.
    grouped: Dict[tuple, List[Dict]] = {}
    for row in all_rows:
        key = (row['neck'], int(float(row['epoch'])))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for (neck, epoch), rows in sorted(grouped.items()):
        s: Dict = {'neck': neck, 'epoch': epoch}
        for col in METRIC_COLS:
            vals = []
            for r in rows:
                v = r.get(col)
                if v is not None and v != '' and v != 'None':
                    vals.append(float(v))
            if vals:
                arr = np.array(vals)
                s[f'{col}_mean'] = round(float(arr.mean()), 6)
                s[f'{col}_std'] = round(float(arr.std()), 6)
            else:
                s[f'{col}_mean'] = None
                s[f'{col}_std'] = None
        summary_rows.append(s)

    if summary_rows:
        sum_path = results_dir / 'convergence_summary.csv'
        cols = list(summary_rows[0].keys())
        with sum_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary_rows)
        print(f'wrote {sum_path} ({len(summary_rows)} rows)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    args = ap.parse_args()
    merge(args.results)


if __name__ == '__main__':
    main()
