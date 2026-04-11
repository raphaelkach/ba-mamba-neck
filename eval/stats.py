"""Statistical tests on the detection results.

Reads ``results/detection_all.csv`` and runs:

    * Friedman test per metric (3 groups x 10 seeds)
    * Nemenyi post-hoc (if p < 0.05) with CD diagrams
    * Kendall's W effect size
    * Pairwise Wilcoxon signed-rank with rank-biserial r

Outputs:
    * ``results/friedman.csv``
    * ``results/nemenyi.csv``
    * ``results/wilcoxon.csv``
    * ``results/figures/cd_diagram_mAP.pdf``
    * ``results/figures/cd_diagram_AP_S.pdf``

Usage:
    PYTHONPATH=. python eval/stats.py [--results results]
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
import pandas as pd
import sciposthocs as sp
from scipy import stats

from eval.constants import NECKS, SEEDS

TEST_METRICS = ['mAP', 'mAP_50', 'mAP_75', 'AP_S', 'AP_M', 'AP_L', 'AR_S']


def _load_all(results_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load detection_all.csv, return {metric: {neck: array}}."""
    path = results_dir / 'detection_all.csv'
    data: Dict[str, Dict[str, List[float]]] = {
        m: {n: [] for n in NECKS} for m in TEST_METRICS
    }
    with path.open() as f:
        for row in csv.DictReader(f):
            neck = row['neck']
            if neck not in NECKS:
                continue
            for m in TEST_METRICS:
                data[m][neck].append(float(row.get(m, 0)))

    return {m: {n: np.array(v) for n, v in d.items()} for m, d in data.items()}


def _kendall_w(chi2: float, k: int, n: int) -> float:
    """Kendall's coefficient of concordance W from Friedman chi2."""
    return chi2 / (k * (n - 1)) if n > 1 else 0.0


def _rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size r = Z / sqrt(n) from Wilcoxon signed-rank."""
    diffs = x - y
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 0.0
    res = stats.wilcoxon(x, y)
    z = stats.norm.ppf(res.pvalue / 2)
    return abs(z) / np.sqrt(n)


def _cd_diagram(data: Dict[str, np.ndarray], metric: str,
                out_path: Path) -> None:
    """Produce a critical difference diagram via scikit-posthocs."""
    records = []
    for neck, vals in data.items():
        for i, v in enumerate(vals):
            records.append({'seed': i, 'neck': neck, 'value': v})
    df = pd.DataFrame(records)
    pivot = df.pivot(index='seed', columns='neck', values='value')

    fig, ax = plt.subplots(figsize=(6, 2))
    try:
        avg_ranks = pivot.rank(axis=1, ascending=False).mean()
        n = len(pivot)
        k = len(NECKS)
        cd = sp.critical_difference_diagram(
            pivot, ax=ax)
    except Exception:
    # Fallback: simple bar plot of average ranks
        avg_ranks = pivot.rank(axis=1, ascending=False).mean()
        ax.barh(range(k), avg_ranks[NECKS], color=['#2196F3', '#9C27B0', '#009688'])
        ax.set_yticks(range(k))
        ax.set_yticklabels(NECKS)
        ax.set_xlabel('Mean rank')
        ax.set_title(f'CD diagram – {metric}')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path}')


def run_stats(results_dir: Path) -> None:
    data = _load_all(results_dir)
    fig_dir = results_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -- Friedman -------------------------------------------------
    friedman_rows: List[Dict] = []
    sig_metrics: List[str] = []
    for m in TEST_METRICS:
        arrays = [data[m][n] for n in NECKS]
        if any(len(a) == 0 for a in arrays):
            continue
        chi2, p = stats.friedmanchisquare(*arrays)
        k = len(NECKS)
        n = len(arrays[0])
        w = _kendall_w(chi2, k, n)
        friedman_rows.append({
            'metric': m, 'chi2': round(chi2, 4), 'p': round(p, 6),
            'W': round(w, 4), 'significant': p < 0.05,
        })
        if p < 0.05:
            sig_metrics.append(m)
    _write_csv(results_dir / 'friedman.csv', friedman_rows)

    # -- Nemenyi --------------------------------------------------
    nemenyi_rows: List[Dict] = []
    for m in sig_metrics:
        vals = pd.DataFrame({n: data[m][n] for n in NECKS})
        pvals = sp.posthoc_nemenyi_friedman(vals)
        pairs = [('fpn', 'aifi'), ('fpn', 'mamba'), ('aifi', 'mamba')]
        for a, b in pairs:
            p_adj = float(pvals.loc[a, b])
            nemenyi_rows.append({
                'metric': m, 'pair': f'{a}_vs_{b}',
                'p_adj': round(p_adj, 6),
                'significant': p_adj < 0.05,
            })
    _write_csv(results_dir / 'nemenyi.csv', nemenyi_rows)

    # -- Wilcoxon -------------------------------------------------
    wilcoxon_rows: List[Dict] = []
    pairs = [('fpn', 'aifi'), ('fpn', 'mamba'), ('aifi', 'mamba')]
    for m in TEST_METRICS:
        for a, b in pairs:
            x, y = data[m][a], data[m][b]
            if len(x) == 0 or len(y) == 0:
                continue
            try:
                stat_val, p = stats.wilcoxon(x, y)
                r = _rank_biserial_r(x, y)
            except ValueError:
                stat_val, p, r = 0, 1.0, 0.0
            wilcoxon_rows.append({
                'metric': m, 'pair': f'{a}_vs_{b}',
                'stat': round(float(stat_val), 4),
                'p': round(float(p), 6),
                'r_effect_size': round(r, 4),
                'significant': p < 0.05,
            })
    _write_csv(results_dir / 'wilcoxon.csv', wilcoxon_rows)

    # -- CD diagrams ----------------------------------------------
    for m in ['mAP', 'AP_S']:
        if m in data and all(len(data[m][n]) > 0 for n in NECKS):
            _cd_diagram(data[m], m,
                        fig_dir / f'cd_diagram_{m}.pdf')


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        print(f'  (no data for {path.name})')
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path} ({len(rows)} rows)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    args = ap.parse_args()
    run_stats(args.results)


if __name__ == '__main__':
    main()
