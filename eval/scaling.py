"""Inference-time scaling benchmark across resolutions.

For each of the three necks, measures latency, peak GPU memory, GFLOPs
(neck-only and full model) and throughput at five input resolutions.

Outputs:
    * ``results/scaling.csv``
    * ``results/neck_params.csv``

Usage:
    PYTHONPATH=. python eval/scaling.py [--results results]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

import necks as _necks  # noqa: F401 – register modules

NECKS_CFG: Dict[str, str] = {
    'fpn': 'configs/fpn.py',
    'aifi': 'configs/aifi.py',
    'mamba': 'configs/mamba.py',
}
RESOLUTIONS = [480, 640, 800, 1024, 1280]
N_WARMUP = 50
N_MEASURE = 500


def _build_detector(cfg_path: str, device: torch.device):
    from mmdet.apis import init_detector
    return init_detector(cfg_path, device=str(device))


def _flops(module: nn.Module, inputs, device: torch.device) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        fca = FlopCountAnalysis(module, inputs)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total() / 1e9
    except Exception:
        return float('nan')


def _dummy_features(res: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Create dummy backbone features for a given square resolution."""
    return tuple(
        torch.randn(1, c, res // s, res // s, device=device)
        for c, s in [(256, 4), (512, 8), (1024, 16), (2048, 32)]
    )


def _measure(det, neck: nn.Module, res: int,
             device: torch.device) -> Dict:
    img = torch.randn(1, 3, res, res, device=device)
    feats = _dummy_features(res, device)

    # ── GFLOPs ───────────────────────────────────────────────────
    gflops_total = _flops(det, img, device)
    gflops_neck = _flops(neck, feats, device)

    # ── Latency + throughput (full model) ────────────────────────
    with torch.no_grad():
        for _ in range(N_WARMUP):
            det(img)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(N_MEASURE):
            det(img)
        end.record()
        torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end) / N_MEASURE
    throughput = 1000.0 / latency_ms if latency_ms > 0 else 0

    # ── Peak memory ──────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        det(img)
    torch.cuda.synchronize()
    memory_gb = torch.cuda.max_memory_allocated() / 1e9

    return dict(
        latency_ms=round(latency_ms, 2),
        memory_gb=round(memory_gb, 3),
        gflops_total=round(gflops_total, 2),
        gflops_neck=round(gflops_neck, 2),
        throughput=round(throughput, 1),
    )


def benchmark(results_dir: Path) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'GPU required for scaling benchmark'
    results_dir.mkdir(parents=True, exist_ok=True)

    scaling_rows: List[Dict] = []
    param_rows: List[Dict] = []

    for neck_name, cfg_path in NECKS_CFG.items():
        print(f'\n=== {neck_name} ===')
        det = _build_detector(cfg_path, device).eval()
        neck_module = det.neck

        params_total = sum(p.numel() for p in det.parameters())
        params_neck = sum(p.numel() for p in neck_module.parameters())
        param_rows.append(dict(
            neck=neck_name,
            params_neck=params_neck,
            params_total=params_total,
            ratio=round(params_neck / params_total * 100, 2),
        ))

        for res in RESOLUTIONS:
            print(f'  {res}x{res} ...', end=' ', flush=True)
            result = _measure(det, neck_module, res, device)
            result.update(
                neck=neck_name,
                resolution=res,
                params_total=params_total,
                params_neck=params_neck,
            )
            scaling_rows.append(result)
            print(f'{result["latency_ms"]}ms  {result["gflops_total"]}G  '
                  f'{result["memory_gb"]}GB')
        del det
        torch.cuda.empty_cache()

    # Write CSVs
    sc_path = results_dir / 'scaling.csv'
    sc_cols = ['neck', 'resolution', 'latency_ms', 'memory_gb',
               'gflops_total', 'gflops_neck', 'throughput',
               'params_total', 'params_neck']
    with sc_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sc_cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(scaling_rows)
    print(f'\nwrote {sc_path}')

    np_path = results_dir / 'neck_params.csv'
    with np_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['neck', 'params_neck',
                                          'params_total', 'ratio'])
        w.writeheader()
        w.writerows(param_rows)
    print(f'wrote {np_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path('results'))
    args = ap.parse_args()
    benchmark(args.results)


if __name__ == '__main__':
    main()
