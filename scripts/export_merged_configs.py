"""Export merged MMDetection configs as JSON for diff-stable comparison.

Loads each top-level config via mmengine.Config.fromfile(), converts
non-serializable values (callables, types) to strings, and writes
deterministic JSON files.

Usage:
    PYTHONPATH=. python scripts/export_merged_configs.py [--outdir configs/baseline_snapshots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmengine.config import Config


CONFIGS = {
    'fpn': 'configs/fpn.py',
    'aifi': 'configs/aifi.py',
    'mamba': 'configs/mamba.py',
}


def _make_serializable(obj):
    """Recursively convert non-JSON-serializable values to strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    return str(obj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', type=Path,
                    default=Path('configs/baseline_snapshots'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for name, cfg_path in CONFIGS.items():
        cfg = Config.fromfile(cfg_path)
        data = _make_serializable(cfg.to_dict())
        out = args.outdir / f'{name}_baseline.json'
        with out.open('w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        lines = out.read_text().count('\n') + 1
        size = out.stat().st_size
        print(f'{out}: {lines} lines, {size} bytes')


if __name__ == '__main__':
    main()
