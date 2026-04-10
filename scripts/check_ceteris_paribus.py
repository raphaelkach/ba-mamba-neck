"""Mechanical ceteris-paribus check for the V1/V2/V3 configs.

Enforces CLAUDE.md's rule: ``configs/fpn.py``, ``configs/aifi.py`` and
``configs/mamba.py`` must differ **only** in ``model.neck``. The script
loads all three configs, redacts ``model.neck`` and compares the rest as
plain Python dicts. Any divergence triggers a non-zero exit so the
notebook pipeline stops immediately.

Usage:
    PYTHONPATH=. python scripts/check_ceteris_paribus.py
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from typing import Dict

from mmengine.config import Config

CFGS: Dict[str, str] = {
    'fpn':   'configs/fpn.py',
    'aifi':  'configs/aifi.py',
    'mamba': 'configs/mamba.py',
}


def _redact_neck(cfg_dict: dict) -> dict:
    """Return a deep copy with ``model.neck`` replaced by a sentinel."""
    c = deepcopy(cfg_dict)
    c.setdefault('model', {})
    c['model']['neck'] = '<REDACTED>'
    return c


def _diff_keys(a: dict, b: dict, path: str = '') -> list[str]:
    """Recursively collect dotted paths where ``a`` and ``b`` differ."""
    diffs: list[str] = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        sub = f'{path}.{k}' if path else k
        if k not in a:
            diffs.append(f'+ {sub} (only in second)')
        elif k not in b:
            diffs.append(f'- {sub} (only in first)')
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            diffs.extend(_diff_keys(a[k], b[k], sub))
        elif a[k] != b[k]:
            diffs.append(f'~ {sub}: {a[k]!r} != {b[k]!r}')
    return diffs


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    loaded: Dict[str, dict] = {}
    for name, rel in CFGS.items():
        loaded[name] = Config.fromfile(str(repo_root / rel)).to_dict()

    # Print the three neck dicts for the reader.
    for name in CFGS:
        print(f'[{name}] model.neck = {loaded[name]["model"]["neck"]}')
    print()

    ref = _redact_neck(loaded['fpn'])
    errors: list[str] = []
    for name in ('aifi', 'mamba'):
        other = _redact_neck(loaded[name])
        diffs = _diff_keys(ref, other)
        if diffs:
            errors.append(
                f'[{name}] differs from [fpn] outside of model.neck:\n    '
                + '\n    '.join(diffs)
            )

    if errors:
        print('CETERIS-PARIBUS VIOLATED:')
        for e in errors:
            print(e)
        return 1

    print('OK: V1/V2/V3 differ only in model.neck')
    return 0


if __name__ == '__main__':
    sys.exit(main())
