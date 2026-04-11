"""Forward-shape and parameter-count check for V1/V2/V3 necks.

Builds each neck in isolation (no backbone, no head), feeds it a set of
dummy C2..C5 feature maps matching the ResNet-50 @ 640x640 shape
contract, and verifies that

    1. each neck returns a 5-tuple,
    2. every output has ``channels == out_channels`` (256),
    3. all three necks produce the **same** output shapes,
    4. parameter counts are within the same order of magnitude
       (max / min < 3x; otherwise a warning is printed).

Runs on CUDA if available, otherwise CPU. Intended to be called from
``notebooks/02_train.ipynb`` on Colab but safe to run locally for
smoke-testing too.

Usage:
    PYTHONPATH=. python scripts/test_necks.py
"""

from __future__ import annotations

import sys
import warnings
from typing import Dict, List, Tuple

import torch

try:
    from mmdet.registry import MODELS
except ImportError:  # pragma: no cover - local smoke tests
    from mmengine.registry import MODELS

# Ensure AifiNeck / MambaNeck are registered before MODELS.build().
import necks  # noqa: F401


IN_CHANNELS = [256, 512, 1024, 2048]
OUT_CHANNELS = 256
NUM_OUTS = 5

NECK_CFGS: Dict[str, dict] = {
    'fpn': dict(
        type='FPN',
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=NUM_OUTS,
    ),
    'aifi': dict(
        type='AifiNeck',
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_outs=NUM_OUTS,
    ),
    'mamba': dict(
        type='MambaNeck',
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_outs=NUM_OUTS,
    ),
}


def _build_neck(name: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    """Build a neck via MODELS.build (FPN falls back to a direct import)."""
    try:
        neck = MODELS.build(cfg)
    except Exception as e:
        if name == 'fpn':
            try:
                from mmdet.models.necks import FPN
            except ImportError:
                raise RuntimeError(
                    'mmdet not installed - cannot build FPN. Install '
                    'mmdet>=3.0 or run this script on Colab.'
                ) from e
            neck = FPN(**{k: v for k, v in cfg.items() if k != 'type'})
        else:
            raise
    return neck.to(device).eval()


def _dummy_inputs(device: torch.device) -> List[torch.Tensor]:
    """ResNet-50 @ 640x640 feature shapes."""
    shapes = [
        (1, 256, 160, 160),   # C2 - dropped by start_level=1
        (1, 512, 80, 80),     # C3
        (1, 1024, 40, 40),    # C4
        (1, 2048, 20, 20),    # C5
    ]
    return [torch.randn(*s, device=device) for s in shapes]


def _param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type != 'cuda':
        warnings.warn(
            'No CUDA device visible. MambaNeck will use its naive '
            'scan fallback - results are for shape-testing only.'
        )

    torch.manual_seed(0)
    inputs = _dummy_inputs(device)

    shapes_per_neck: Dict[str, List[Tuple[int, int, int, int]]] = {}
    param_counts: Dict[str, int] = {}

    for name, cfg in NECK_CFGS.items():
        neck = _build_neck(name, cfg, device)
        with torch.no_grad():
            outs = neck(inputs)
        assert isinstance(outs, tuple), (
            f'[{name}] forward must return tuple, got {type(outs).__name__}'
        )
        assert len(outs) == NUM_OUTS, (
            f'[{name}] expected {NUM_OUTS} outputs, got {len(outs)}'
        )
        for i, t in enumerate(outs):
            assert t.shape[1] == OUT_CHANNELS, (
                f'[{name}] level {i}: channel mismatch '
                f'({t.shape[1]} != {OUT_CHANNELS})'
            )
        shapes_per_neck[name] = [tuple(t.shape) for t in outs]
        param_counts[name] = _param_count(neck)
        print(f'[{name}] shapes: {shapes_per_neck[name]}')

    # Check that output shapes are identical across all three necks.
    ref_name = next(iter(shapes_per_neck))
    ref_shapes = shapes_per_neck[ref_name]
    mismatched = [n for n, s in shapes_per_neck.items() if s != ref_shapes]
    if mismatched:
        print('SHAPE MISMATCH:')
        for n in mismatched:
            print(f'  [{n}] {shapes_per_neck[n]} != [{ref_name}] {ref_shapes}')
        return 1
    print(f'\nall three necks produce identical shapes: {ref_shapes}')

    # Parameter-count table and spread sanity check.
    min_count = min(param_counts.values())
    max_count = max(param_counts.values())
    spread = max_count / max(min_count, 1)
    print('\nparameter counts:')
    print(f'  {"neck":<8} {"params":>14}  {"ratio/min":>10}')
    for name, cnt in param_counts.items():
        print(f'  {name:<8} {cnt:>14,}  {cnt / min_count:>9.2f}x')

    if spread > 3.0:
        print(
            f'\nWARNING: parameter-count spread {spread:.2f}x > 3x. '
            'The three necks are not in the same order of magnitude - '
            'consider tuning num_ccfm_blocks (AifiNeck) or '
            'num_vss_blocks / expand (MambaNeck).'
        )
    else:
        print(f'\nspread {spread:.2f}x <= 3x - all necks within same order of magnitude.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
