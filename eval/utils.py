"""Shared utilities for the evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .constants import SEEDS


def get_best_checkpoint(ckpt_dir: Path, neck: str,
                        seeds: List[int] | None = None) -> Tuple[int, Path]:
    """Find the first available best checkpoint across seeds.

    Args:
        ckpt_dir: Base checkpoint directory (e.g. /content/drive/MyDrive/ba).
        neck: Neck name (fpn, aifi, mamba).
        seeds: Seed list to search. Defaults to ``constants.SEEDS``.

    Returns:
        Tuple of (seed, checkpoint_path).

    Raises:
        FileNotFoundError: If no best checkpoint is found for any seed.
    """
    if seeds is None:
        seeds = SEEDS
    for seed in seeds:
        ckpts = sorted(
            (ckpt_dir / neck / f'seed_{seed}').glob('best_*.pth'))
        if ckpts:
            return seed, ckpts[0]
    raise FileNotFoundError(
        f'No best checkpoint found for {neck} in {ckpt_dir / neck}')
