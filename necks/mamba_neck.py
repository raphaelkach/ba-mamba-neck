"""V3 - MambaNeck (SSM-based FPN, MambaFPN-style).

Drop-in MMDetection neck that replaces FPN's convolutional lateral
refinement by VSS-blocks (Visual State Space, Liu et al. 2024 / VMamba).
Each VSS-block performs a 2D selective scan in four directions
(row forward/backward, column forward/backward) via the ``mamba-ssm``
CUDA kernels and sums them (Cross-Scan merge).

Shape contract - identical to :class:`mmdet.models.necks.FPN` with
``start_level=1`` and ``num_outs=5``.

Correctness notes
-----------------
The selective scan is numerically well-defined only with the CUDA
kernels from the ``mamba-ssm`` package. This module does ship a naive
PyTorch reference implementation but it is **only** suitable for CPU
shape tests. Training and evaluation **must** use the CUDA path - the
module emits a warning when either ``mamba-ssm`` is unavailable or no
CUDA device is visible, so that a misconfigured Colab runtime is
detected immediately.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.registry import MODELS
except ImportError:  # pragma: no cover - local config-parse only
    from mmengine.registry import MODELS

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _MAMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    selective_scan_fn = None
    _MAMBA_AVAILABLE = False
    warnings.warn(
        'mamba-ssm not installed. MambaNeck will fall back to a naive '
        'PyTorch selective-scan implementation that is ONLY valid for '
        'CPU unit tests. Install mamba-ssm (and causal-conv1d) before '
        'running training or evaluation.'
    )

if _MAMBA_AVAILABLE and not torch.cuda.is_available():
    warnings.warn(
        'mamba-ssm requires CUDA. Running on CPU will still invoke the '
        'naive PyTorch fallback scan. Use a GPU runtime for training.'
    )


# -----------------------------------------------------------------------------
# Selective scan - reference implementation for CPU unit tests
# -----------------------------------------------------------------------------

def _selective_scan_naive(u: torch.Tensor, delta: torch.Tensor,
                          A: torch.Tensor, B: torch.Tensor,
                          C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Reference Mamba selective scan (slow; correct output shape).

    Shapes:
        u, delta: (Bsz, d_inner, L)
        A:        (d_inner, d_state)
        B, C:     (Bsz, d_state, L)
        D:        (d_inner,) or None
    Returns:
        y: (Bsz, d_inner, L)
    """
    Bsz, d, L = u.shape
    n = A.shape[1]
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
    B_perm = B.permute(0, 2, 1).unsqueeze(1)              # (Bsz, 1, L, n)
    deltaB_u = (delta.unsqueeze(-1) * B_perm) * u.unsqueeze(-1)
    C_perm = C.permute(0, 2, 1)                           # (Bsz, L, n)
    h = torch.zeros(Bsz, d, n, device=u.device, dtype=u.dtype)
    ys: List[torch.Tensor] = []
    for t in range(L):
        h = deltaA[:, :, t] * h + deltaB_u[:, :, t]
        ys.append((h * C_perm[:, t].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + D.view(1, -1, 1) * u
    return y


def _run_scan(u, delta, A, B, C, D):
    """Dispatch to mamba-ssm CUDA kernel or the naive reference."""
    if _MAMBA_AVAILABLE and u.is_cuda:
        return selective_scan_fn(u, delta, A, B, C, D)
    return _selective_scan_naive(u, delta, A, B, C, D)


# -----------------------------------------------------------------------------
# SS2D - 2D selective scan with 4-direction cross-scan
# -----------------------------------------------------------------------------

class SS2D(nn.Module):
    """Selective scan 2D with cross-scan over 4 directions (VMamba)."""

    K = 4  # row fwd, col fwd, row rev, col rev

    def __init__(self, d_model: int, d_state: int = 16,
                 expand: int = 2, dt_rank: int | str = 'auto') -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = (math.ceil(self.d_inner / 16)
                        if dt_rank == 'auto' else int(dt_rank))

        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner,
                      self.dt_rank + 2 * self.d_state, bias=False)
            for _ in range(self.K)
        ])
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True)
            for _ in range(self.K)
        ])
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.repeat(self.d_inner, 1)                     # (d_inner, d_state)
        self.A_logs = nn.ParameterList([
            nn.Parameter(torch.log(A.clone())) for _ in range(self.K)
        ])
        self.Ds = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(self.K)
        ])

    def _scan_direction(self, k: int, u: torch.Tensor) -> torch.Tensor:
        # u: (Bsz, d_inner, L)
        dblproj = self.x_proj[k](u.transpose(1, 2))        # (B, L, dt+2n)
        dt, B_raw, C_raw = torch.split(
            dblproj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = self.dt_proj[k](dt).transpose(1, 2)           # (B, d_inner, L)
        dt = F.softplus(dt)
        B_raw = B_raw.transpose(1, 2).contiguous()         # (B, d_state, L)
        C_raw = C_raw.transpose(1, 2).contiguous()
        A = -torch.exp(self.A_logs[k])                     # (d_inner, d_state)
        return _run_scan(u.contiguous(), dt, A, B_raw, C_raw, self.Ds[k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, d, H, W = x.shape
        L = H * W

        x_row = x.flatten(2)                               # row-major
        x_col = x.transpose(2, 3).flatten(2)               # col-major
        scans = [
            x_row,
            x_col,
            x_row.flip(-1),
            x_col.flip(-1),
        ]
        ys = [self._scan_direction(k, scans[k]) for k in range(self.K)]

        # reverse the scan orders back into row-major (H, W)
        y_row = ys[0]
        y_col = ys[1]
        y_row_r = ys[2].flip(-1)
        y_col_r = ys[3].flip(-1)

        def _col_to_row(t: torch.Tensor) -> torch.Tensor:
            return (t.view(Bsz, d, W, H).transpose(2, 3).reshape(Bsz, d, L))

        y = y_row + _col_to_row(y_col) + y_row_r + _col_to_row(y_col_r)
        return y.view(Bsz, d, H, W)


# -----------------------------------------------------------------------------
# VSSBlock - VMamba Fig. 2c
# -----------------------------------------------------------------------------

class VSSBlock(nn.Module):
    """Single Visual State Space block (pre-LN residual)."""

    def __init__(self, c: int, d_state: int = 16,
                 expand: int = 2, d_conv: int = 3) -> None:
        super().__init__()
        self.d_inner = int(expand * c)
        self.norm = nn.LayerNorm(c)
        self.in_proj = nn.Linear(c, 2 * self.d_inner)
        self.dwconv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner,
        )
        self.ss2d = SS2D(c, d_state=d_state, expand=expand)
        self.out_proj = nn.Linear(self.d_inner, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bsz, c, H, W = x.shape
        residual = x
        y = x.flatten(2).transpose(1, 2)                   # (B, HW, C)
        y = self.norm(y)
        y = self.in_proj(y)
        a, z = y.chunk(2, dim=-1)                          # each (B, HW, d_inner)
        a = a.transpose(1, 2)                              # (B, d_inner, HW)
        a = F.silu(self.dwconv(a))
        a = a.view(Bsz, self.d_inner, H, W)
        a = self.ss2d(a)
        a = a.flatten(2).transpose(1, 2)                   # (B, HW, d_inner)
        a = a * F.silu(z)
        a = self.out_proj(a)
        a = a.transpose(1, 2).view(Bsz, c, H, W)
        return residual + a


# -----------------------------------------------------------------------------
# MMDet neck
# -----------------------------------------------------------------------------


@MODELS.register_module()
class MambaNeck(BaseModule):
    """VSS-block FPN (MambaFPN-style) as an MMDetection neck.

    Args:
        in_channels: Backbone out-channels (length 4; C2..C5).
        out_channels: Unified channel width of all outputs.
        num_outs: Number of output levels (must be 5 for the ATSS head).
        num_vss_blocks: VSS blocks per pyramid level (P3/P4/P5).
        d_state: Mamba state dimension.
        expand: Mamba inner-channel expansion factor.
        d_conv: Depthwise Conv1d kernel size inside each VSS block.
        start_level: Lowest backbone level to use (1 drops C2, matches FPN).
        init_cfg: Standard MMEngine init config.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 num_vss_blocks: int = 2,
                 d_state: int = 16,
                 expand: int = 2,
                 d_conv: int = 3,
                 start_level: int = 1,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert num_outs == 5, 'MambaNeck produces exactly 5 levels.'
        assert start_level == 1, 'start_level must be 1 (drop C2).'
        assert len(in_channels) == 4, 'MambaNeck expects 4 backbone levels.'
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level

        used_ch = self.in_channels[start_level:]
        self.input_proj = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in used_ch]
        )
        self.vss_per_level = nn.ModuleList([
            nn.Sequential(*[
                VSSBlock(out_channels,
                         d_state=d_state, expand=expand, d_conv=d_conv)
                for _ in range(num_vss_blocks)
            ])
            for _ in range(len(used_ch))
        ])
        self.extra_p6 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.extra_p7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels), (
            f'Expected {len(self.in_channels)} inputs, got {len(inputs)}'
        )
        feats = [proj(f) for proj, f in
                 zip(self.input_proj, inputs[self.start_level:])]

        # Top-down FPN pass with VSS blocks as the refinement at each level.
        outs: List[torch.Tensor | None] = [None] * len(feats)
        outs[-1] = self.vss_per_level[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(outs[i + 1],
                               size=feats[i].shape[-2:], mode='nearest')
            outs[i] = self.vss_per_level[i](feats[i] + up)

        p3, p4, p5 = outs  # type: ignore[misc]
        p6 = self.extra_p6(p5)
        p7 = self.extra_p7(p6)
        return (p3, p4, p5, p6, p7)
