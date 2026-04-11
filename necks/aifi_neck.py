"""V2 - AIFI + CCFM Neck (adapted from RT-DETR, Zhao et al. 2024).

Efficient Hybrid Encoder as a drop-in MMDetection neck:
    - AIFI: 1 Multi-Head Self-Attention layer on C5 with 2D sinusoidal PE
    - CCFM: PAFPN-style RepConv Cross-Scale Feature-Merge on (C3, C4, C5)
    - 2 extra stride-2 convs for P6 and P7

Shape contract (identical to mmdet FPN with ``start_level=1``, ``num_outs=5``):

    Input : List[Tensor] of length 4 with shapes
            (B,  256, H/4,  W/4)   # C2  - discarded via start_level=1
            (B,  512, H/8,  W/8)   # C3
            (B, 1024, H/16, W/16)  # C4
            (B, 2048, H/32, W/32)  # C5
    Output: Tuple[Tensor] of length 5, all channels == out_channels,
            strides {8, 16, 32, 64, 128}.

Notes:
    * ``num_ccfm_blocks=1`` keeps the parameter count within the same order
      of magnitude as MMDet-FPN (see ``scripts/test_necks.py``).
    * Reparameterization merge of the RepConvBlock at inference is
      intentionally not implemented - the BA compares architectures, not
      deployment-optimized forms.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.registry import MODELS
except ImportError:  # pragma: no cover - local config-parse only
    from mmengine.registry import MODELS


# Building blocks

def _conv_bn_act(in_ch: int, out_ch: int, k: int, s: int = 1,
                 p: int | None = None, act: bool = True) -> nn.Sequential:
    """Conv2d + BatchNorm2d + (optional) SiLU."""
    if p is None:
        p = k // 2
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class SinusoidalPosEmbed2D(nn.Module):
    """2D sin/cos positional encoding (DETR-style, Carion et al. 2020)."""

    def __init__(self, embed_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4'
        self.embed_dim = embed_dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        dim_t = torch.arange(self.embed_dim // 4,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (4 * (dim_t // 2) / self.embed_dim)
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)
        y_embed = y_embed[:, None].expand(H, W)
        x_embed = x_embed[None, :].expand(H, W)
        pos_y = y_embed[..., None] / dim_t
        pos_x = x_embed[..., None] / dim_t
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1)              # (H, W, C)
        return pos.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)


class AIFIBlock(nn.Module):
    """Single Transformer encoder layer (MHSA + FFN) on flattened C5."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 dim_ff: int = 1024, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pe = SinusoidalPosEmbed2D(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe(x)
        tokens = x.flatten(2).transpose(1, 2)         # (B, HW, C)
        q = k = (x + pe).flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(q, k, tokens, need_weights=False)
        y = self.norm1(tokens + attn_out)
        y = self.norm2(y + self.ffn(y))
        return y.transpose(1, 2).view_as(x)


class RepConvBlock(nn.Module):
    """RepVGG-style (3x3 + 1x1) + BN + SiLU."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv3 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn3(self.conv3(x)) + self.bn1(self.conv1(x)))


class CCFMFusion(nn.Module):
    """PAFPN-style Cross-Scale Feature-Merge used in RT-DETR.

    Top-down path fuses F5 -> F4 -> F3; bottom-up path fuses
    F3 -> F4 -> F5 afterwards. Concat+1x1 reduces channels, a
    ``RepConvBlock`` stack refines each fusion output.
    """

    def __init__(self, c: int, num_blocks: int = 1) -> None:
        super().__init__()

        def _stack() -> nn.Sequential:
            return nn.Sequential(*[RepConvBlock(c) for _ in range(num_blocks)])

        self.td_45 = _stack()
        self.td_34 = _stack()
        self.bu_34 = _stack()
        self.bu_45 = _stack()
        self.fuse_td_45 = _conv_bn_act(2 * c, c, 1)
        self.fuse_td_34 = _conv_bn_act(2 * c, c, 1)
        self.fuse_bu_34 = _conv_bn_act(2 * c, c, 1)
        self.fuse_bu_45 = _conv_bn_act(2 * c, c, 1)
        self.down_34 = _conv_bn_act(c, c, 3, s=2)
        self.down_45 = _conv_bn_act(c, c, 3, s=2)

    def forward(self, f3: torch.Tensor, f4: torch.Tensor,
                f5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                # Top-down
        f5_up = F.interpolate(f5, size=f4.shape[-2:], mode='nearest')
        f4_td = self.td_45(self.fuse_td_45(torch.cat([f5_up, f4], dim=1)))
        f4_up = F.interpolate(f4_td, size=f3.shape[-2:], mode='nearest')
        f3_td = self.td_34(self.fuse_td_34(torch.cat([f4_up, f3], dim=1)))
        # Bottom-up
        f3_dn = self.down_34(f3_td)
        f4_out = self.bu_34(self.fuse_bu_34(torch.cat([f3_dn, f4_td], dim=1)))
        f4_dn = self.down_45(f4_out)
        f5_out = self.bu_45(self.fuse_bu_45(torch.cat([f4_dn, f5], dim=1)))
        return f3_td, f4_out, f5_out


# MMDet neck


@MODELS.register_module()
class AifiNeck(BaseModule):
    """RT-DETR Efficient Hybrid Encoder as an MMDetection neck.

    Args:
        in_channels: Backbone out-channels (length 4; C2..C5 strides).
        out_channels: Unified channel width of all outputs.
        num_outs: Number of output levels (must be 5 for the ATSS head).
        num_encoder_layers: Number of AIFI blocks on C5.
        num_heads: MHSA heads per AIFI block.
        dim_feedforward: FFN hidden dim per AIFI block.
        num_ccfm_blocks: RepConv blocks per CCFM fusion stage.
        start_level: Lowest backbone level to use (1 drops C2, matches FPN).
        init_cfg: Standard MMEngine init config (passed to BaseModule).
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 num_encoder_layers: int = 1,
                 num_heads: int = 8,
                 dim_feedforward: int = 1024,
                 num_ccfm_blocks: int = 1,
                 start_level: int = 1,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert num_outs == 5, 'AifiNeck produces exactly 5 levels.'
        assert start_level == 1, 'start_level must be 1 (drop C2).'
        assert len(in_channels) == 4, 'AifiNeck expects 4 backbone levels.'
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level

        used_ch = self.in_channels[start_level:]
        self.input_proj = nn.ModuleList(
            [_conv_bn_act(c, out_channels, 1) for c in used_ch]
        )
        self.aifi = nn.Sequential(*[
            AIFIBlock(out_channels, num_heads=num_heads, dim_ff=dim_feedforward)
            for _ in range(num_encoder_layers)
        ])
        self.ccfm = CCFMFusion(out_channels, num_blocks=num_ccfm_blocks)
        self.extra_p6 = _conv_bn_act(out_channels, out_channels, 3, s=2)
        self.extra_p7 = _conv_bn_act(out_channels, out_channels, 3, s=2)

    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels), (
            f'Expected {len(self.in_channels)} inputs, got {len(inputs)}'
        )
        feats = inputs[self.start_level:]                    # [C3, C4, C5]
        c3, c4, c5 = [proj(f) for proj, f in zip(self.input_proj, feats)]
        c5 = self.aifi(c5)
        p3, p4, p5 = self.ccfm(c3, c4, c5)
        p6 = self.extra_p6(p5)
        p7 = self.extra_p7(p6)
        return (p3, p4, p5, p6, p7)
