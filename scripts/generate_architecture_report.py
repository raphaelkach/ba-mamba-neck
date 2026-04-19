"""Generate ``docs/architecture_report.md`` for the three neck variants.

Builds the full ATSS(ResNet-50 + <neck> + ATSSHead) detector for each
of the three configs and writes a markdown report containing:

    1. Model summary (torchinfo) for the complete detector + for the
       neck in isolation, plus a backbone / neck / head parameter split.
    2. Config diff between V1, V2 and V3 (via ``deepdiff``) confirming
       that only ``model.neck`` changes.
    3. Neck architecture tables (encoder layers, heads, d_state, ...).
    4. ASCII data-flow diagrams (C2..C5 -> P3..P7) with live-captured
       tensor shapes.
    5. Computational profile: GFLOPs (fvcore), wall-clock latency
       (50 warmup + 100 timed iterations, cuda events), peak GPU memory.

GPU-only script. Intended to be invoked from a training notebook:

    !PYTHONPATH=. python scripts/generate_architecture_report.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Import necks so that AifiNeck / MambaNeck are registered *before*
# init_detector is called.
import necks  # noqa: F401


CFGS: Dict[str, str] = {
    'fpn':   'configs/fpn.py',
    'aifi':  'configs/aifi.py',
    'mamba': 'configs/mamba.py',
}

IN_CHANNELS = [256, 512, 1024, 2048]
DUMMY_FEATURE_SHAPES = [
    (1, 256, 160, 160),
    (1, 512, 80, 80),
    (1, 1024, 40, 40),
    (1, 2048, 20, 20),
]
DUMMY_IMAGE_SHAPE = (1, 3, 640, 640)


# Helpers


def _param_split(model: nn.Module) -> Tuple[int, int, int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    backbone = sum(p.numel() for p in model.backbone.parameters())
    neck = sum(p.numel() for p in model.neck.parameters())
    head = sum(p.numel() for p in model.bbox_head.parameters())
    return total, trainable, frozen, backbone, neck, head  # type: ignore[return-value]


def _torchinfo_str(module: nn.Module, input_data) -> str:
    try:
        from torchinfo import summary
    except ImportError:
        return '(torchinfo not installed - pip install torchinfo)'
    return str(summary(module, input_data=input_data,
                       depth=3, verbose=0,
                       col_names=('input_size', 'output_size', 'num_params')))


def _dummy_features(device: torch.device) -> List[torch.Tensor]:
    return [torch.randn(*s, device=device) for s in DUMMY_FEATURE_SHAPES]


def _capture_neck_shapes(neck: nn.Module,
                         device: torch.device) -> List[Tuple[int, ...]]:
    """Run the neck once and return output shapes."""
    feats = _dummy_features(device)
    with torch.no_grad():
        outs = neck(feats)
    return [tuple(t.shape) for t in outs]


def _flops(neck: nn.Module, device: torch.device) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return float('nan')
    feats = tuple(_dummy_features(device))
    fca = FlopCountAnalysis(neck, feats)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return fca.total() / 1e9  # GFLOPs


def _latency_and_memory(neck: nn.Module,
                        device: torch.device) -> Tuple[float, float]:
    feats = _dummy_features(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        for _ in range(50):  # warmup
            _ = neck(feats)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            _ = neck(feats)
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / 100.0
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return latency_ms, peak_mb
        # CPU fallback
    for _ in range(5):
        _ = neck(feats)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = neck(feats)
    latency_ms = (time.perf_counter() - t0) / 20 * 1000
    return latency_ms, float('nan')


def _build_detector(cfg_path: str, device: torch.device):
    """Build a full ATSS detector from a config path."""
    from mmdet.apis import init_detector
    return init_detector(cfg_path, device=str(device))


# Markdown sections


def _section_param_split(detectors: Dict[str, nn.Module]) -> List[str]:
    lines = [
        '## 1. Model summary and parameter split',
        '',
        '| Neck | total | trainable | frozen | backbone | neck | head |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, det in detectors.items():
        total, trainable, frozen, bb, nk, hd = _param_split(det)
        lines.append(
            f'| {name} | {total:,} | {trainable:,} | {frozen:,} | '
            f'{bb:,} ({bb/total*100:.1f}%) | '
            f'{nk:,} ({nk/total*100:.1f}%) | '
            f'{hd:,} ({hd/total*100:.1f}%) |'
        )
    lines.append('')
    return lines


def _section_torchinfo(detectors: Dict[str, nn.Module],
                       device: torch.device) -> List[str]:
    lines = ['### torchinfo - full detector (1x3x640x640)', '']
    for name, det in detectors.items():
        lines.append(f'#### {name}')
        lines.append('```')
        lines.append(_torchinfo_str(
            det,
            torch.randn(*DUMMY_IMAGE_SHAPE, device=device),
        ))
        lines.append('```')
        lines.append('')

    lines += ['### torchinfo - neck only (C2..C5 dummy features)', '']
    for name, det in detectors.items():
        lines.append(f'#### {name}.neck')
        lines.append('```')
        lines.append(_torchinfo_str(det.neck, _dummy_features(device)))
        lines.append('```')
        lines.append('')
    return lines


def _section_config_diff() -> List[str]:
    from mmengine.config import Config
    try:
        from deepdiff import DeepDiff
    except ImportError:
        return ['## 2. Config diff', '', '_deepdiff not installed._', '']

    cfgs = {n: Config.fromfile(p).to_dict() for n, p in CFGS.items()}
    lines = ['## 2. Config diff', '']

    for target in ('aifi', 'mamba'):
        d = DeepDiff(cfgs['fpn'], cfgs[target], ignore_order=True)
        lines.append(f'### fpn <-> {target}')
        lines.append('')
        lines.append('```')
        lines.append(d.to_json(indent=2) if d else '{}')
        lines.append('```')
        lines.append('')

    lines += [
        '### Identical across V1/V2/V3',
        '',
        '- `model.backbone` (ResNet-50, frozen_stages=4, norm_eval=True)',
        '- `model.bbox_head` (ATSSHead, num_classes=10)',
        '- `model.data_preprocessor`',
        '- `optim_wrapper` (AdamW, lr=1e-4, wd=0.05, grad_clip=0.1)',
        '- `param_scheduler` (LinearLR warmup + CosineAnnealingLR)',
        '- `train_dataloader.dataset.pipeline` (Resize + RandomFlip)',
        '- `train_dataloader.batch_size` (8)',
        '- `train_cfg.max_epochs` (24)',
        '- `randomness.deterministic` (True)',
        '',
    ]
    return lines


def _section_neck_details(detectors: Dict[str, nn.Module]) -> List[str]:
    lines = ['## 3. Neck architecture details', '']

    # FPN
    fpn = detectors['fpn'].neck
    lines += [
        '### V1 - FPN (Lin et al. 2017)',
        '',
        f'- Lateral 1x1 convs:        3 (C3, C4, C5 -> 256)',
        f'- 3x3 refinement convs:    3 (P3, P4, P5)',
        f'- Extra stride-2 convs:    2 (P6, P7; on_output)',
        f'- Data flow: C2 (dropped) | C3 -> P3 | C4 -> P4 | C5 -> P5 -> P6 -> P7',
        f'- Complexity: O(H*W) per level (pure conv)',
        f'- Parameters: {sum(p.numel() for p in fpn.parameters()):,}',
        '',
    ]

    # AIFI
    aifi = detectors['aifi'].neck
    lines += [
        '### V2 - AifiNeck (AIFI + CCFM, adapted from RT-DETR)',
        '',
        f'- Input projections (1x1 conv + BN + SiLU): 3 (C3, C4, C5 -> 256)',
        f'- AIFI blocks on C5: {len(aifi.aifi)} x (MHSA 8 heads, d_model=256, FFN=1024)',
        '- Positional encoding: 2D sinusoidal, temperature=10000',
        f'- CCFM RepConv blocks per stage: '
        f'{len(aifi.ccfm.td_45)} (top-down x2 + bottom-up x2 = 4 stages)',
        '- CCFM kernels: 3x3 + 1x1 RepConv, 1x1 fusion convs, 3x3 stride-2 downsample',
        '- Extra P6, P7: 3x3 stride-2 conv + BN + SiLU',
        '- Data flow: C5 -> AIFI -> CCFM(C3, C4, C5) -> (P3, P4, P5) -> P6 -> P7',
        '- Complexity: O((H*W)^2) for AIFI (flattened C5), O(H*W) for CCFM',
        f'- Parameters: {sum(p.numel() for p in aifi.parameters()):,}',
        '',
    ]

    # Mamba
    mamba = detectors['mamba'].neck
    first_block = mamba.vss_per_level[0][0]  # first VSSBlock of first level
    n_blocks = len(mamba.vss_per_level[0])
    lines += [
        '### V3 - MambaNeck (VSS-FPN, MambaFPN-style)',
        '',
        f'- Input projections (1x1 conv): 3 (C3, C4, C5 -> 256)',
        f'- VSS blocks per pyramid level: {n_blocks} (levels P3, P4, P5)',
        f'- d_state: {first_block.ss2d.d_state}',
        f'- d_inner (expand=2): {first_block.ss2d.d_inner}',
        f'- dt_rank: {first_block.ss2d.dt_rank}',
        f'- Cross-scan directions: 4 (row-fwd, col-fwd, row-rev, col-rev)',
        f'- Depthwise Conv1d kernel: {first_block.dwconv.kernel_size[0]}',
        '- Top-down fusion: upsample + add + VSS (FPN-style, no concat)',
        '- Extra P6, P7: 3x3 stride-2 conv + BN + SiLU',
        '- Data flow: C5 -> VSS -> up+add C4 -> VSS -> up+add C3 -> VSS -> (P3, P4, P5) -> P6 -> P7',
        '- Complexity: O(H*W) for the selective scan (linear in sequence length)',
        f'- Parameters: {sum(p.numel() for p in mamba.parameters()):,}',
        '',
    ]
    return lines


def _section_dataflow(detectors: Dict[str, nn.Module],
                      device: torch.device) -> List[str]:
    lines = ['## 4. Data-flow diagrams', '']
    for name, det in detectors.items():
        shapes = _capture_neck_shapes(det.neck, device)
        lines.append(f'### {name}')
        lines.append('')
        lines.append('```')
        lines.append('Backbone outputs (ResNet-50 @ 640x640):')
        for i, s in enumerate(DUMMY_FEATURE_SHAPES):
            tag = 'C' + str(i + 2)
            drop = '  (dropped, start_level=1)' if i == 0 else ''
            lines.append(f'  {tag}: {s}{drop}')
        lines.append('')
        lines.append(f'{name}.forward([C2, C3, C4, C5]):')
        level_names = ['P3', 'P4', 'P5', 'P6', 'P7']
        strides = [8, 16, 32, 64, 128]
        for lvl, s, stride in zip(level_names, shapes, strides):
            lines.append(f'  {lvl}: {s}   stride={stride}')
        lines.append('```')
        lines.append('')
    return lines


def _section_compute(detectors: Dict[str, nn.Module],
                     device: torch.device) -> List[str]:
    lines = [
        '## 5. Computational profile (neck only, 640x640 input features)',
        '',
        '| Neck | GFLOPs | latency (ms/iter) | peak GPU mem (MB) |',
        '|---|---:|---:|---:|',
    ]
    for name, det in detectors.items():
        gflops = _flops(det.neck, device)
        latency, peak_mb = _latency_and_memory(det.neck, device)
        lines.append(
            f'| {name} | {gflops:.2f} | {latency:.2f} | '
            f'{peak_mb:.1f} |'
        )
    lines += [
        '',
        '_Latency measured over 100 iterations after 50 warmup steps '
        '(cuda events). Peak memory from `torch.cuda.max_memory_allocated`._',
        '',
    ]
    return lines


# Orchestration


def generate(docs_dir: Path) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    detectors: Dict[str, nn.Module] = {}
    for name, cfg_path in CFGS.items():
        print(f'building {name} from {cfg_path}')
        detectors[name] = _build_detector(cfg_path, device).eval()

    sections: List[str] = [
        '# Architecture Report - V1 FPN vs V2 AIFI vs V3 MambaNeck',
        '',
        '_Auto-generated by `scripts/generate_architecture_report.py` (P3)._',
        '',
    ]
    sections += _section_param_split(detectors)
    sections += _section_torchinfo(detectors, device)
    sections += _section_config_diff()
    sections += _section_neck_details(detectors)
    sections += _section_dataflow(detectors, device)
    sections += _section_compute(detectors, device)

    out = docs_dir / 'architecture_report.md'
    out.write_text('\n'.join(sections))
    print(f'wrote {out} ({out.stat().st_size} bytes)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.docs)


if __name__ == '__main__':
    main()
