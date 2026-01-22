# Tiny ViT on CIFAR-10

## Overview

Train a tiny ViT-like encoder on CIFAR-10 with a PyTorch reference and a VibeTensor implementation.

## Scripts

| Script | Framework | Purpose |
|---|---|---|
| `train_pytorch_vit_cifar.py` | PyTorch | Reference training loop |
| `train_vbt_vit_cifar.py` | VibeTensor + Triton | VibeTensor forward/backward using `vibe_kernels.*` |

## Requirements

- **PyTorch**: required for `train_pytorch_vit_cifar.py`.
- **CUDA**: required for `train_vbt_vit_cifar.py`.

## Quickstart

```bash
CUDA_VISIBLE_DEVICES=0 python examples/vit_cifar/train_pytorch_vit_cifar.py \
  --metrics-json tmp/metrics/vit_cifar_pytorch.json

CUDA_VISIBLE_DEVICES=0 python examples/vit_cifar/train_vbt_vit_cifar.py \
  --metrics-json tmp/metrics/vit_cifar_vbt.json
```

## Outputs

- Dataset: `tmp/data/CIFAR10/` (gitignored)
- Metrics: pass `--metrics-json tmp/metrics/<name>.json`

## Default configuration

| Parameter | Value |
|---|---:|
| Batch size | 256 |
| Eval batch size | 256 |
| Epochs | 50 |
| LR | 3e-4 |
| Weight decay | 0.0 |
| Patch size | 4 |
| Dim | 256 |
| Depth | 6 |
| Heads | 8 |
| MLP ratio | 4 |
| Dropout | 0.1 |
| Emb dropout | 0.1 |

## Notes

- Patch size is 4 (32×32 → 8×8 = 64 patches).
- Uses multiple CLS tokens (default 16) averaged before the classifier head.

## File layout

```
examples/vit_cifar/
├── README.md
├── common.py
├── train_pytorch_vit_cifar.py
└── train_vbt_vit_cifar.py
```
