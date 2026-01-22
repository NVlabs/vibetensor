# miniGPT Reverse (toy causal next-token task)

## Overview

This example trains a small GPT-style causal Transformer on a synthetic “reverse after `<SEP>`” sequence task.

Format:

`<BOS> a1 a2 ... an <SEP> an ... a2 a1 <EOS>`

Loss is computed only for positions after `<SEP>`.

## Scripts

| Script | Framework | Purpose |
|---|---|---|
| `train_pytorch_reverse.py` | PyTorch | Reference training loop (matches the VibeTensor math/update choices) |
| `train_vbt_reverse.py` | VibeTensor + Triton | VibeTensor training using `vibe_kernels.ops.Transformer` |

## Requirements

- **PyTorch**: required for `train_pytorch_reverse.py`.
- **CUDA**: required for `train_vbt_reverse.py`.

## Quickstart

```bash
CUDA_VISIBLE_DEVICES=0 python examples/minigpt_reverse/train_pytorch_reverse.py \
  --metrics-json tmp/metrics/minigpt_reverse_pytorch.json

CUDA_VISIBLE_DEVICES=0 python examples/minigpt_reverse/train_vbt_reverse.py \
  --metrics-json tmp/metrics/minigpt_reverse_vbt.json
```

## Outputs

- Metrics: pass `--metrics-json tmp/metrics/<name>.json` to write per-log-step metrics.

## Default configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 36 |
| Max \(N\) | 16 |
| Max seq len | 64 |
| Dim | 128 |
| Heads | 4 |
| Layers | 2 |

## File layout

```
examples/minigpt_reverse/
├── README.md
├── common.py
├── train_pytorch_reverse.py
└── train_vbt_reverse.py
```
