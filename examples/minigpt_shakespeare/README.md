# miniGPT Shakespeare (train & generate)

## Overview

This example trains and runs inference for a GPT-style causal Transformer on the Complete Works of Shakespeare (Project Gutenberg #100) using `vibe_kernels.ops.Transformer`.

## Scripts

| Script | Framework | Purpose |
|---|---|---|
| `train_pytorch_shakespeare.py` | PyTorch | Reference training loop |
| `train_vbt_shakespeare.py` | VibeTensor + vibe_kernels | Train via `Transformer.fwd()` / `compute_loss()` / `bwd()` + VBT AdamW |
| `generate_vbt_shakespeare.py` | VibeTensor | Generate text from a VBT checkpoint |
| `vbt_io.py` | Python | Checkpoint I/O helpers |

## Requirements

- **CUDA**: required for VibeTensor training/inference.
- **PyTorch**: required for the PyTorch baseline.

## Quickstart

Train (VibeTensor):

```bash
CUDA_VISIBLE_DEVICES=0 python examples/minigpt_shakespeare/train_vbt_shakespeare.py \
  --metrics-json tmp/metrics/minigpt_shakespeare_vbt.json \
  --ckpt-path tmp/ckpt/shakespeare_vbt.pt
```

Generate:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/minigpt_shakespeare/generate_vbt_shakespeare.py \
  --checkpoint tmp/ckpt/shakespeare_vbt.pt \
  --prompt "to be or" \
  --max-new-tokens 200
```

Streaming (optional):

```bash
CUDA_VISIBLE_DEVICES=0 python examples/minigpt_shakespeare/generate_vbt_shakespeare.py \
  --checkpoint tmp/ckpt/shakespeare_vbt.pt \
  --prompt "to be or" \
  --max-new-tokens 200 \
  --stream --delay 0.02
```

Optional (PyTorch baseline):

```bash
CUDA_VISIBLE_DEVICES=0 python examples/minigpt_shakespeare/train_pytorch_shakespeare.py \
  --metrics-json tmp/metrics/minigpt_shakespeare_pytorch.json
```

## Outputs

On first run, this example will populate `tmp/` (gitignored):

- Dataset: `tmp/data/shakespeare_full/shakespeare.txt` (auto-download)
- Tokenizer JSON: `tmp/tokenizers/shakespeare_full_bpe/shakespeare_full_bpe_v4096.json`
- Token ids cache: `tmp/data/shakespeare_full/tokens_shakespeare_full_bpe_v4096.npy`
- Metrics: `--metrics-json tmp/metrics/<name>.json`
- Checkpoints: `--ckpt-path tmp/ckpt/<name>.pt`

## Default configuration

| Parameter | Value |
|---|---:|
| Dataset | Project Gutenberg #100 (full Shakespeare) |
| Tokenizer | byte-level BPE |
| Vocab size | 4096 |
| Max seq len | 128 |
| Dim | 256 |
| Heads | 8 |
| Layers | 6 |
| Batch size | 64 |
| Iters | 100000 |
| Optimizer | AdamW |
| LR | 1e-3 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |
| Grad clip | 1.0 |

## Notes

- By default there is no train/val split (`train_frac=1.0`); the full token stream is used for training.

## File layout

```
examples/minigpt_shakespeare/
├── README.md
├── common.py
├── train_pytorch_shakespeare.py
├── train_vbt_shakespeare.py
├── generate_vbt_shakespeare.py
└── vbt_io.py
```
