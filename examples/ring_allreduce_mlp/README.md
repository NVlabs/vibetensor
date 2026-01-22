# Ring allreduce MLP (single-process multi-GPU)

## Overview

A minimal single-process, multi-GPU data-parallel training demo on synthetic data.

- Model: 2-layer MLP (`Linear → GELU → Linear`)
- Data: synthetic classification (labels from a fixed “teacher” linear model)
- Communication: ring reduction (sum) of gradients + local SGD update using the averaged gradient

## Scripts

| Script | Framework | Purpose |
|---|---|---|
| `train_vbt_ring_allreduce_mlp.py` | VibeTensor | Training loop with gradient ring reduction |
| `benchmark_scaling.py` | Python | Convenience harness to benchmark 1–4 GPU scaling |

## Requirements

- **CUDA**: required.
- **Triton**: required (this example uses `vibe_kernels.*.vbt_native`).

Optional plugin backend:

- Backend `plugin` uses a CUTLASS Blackwell ring allreduce plugin and requires **SM103** GPUs.
- `libvbt_ring_allreduce.so` must be discoverable via one of:
  - `VBT_RING_ALLREDUCE_PLUGIN_PATH=/path/to/libvbt_ring_allreduce.so`
  - `./libvbt_ring_allreduce.so`
  - `./build-py/libvbt_ring_allreduce.so`

## Quickstart

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/ring_allreduce_mlp/train_vbt_ring_allreduce_mlp.py \
  --world-size 4 --steps 200 --batch-size 256 --lr 0.5
```

Backend selection:

- `--backend auto`: uses `plugin` when available, otherwise falls back to `fabric_ring`.
- `--backend fabric_ring`: portable Python ring reduction (works on any CUDA GPUs).
- `--backend plugin`: SM103-only plugin backend.

## Benchmark (scaling)

```bash
# Weak scaling (fixed per-GPU batch size)
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/ring_allreduce_mlp/benchmark_scaling.py --batch-size 65536

# Strong scaling (fixed global batch size; must be divisible by 1/2/3/4)
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/ring_allreduce_mlp/benchmark_scaling.py --global-batch 196608
```

## Notes

- `--backend plugin` only supports `--world-size` in `{2, 4, 8}`.
- `--backend fabric_ring` supports any `--world-size >= 1`.

## How it works (high level)

1. Replicate weights on each GPU.
2. Each GPU runs forward/backward on its own mini-batch.
3. Ring-reduce gradients (sum).
4. Each GPU applies the same SGD update using the averaged gradient.
