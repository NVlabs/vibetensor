# PyTorch-Free Transformer Training

This directory contains transformer training examples using **pure VibeTensor + Triton kernels** - completely free of PyTorch and NumPy dependencies at runtime.

## Key Features

- **Zero PyTorch dependency** - Uses `vibetensor.torch` API (not PyTorch)
- **Zero NumPy in training loop** - All ops run on GPU via Triton kernels
- **Pure Triton kernels** - All forward/backward ops implemented in Triton
- **VBT-Triton Flash Attention** - Uses optimized attention kernel from `vibe_kernels.attention.vbt_native`
- **Modular ops design** - Clean `fwd()` / `bwd()` / `update()` interface

## Performance (H100, CUDA 13)

| Metric | Value |
|--------|-------|
| Mean step time | 27.6 ms |
| Median step time | 27.0 ms |
| Tokens/sec | ~4,600 |
| Config | batch=4, seq=32, dim=128, heads=4, layers=2 |

The transformer uses VBT-Triton Flash Attention which provides:
- **2-3x faster** than PyTorch SDPA on small batches (float32)
- **No PyTorch dependency** - pure VibeTensor tensors with Triton kernels
- Numerically equivalent to PyTorch reference (allclose passes)

## Scripts

### train_modular.py (Recommended)

A clean, minimal transformer training script using modular ops from `vibe_kernels.ops`.

**Usage:**
```bash
python train_modular.py
```

**Config:**
| Parameter | Value |
|-----------|-------|
| vocab_size | 256 |
| hidden_dim | 128 |
| n_heads | 4 |
| n_layers | 2 |
| batch_size | 4 |
| seq_len | 32 |
| steps | 100 |

**Task:** Learn `(token + 1) % vocab_size` - a sanity check to verify the training pipeline.

### train_vbt_transformer.py

A more detailed implementation showing how to use individual kernel_factory ops directly.

## Architecture

```
Embedding → [TransformerBlock × N] → Output Projection → CrossEntropyLoss

TransformerBlock:
  RMSNorm → Attention (with RoPE) → RMSNorm → FFN (SwiGLU)
```

All ops have explicit `fwd()` and `bwd()` methods backed by Triton kernel implementations from `kernel_factory`.

## Dependencies

- **VibeTensor** - PyTorch-like API without PyTorch
- **vibe_kernels** - Triton-based ops (attention, rmsnorm, gemm, etc.)
- **Triton** - GPU kernel compiler

## Why PyTorch-Free?

This demonstrates that transformer training can run entirely on:
1. VibeTensor tensors (C++ tensor library with Python bindings)
2. Triton kernels (compiled directly to PTX, no CUDA runtime from PyTorch)

Potential benefits:
- Smaller deployment footprint (no PyTorch installation needed)
- Direct control over all GPU operations
- Educational - understand every op in the training loop
