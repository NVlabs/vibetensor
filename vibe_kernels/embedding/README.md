# Fused Embedding + RMSNorm Kernel Report

## Overview
This Triton kernel combines token embedding lookup with RMS normalization, replacing the
separate `nn.Embedding` + `F.rms_norm` pair used in NanoChat. It accumulates in `float32`,
optionally applies a learnable scale (`gamma`), and returns activations in the original
dtype (bf16 or fp16).

## Benchmark Setup
- GPU: NVIDIA H100 PCIe
- Command: `python -m ai_kernel_factory.embedding.benchmark --tokens 4096 --vocab 50304 --hidden 768 --iters 50 --warmup 10 --dtype bfloat16`
- Baseline: PyTorch embedding followed by `torch.nn.functional.rms_norm`

| Tokens | Vocab | Hidden | Dtype   | Baseline (ms) | Fused (ms) | Speedup | Max |Δ| | Allclose |
|--------|-------|--------|---------|---------------|------------|---------|--------|----------|
| 4096   | 50304 | 768    | bf16    | 0.084         | 0.112      | 0.75×   | 0.0078 | True |

While the fused kernel is slower for this configuration (due to the additional atomic
graduation bookkeeping), it consolidates two memory-bound operations and prepares the
path for future fusion with downstream attention blocks.

## Usage Notes
- Module: `ai_kernel_factory.embedding.kernel.FusedEmbeddingRMSNorm`
- Supports optional learnable gamma; gradients accumulate in `float32`
- Requires CUDA; fall back to standard PyTorch ops on CPU

## Next Steps
- Explore shared-memory staging to reduce atomic contention and improve forward latency
- Add batched lookup fusion with token dropout / masking when available
