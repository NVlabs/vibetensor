# Triton Attention Kernel Report

## Overview
This report covers the current fused scaled dot-product attention implementation
in `vibe_kernels.attention`. The kernel targets Hopper-class GPUs and supports:

- Native BF16 / FP16 pipeline (FP32 inputs are automatically downcast) with optional causal masking
- Grouped Query Attention (GQA) via `n_kv_head`
- Pointer-based tiled loads (warp specialization is disabled because the Triton
  3.5 wheels used here do not yet expose the `num_consumer_groups` /
  `num_buffers_warp_spec` configuration hooks)
- Utility helpers for grouped-query attention reshaping and KV cache management
  (`vibe_kernels.attention.utils`)

The kernel shares the same API surface as PyTorch's FlashAttention and is
designed as a drop-in replacement for NanoChat-sized models.

## Environment
- GPU: NVIDIA H100 PCIe (SM90)
- Software: CUDA 12, PyTorch 2.9 nightly (`torch==2.9.0+cu128`), `triton==3.5.0`
- Benchmark entry point: `python -m vibe_kernels.attention.benchmark`

Unless specified, all measurements were taken with the benchmark's default
warmup/iteration counts and BF16 accumulation. Each run seeds PyTorch, warms up
both kernels equally, and validates `torch.allclose(atol=2e-2, rtol=0)`.

## Benchmarks

### Small-batch causal GQA (NanoChat prompt prefill)
Command:
```bash
python -m vibe_kernels.attention.benchmark \
  --batch 2 --heads 8 --kv-heads 4 --seqlen 1024 --headdim 128 \
  --dtype bfloat16 --causal --both --warmup 50 --iters 100
```

| Pass | Torch Flash (ms) | Triton (ms) | Speedup | `max|Δ|` | `allclose` |
|------|------------------|-------------|---------|----------|------------|
| Forward | 0.14 | 0.22 | 0.67x | 0.0039 | True |
| Backward | 0.47 | 0.71 | 0.66x | 0.031 | True |

### Default NanoChat training batch
Command:
```bash
python -m vibe_kernels.attention.benchmark \
  --batch 32 --heads 10 --kv-heads 10 --seqlen 2048 --headdim 128 \
  --dtype bfloat16 --causal --both --warmup 50 --iters 100
```

| Pass | Torch Flash (ms) | Triton (ms) | Speedup | `max|Δ|` | `allclose` |
|------|------------------|-------------|---------|----------|------------|
| Forward | 2.24 | 1.46 | **1.54x** | 0.0078 | True |
| Backward | 8.78 | 6.97 | **1.26x** | 0.031 | True |

### Discussion
- The pointer-path kernel trails PyTorch on the small GQA prefill case but
  outperforms FlashAttention on the full NanoChat training batch (1.54x forward,
  1.26x backward), even without Hopper warp specialization.
- Numerical parity is maintained, with maximum absolute deviation ≤ 0.031
  (comfortably within BF16 tolerance).
- Hopper-specific descriptor plumbing is in place, but the Triton 3.5 wheel
  lacks the configuration knobs necessary to enable warp specialization, so the
  optimized path remains dormant.

## Reproducing Benchmarks
Invoke the benchmark module with the desired problem size:

```bash
python -m vibe_kernels.attention.benchmark \
  --batch <B> --heads <Hq> --kv-heads <Hkv> --seqlen <T> --headdim <D> \
  --dtype {float16,bfloat16} [--causal] [--warmup N] [--iters M]
```

Additional flags:
- `--dtype`: switch between FP16 and BF16 operands
- `--causal`: enable or disable causal masking
- `--warmup` / `--iters`: control compilation warmup and timed repetitions
- `--backward`: benchmark backward pass only
- `--both`: benchmark both forward and backward passes
- Environment flag `AIKF_DISABLE_WARP_SPECIALIZE=1` forces the descriptor-based
  path (useful for debugging Hopper-specific issues)

## Tests
```
python -m pytest vibe_kernels/attention/tests/test_attention_utils.py
python -m pytest vibe_kernels/attention/tests/test_attention_kernel.py
```
These suites cover GQA reshape round-trips, KV cache utilities, BF16/FP16 forward
and backward parity, and regression checks with and without warp specialization.

The benchmark script prints mean latencies for Torch FlashAttention and the
Triton kernel, followed by the computed speedup and max absolute difference.

## PyTorch vs Triton vs VBT-Triton Benchmarks

Performance comparison of attention implementations (H100, CUDA 13).

### Forward Pass

| Batch | Heads | KV | SeqLen | Dim | Dtype | PyTorch (ms) | Triton (ms) | VBT (ms) | Triton Speedup | VBT Speedup | Numerics |
|-------|-------|-----|--------|-----|-------|--------------|-------------|----------|----------------|-------------|----------|
| 1 | 8 | 8 | 512 | 64 | float32 | 0.130 | 0.165 | 0.048 | 0.79x | **2.73x** | T:OK V:OK |
| 1 | 8 | 8 | 512 | 64 | float16 | 0.097 | 0.115 | N/A | 0.85x | - | T:OK |
| 4 | 32 | 32 | 1024 | 64 | float32 | 0.938 | 0.263 | 0.444 | **3.56x** | **2.11x** | T:OK V:OK |
| 4 | 32 | 32 | 1024 | 64 | float16 | 0.200 | 0.192 | N/A | 1.05x | - | T:OK |
| 8 | 32 | 32 | 2048 | 128 | float32 | 9.654 | 2.026 | 8.098 | **4.76x** | 1.19x | T:OK V:OK |
| 8 | 32 | 32 | 2048 | 128 | float16 | 1.631 | 1.209 | N/A | **1.35x** | - | T:OK |

### Backward Pass

| Batch | Heads | KV | SeqLen | Dim | Dtype | PyTorch (ms) | Triton (ms) | VBT (ms) | Triton Speedup | VBT Speedup | Numerics |
|-------|-------|-----|--------|-----|-------|--------------|-------------|----------|----------------|-------------|----------|
| 1 | 8 | 8 | 512 | 64 | float32 | 0.438 | 1.065 | 0.274 | 0.41x | **1.60x** | T:OK V:OK |
| 4 | 32 | 32 | 1024 | 64 | float32 | 3.409 | 1.104 | 5.156 | **3.09x** | 0.66x | T:OK V:OK |

**Legend:**
- **PyTorch**: `torch.nn.functional.scaled_dot_product_attention` (baseline)
- **Triton**: Flash attention kernel using Triton with PyTorch tensors (`vibe_kernels.attention.kernel`)
- **VBT**: Pure VibeTensor implementation with no PyTorch dependency (`vibe_kernels.attention.vbt_native`, float32 only)
- **Numerics**: T=Triton, V=VBT; OK means allclose to PyTorch reference

### Reproduce Commands

```bash
# Run full forward benchmark suite
python -m vibe_kernels.attention.benchmark --suite

# Single config with both forward and backward
python -m vibe_kernels.attention.benchmark \
  --batch 4 --heads 32 --seqlen 1024 --headdim 64 \
  --dtype float32 --causal --both --warmup 50 --iters 100

# Forward only
python -m vibe_kernels.attention.benchmark \
  --batch 1 --heads 8 --seqlen 512 --headdim 64 \
  --dtype float32 --causal --warmup 50 --iters 100

# Backward only
python -m vibe_kernels.attention.benchmark \
  --batch 4 --heads 32 --seqlen 1024 --headdim 64 \
  --dtype float32 --causal --backward --warmup 50 --iters 100
```

## Next Steps
- Re-run the benchmarks once an official Triton build exposes
  `num_consumer_groups` / `num_buffers_warp_spec` so the Hopper warp-specialized
  path can be validated.
- Investigate alternative tiling choices (e.g., smaller `BLOCK_N`) to better fit
  medium batch sizes in the absence of warp specialization.
- Extend the benchmark suite with incremental-decoding workloads to capture KV
  cache behaviour in streaming inference scenarios.
