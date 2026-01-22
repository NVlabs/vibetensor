# Rotary Embedding Kernel Report

## Overview

This module provides rotary position embedding kernels for query and key projections, with
both a Triton implementation and a CuTeDSL implementation. The kernels operate on
`(batch, heads, seq, dim)` tensors, gather the appropriate cos/sin coefficients per token
position, and perform the complex rotation in `float32` before writing results back in the
original dtype. Backward propagation uses analytical gradients on the host for simplicity.

- **Triton backend**: high-performance rotary kernel used via
  `apply_rotary_embedding(..., backend="triton")`.
- **CuTeDSL backend**: standalone CuTeDSL ROPE kernel (no Quack dependency) wired through
  `apply_rotary_embedding(..., backend="cutedsl")` using a thin autograd wrapper and a
  fused Q/K CuTeDSL kernel.

`apply_rotary_embedding` is defined in `tmp/kernel_factory/rotary/kernel.py` and is the
entry point used by the NanoChat attention stack.

## Benchmark Setup (current measurements)

All benchmarks below were run in this environment:

- GPU: **NVIDIA H100 PCIe**
- Dtype: **bfloat16** for performance tables (compute in `float32`, cast back to `bf16`)
- Shapes: Q/K shaped `(B, H, S, D)` as noted per configuration

Reproduction commands for the measurements in this README:

- **QK optimized performance – Torch vs Triton vs CuTeDSL (Optimized)**
  ```bash
  cd /workspace/terry/nano-cursor
  PYTHONPATH=tmp python -m kernel_factory.rotary.bench_compare_all
  ```

- **QK rotary performance – Torch vs Triton**
  ```bash
  cd /workspace/terry/nano-cursor
  python tmp/kernel_factory/rotary/bench_qk_triton.py
  ```

- **QK rotary performance – Torch vs CuTeDSL**
  ```bash
  cd /workspace/terry/nano-cursor
  python tmp/kernel_factory/rotary/bench_qk_cutedsl.py
  ```

- **Cross-backend numerics – Triton vs CuTeDSL (float32)**
  ```bash
  cd /workspace/terry/nano-cursor
  python tmp/kernel_factory/rotary/compare_triton_cutedsl.py
  ```

- **Cross-backend numerics – Triton vs Torch reference (float32)**
  ```bash
  cd /workspace/terry/nano-cursor
  python tmp/kernel_factory/rotary/compare_triton_torch.py
  ```

The helper scripts live under `tmp/` and use CUDA wall-clock timing with a warmup phase
followed by repeated iterations.

## QK Rotary Performance: Torch vs Triton vs CuTeDSL (Optimized)

All numbers below are for `dtype=bfloat16` on an NVIDIA H100 PCIe. The reference PyTorch
implementation uses a pure PyTorch ROPE in `float32` with a final cast back to `bf16`.

**Configuration:** Q/K shaped `(B=32, H=32, S=4096, D=128)`.

| Implementation | Latency (ms) | Speedup vs Torch | Rel. to Triton |
| :--- | :--- | :--- | :--- |
| **Triton (Baseline)** | **2.40 ms** | **9.1x** | **1.00x** |
| **CuTeDSL (Optimized)**| **2.95 ms** | **7.4x** | **1.23x** |
| PyTorch (Native) | 21.86 ms | 1.0x | 9.10x |

**Analysis:**
1.  **Massive Speedup over PyTorch**: CuTeDSL is **7.4x faster** than the native PyTorch implementation.
2.  **Near-Triton Performance**: The optimized CuTeDSL kernel is now only **~1.2x slower** than the highly-tuned Triton kernel.
    *   Initial Gap: ~4.6x
    *   Final Gap: ~1.2x
3.  **Optimizations Applied**:
    *   **Native Layout (BHSD)**: Removed expensive `permute` operations.
    *   **Flat Indexing**: Minimized integer arithmetic overhead.
    *   **Bitwise Arithmetic**: Specialized path for power-of-2 head dimensions.

## QK Rotary Performance (Legacy Benchmarks)

All numbers below are for `dtype=bfloat16` on an H100 PCIe. The reference PyTorch
implementation (`torch` columns) uses a pure PyTorch ROPE in `float32` with a final cast
back to `bf16`, using explicit `cos`, `sin`, and `positions` tensors.

**Configuration:** Q/K tensors shaped `(B, H, S, D)`.

| Config | B | H  | S    | D   | torch (ms) | Triton (ms) | CuTeDSL (ms) | Triton speedup vs torch | CuTeDSL speedup vs torch |
|--------|---|----|------|-----|-----------:|------------:|-------------:|-------------------------:|--------------------------:|
| A      | 1 | 8  | 2048 | 128 |    0.2662 |     0.1056  |      0.3571 |                    2.52× |                     0.75× |
| B      | 4 | 8  | 2048 | 128 |    0.7271 |     0.1051  |      0.4110 |                    6.92× |                     1.77× |
| C      | 4 | 16 | 2048 | 128 |    1.4909 |     0.1571  |      0.5309 |                    9.49× |                     2.81× |

- **Torch baseline**: pure PyTorch rotary with `float32` math and final cast to `bf16`.
- **Triton backend**: fused Q/K rotary Triton kernel used by
  `apply_rotary_embedding(..., backend="triton")`.
- **CuTeDSL backend**: fused Q/K CuTeDSL kernel used by
  `apply_rotary_embedding(..., backend="cutedsl")`.

Observations:

- Triton delivers **2.5–9.5×** speedup over the PyTorch baseline in these configs.
- CuTeDSL delivers **~1.7–2.8×** speedup over PyTorch once the shape is large enough
  (Configs B/C), but is still slower than Triton on this GPU.

## QK Rotary Numerics

### CuTeDSL vs Torch (bf16)

From `tmp/kernel_factory/rotary/bench_qk_cutedsl.py`, comparing CuTeDSL Q/K outputs against the
PyTorch reference (`_torch_rotary`) in `bfloat16`:

| Config | B,H,S,D         | dtype | max |Δq| (CuTeDSL vs torch) | max |Δk| (CuTeDSL vs torch) |
|--------|-----------------|:-----:|---------------------------:|---------------------------:|
| A      | 1, 8, 2048, 128 | bf16  |                     0.0625 |                     0.0625 |
| B      | 4, 8, 2048, 128 | bf16  |                     0.0625 |                     0.0625 |
| C      | 4, 16, 2048,128 | bf16  |                     0.0625 |                     0.1250 |

These differences are on the order of a few ULPs in `bfloat16` and are consistent with
expected rounding noise for this dtype.

### Triton vs CuTeDSL (float32)

From `tmp/tmp_compare_triton_cutedsl_once.py`, comparing `apply_rotary_embedding`
backends directly in `float32` for `(B=4,H=8,S=2048,D=128)`:

- `max_diff_q ≈ 9.54e-07`
- `max_diff_k ≈ 9.54e-07`
- `allclose(atol=1e-6, rtol=0) = True`

This shows that the CuTeDSL fused Q/K kernel matches the Triton backend to within
`~1e-6` in `float32` for this configuration.

### Triton vs Torch Reference (float32)

From `tmp/tmp_compare_triton_torch_once.py`, comparing the Triton backend against the
same PyTorch reference implementation in `float32` for `(B=4,H=8,S=2048,D=128)`:

- `max_diff_q ≈ 1.9e-06`
- `max_diff_k ≈ 9.5e-07`
- With `atol=1e-6, rtol=0`, `torch.allclose` is slightly `False`, but all differences are
  ≤ `2e-6`.

In practice this means Triton and CuTeDSL both agree with the PyTorch reference up to
`~1–2e-6` in `float32`, and all three agree up to typical `bf16` rounding noise in the
end-to-end attention stack.

## Usage Notes

- Entry point (Triton backend):
  ```python
  from kernel_factory import apply_rotary_embedding

  q_rot, k_rot = apply_rotary_embedding(q, k, cos, sin, positions, backend="triton")
  ```

- Entry point (CuTeDSL backend):
  ```python
  from kernel_factory import apply_rotary_embedding

  q_rot, k_rot = apply_rotary_embedding(q, k, cos, sin, positions, backend="cutedsl")
  ```

- `q` and `k` are typically shaped `(batch, heads, seqlen, head_dim)` with `head_dim` even.
- `positions` is shaped `(batch, heads, seqlen)` and indexes into the first dimension of
  the `cos`/`sin` caches, which are shaped `(seq, head_dim // 2)`.
- Both backends compute the rotation in `float32` internally and cast back to the original
dtype (`bf16` or `fp16`) for outputs.

## Latest Verification (2025-11-19)

Verified performance on current environment (B=4, H=8, S=2048, D=128, bf16).

| Backend | Latency (ms) | Speedup vs Torch |
|---------|--------------|------------------|
| Triton  | 0.1375       | 5.33x            |
| CuTeDSL | 0.7400       | 0.99x            |
| Torch   | 0.7334       | 1.00x            |

