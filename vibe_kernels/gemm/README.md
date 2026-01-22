# Triton GEMM Kernel Report

## Overview
This document summarizes the Hopper-optimized Triton GEMM implementation that backs
`ai_kernel_factory.gemm`, now covering both forward and backward passes. The kernels
select between:

- A Tensor Memory Accelerator (TMA) / warp-specialized forward pipeline for large
  square problems on Hopper-class GPUs (compute capability ≥ 9.0).
- Classic, fully cached Triton kernels for smaller workloads and for backward
gradient computations on all devices.

Both paths accumulate in FP32, support `torch.float16` and `torch.bfloat16` operands,
optionally fuse bias addition in the forward pass, and expose an explicit backward
API returning gradients for inputs, weights, and bias.

## Environment
- GPU: NVIDIA H100 PCIe (torch reports `sm_90`)
- Driver stack: CUDA 12 (via the Triton wheel bundled with PyTorch 2.4)
- Python dependencies: `torch`, `triton`
- Benchmark script: `python -m ai_kernel_factory.gemm.benchmark`

Benchmarks were executed with the default CLI parameters:

```bash
python -m ai_kernel_factory.gemm.benchmark --dtype float16
```

The script seeds random generators, performs equal warmup for Triton and
`torch.matmul`, and reports forward/backward latencies alongside numerical parity
statistics.

## Results
All tests were performed with FP16 operands and fused bias.

| Size (M=N=K) | Triton fwd (ms) | Torch fwd (ms) | Fwd speedup | Triton bwd (ms) | Torch bwd (ms) | Bwd speedup |
|--------------|-----------------|----------------|-------------|-----------------|----------------|-------------|
| 2048         | 0.167           | 0.071          | 0.42×       | 0.286           | 0.117          | 0.41× |
| 4096         | 0.408           | 0.407          | 1.00×       | 0.905           | 0.705          | 0.78× |
| 8192         | 3.192           | 3.370          | 1.06×       | 6.406           | 5.340          | 0.83× |

_Previous backward baseline (classic kernel)_:

| Size (M=N=K) | Triton bwd (ms) | Torch bwd (ms) | Bwd speedup |
|--------------|-----------------|----------------|-------------|
| 2048         | 0.289           | 0.124          | 0.43× |
| 4096         | 1.135           | 0.694          | 0.61× |
| 8192         | 13.503          | 6.776          | 0.50× |

### Discussion
- The forward Hopper TMA kernel reaches parity with cuBLAS at 8192² while trailing at
  2048² due to JIT overhead; this matches prior measurements.
- The backward path now uses a Hopper TMA persistent kernel for both dA and dW,
  cutting 8192² latency from ~13.5 ms to ~6.4 ms; it still trails cuBLAS by ~18%, leaving
  room for future specialization.
- All forward and backward results matched PyTorch within `atol=1e-2`, with every run
  reporting `allclose=True` across outputs and gradients.

## Numerical Validation
Gradient parity is enforced via `tests/test_gemm_backward.py`, which compares Triton
outputs against the reference `torch.matmul` expressions across multiple shapes and
both FP16/BF16. The suite also exercises zero-length edge cases to ensure the API
returns correctly shaped tensors when dimensions collapse. To run the checks:

```bash
python -m pytest tests/test_gemm_backward.py
```

## Benchmark Script
The benchmark CLI now captures both forward and backward metrics while emitting a
CSV summary containing GFLOP/s, speedups, and maximum absolute differences for the
outputs and gradients. Invoke it as shown above to reproduce the table, or adjust the
shape presets by editing `BenchmarkCase` in `ai_kernel_factory/gemm/benchmark.py`.

## Next Steps
- Implement Hopper TMA variants for the backward kernels to close the performance gap
  with cuBLAS on large batches.
- Extend benchmarking with rectangular shapes and mixed-precision (FP8/INT8) paths.
- Add integration tests that differentiate the Triton backward kernel inside end-to-
  end training loops, ensuring autograd compatibility when swapped into larger models.

## Latest Verification (2025-11-19)

Verified on current environment.

| Size (M=N=K) | Triton fwd (ms) | Torch fwd (ms) | Speedup |
|--------------|-----------------|----------------|---------|
| 2048         | 0.1523          | 0.0718         | 0.47x   |
| 4096         | 0.4075          | 0.3846         | 0.94x   |
| 8192         | 3.2049          | 3.3082         | 1.03x   |

*CuTeDSL backend was also verified but remains experimental (approx. 0.93x speedup at 8192).*

