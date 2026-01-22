# CuTeDSL Cross-Entropy

CuTeDSL implementations of the masked cross-entropy forward/backward pair now ship
alongside the original Triton kernels under `kernel_factory.loss.cross_entropy_loss`.
The Python entrypoint takes a `backend` keyword (defaulting to `"triton"`) so the
CuTeDSL path can be enabled per-call without removing the Triton fallback. Tests and
benchmarks cover the Triton, CuTeDSL, Torch reference, and optional Quack kernels.

## Reproducing Results

```bash
# Benchmarks (requires tmp/quack on PYTHONPATH for Quack rows)
PYTHONPATH=$(pwd)/tmp:$(pwd)/tmp/quack:$(pwd) \
  python -m kernel_factory.loss.run_cross_entropy_bench --iters 30 --warmup 5

# Numerical checks vs PyTorch across all backends
PYTHONPATH=$(pwd)/tmp:$(pwd) \
  python -m pytest tmp/kernel_factory/loss/tests/test_cross_entropy.py
```

Benchmarking was performed on an NVIDIA H100 PCIe system (CUDA 12.4) with warmup=5
and iters=30. Speedup columns report `torch_time / backend_time` (>1.0 means faster
than PyTorch for that configuration).

## Cross-Entropy Forward / Backward

### Latest Verification (2025-11-19)

| Backend | Batch | Vocab | Dtype    | Forward (ms) | Speedup | Backward (ms) | Speedup |
|---------|------:|------:|----------|-------------:|--------:|--------------:|--------:|
| torch   |  4096 |  8192 | float16  | 0.1673       | 1.00×   | 0.3998        | 1.00×   |
| triton  |  4096 |  8192 | float16  | 0.5066       | 0.33×   | 0.7587        | 0.53×   |
| cutedsl |  4096 |  8192 | float16  | 0.2718       | 0.62×   | 0.5152        | 0.78×   |

*Note: PyTorch's native fused implementation is highly optimized for these dimensions.*

## Notes

- Entire `tmp/kernel_factory/loss/tests/test_cross_entropy.py` suite now runs each
  scenario against both Triton and CuTeDSL to ensure matching losses and gradients,
  including `byte_mean` reduction coverage and ignore-index edge cases.
- CuTeDSL forward now requests per-row gradients so the autograd backward pass only
  rescales cached `dx`, eliminating the second CuTeDSL kernel launch.
- When gradients are disabled (e.g., inference benchmarks), the CuTeDSL path now skips
  materializing `dx`, so the forward timings above reflect pure loss computation.
- cp.async/TMA tiling now keeps 32/64/128-bit loads whenever the vocab is divisible,
  falling back only when divisibility would be violated.
- The CuTeDSL kernel still drops to synchronous copies for very narrow vocabularies to
  satisfy cp.async alignment rules, removing the need for Quack runtime helpers.
- Quack remains the fastest backend for the measured shapes; further tiling/TMA
  tuning plus persistent caches will be required to close the gap.
