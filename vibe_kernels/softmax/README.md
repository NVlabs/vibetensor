# CuTeDSL Softmax

CuTeDSL-backed softmax and log-softmax kernels providing optional drop-in
replacements for the existing Triton implementations in `kernel_factory.loss`.
The Python wrappers accept a `backend` keyword (`"triton"` default) so CuTeDSL
can be enabled without removing the Triton path. Tests remain identical across
backends (see below).

## Reproducing Results

```bash
# Benchmarks (requires Quack checkout under tmp/quack for Quack numbers)
PYTHONPATH=$(pwd)/tmp:$(pwd)/tmp/quack:$(pwd) \
  python tmp/kernel_factory/loss/run_softmax_bench.py

# Numerical checks vs PyTorch across backends
PYTHONPATH=$(pwd)/tmp:$(pwd) \
  python -m pytest tmp/kernel_factory/loss/tests/test_softmax_ops.py
```

Runs were collected on an NVIDIA H100 PCIe system (CUDA 12.4), warmup 5/iters 20
per measurement. Speedup columns show `torch_time / backend_time` (values > 1
mean faster than PyTorch).

## Softmax Forward / Backward

| Backend | Rows | Cols  | Dtype    | Forward (ms) | Speedup | Backward (ms) | Speedup |
|---------|-----:|------:|----------|-------------:|--------:|--------------:|--------:|
| torch   | 4096 |  8192 | float16  | 0.1716       | 1.00×   | 0.5815        | 1.00×   |
| triton  | 4096 |  8192 | float16  | 0.2841       | 0.60×   | 0.8467        | 0.69×   |
| cutedsl | 4096 |  8192 | float16  | 0.1559       | 1.10×   | 0.4579        | 1.27×   |
| quack   | 4096 |  8192 | float16  | 0.1388       | 1.24×   | 0.4724        | 1.23×   |
| torch   | 4096 |  8192 | bfloat16 | 0.1711       | 1.00×   | 0.5797        | 1.00×   |
| triton  | 4096 |  8192 | bfloat16 | 0.2851       | 0.60×   | 0.8501        | 0.68×   |
| cutedsl | 4096 |  8192 | bfloat16 | 0.1540       | 1.11×   | 0.4903        | 1.18×   |
| quack   | 4096 |  8192 | bfloat16 | 0.1351       | 1.27×   | 0.4431        | 1.31×   |
| torch   | 4096 | 16384 | float16  | 0.1741       | 1.00×   | 0.9806        | 1.00×   |
| triton  | 4096 | 16384 | float16  | 0.6846       | 0.25×   | 1.7769        | 0.55×   |
| cutedsl | 4096 | 16384 | float16  | 0.1614       | 1.08×   | 0.7349        | 1.33×   |
| quack   | 4096 | 16384 | float16  | 0.1561       | 1.12×   | 0.7351        | 1.33×   |

## Log-Softmax Forward / Backward

(Quack lacks a log-softmax kernel.)

| Backend | Rows | Cols  | Dtype    | Forward (ms) | Speedup | Backward (ms) | Speedup |
|---------|-----:|------:|----------|-------------:|--------:|--------------:|--------:|
| torch   | 4096 |  8192 | float16  | 0.1462       | 1.00×   | 0.4447        | 1.00×   |
| triton  | 4096 |  8192 | float16  | 0.2787       | 0.52×   | 0.8082        | 0.55×   |
| cutedsl | 4096 |  8192 | float16  | 0.1514       | 0.97×   | 0.4838        | 0.92×   |
| torch   | 4096 |  8192 | bfloat16 | 0.1456       | 1.00×   | 0.4443        | 1.00×   |
| triton  | 4096 |  8192 | bfloat16 | 0.2795       | 0.52×   | 0.8087        | 0.55×   |
| cutedsl | 4096 |  8192 | bfloat16 | 0.1568       | 0.93×   | 0.4869        | 0.91×   |
| torch   | 4096 | 16384 | float16  | 0.1562       | 1.00×   | 0.7391        | 1.00×   |
| triton  | 4096 | 16384 | float16  | 0.6802       | 0.23×   | 1.7073        | 0.43×   |
| cutedsl | 4096 | 16384 | float16  | 0.1588       | 0.98×   | 0.7355        | 1.00×   |

## Notes

- CuTeDSL matches PyTorch within the tolerances checked in
  `tmp/kernel_factory/loss/tests/test_softmax_ops.py` (24 permutations covering
  dtype, dimension, and backend).
- The benchmark helper emits all table values (`tmp/kernel_factory/loss/run_softmax_bench.py`).
- Triton remains the default backend to avoid behavior changes; opt-in to
  CuTeDSL via `backend="cutedsl"` when desired.

## Latest Verification (2025-11-19)

Verified performance on current environment (4096x8192, bf16).

| Backend | Latency (ms) | Speedup vs Torch |
|---------|--------------|------------------|
| Triton  | 0.3372       | 1.35x            |
| CuTeDSL | 0.1996       | 2.27x            |
| Torch   | 0.4536       | 1.00x            |

