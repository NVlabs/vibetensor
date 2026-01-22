# Triton Activation Kernel Report

## Overview
This module provides activation-focused Triton kernels used throughout NanoChat's
MLP stack. The package currently includes

- `relu_squared`: elementwise ReLU² with fused backward support
- `softcap_tanh_projection`: biasless feature projection that emits both softcap
  and tanh activations in a single pass
- Elementwise helpers (`add`, `mul`, `where`, `lerp`) and a row-wise L2 norm
  primitive shared across optimizers and residual connections

All kernels run on NVIDIA GPUs via Triton and fall back to PyTorch to validate
numerics.

## Directory Layout

```text
ai_kernel_factory/activation/
├── benchmark.py   # CLI benchmarking harness against PyTorch baselines
├── kernel.py      # Triton implementations and Python wrappers
├── tests/         # CUDA unit tests covering numerics & backward parity
└── README.md      # (this file)
```

## Benchmarks
Benchmarks compare Triton kernels to their PyTorch counterparts using
`python -m ai_kernel_factory.activation.benchmark` on an NVIDIA H100 PCIe with
CUDA 12, PyTorch 2.9 nightly, and `triton==3.5.0`.

All runs: batch = 4096, features = 4096, dtype = BF16, warmup = 10, iterations = 50.

| Activation         | PyTorch (ms) | Triton (ms) | Speedup | `max|Δ|` |
|--------------------|--------------|-------------|---------|----------|
| `relu_squared`     | 0.0816       | 0.0417      | 1.96×   | 0.0e+00  |
| `softcap_tanh`     | 0.2341       | 0.2251      | 1.04×   | 3.9e-03  |

Commands:

```bash
python -m ai_kernel_factory.activation.benchmark --activation relu_squared --batch 4096 --features 4096 --dtype bfloat16
python -m ai_kernel_factory.activation.benchmark --activation softcap_tanh --batch 4096 --features 4096 --dtype bfloat16
```

## Tests

```
python -m pytest ai_kernel_factory/activation/tests
```

The suite verifies forward parity, gradients, and utility helpers against
PyTorch references under CUDA.
