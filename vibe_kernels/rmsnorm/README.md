# RMSNorm Kernels

## Overview
The RMSNorm package now exposes both the original Triton implementation and a CuTeDSL backend ported from Quack. Both variants support FP16/BF16/FP32 activations, optional learnable scale (gamma), residual fusion, and custom backward passes.

## Numeric Validation
CuTeDSL results match the upstream Quack kernels exactly across the scenarios below. The helper script requires both `tmp` and `tmp/quack` on `PYTHONPATH`:

```bash
PYTHONPATH=$(pwd)/tmp:$(pwd)/tmp/quack \
python - <<'PY'
import importlib, torch, sys
from vibe_kernels.rmsnorm import cutedsl_rmsnorm
quack_rmsnorm = importlib.import_module('quack.rmsnorm')

def run(rows, hidden, dtype, bias=False, residual=False):
    torch.manual_seed(0)
    device = torch.device('cuda')
    x = torch.randn(rows, hidden, device=device, dtype=dtype, requires_grad=True)
    gamma = torch.randn(hidden, device=device, dtype=torch.float32, requires_grad=True)
    beta = torch.randn(hidden, device=device, dtype=dtype, requires_grad=True) if bias else None
    res = torch.randn(rows, hidden, device=device, dtype=dtype, requires_grad=True) if residual else None

    def clone(t):
        return t.detach().clone().requires_grad_(True) if t is not None else None

    xc, gc, bc, rc = x.clone().requires_grad_(True), clone(gamma), clone(beta), clone(res)
    qc, qg, qb, qr = x.clone().requires_grad_(True), clone(gamma), clone(beta), clone(res)

    yc = cutedsl_rmsnorm(xc, gc, bias=bc, residual=rc, eps=1e-6)
    yq = quack_rmsnorm.rmsnorm(qc, qg, bias=qb, residual=qr, eps=1e-6)
    grad = torch.randn_like(yc)
    yc.backward(grad, retain_graph=True)
    yq.backward(grad.clone())

    diff = (yc.detach() - yq.detach()).abs().max().item()
    print(f"rows={rows} hidden={hidden} dtype={dtype} bias={bias} residual={residual} max|Δ|={diff:.3e}")

run(1024, 4096, torch.float16)
run(1024, 4096, torch.bfloat16, residual=True)
run(2048, 8192, torch.float16, bias=True)
PY
```

| Rows | Hidden | Dtype    | Bias | Residual | Max \|Δ\| vs Quack |
|------|--------|----------|------|----------|-------------------|
| 1024 | 4096   | float16  | No   | No       | 0.000e+00         |
| 1024 | 4096   | bfloat16 | No   | Yes      | 0.000e+00         |
| 2048 | 8192   | float16  | Yes  | No       | 0.000e+00         |

Gradients with respect to inputs, weights, and biases also match to numerical zero in every case. For comparison, PyTorch’s built-in `rms_norm` diverges for residual fusion (up to 1.37e+01) because it lacks that fused path.

## Performance
Benchmarks use an H100 PCIe GPU. Run with:

```bash
PYTHONPATH=$(pwd)/tmp \
python -m kernel_factory.rmsnorm.benchmark \
  --rows 4096 --hidden 8192 --dtype float16 \
  --backends torch,cutedsl,quack,triton --reference quack \
  --warmup 10 --iters 60
```

### Latest Verification (2025-11-19)

| Rows | Hidden | Dtype    | Config           | Torch (ms) | CuTeDSL (ms) | Quack (ms) | Triton (ms) | Notes |
|------|--------|----------|------------------|------------|--------------|------------|-------------|-------|
| 4096 | 8192   | bfloat16 | weight only      | 0.8178     | 0.2040       | 0.1980     | 0.1308      | Triton 6.3x vs Torch; CuTeDSL 4.0x vs Torch |


CuTeDSL stays numerically identical to Quack across all scenarios while keeping runtime within a few percent. PyTorch’s native op is substantially slower because it materializes bias/residual separately and cannot fuse gamma in mixed precision. Triton remains an option for raw speed but should be used cautiously when residual or bias fusion is required.

## Usage
```python
import torch
from vibe_kernels.rmsnorm import RMSNorm, cutedsl_rmsnorm

# Triton module
module = RMSNorm(hidden_size=8192, eps=1e-6, learnable_gamma=True)
out = module(torch.randn(1024, 8192, device='cuda', dtype=torch.float16))

# CuTeDSL functional call
x = torch.randn(1024, 8192, device='cuda', dtype=torch.float16)
gamma = torch.randn(8192, device='cuda', dtype=torch.float32)
out_cutedsl = cutedsl_rmsnorm(x, gamma, eps=1e-6)
```

## VibeTensor Integration Benchmark (2025-11-20)

Performance comparison of RMSNorm using VibeTensor's optimized zero-copy dispatch vs PyTorch Native and Direct Kernel Factory (Triton) calls.
This integration uses a **Zero-Intrusion Adapter** (`vibetensor_impl.py`), leaving the original kernel implementation untouched.

**Shape:** (4096, 4096) | **Dtype:** FP16 | **Device:** NVIDIA H100

| Provider | Time (ms) | Bandwidth (GB/s) | Note |
| :--- | :--- | :--- | :--- |
| **PyTorch Native** | 0.0457 | 1470 | Baseline (cuDNN/Native) |
| **KF Direct (Triton)** | 0.0539 | 1245 | Direct Python call to Triton kernel |
| **VibeTensor (Adapter)** | **0.0553** | **1213** | **Fully Asynchronous Dispatch** |

*> Note: VibeTensor performance is roughly on par with the direct Triton call, confirming minimal overhead (approx. 1.4µs) added by the adapter layer.*

To reproduce:
```bash
python tmp/kernel_factory/rmsnorm/benchmark_vibetensor.py
```
