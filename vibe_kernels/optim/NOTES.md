# Optimizer Development Notes

## Performance Benchmark
- **CPU (H100 host) Command:**
  ```bash
  python - <<'PY'
  import time
  import torch
  from ai_kernel_factory.optim import TritonAdamW

  device = 'cpu'
  param_shape = (4096, 1024)
  steps = 200

  param_torch = torch.randn(*param_shape, device=device, dtype=torch.float32, requires_grad=True)
  param_triton = param_torch.detach().clone().requires_grad_(True)
  base_grad = torch.randn_like(param_torch)

  opt_torch = torch.optim.AdamW([param_torch], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)
  opt_triton = TritonAdamW([param_triton], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)

  for _ in range(20):
      param_torch.grad = base_grad.clone()
      opt_torch.step()
      param_triton.grad = base_grad.clone()
      opt_triton.step()

  start = time.perf_counter()
  for _ in range(steps):
      param_torch.grad = base_grad.clone()
      opt_torch.step()
  torch_time = (time.perf_counter() - start) / steps

  start = time.perf_counter()
  for _ in range(steps):
      param_triton.grad = base_grad.clone()
      opt_triton.step()
  triton_time = (time.perf_counter() - start) / steps

  print(f"torch.AdamW avg step: {torch_time*1000:.3f} ms")
  print(f"TritonAdamW avg step: {triton_time*1000:.3f} ms")
  print(f"speedup (torch/triton): {torch_time / triton_time:.3f}x")
  PY
  ```
  - **Result:** `torch.AdamW` 7.714 ms/step vs `TritonAdamW` 3.356 ms/step (≈2.30× speedup).

- **GPU (NVIDIA H100 PCIe) Command:**
  ```bash
  python - <<'PY'
  import time
  import torch
  from ai_kernel_factory.optim import TritonAdamW

  device = 'cuda'
  param_shape = (4096, 1024)
  steps = 200

  param_torch = torch.randn(*param_shape, device=device, dtype=torch.float32, requires_grad=True)
  param_triton = param_torch.detach().clone().requires_grad_(True)
  base_grad = torch.randn_like(param_torch)

  opt_torch = torch.optim.AdamW([param_torch], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)
  opt_triton = TritonAdamW([param_triton], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)

  for _ in range(20):
      param_torch.grad = base_grad.clone()
      opt_torch.step()
      param_triton.grad = base_grad.clone()
      opt_triton.step()

  torch.cuda.synchronize()
  start = time.perf_counter()
  for _ in range(steps):
      param_torch.grad = base_grad.clone()
      opt_torch.step()
      torch.cuda.synchronize()
  torch_time = (time.perf_counter() - start) / steps

  start = time.perf_counter()
  for _ in range(steps):
      param_triton.grad = base_grad.clone()
      opt_triton.step()
      torch.cuda.synchronize()
  triton_time = (time.perf_counter() - start) / steps

  print(f"torch.AdamW avg step: {torch_time*1000:.3f} ms")
  print(f"TritonAdamW avg step: {triton_time*1000:.3f} ms")
  print(f"speedup (torch/triton): {torch_time / triton_time:.3f}x")
  PY
  ```
  - **Result:** `torch.AdamW` 0.308 ms/step vs `TritonAdamW` 0.214 ms/step (≈1.44× speedup).

## Numerical Equivalence Check
- **Command:**
  ```bash
  python - <<'PY'
  import torch
  from ai_kernel_factory.optim import TritonAdamW

  shape = (1024, 1024)
  steps = 50
  param_torch = torch.randn(*shape, dtype=torch.float32, requires_grad=True)
  param_triton = param_torch.detach().clone().requires_grad_(True)

  opt_torch = torch.optim.AdamW([param_torch], lr=2e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=3e-2)
  opt_triton = TritonAdamW([param_triton], lr=2e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=3e-2)

  for _ in range(steps):
      grad = torch.randn_like(param_torch)
      param_torch.grad = grad
      param_triton.grad = grad.clone()
      opt_torch.step()
      opt_triton.step()

  diff_param = (param_torch - param_triton).abs()
  diff_exp_avg = (opt_torch.state[param_torch]['exp_avg'] - opt_triton.state[param_triton]['exp_avg']).abs()
  diff_exp_avg_sq = (opt_torch.state[param_torch]['exp_avg_sq'] - opt_triton.state[param_triton]['exp_avg_sq']).abs()

  print(f"param max abs diff: {diff_param.max().item():.3e}")
  print(f"param mean abs diff: {diff_param.mean().item():.3e}")
  print(f"exp_avg max abs diff: {diff_exp_avg.max().item():.3e}")
  print(f"exp_avg_sq max abs diff: {diff_exp_avg_sq.max().item():.3e}")
  PY
  ```
- **Result:** All reported differences were `0.000e+00`, indicating bitwise agreement under the sampled workload.

## Reproducibility
- **Unit tests:**
  ```bash
  pytest ai_kernel_factory/optim/tests/test_adamw.py
  ```
- **Distributed specific test:**
  ```bash
  pytest ai_kernel_factory/optim/tests/test_adamw.py::test_triton_dist_adamw_matches_torch_distributed
  ```

## Implementation Notes
- `TritonDistAdamW` currently averages gradients with `dist.all_reduce` and then gathers parameter shards via `dist.all_gather`. A true `reduce_scatter`/`all_gather` kernel integration will follow once metadata packing is ready.
- Gradients and parameters must be contiguous and have matching shapes; the implementation raises explicit errors otherwise.
- Optimizer states (`exp_avg`, `exp_avg_sq`) are stored per shard using `ShardMetadata` to track offsets.
- Tests spawn two Gloo ranks via `torch.multiprocessing.spawn`; ensure `torch.distributed` with Gloo backend is available on the host.

## Next Steps
- Replace the temporary all-reduce/all-gather fallback with fused Triton reduce-scatter + all-gather once kernels land.
- Extend benchmarks to cover GPU runs and larger parameter groups.
- Add Muon optimizer benchmarks and distributed tests following the same pattern.

## Muon Newton–Schulz Kernel (flash-muon parity)
- **Numerical comparison:**
  ```bash
  python - <<'PY'
  import torch
  from ai_kernel_factory.optim.muon_kernels import fast_newton_schulz

  torch.manual_seed(42)
  mat = torch.randn(128, 96, device='cuda', dtype=torch.float32)

  def reference_newton_schulz(matrix, steps=5):
      a, b, c = (3.4445, -4.7750, 2.0315)
      X = matrix.to(torch.bfloat16)
      transposed = False
      if X.size(-2) > X.size(-1):
          X = X.mT
          transposed = True
      X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
      for _ in range(steps):
          buf1 = X @ X.transpose(-1, -2)
          buf2 = buf1 @ buf1.transpose(-1, -2)
          B = b * buf1 + c * buf2
          X = a * X + torch.matmul(B, X)
      if transposed:
          X = X.mT
      return X.to(matrix.dtype)

  result = fast_newton_schulz(mat, steps=5)
  ref = reference_newton_schulz(mat, steps=5)
  print("max abs diff:", (result - ref).abs().max().item())
  print("mean abs diff:", (result - ref).abs().mean().item())
  PY
  ```
  - **Result:** Both max/mean absolute differences were `0.0` on the sampled workload (bitwise agreement).

- **Performance benchmark:** (square matrices, 5 iterations, H100 PCIe)
  ```bash
  python - <<'PY'
  import time
  import torch
  from ai_kernel_factory.optim.muon_kernels import fast_newton_schulz

  steps = 5
  for dim in (2048, 4096, 8192):
      mat = torch.randn(dim, dim, device='cuda', dtype=torch.float32)
      for _ in range(3):
          fast_newton_schulz(mat, steps=steps)
      torch.cuda.synchronize()
      start = time.perf_counter()
      fast_newton_schulz(mat, steps=steps)
      torch.cuda.synchronize()
      print(f"dim={dim}: triton {time.perf_counter() - start:.4f} s")
  PY
  ```
  - **Result (H100 PCIe):** `dim=2048` → **1.8 ms**, `dim=4096` → **7.2 ms**, `dim=8192` → **53.0 ms** after routing large shapes through the GEMM Hopper (TMA) path.

- **Hopper vs classic vs pilot comparison (single run, warmed-up, 5 iterations):**

  | Size | Hopper (ms) | Classic (ms) | Pilot/torch (ms) |
  |------|-------------|--------------|------------------|
  | 2048 | 1.8         | 1.2          | 0.9              |
  | 4096 | 7.2         | 7.5          | 6.2              |
  | 8192 | 50.9        | 98.4         | 45.3             |

  **Notes:**
  - The Hopper numbers include the full Newton–Schulz loop (two Gram passes, the linear combination, `B @ X`, and bf16/float32 transitions) rather than a bare `matmul_transpose` kernel. Flash-Muon's `0.0124 ms` data point covers only a single Gram kernel.
  - The classic path still mirrors the Flash baseline's upper-triangular writeback, but for 2k–4k sizes PyTorch/cuBLAS tuned matmul carries lower launch overhead, so the torch baseline stays slightly faster.
  - The Hopper/TMA variant adds tensor descriptors plus warp specialization; once the matrix reaches 8k, the workload is large enough to amortize the extra setup and we see a 2× speedup.
