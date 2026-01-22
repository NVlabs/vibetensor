# ai_kernel_factory.optim

Triton-backed optimizer utilities intended as drop-in replacements for the NanoChat training stack. The module exposes both single-rank and distributed AdamW variants along with supporting gradient utilities.

## Components
- **`TritonAdamW`** – API-compatible with `torch.optim.AdamW`, now executing its update step through a fused Triton kernel for CUDA tensors (fp32/fp16/bf16). Unsupported configurations automatically fall back to the PyTorch functional path.
- **`TritonDistAdamW`** – ZeRO-2 style sharded optimizer that synchronizes gradients and parameters across ranks via `torch.distributed` primitives (still reusing PyTorch functional updates inside the communication loop).
- **`TritonMuon`** – Single-rank Muon optimizer that applies the Triton Newton–Schulz kernels shipped in `muon_kernels.py`.
- **`clip` utilities** – Helpers for gradient norm computation and clipping (`compute_global_grad_norm`, `clip_grad_norm_`).
- **`muon_kernels`** – Standalone Triton Newton–Schulz primitives (forward/backward-ready Gram builder plus iterative orthogonalization helper).

## Usage
```python
import torch
from ai_kernel_factory.optim import TritonAdamW

model = torch.nn.Linear(1024, 1024)
optimizer = TritonAdamW(model.parameters(), lr=1e-3)

loss = model(torch.randn(16, 1024)).sum()
loss.backward()
optimizer.step()
```

For distributed training, initialize `torch.distributed` before constructing `TritonDistAdamW` and ensure the leading dimension of each parameter is divisible by `world_size`.

## Benchmarks
Use the bundled script to compare TritonAdamW with the PyTorch baseline:

```bash
PYTHONPATH=/path/to/repo python -m ai_kernel_factory.optim.benchmarks.adamw_benchmark \
  --shape 4096 1024 --dtype float16 --steps 200 --warmup 20
```

On an H100 PCIe the following command yielded PyTorch AdamW ≈ 0.196 ms/step vs TritonAdamW ≈ 0.120 ms/step (~1.64× speedup).

Muon Newton–Schulz timings can be measured with:

```bash
PYTHONPATH=/path/to/repo python -m ai_kernel_factory.optim.benchmarks.muon_benchmark \
  --dim 2048 --dtype float16 --steps 5 --warmup 3 --runs 10

PYTHONPATH=/path/to/repo python -m ai_kernel_factory.optim.benchmarks.muon_benchmark \
  --dim 4096 --dtype float16 --steps 5 --warmup 3 --runs 10
```

On the same H100 setup the retuned kernels now report Triton ≈ 0.86 ms/iteration vs PyTorch ≈ 0.82 ms/iteration at dim=2048 (0.95×) and Triton ≈ 5.38 ms/iteration vs PyTorch ≈ 6.34 ms/iteration at dim=4096 (1.18×). Further tuning for ≥8K matrices continues to use the Hopper TMA path.

For Muon, the Triton Newton–Schulz primitives match the Flash-Muon reference within numerical tolerance; see
`pytest ai_kernel_factory/optim/tests/test_muon.py` for the validation suite.

Extended benchmarking notes live in [`NOTES.md`](./NOTES.md).

## Numerical Checks
- `pytest ai_kernel_factory/optim/tests/test_adamw.py -k "not distributed"` validates TritonAdamW against `torch.optim.AdamW` across fp32/fp16/bf16 tensors, including `maximize=True` and multi-parameter groups.
- `pytest ai_kernel_factory/optim/tests/test_muon.py` exercises the Triton matmul/Gram kernels and compares the Muon update step with a pure PyTorch reference implementation.

## Testing
```bash
pytest ai_kernel_factory/optim/tests/test_adamw.py
pytest ai_kernel_factory/optim/tests/test_clip.py
pytest ai_kernel_factory/optim/tests/test_muon.py
```

The distributed optimizer test (`test_triton_dist_adamw_matches_torch_distributed`) spawns two Gloo ranks via `torch.multiprocessing.spawn`; ensure a working `torch.distributed` environment.

## Reproduction Commands
- Run all optimizer tests: `pytest ai_kernel_factory/optim/tests/test_adamw.py`
- Focused distributed check: `pytest ai_kernel_factory/optim/tests/test_adamw.py::test_triton_dist_adamw_matches_torch_distributed`
- Muon kernel validation: `pytest ai_kernel_factory/optim/tests/test_muon.py`
- Performance benchmark: see the embedded script in [`NOTES.md`](./NOTES.md).

## Limitations
- `TritonDistAdamW` currently performs gradient averaging through `all_reduce` followed by `all_gather`. Integration with fused reduce-scatter kernels is planned.
- Distributed Muon (`TritonDistMuon`) is not yet implemented.
- Parameters and gradients must be contiguous and reside on identical devices; sparse gradients are unsupported.
- Hopper-specific tuning (e.g., warp-specialisation) can further improve the Muon kernels on small matrices.
- GPU benchmarks currently cover a single H100 configuration; broaden device/shape coverage.

## Next Steps
- Swap in Triton reduce-scatter/all-gather primitives once kernels are available.
- Layer Hopper-specific optimisations (TMA pipelines, warp specialisation, fused iteration steps) onto the flash-muon baseline kernels.
- Extend benchmarking harnesses to GPUs and additional optimizers.
- Harden distributed tests across NCCL backend once accessible.
