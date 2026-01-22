# Top-K Benchmark Reproduction Guide

## Hardware & Software
- GPU: NVIDIA H100 PCIe (or comparable Hopper GPU)
- CUDA: 12.8+
- Python: 3.12
- PyTorch: 2.9.0+cu128
- Triton: 3.5.0
- CuTe / CUTLASS: 3.x with DSL support

## Setup
```bash
cd /workspace/terry/nano-cursor
pip install -r requirements.txt  # includes torch, triton, cutlass bindings
```

## Running the Benchmarks
### Quick sanity check
```bash
python ai_kernel_factory/sampling/cute_topk.py
```
Validates CuTe correctness against PyTorch.

### Full benchmark suite
```bash
python ai_kernel_factory/sampling/benchmark_final.py --dtype bfloat16 --batch 32 --vocab 50304 --k 50
```
Key flags:
- `--batch` defaults to 32 (roughly production batch size).
- `--vocab` can be adjusted; 50k replicates GPT-2 style workloads.
- `--k` selects the top-k value to test (5–128).

The script prints latency for PyTorch, CuTe, and Triton, plus the relative slowdown.

### Optional stress sweep
To reproduce the small-vocabulary stress numbers:
```bash
python ai_kernel_factory/sampling/benchmark_final.py --batch 4096 --vocab 1024 --k 32
```

## Expected Output
```
====================================================================================================
TOP-K BENCHMARK (Corrected)
Comparing: PyTorch vs CuTe (ours) vs Triton (stream)
====================================================================================================
Batch=32, Vocab=50304, k=50
PyTorch: 0.045 ms
CuTe:    N/A (vocab limit)
Triton:  0.181 ms (4.0× slower)
...
```

## Interpreting Results
- **Production workloads (large vocab, small batch):** expect Triton to be 2–15× slower than PyTorch, with sub-millisecond absolute gaps.
- **Small vocab sweeps (≤4096, power-of-two):** expect CuTe to beat PyTorch by 1.1–2.4×; Triton may be tens of times slower.

Refer to [FINAL_CORRECTED_SUMMARY.md](FINAL_CORRECTED_SUMMARY.md) for the consolidated tables and discussion.

---
**Updated:** 2025-11-06  
**Status:** ✅ Current
