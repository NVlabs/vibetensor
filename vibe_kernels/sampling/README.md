# Top-K Sampling Module Overview

The sampling module bundles several top-k kernels and the scripts used to benchmark them. It primarily serves the NanoChat runtime, but the components can be reused elsewhere.

## Directory Snapshot
```
ai_kernel_factory/sampling/
├── cute_topk.py            # Standalone CuTe implementation
├── kernel.py               # Triton kernels (two-stage, streaming, etc.)
├── benchmark_final.py      # Benchmark driver
├── test_all_topk.py        # Quick correctness/performance smoke test
└── docs/                   # Detailed documentation and archived reports
```

## Recommended Reading
1. **[README_START_HERE.md](docs/README_START_HERE.md)** — Quick orientation and decision guide.
2. **[FINAL_CORRECTED_SUMMARY.md](docs/FINAL_CORRECTED_SUMMARY.md)** — Consolidated benchmark tables.
3. **[QUICKSTART_CORRECTED.md](docs/QUICKSTART_CORRECTED.md)** — How to run CuTe vs PyTorch in two minutes.
4. **[BENCHMARK_REPRODUCTION.md](docs/BENCHMARK_REPRODUCTION.md)** — Reproduce the corrected results.
5. **[TRITON_PERFORMANCE_CORRECTION.md](docs/TRITON_PERFORMANCE_CORRECTION.md)** — Lessons learned from the Triton attempt.

## Quick Commands
```bash
# Fast smoke test for every implementation (requires CUDA)
python ai_kernel_factory/sampling/test_all_topk.py

# Production-style benchmark (batch 32, vocab 50k, k=50)
python ai_kernel_factory/sampling/benchmark_final.py --batch 32 --vocab 50304 --k 50
```

## High-Level Findings
- **PyTorch** remains the production default for most shapes; with PyTorch 2.9.0+cu128 it
  outperforms the CuTe/QuACK kernels on small and medium vocabularies (`N ≤ 1024`).
- **CuTe / QuACK** only pull ahead (up to ~1.7× speedup) once the column dimension reaches
  ≥2048 and stays power-of-two friendly; otherwise expect ≤1×.
- **Triton** streaming kernels are still functional but trail PyTorch significantly (≈0.01–0.41×
  across the latest sweep).

## Status
- CuTe kernel: ✅ production-ready (self-contained, documented).
- Triton kernels: ⚠️ experimental, useful for research only.
- Documentation: ✅ updated 2025-11-06 with corrected benchmarks.
