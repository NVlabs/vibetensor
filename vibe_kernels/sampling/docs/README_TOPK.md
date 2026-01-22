# Top-K Documentation Index

## 📚 Primary Guides
| Document | Purpose |
| --- | --- |
| **[README_START_HERE.md](README_START_HERE.md)** | High-level overview and quick answers. |
| **[FINAL_CORRECTED_SUMMARY.md](FINAL_CORRECTED_SUMMARY.md)** | Complete benchmark data and recommendations. |
| **[QUICKSTART_CORRECTED.md](QUICKSTART_CORRECTED.md)** | Two-minute onboarding for CuTe vs PyTorch. |
| **[TRITON_PERFORMANCE_CORRECTION.md](TRITON_PERFORMANCE_CORRECTION.md)** | Details on the Triton measurement fix. |
| **[BENCHMARK_REPRODUCTION.md](BENCHMARK_REPRODUCTION.md)** | Step-by-step instructions to rerun benchmarks. |

## 🧰 Implementation Highlights
- **CuTe DSL Top-K** — `sampling/cute_topk.py`
  - Standalone CUDA/CuTe implementation, up to 2.4× faster than PyTorch when constraints are met.
- **Triton Streaming Top-K** — `sampling/kernel.py`
  - Educational variant; slower than PyTorch but useful for experimentation.
- **Benchmark Suite** — `sampling/benchmark_final.py`
  - Reproduces the corrected measurements across CuTe, PyTorch, Triton, and QuACK.

## 📈 Key Metrics (H100 PCIe)
| Scenario | Winner | Notes |
| --- | --- | --- |
| Real LLM inference (vocab ≥ 32k) | **PyTorch** | CuTe unsupported; Triton 2–15× slower. |
| Small vocab stress test (vocab ≤ 4096, power-of-two) | **CuTe / QuACK** | 1.1–2.4× faster than PyTorch. |

## 🚀 Quick Start Snippet
```python
from cute_topk import topk

x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
values, indices = topk(x, k=64)  # Works when vocab and k are powers of two
```

## 📂 Directory Layout
```
sampling/
├── cute_topk.py           # CuTe implementation
├── kernel.py              # Triton variants (streaming, 2-stage, etc.)
├── benchmark_final.py     # Comprehensive benchmark harness
└── docs/                  # Documentation and archived reports
```

---
**Updated:** 2025-11-06  
**Status:** ✅ Current
