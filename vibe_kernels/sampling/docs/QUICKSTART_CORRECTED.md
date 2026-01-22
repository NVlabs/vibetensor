# Top-K Kernel Quick Start (Corrected)

## TL;DR
We maintain four top-k paths. Choose based on vocabulary size and constraints:
1. **CuTe (ours)** – fastest when `vocab` and `k` are powers of two, `vocab ≤ 4096`, and `k ≤ 128`.
2. **QuACK (original)** – broadly the same performance envelope as CuTe under the same constraints.
3. **PyTorch** – dependable baseline that works for every shape.
4. **Triton (ours)** – educational implementation; slower but useful for experimentation.

```python
import math

N = x.shape[1]
if N >= 1024 and N == 2 ** int(math.log2(N)) and k <= 128 and k == 2 ** int(math.log2(k)):
    from cute_topk import topk
    values, indices = topk(x, k)  # 1.4–2.4× faster than PyTorch
else:
    values, indices = torch.topk(x, k, dim=-1)  # production default
```

## Performance Snapshot (H100 PCIe)
| Scenario | Config | Best Choice | Notes |
| --- | --- | --- | --- |
| Real LLM inference | vocab=50304, batch=1, k=50 | **PyTorch** | CuTe unsupported; Triton 4× slower. |
| Small vocab sweep | vocab=1024, batch=4096, k=32 | **CuTe** | 0.103 ms vs PyTorch 0.114 ms. |
| Small vocab sweep | vocab=2048, batch=4096, k=64 | **CuTe** | 0.111 ms vs PyTorch 0.159 ms. |

## Quick Benchmarks
```bash
# Validate every implementation (requires CUDA)
python ai_kernel_factory/sampling/test_all_topk.py

# Full benchmark run (adds Triton stream kernel)
python ai_kernel_factory/sampling/benchmark_final.py
```

## Best Practices
- Use **PyTorch** for production workloads (large vocabularies, flexible shapes).
- Use **CuTe/QuACK** only when the vocabulary is ≤ 4096 and a power of two.
- Use **Triton** when you need kernel introspection or parity checks; heed the performance warning the script prints.

## Related Documents
- [FINAL_CORRECTED_SUMMARY.md](FINAL_CORRECTED_SUMMARY.md)
- [README_START_HERE.md](README_START_HERE.md)
- [TRITON_PERFORMANCE_CORRECTION.md](TRITON_PERFORMANCE_CORRECTION.md)
