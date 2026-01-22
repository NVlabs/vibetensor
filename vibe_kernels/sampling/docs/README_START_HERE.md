# Start Here – Top-K Kernel Guide

## ⚡ Quick Answers
- **Running real LLM inference?** (vocab 32k–100k) → Use **PyTorch** `torch.topk(x, k, dim=-1)` — fastest and production-safe.
- **Exploring small vocabularies?** (vocab ≤ 4096, power-of-two) → Use **CuTe** `from cute_topk import topk; topk(x, k)` — 1.4–2.4× faster than PyTorch.

## 📊 Performance at a Glance (H100 PCIe)
| Scenario | Config | Best Choice | Notes |
| --- | --- | --- | --- |
| Real LLM inference | vocab=50304, batch=1, k=50 | **PyTorch** | 0.045 ms; CuTe unsupported; Triton is 4× slower. |
| Small vocabulary stress test | vocab=1024, batch=4096, k=32 | **CuTe** | 0.103 ms vs PyTorch 0.114 ms; Triton is 40× slower. |

## 🎯 Decision Tree
```python
batch, vocab = x.shape

if vocab >= 10000:
    return torch.topk(x, k, dim=-1)  # fastest, production default
elif vocab <= 4096 and is_power_of_2(vocab):
    from cute_topk import topk
    return topk(x, k)  # 1.4-2.4x faster
else:
    return torch.topk(x, k, dim=-1)  # reliable fallback
```

## 🧪 Quick Checks
```bash
# Validate every implementation (cuDNN, CuTe, Triton)
python ai_kernel_factory/sampling/test_all_topk.py

# Run the full benchmark suite (includes Triton variants)
python ai_kernel_factory/sampling/benchmark_final.py
```

## 📚 Documentation Map
**Corrected (read these):**
- [FINAL_CORRECTED_SUMMARY.md](FINAL_CORRECTED_SUMMARY.md) — definitive performance summary.
- [QUICKSTART_CORRECTED.md](QUICKSTART_CORRECTED.md) — two-minute onboarding guide.
- [TRITON_PERFORMANCE_CORRECTION.md](TRITON_PERFORMANCE_CORRECTION.md) — explains Triton adjustments.
- [README.md](README.md) — umbrella summary of every resource.

**Legacy (kept for reference):**
- ~~COMPLETE_BENCHMARK_WITH_TRITON.md~~ — used non-representative configs.
- ~~PERFORMANCE_SUMMARY.md~~ — superseded by the corrected summary.

## ⚠️ Key Reminders
### CuTe / QuACK Limitations
- vocab must be ≤ 4096.
- vocab must be a power-of-two.
- Models such as LLaMA (32k) and GPT-2 (50k) exceed these constraints.

### Triton Performance
- **Real LLM workloads:** 2–15× slower than PyTorch but still usable for experimentation.
- **Small vocabulary sweeps:** 37–200× slower; avoid Triton here.

## ✅ Final Recommendations
| Scenario | Recommendation | Rationale |
| --- | --- | --- |
| Production LLM inference | **PyTorch** | Fastest and most reliable across shapes. |
| Learning / research | **Triton** | Understand kernel behavior; performance is acceptable. |
| Power-of-two small vocabularies | **CuTe** | Only option that beats PyTorch. |

---
**Updated:** 2025-11-06  
**Status:** ✅ Verified  
**Thanks:** Community feedback for surfacing realistic inference configs.
