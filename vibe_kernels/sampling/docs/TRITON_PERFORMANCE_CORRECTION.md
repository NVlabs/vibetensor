# Triton Top-K Performance Correction

## ⚠️ Why This Exists
The old benchmark results used unrealistic shapes (batch 4096, vocab 1024) and made Triton appear catastrophically slow. Real LLM inference never looks like that, so we recomputed every measurement with production-style shapes.

## 📊 Realistic Results (H100 PCIe)
| Batch | Vocab | k | dtype | PyTorch | Triton | Relative |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 50304 | 50 | fp16 | 0.045 ms | 0.181 ms | Triton 4× slower |
| 4 | 50304 | 5 | fp16 | 0.049 ms | 0.109 ms | Triton 2.2× slower |
| 4 | 50304 | 128 | fp16 | 0.048 ms | 0.729 ms | Triton 15× slower |
| 32 | 32000 | 40 | fp32 | 0.080 ms | 0.159 ms | Triton 2× slower |

Even though Triton lags behind PyTorch, the absolute gap is only 0.05–0.7 ms per call, which is acceptable for experimentation.

## 🚫 What Went Wrong Before
| Batch | Vocab | k | dtype | PyTorch | Triton | Relative |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 32 | bf16 | 0.114 ms | 4.217 ms | Triton 37× slower |
| 4096 | 2048 | 64 | bf16 | 0.159 ms | 8.847 ms | Triton 56× slower |

Those configurations inflate the gap because:
- Batch sizes of 4096 are unrealistic for inference (typical ≤ 32).
- Vocabulary sizes of 1024 are far smaller than real models (32k–100k).
- CuTe only wins because the scenario was tailor-made for its constraints.

## 🎯 Updated Guidance
- **Production LLM inference (vocab ≥ 32k):** stick with **PyTorch**. CuTe cannot run; Triton is slower but usable for validation.
- **Small-vocabulary microbenchmarks:** **CuTe/QuACK** deliver 1.1–2.4× speedups when the power-of-two requirement is met.
- **Learning / kernel exploration:** **Triton** is perfectly fine—expect a 2–15× slowdown but keep the absolute cost in perspective.

## 🧾 Quick Checklist
- ✅ Benchmark the shapes you actually care about (small batches, large vocabularies).
- ✅ Use CuTe only when `vocab ≤ 4096` and `vocab` is a power of two.
- ✅ Use Triton when you need GPU-side customisation or diagnostics.
- ✅ Default to PyTorch for anything customer-facing.

## 🔁 Reproduce the Realistic Numbers
```bash
python ai_kernel_factory/sampling/benchmark_final.py \
  --batch 32 --vocab 50304 --k 50 --dtype float16
```
The script prints PyTorch vs Triton latency side by side and highlights the relative slowdown.

---
**Updated:** 2025-11-06  
**Status:** ✅ Corrected  
**Takeaway:** Triton is slower, not broken—always benchmark with production shapes.
