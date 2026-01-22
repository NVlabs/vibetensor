# Final Top-K Summary (Corrected)

## ⚠️ Critical Correction
Earlier benchmarks relied on an unrealistic configuration (batch=4096, vocab=1024), which overstated the gap between implementations. Real LLM inference workloads behave very differently, so the entire analysis has been rebuilt around production-like shapes.

## 📊 Updated Performance Comparisons
### Scenario 1: Production LLM Inference (batch 1–32, vocab 32k–100k)
| Batch | Vocab | k | dtype | PyTorch | Triton | CuTe/QuACK | Fastest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 50304 | 50 | fp16 | **0.046 ms** | 0.181 ms | N/A | **PyTorch** |
| 4 | 50304 | 5 | fp16 | **0.074 ms** | 0.155 ms | N/A | **PyTorch** |
| 4 | 50304 | 128 | fp16 | **0.048 ms** | 0.729 ms | N/A | **PyTorch** |
| 32 | 32000 | 40 | fp32 | **0.082 ms** | 0.159 ms | N/A | **PyTorch** |

Key takeaways:
- ✅ **PyTorch wins** at 0.04–0.08 ms per call.
- ⚠️ **Triton is usable** but 2–15× slower (absolute latency is still sub-millisecond).
- ❌ **CuTe/QuACK cannot run** because vocab exceeds the 4096 limit.

### Scenario 2: Small Vocabulary Stress Tests (batch 4096, vocab ≤ 4096)
| Batch | Vocab | k | PyTorch | CuTe | QuACK | Triton |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 32 | 0.114 ms | **0.103 ms** | 0.105 ms | 4.22 ms |
| 4096 | 2048 | 64 | 0.159 ms | **0.111 ms** | 0.119 ms | 8.85 ms |
| 4096 | 4096 | 64 | 0.250 ms | 0.111 ms | **0.108 ms** | 9.15 ms |

Key takeaways:
- ⭐ **CuTe/QuACK lead** with 1.1–2.4× speedups over PyTorch when their constraints are met.
- ✅ **PyTorch remains a solid baseline**.
- ❌ **Triton is 37–85× slower** here and should be avoided.

## 🎯 Recommendation Matrix
| Scenario | Vocab | Implementation | Performance | Rationale |
| --- | --- | --- | --- | --- |
| Production LLM inference | 32k–100k | **PyTorch** | Fastest | CuTe unsupported; Triton 2–15× slower. |
| Research / learning | 32k–100k | Triton | 2–15× slower | Good for kernel experimentation. |
| Small-vocab workloads | ≤ 4096 | CuTe | 1.1–2.4× faster | Only scenario where CuTe shines. |
| General fallback | Any | PyTorch | Baseline | Most reliable overall. |

## 💡 Lessons Learned
1. **CuTe limitations are severe.**
   - vocab must be ≤ 4096 and a power-of-two; k ≤ 128.
   - LLaMA (32k) and GPT-2 (50k) are out of scope, so production models cannot use CuTe.

2. **Triton is viable for real workloads.**
   - Earlier “37–202× slower” claims stemmed from unrealistic test shapes.
   - With production shapes (small batch, huge vocab), Triton is only 2–15× slower.

3. **PyTorch is the safest default.**
   - Fastest in real scenarios.
   - Supports every shape without constraints.
   - Only falls behind for synthetic, power-of-two micro-benchmarks.

## 🔁 Reproduction Commands
### Real LLM Shapes (Recommended)
```bash
cd /workspace/terry/nano-cursor
python -c "
import torch, time, os, sys
sys.path.insert(0, 'ai_kernel_factory')
os.environ['AIKF_TOPK_IMPL'] = 'stream'
from sampling.kernel import _select_topk

def bench(fn, n=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

configs = [
    (1, 50304, 50, torch.float16),
    (4, 50304, 5, torch.float16),
    (32, 32000, 40, torch.float32),
]

print('Real LLM scenario:')
for batch, vocab, k, dtype in configs:
    x = torch.randn(batch, vocab, device='cuda', dtype=dtype)
    pt = bench(lambda: torch.topk(x, k, dim=-1))
    tr = bench(lambda: _select_topk(x, k))
    print(f'B={batch} V={vocab} k={k}: PyTorch={pt:.3f}ms Triton={tr:.3f}ms ({pt/tr:.2f}x)')
"
```

### Small Vocabulary Shapes
```bash
python ai_kernel_factory/sampling/test_all_topk.py
```

## 🧠 Smart Selection Helper
```python
def topk_smart(x, k):
    """Dispatch to the best top-k implementation for the given shape."""
    batch, vocab = x.shape

    if vocab >= 10000:
        return torch.topk(x, k, dim=-1)  # fastest and most robust
    elif vocab <= 4096 and is_power_of_2(vocab) and k <= 128:
        from cute_topk import topk
        return topk(x, k)  # 1.1-2.4x faster than PyTorch
    else:
        return torch.topk(x, k, dim=-1)  # reliable fallback
```

## 🗂️ Documentation Status
- ✅ `README.md` — master index (updated).
- ✅ `QUICKSTART_CORRECTED.md` — onboarding guide (updated).
- ✅ `TRITON_PERFORMANCE_CORRECTION.md` — details on fixups.
- ✅ `FINAL_CORRECTED_SUMMARY.md` — this document.

## 🙏 Acknowledgements
Thanks to the community members who highlighted the unrealistic benchmark configuration and helped recalibrate Triton expectations. The real-world data shows:
- Triton is **usable** for inference despite being slower.
- CuTe is **blazing fast but impractical** for real vocabularies.
- PyTorch remains the **default choice**.

---
**Updated:** 2025-11-06  
**Status:** ✅ Corrected & verified  
**Key takeaway:** Always benchmark with shapes that mirror production workloads.
