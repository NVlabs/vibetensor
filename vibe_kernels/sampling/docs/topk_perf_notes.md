# Streaming Top-k Benchmarks (Nanochat Workload)

_Last updated: (agent) current iteration_

## Test Matrix

All experiments run on an NVIDIA GPU via `python -m ai_kernel_factory.sampling.topk_benchmark` unless noted. Logits sampled from `torch.randn` with seed 0. Precision set to `bfloat16` to match nanochat defaults.

| Batch | Vocab | Top-k | dtype      | torch.topk (ms) | Triton streaming (ms) | Speedup (torch/Triton) | max |diff| | indices equal |
|-------|-------|-------|------------|-----------------|------------------------|-------------------------|-------------|----------------|
| 1     | 50,304| 50    | bfloat16   | 0.0453          | 0.1003                 | 0.452×                  | 0.000e+00    | True           |
| 1     | 50,304| 100   | bfloat16   | 0.0466          | 0.2578                 | 0.181×                  | 0.000e+00    | True           |
| 1     | 50,304| 200   | bfloat16   | 0.0567          | 0.5179                 | 0.110×                  | 0.000e+00    | True           |
| 4     | 50,304| 50    | bfloat16   | 0.0493          | 0.1155                 | 0.427×                  | 0.000e+00    | True           |
| 16    | 50,304| 50    | bfloat16   | 0.0592          | 0.1133                 | 0.523×                  | 0.000e+00    | True           |
| 32    | 50,304| 50    | bfloat16   | 0.0648          | 0.1512                 | 0.429×                  | 0.000e+00    | True           |
| 32    | 50,304| 200   | bfloat16   | 0.0747          | 0.8114                 | 0.092×                  | 0.000e+00    | True           |

Commands follow pattern:

```bash
PYTHONPATH=. python -m ai_kernel_factory.sampling.topk_benchmark \
  --batch <B> --vocab 50304 --top-k <K> --dtype bfloat16 --iters 200
```

## Profiling Snapshots

Using `torch.profiler` over 20 iterations (batch=1, vocab=50,304, dtype=bfloat16):

| Top-k | Total CUDA time (ms/call) | Stage-1 kernel (ms) | Stage-2 kernel (ms) | Stage-2 share |
|-------|---------------------------|---------------------|---------------------|---------------|
| 50    | 0.0897                    | 0.0201              | 0.0508              | ~57%          |
| 100   | 0.2668                    | 0.0407              | 0.2072              | ~78%          |
| 200   | 0.5255                    | 0.0897              | 0.4173              | ~79%          |

Profiler output also surfaces repeated `aten::to/_to_copy` launches (bfloat16 → float32), indicating dtype conversions add measurable overhead.

## Numerical Checks

- `max |diff| = 0.0` and `indices_equal = True` for every benchmark above.
- `pytest ai_kernel_factory/sampling/tests/test_sampling.py` passes (7 tests).

## Key Insights

1. **Stage-2 dominates runtime** once `top_k` exceeds ~50. For `top_k=200`, the merge kernel consumes ~80% of device time, yielding only ~0.09× torch performance.
2. **Stage-1 scales reasonably** with `K`; its contribution roughly doubles when `K` quadruples (50 → 200), but remains secondary to stage-2.
3. **Batch size growth** hurts Triton more than torch. `torch.topk` latency is nearly flat from batch 1 to 32, while Triton increases 1.4×–1.5× for `K=50` and 8× for `K=200`.
4. **dtype conversions** (bf16 → fp32) appear in the profile. Optimizing to avoid redundant conversions could shave a few microseconds per call.
5. **Numerical parity** is solid; the main gap is pure performance. Focus should stay on algorithmic improvements rather than correctness fixes.

## New default heuristic

- Target stage-2 candidate pool near ~500 by default (env: `AIKF_TARGET_STAGE2_CANDIDATES`, default 500). This balances stage‑1 scan cost and stage‑2 merge cost for nanochat (e.g., k=50 → ~10 chunks).
- Launch hints updated:
  - Stage‑1 warps prefers 8 for very large chunks (chunk_size ≥ 8192) when rows ≤ 4.
  - Stage‑2 warps prefers 8 once block_size ≥ 512.

## Immediate Optimization Targets

- Replace the iterative stage-2 loop with a GPU-friendly reducer (e.g., multi-tile merge, packed keys + partial sort, or a heap-based approach) to cut the O(K·candidates) passes.
- Investigate keeping values in fp32 throughout stage-1 to reduce `aten::to` churn, or load logits in fp32 when bf16 cost is acceptable.
- After stage-2 speedup, revisit stage-1 to remove K-pass `tl.max` via tile-level selection.
