# Streaming Top-K Implementation Summary

## What Was Attempted

Implemented a streaming top-k kernel inspired by FlashInfer's approach:
- Iterates through vocabulary tiles
- Maintains K-element register accumulator
- Merges tile top-k results into accumulator
- Uses cache hints (`.cg`) for streaming loads
- Deterministic tie-breaking (smallest index wins)

## Implementation Status

✅ **Working features**:
- Correctness: Matches PyTorch `topk` exactly
- Proper sorted output in descending order  
- Deterministic ordering for ties
- Configurable via environment variables:
  - `AIKF_TOPK_IMPL=streaming` to enable
  - `AIKF_STREAMING_BLOCK_N` for tile size (64, 128, 256)
  - `AIKF_STREAMING_WARPS` for warp count (4, 8)

❌ **Performance issues**:
- Manual min/max selection loops are very slow
- ~14-28ms vs PyTorch's ~0.08ms (b=1, vocab=50k, k=50)
- 100x+ slower than baseline

## Why It's Slow

Triton constraints prevented using efficient hardware primitives:
1. `tl.topk(x, k)` requires k to be power-of-2
2. `tl.join(a, b)` creates non-power-of-2 arrays
3. `tl.bitonic_merge` operates on single bitonic sequence, not two sorted arrays
4. Fallback to manual `tl.static_range(K)` loops with min/max creates large kernels

## Current Production Approach

The existing two-stage kernel remains faster:
- **Stage 1**: Extract K candidates per chunk (parallel across chunks)
- **Stage 2**: Reduce candidates to final K
- Simple tl.max operations, highly parallel
- Competitive with PyTorch (~0.16ms vs ~0.08ms)

## Recommendations

1. **Keep two-stage approach** as default (already in codebase)
2. **Use PyTorch fallback** for b=1 cases (add via `AIKF_TOPK_IMPL=torch`)
3. **Future work**: Implement packed-key optimization in stage-1/stage-2:
   - Pack float value + index into u64 in stage-1
   - Use single tl.topk on packed keys in stage-2
   - Reduces memory traffic and ensures deterministic ordering

## Key Learnings from FlashInfer

Applicable optimizations already added:
- ✅ Cache modifier `.cg` for streaming loads
- ✅ Deterministic tie-breaking (smallest index)
- ✅ Autotune knobs (BLOCK_N, num_warps)
- ✅ Multiple backend selection (torch/stream/singlepass)

Difficult to implement in pure Triton:
- ❌ True streaming bitonic merge
- ❌ Packed key sorting with flexible K
- ❌ Hardware radix selection

## Files Added

- `sampling/kernel.py`: Added `_streaming_topk_kernel` implementation
- `sampling/benchmark_streaming.py`: Autotune benchmark script
- `sampling/STREAMING_TOPK_NOTES.md`: Detailed implementation notes
- Environment variables for runtime configuration

## Conclusion

The streaming approach is implemented and correct but not performant enough for production use. The existing two-stage approach remains the recommended default. The streaming implementation serves as a reference and may become useful if Triton adds more flexible topk/merge primitives in future versions.

For immediate use, recommend:
- Default: Two-stage approach (current behavior)
- b=1: PyTorch fallback via `AIKF_TOPK_IMPL=torch`
- Expose autotune knobs for stage-1/stage-2 tile sizes
