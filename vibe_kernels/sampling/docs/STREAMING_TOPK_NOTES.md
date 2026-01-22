# Streaming Top-K Implementation Notes

## Implementation Attempt Summary

### Goal
Implement a streaming top-k kernel inspired by FlashInfer that:
1. Packs float values + indices into u64 keys
2. Uses tl.topk per tile then bitonic merge into register accumulator
3. Reduces memory traffic and ensures deterministic ordering

### Challenges Encountered

1. **Triton Power-of-2 Requirements**
   - `tl.arange(0, N)` requires N to be power of 2
   - `tl.topk(x, k)` requires k to be power of 2
   - `tl.join(a, b)` creates array of size len(a) + len(b), which may not be power of 2
   - This makes it difficult to use tl.topk on joined arrays

2. **Bitonic Merge API**
   - `tl.bitonic_merge` operates on a single bitonic sequence, not merging two sorted arrays
   - Can't directly merge two top-k results

3. **Manual Selection Loops**
   - Fallback to manual min/max loops is very slow (~14-28ms vs 0.08ms PyTorch)
   - Static unrolling of K iterations creates large kernels

### Trade-offs

**Streaming Approach (attempted)**:
- Pro: Single-pass through data
- Pro: Keeps only K elements in registers
- Con: Requires many tl.static_range iterations (K per tile)
- Con: Manual selection is slow without hardware topk support
- Result: 100x+ slower than PyTorch for k=50

**Current Two-Stage Approach**:
- Stage 1: Extract K candidates per chunk (parallel)
- Stage 2: Reduce candidates to final K
- Pro: Uses simple tl.max operations
- Pro: Highly parallel (many chunks processed in parallel)
- Con: Intermediate storage for candidates (but small: ~500-2K elements)
- Result: Competitive with PyTorch

**PyTorch**:
- Uses highly optimized CUB/Thrust primitives
- ~0.08ms for b=1, vocab=50k, k=50
- Hard to beat without similar low-level optimization

## Recommendations

1. **Keep current two-stage approach** as default for b≤8 use cases
2. **Add torch fallback** for very small batches (b=1) where PyTorch overhead is low
3. **Future optimization**: Implement packed-key approach in stage-1/stage-2 kernels:
   - Pack value+index in stage-1 output
   - Use single tl.topk call in stage-2 on packed keys
   - Unpack at the end

4. **Autotune knobs** to expose:
   - AIKF_TOPK_IMPL: "stream" (default) | "torch" | "singlepass" | "streaming"
   - AIKF_STREAMING_BLOCK_N: tile size (64, 128, 256)
   - AIKF_STREAMING_WARPS: warp count (4, 8)

## Key Insights from FlashInfer

What we can still mirror:
- **Cache hints**: Use `.cg` for streaming loads (already done)
- **FP16 path**: Keep values in FP16 until final stage when possible
- **Early exit**: Add threshold-based early exit in stage-1
- **Deterministic ordering**: Tie-break by smallest index (already done in stage-2)
- **Autotune**: Expose block size and warp count knobs

What's difficult without CUB/custom CUDA:
- True streaming merge with bitonic networks
- Packed key sorting (Triton's tl.topk has power-of-2 constraints)
- Hardware-efficient radix selection
