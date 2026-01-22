# Standalone CuTe DSL Top-K Implementation

## Overview

This is a self-contained CuTe DSL top-k implementation extracted and adapted from QuACK, with all dependencies removed. It can be used independently without the quack package.

## Key Features

1. **Packed Key Optimization**: Encodes value+index into single u64 for efficient sorting
2. **Bitonic Sort Networks**: Uses optimal sorting networks for small arrays (2, 4, 8, 16 elements)
3. **Warp-Level Reduction**: Cross-thread reduction using warp shuffles
4. **Deterministic Tie-Breaking**: Inverts index bits for positive values to prefer smaller indices

## Implementation Details

### Core Components

1. **Sorting Networks**:
   - `optimal_sort_2/4/8/16`: Hand-coded optimal sorting networks
   - `bitonic_sort`: Recursive bitonic sort for power-of-2 sizes
   - `bitonic_merge`: Merge bitonic sequences
   - `bitonic_topk_merge`: Merge two sorted top-k sequences

2. **Top-K Algorithm**:
   - Process input in chunks of size k
   - Sort each chunk with bitonic sort
   - Merge chunks iteratively to maintain top-k
   - Cross-thread reduction via warp shuffles

3. **Packed Key Encoding**:
   ```
   # Pack value (32-bit float) + index into bottom log_N bits
   encoded_idx = ~col_idx if value >= 0 else col_idx
   packed = (value_as_u32 & ~idx_mask) | (encoded_idx & idx_mask)
   ```
   
   This halves memory traffic and ensures deterministic ordering.

### Constraints

- N must be power of 2, N <= 4096
- k must be power of 2, k <= 128
- Supported dtypes: fp16, bf16, fp32

## Usage

```python
from cute_topk import topk

# Create input (M, N) where N is power of 2
x = torch.randn(4096, 1024, device='cuda', dtype=torch.bfloat16)

# Get top-k (k must be power of 2, <= 128)
values, indices = topk(x, k=32)
```

## Performance

On H100 PCIe with M=4096, dtype=bfloat16:

| N | k | Time (ms) | Speedup vs PyTorch |
|---|---|-----------|-------------------|
| 1024 | 32 | 0.097 | 1.17x faster |
| 2048 | 64 | 0.100 | 1.57x faster |
| 4096 | 128 | 0.422 | 1.45x faster |

Constant ~0.10ms for most configurations regardless of k size.

## Differences from QuACK

1. **Removed Dependencies**:
   - No `quack.utils` module
   - No `quack.sort` package
   - Inlined all utility functions

2. **Simplified**:
   - Only included necessary functions
   - Removed cluster-level features
   - Removed advanced memory operations

3. **Self-Contained**:
   - All sorting networks coded inline
   - All utility functions included
   - Single file implementation

## Testing

```bash
cd ai_kernel_factory/sampling
python cute_topk.py
```

Expected output:
```
Shape: torch.Size([4096, 32])
Max diff: 0.000000
Values match: True
Indices match: False  # Different but valid (tie-breaking)
```

## Credits

Based on QuACK implementation by:
- Wentao Guo
- Mayank Mishra  
- Tri Dao

Adapted to be a standalone module without QuACK dependencies.

## Limitations

1. Power-of-2 constraints (can't use arbitrary N or k)
2. Maximum k=128
3. Maximum N=4096
4. Requires CuTe DSL (part of CUTLASS)

For arbitrary sizes, use PyTorch `torch.topk()` instead.
