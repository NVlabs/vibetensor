# Triton Indexing Kernel Report

## Overview
This module provides indexing operations implemented as Triton kernels:

- `gather`: Select elements from a tensor along a dimension using an index tensor
- `gather_with_grad`: Gather with autograd backward support (uses scatter_add for gradient)
- `scatter_add`: Scatter values into a tensor with accumulation using an index tensor
- `scatter_add_`: In-place version of scatter_add

These operations are essential for:
- Cross-entropy loss computation (gather logits at target indices)
- Embedding lookups
- Sparse gradient accumulation
- Advanced indexing patterns

All kernels run on NVIDIA GPUs via Triton and match PyTorch semantics exactly.

## Directory Layout

```text
kernel_factory/indexing/
├── benchmark.py   # CLI benchmarking harness against PyTorch baselines
├── kernel.py      # Triton implementations and Python wrappers
├── tests/         # CUDA unit tests covering numerics & backward parity
└── README.md      # (this file)
```

## API

### gather
```python
from vibe_kernels.indexing import gather

# out[i, j] = src[idx[i], j]
out = gather(src, dim=0, index=idx)
```

### gather_with_grad
```python
from vibe_kernels.indexing import gather_with_grad

# Same as gather but supports autograd backward
out = gather_with_grad(src, dim=0, index=idx)
loss = out.sum()
loss.backward()  # Gradients computed via scatter_add
```

### scatter_add
```python
from vibe_kernels.indexing import scatter_add

# out[idx[i], j] += src[i, j]
out = torch.zeros(out_size, inner_size, device='cuda')
scatter_add(out, dim=0, index=idx, src=values)
```

## Benchmarks

Benchmarks compare Triton kernels to PyTorch native operations using
`python -m kernel_factory.indexing.benchmark` on NVIDIA H100.

**Gather** (dtype=fp16):

| Src Shape      | Idx Size | Triton (ms) | PyTorch (ms) | Notes |
|----------------|----------|-------------|--------------|-------|
| (1024, 256)    | 128      | 0.028       | 0.009        | PyTorch native is highly optimized |
| (4096, 1024)   | 512      | 0.029       | 0.008        | |
| (16384, 1024)  | 2048     | 0.026       | 0.008        | |

**Scatter Add** (dtype=fp16):

| Out Shape      | Src Size | Triton (ms) | PyTorch (ms) | Notes |
|----------------|----------|-------------|--------------|-------|
| (1024, 256)    | 128      | 0.040       | 0.018        | Atomic operations add overhead |
| (4096, 1024)   | 512      | 0.036       | 0.018        | |
| (16384, 1024)  | 2048     | 0.035       | 0.032        | Gap narrows at larger sizes |

**Note**: PyTorch's native CUDA kernels are highly optimized for these operations.
The Triton kernels provide correct implementations that can be further optimized
or fused into larger kernels (e.g., fused cross-entropy loss).

Commands:

```bash
python -m kernel_factory.indexing.benchmark --iterations 100 --warmup 10 --dtype fp16
python -m kernel_factory.indexing.benchmark --dtype bf16
python -m kernel_factory.indexing.benchmark --dtype fp32
```

## Tests

```bash
python -m pytest kernel_factory/indexing/tests -v
```

Test coverage:
- `test_gather_1d_simple`: Basic 1D gather
- `test_gather_2d`: 2D tensor gather along dim 0
- `test_gather_3d`: 3D tensor gather
- `test_gather_fp16/bf16`: Half-precision support
- `test_gather_large`: Large tensor correctness
- `test_scatter_add_simple`: Basic scatter_add
- `test_scatter_add_accumulate`: Duplicate index accumulation
- `test_scatter_add_values`: Non-uniform value scattering
- `test_scatter_add_fp16`: Half-precision scatter
- `test_scatter_add_large`: Large tensor with random duplicates
- `test_gather_backward`: Autograd gradient correctness
- `test_gather_backward_duplicate_idx`: Gradient accumulation at duplicates
- `test_gather_single_element`: Edge case
- `test_scatter_add_empty`: Empty tensor edge case
- `test_gather_int32_index`: Int32 index support

All 16 tests pass with exact PyTorch parity.

## Implementation Details

### Gather
- Uses a simple 1D kernel where each program handles one index
- Processes inner dimensions in blocks for memory coalescing
- Supports batched operation via 2D kernel variant

### Scatter Add
- Uses `tl.atomic_add` for thread-safe accumulation
- Handles duplicate indices correctly (values accumulate)
- Clamps indices to valid range for safety

### Autograd
- `gather_with_grad` implements `torch.autograd.Function`
- Backward pass uses `scatter_add` to accumulate gradients
- Correctly handles duplicate indices in backward (gradient accumulation)
