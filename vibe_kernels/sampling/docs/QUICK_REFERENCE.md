# Quick Reference: Top-K Backend Selection

## Usage

Set environment variable before importing:
```python
import os
os.environ['AIKF_TOPK_IMPL'] = 'torch'  # or 'stream', 'singlepass', 'streaming'

from ai_kernel_factory.sampling import sample_logits
```

## Available Backends

| Backend | Speed | Use Case | Notes |
|---------|-------|----------|-------|
| `torch` | **Fastest** (0.08ms) | b=1, any vocab | PyTorch native topk |
| `stream` | **Fast** (0.16ms) | b≥2, large vocab | **Default**, two-stage Triton |
| `singlepass` | Medium | Medium batch | Single-kernel Triton |
| `streaming` | **Slowest** (15ms) | Reference only | New implementation, not for production |

Benchmark: b=1, vocab=50k, k=50 on H100 PCIe

## Configuration Knobs

### For 'streaming' backend:
```bash
export AIKF_STREAMING_BLOCK_N=128    # Tile size: 64, 128, or 256
export AIKF_STREAMING_WARPS=4        # Warp count: 4 or 8
```

### For 'stream' backend (default):
```bash
export AIKF_STAGE1_WARPS=8
export AIKF_STAGE1_STAGES=2
export AIKF_STAGE2_IMPL=tiles        # or default
export AIKF_STAGE2_TILE=256
export AIKF_STAGE2_WARPS=8
```

### Parallelism hints:
```bash
export AIKF_STAGE1_PARALLEL_HINT=1   # Favor parallelism over candidate count
export AIKF_STAGE1_TARGET_CTAS=128   # Target CTA count
export AIKF_TARGET_STAGE2_CANDIDATES=500  # Target intermediate candidates
```

## Recommendation

**For production**:
```python
# b=1: Use PyTorch
if batch_size == 1:
    os.environ['AIKF_TOPK_IMPL'] = 'torch'
else:
    # b≥2: Use default (stream)
    os.environ['AIKF_TOPK_IMPL'] = 'stream'
```

**For development/debugging**:
- Use `streaming` to test new approach (slow but correct)
- Use `singlepass` for single-kernel simplicity
- Use `torch` as ground truth reference

## Testing

All backends pass correctness tests:
```bash
cd /workspace/terry/nano-cursor
python -m pytest ai_kernel_factory/sampling/tests/test_sampling.py -v
```

## Benchmarking

```bash
# Autotune streaming backend
cd ai_kernel_factory/sampling
python benchmark_streaming.py

# Compare all backends
python topk_benchmark.py --batch 1 --vocab 50000 --top-k 50
```
