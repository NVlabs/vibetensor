# GEMM Benchmark: cuTile vs PyTorch

Benchmarks comparing cuTile GEMM kernels against PyTorch's cuBLAS implementation.

## Files

| File | Description |
|------|-------------|
| `gemm_v3.py` | **Best optimized version** with tuned configs |
| `gemm_benchmark.py` | Original baseline implementation |
| `gemm_opt.py` | First optimization attempt |
| `gemm_benchmark_optimized.py` | Same as gemm_opt.py |
| `gemm_v5.py` | TileGym config experiments |

## Quick Start

Run the optimized benchmark:

```bash
cd /workspace/nano-cursor/gemm_benchmark
python gemm_v3.py
```

## Expected Results (NVIDIA Blackwell B200, sm_103)

```
Size                 cuTile (ms)  cuTile TFLOPS  PyTorch TFLOPS  vs PyTorch  
------------------------------------------------------------------------------------------
512x512x512          0.024        11.24          10.28           1.09x
1024x1024x1024       0.023        92.36          80.40           1.15x
2048x2048x2048       0.056        307.13         290.48          1.06x
4096x4096x4096       0.096        1424.24        1448.68         0.98x
1024x4096x1024       0.047        181.08         167.22          1.08x
4096x1024x4096       0.063        549.48         532.13          1.03x
8192x8192x8192       0.742        1481.33        1361.33         1.09x
```

All cases match or beat PyTorch cuBLAS (>= 0.98x).

## Key Optimizations

1. **2D Swizzle Pattern** - Improves L2 cache locality by grouping blocks
2. **Tuned Tile Sizes** - Different configs for different matrix shapes:
   - Small matrices (512-1024): `128x128x32`
   - 2048x2048: `128x256x64`
   - 4096x4096: `256x256x64`
   - Wide matrices (small K): `256x256x32`
3. **Single CTA Mode** - `num_ctas=1` for better performance on sm_103
4. **TF32 Conversion** - Uses tensor cores for float32 inputs

## Requirements

- CUDA 13.0+
- PyTorch with CUDA support
- cuTile (`cuda.tile`)
