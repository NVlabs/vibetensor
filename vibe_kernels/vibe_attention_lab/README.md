
# VibeAttention Lab

This directory contains a standalone, zero-dependency version of FlexAttention (based on CuTe DSL) extracted from FlashAttention, now renamed to **VibeAttention**. It runs on NVIDIA H100 GPUs without requiring any C++ compilation or the original `flash_attn` package.

## 📂 Structure

*   `benchmark_suite.py`: The main script that validates correctness and benchmarks performance against PyTorch native implementations.
*   `block_sparsity_utils.py`: A Python helper to generate block sparsity masks for VibeAttention.

## 🛠️ Prerequisites

This library requires the NVIDIA CuTe DSL compiler. We recommend version **4.3.0** or newer (which includes critical fixes for boolean masking logic).

```bash
pip install "nvidia-cutlass-dsl>=4.3.0"
pip install torch einops
```

## 🚀 Benchmark Results (H100 PCIe)

We compared the JIT-compiled CuTe DSL kernels against standard PyTorch implementations across 6 different attention variants.

| Experiment | Status | CuTe Time | PyTorch Time | Speedup | Numerics |
|------------|--------|-----------|--------------|---------|----------|
| **T5 Relative Bias** | ✅ **Success** | **0.66 ms** | 118.90 ms | **181.3x** | ✅ Match |
| **ALiBi** | ✅ **Success** | **0.62 ms** | 0.76 ms | **1.23x** | ✅ Match |
| **Document Masking** | ✅ **Success** | **0.74 ms** | 35.92 ms | **48.6x** | ✅ Match |
| **PrefixLM** | ✅ **Success** | **0.58 ms** | 9.97 ms | **17.1x** | ✅ Match |
| **Causal** | ✅ **Success** | **0.52 ms** | 0.55 ms | **1.06x** | ✅ Match |
| **Sliding Window** | ✅ **Success** | **0.63 ms** | 0.69 ms | **1.10x** | ✅ Match |


## 🛠️ Reproduction Steps

To reproduce the benchmarks and verify all implementations:

1.  **Install Dependencies**:
    ```bash
    pip install "nvidia-cutlass-dsl>=4.3.0"
    pip install torch einops
    ```

2.  **Run the Suite**:
    ```bash
    # Ensure you are in the parent directory of 'vibe_attention_lab'
    export PYTHONPATH=$PYTHONPATH:$(pwd)/../
    python benchmark_suite.py
    ```
