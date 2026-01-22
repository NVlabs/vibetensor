# MNIST Training: Kernel-Level Operations

All kernel-level operations required for training a 2-layer MLP on MNIST.

---

## 1. Forward Pass

### Linear Layer (fc1, fc2)
| Op | Description | Shapes (example) |
|----|-------------|------------------|
| `gemm` / `mm` / `addmm` | Matrix multiplication + bias | [B,784]×[784,256] → [B,256] |
| `add` | Bias addition (if not fused) | [B,256] + [256] → [B,256] |

### Activation
| Op | Description |
|----|-------------|
| `relu` | Elementwise max(x, 0) |

### Cross Entropy Loss
| Op | Description |
|----|-------------|
| `exp` | Elementwise exponential (softmax numerator) |
| `sum` | Reduction along axis (softmax denominator) |
| `log` | Elementwise logarithm |
| `sub` | Elementwise subtraction (log_softmax) |
| `gather` | Index select for target class |
| `neg` | Negate values |
| `mean` | Reduction for loss averaging |

**Fused alternatives:**
| Op | Description |
|----|-------------|
| `log_softmax` | Fused log + softmax (numerically stable) |
| `nll_loss` | Negative log likelihood loss |
| `cross_entropy` | Fully fused cross entropy |

---

## 2. Backward Pass

### Cross Entropy Backward
| Op | Description |
|----|-------------|
| `softmax` | Compute softmax probabilities |
| `scatter_add` / `index_put` | Create one-hot or accumulate gradients |
| `sub` | softmax - one_hot |
| `div` | Scale by batch size |

### Linear Layer Backward
| Op | Description | Purpose |
|----|-------------|---------|
| `gemm` (mm) | grad @ W.T | dX (input gradient) |
| `gemm` (mm) | X.T @ grad | dW (weight gradient) |
| `sum` | Reduce grad along batch | db (bias gradient) |

### ReLU Backward
| Op | Description |
|----|-------------|
| `mul` | grad * (x > 0) mask |
| `gt` / `threshold_backward` | Create gradient mask |

---

## 3. Optimizer (Adam)

| Op | Description | Formula |
|----|-------------|---------|
| `mul` | Scale moments | β₁ * m, β₂ * v |
| `add` | Accumulate | m + (1-β₁) * g |
| `addcmul` | Fused multiply-add | m = β₁*m + (1-β₁)*g |
| `pow` / `square` | Square gradients | g² |
| `div` | Bias correction | m / (1 - β₁ᵗ) |
| `sqrt` | Second moment sqrt | √v̂ |
| `addcdiv` | Fused param update | θ -= lr * m̂ / (√v̂ + ε) |

---

## 4. Utility Operations

| Op | Description |
|----|-------------|
| `copy` / `clone` | Memory copy |
| `contiguous` | Ensure contiguous layout |
| `view` / `reshape` | Change tensor shape (metadata only) |
| `transpose` | Transpose dimensions |
| `fill` | Initialize with constant |
| `zeros` / `ones` | Tensor factories |
| `random` | Random initialization |

---

## 5. Summary: Required Kernel Operations

### Compute-Intensive
- `gemm` / `mm` / `addmm` — Matrix multiplication

### Elementwise (Unary)
- `exp`, `log`, `sqrt`, `neg`, `relu`, `copy`, `fill`

### Elementwise (Binary)
- `add`, `sub`, `mul`, `div`, `pow`, `gt` (comparison)

### Reductions
- `sum`, `mean`, `max` (for softmax stability)

### Indexing
- `gather`, `scatter_add`, `index_select`, `index_put`

### Fused Operations
- `addmm` (gemm + bias)
- `addcmul` (a + b * c)
- `addcdiv` (a + b / c)
- `log_softmax`
- `cross_entropy`

---

# Implementation Status

## VibeTensor (C++ Core)

### ✅ Implemented (CPU + CUDA)

**Elementwise Binary:**
- `add`, `sub`, `mul`, `div`, `pow`

**Elementwise Unary:**
- `relu`, `relu6`, `neg`, `abs`, `exp`, `log`, `sqrt`, `rsqrt`
- `sin`, `cos`, `tanh`, `sigmoid`, `square`
- `floor`, `ceil`, `trunc`, `round`, `frac`, `reciprocal`, `sign`

**Comparisons:**
- `eq`, `ne`, `lt`, `gt`, `le`, `ge`

**Fused Ops:**
- `addcmul`, `addcdiv`, `lerp`, `where`, `clamp`, `threshold`

**Activations:**
- `silu`, `gelu`, `mish`, `selu`, `softplus`, `hardtanh`, `hardsigmoid`, `celu`, `elu`

**Indexing:**
- `index`, `index_put` (CUDA)

**Reductions (CPU + CUDA):**
- `sum`, `mean`, `max`, `min`, `amax`, `amin`, `prod`, `argmax`, `argmin`

**Indexing:**
- `index` (advanced indexing, CUDA), `index_put_`, `select`

### ❌ Missing in VibeTensor

| Op | Status | Priority |
|----|--------|----------|
| `gemm` / `mm` / `matmul` | **NOT IMPLEMENTED** | 🔴 Critical |
| `addmm` | **NOT IMPLEMENTED** | 🔴 Critical |
| `gather` (standalone) | Via advanced indexing (partial) | 🟡 High |
| `scatter_add` | **NOT IMPLEMENTED** | 🟡 High |
| `softmax` | **NOT IMPLEMENTED** | 🟡 High |
| `log_softmax` | **NOT IMPLEMENTED** | 🟡 High |
| `cross_entropy` | **NOT IMPLEMENTED** | 🟡 High |
| `transpose` | **NOT IMPLEMENTED** (for gemm) | 🟡 High |

---

## Kernel Factory (Triton Kernels)

### ✅ Implemented

| Module | Ops | Notes |
|--------|-----|-------|
| `gemm` | `triton_gemm`, `triton_gemm_backward` | Hopper TMA, FP16/BF16 |
| `indexing` | `gather`, `gather_with_grad`, `scatter_add`, `scatter_add_` | **NEW** - With autograd support |
| `softmax` | `softmax`, `log_softmax` | Triton + CuTeDSL backends |
| `loss` | `cross_entropy_loss` | With ignore_index, byte_mean |
| `activation` | `relu_squared`, `elementwise_add`, `elementwise_mul`, `elementwise_where`, `elementwise_lerp`, `rowwise_l2_norm` | Triton |
| `optim` | `TritonAdamW`, `TritonMuon` | Fused optimizer steps |
| `rmsnorm` | `RMSNorm` | With backward |
| `layernorm` | `layernorm`, `CuTeDSLLayerNorm` | Triton + CuTeDSL |
| `rotary` | `apply_rotary_embedding` | For transformers |
| `attention` | `fused_attention` | Causal + GQA |
| `embedding` | `FusedEmbeddingRMSNorm` | Fused lookup + norm |
| `sampling` | `sample_logits` | Top-k/top-p |

### ✅ All Critical Ops Now Implemented

| Module | Ops | Notes |
|--------|-----|-------|
| `indexing` | `gather`, `gather_with_grad`, `scatter_add`, `scatter_add_` | **NEW** - Triton kernels with autograd |

### 🟢 Low Priority (Not Needed)

| Op | Status | Priority |
|----|--------|----------|
| `relu` (standalone) | Only `relu_squared` | 🟢 Low (use PyTorch/VibeTensor) |
| `sum` / `mean` reductions | Use VibeTensor (has CUDA impl) | 🟢 Low |
| `nll_loss` (standalone) | Fused into cross_entropy | 🟢 Low |

---

## Gap Analysis for MNIST Training

### Critical Missing Ops (Must Have)

| Op | VibeTensor | Kernel Factory | Solution |
|----|-----------|----------------|----------|
| `gemm/mm/matmul` | ❌ | ✅ `triton_gemm` | Use kernel_factory or add to VibeTensor |
| `addmm` | ❌ | ✅ (gemm + add) | Compose from existing |

### ✅ All High Priority Ops Available

| Op | VibeTensor | Kernel Factory | Status |
|----|-----------|----------------|--------|
| `softmax` | ❌ | ✅ | Use kernel_factory |
| `log_softmax` | ❌ | ✅ | Use kernel_factory |
| `cross_entropy` | ❌ | ✅ | Use kernel_factory |
| `gather` | Partial | ✅ **NEW** | `kernel_factory.indexing.gather` |
| `scatter_add` | ❌ | ✅ **NEW** | `kernel_factory.indexing.scatter_add` |

### Already Covered

| Op | VibeTensor | Kernel Factory |
|----|-----------|----------------|
| `add` | ✅ | ✅ |
| `mul` | ✅ | ✅ |
| `sub` | ✅ | ❌ |
| `div` | ✅ | ❌ |
| `relu` | ✅ | (relu_squared) |
| `exp` | ✅ | ❌ |
| `log` | ✅ | ❌ |
| `sqrt` | ✅ | ❌ |
| `neg` | ✅ | ❌ |
| `gt` | ✅ | ❌ |
| `sum` | ✅ (CPU+CUDA) | ❌ |
| `mean` | ✅ (CPU+CUDA) | ❌ |
| `max/min` | ✅ (CPU+CUDA) | ❌ |
| `addcmul` | ✅ | ❌ |
| `addcdiv` | ✅ | ❌ |
| `where` | ✅ | ✅ |
| `gather` | Partial | ✅ **NEW** |
| `scatter_add` | ❌ | ✅ **NEW** |

---

## Status: ✅ ALL OPS AVAILABLE FOR MNIST TRAINING

All required operations are now implemented across VibeTensor and Kernel Factory.

### Available in Kernel Factory (Triton)

| Op | Import |
|----|--------|
| GEMM | `from kernel_factory import triton_gemm` |
| Gather | `from kernel_factory.indexing import gather` |
| Scatter Add | `from kernel_factory.indexing import scatter_add` |
| Softmax | `from kernel_factory.loss import softmax` |
| Log Softmax | `from kernel_factory.loss import log_softmax` |
| Cross Entropy | `from kernel_factory.loss import cross_entropy_loss` |
| AdamW | `from kernel_factory.optim import TritonAdamW` |

### Available in VibeTensor (C++ Core)

- Elementwise: `add`, `sub`, `mul`, `div`, `relu`, `exp`, `log`, `sqrt`, `neg`
- Reductions: `sum`, `mean`, `max`, `min`, `argmax`, `argmin`
- Fused: `addcmul`, `addcdiv`, `where`, `clamp`
- Comparisons: `eq`, `ne`, `lt`, `gt`, `le`, `ge`

### Training Approach

Use **PyTorch tensors** with `kernel_factory` Triton kernels for:
- Forward: `triton_gemm` → `relu` → `triton_gemm` → `cross_entropy_loss`
- Backward: Handled by autograd (`gather_with_grad` uses `scatter_add`)
- Optimizer: `TritonAdamW`

Or use **VibeTensor** for tensor management with kernel_factory for compute-heavy ops.
