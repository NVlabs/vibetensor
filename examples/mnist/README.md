# MNIST Training Examples

Two implementations of a 2-layer MLP for MNIST digit classification.

## Scripts

| Script | Framework | Description |
|--------|-----------|-------------|
| `train_pytorch.py` | PyTorch | PyTorch with raw NumPy data loading |
| `train_vbt.py` | Pure VibeTensor | Manual gradients, uses kernel_factory Triton kernels |
| `train_vbt_autograd.py` | VibeTensor + Autograd | Automatic differentiation via `vag.register()` |

## Quick Start

```bash
# Download MNIST data (auto-downloads on first run of PyTorch version)
python examples/mnist/train_pytorch.py

# Run pure VibeTensor version (manual gradients)
python examples/mnist/train_vbt.py

# Run VibeTensor with autograd (automatic differentiation)
python examples/mnist/train_vbt_autograd.py
```

## Data Location

MNIST data is stored in `tmp/data/MNIST/` (gitignored). Both scripts use this path.

## Model Architecture

```
Input (784) -> Linear (256) -> ReLU -> Linear (10) -> Output
```

- Input: 28x28 flattened images (784 features)
- Hidden: 256 units with ReLU activation
- Output: 10 classes (digits 0-9)

## Training Configuration

| Parameter | PyTorch | VibeTensor |
|-----------|---------|-----------|
| Batch Size | 128 | 128 |
| Epochs | 50 | 50 |
| Learning Rate | 0.1 | 0.1 |
| Optimizer | SGD | SGD |
| Batches/Epoch | 468 | 468 |

## Benchmark Results

Tested on NVIDIA GPU (with Triton kernels pre-compiled):

| Metric | PyTorch | VibeTensor (Manual) | VibeTensor (Autograd) |
|--------|---------|--------------------|-----------------------|
| **Test Accuracy** | ~98.0% | ~98.0% | ~98.1% |
| **Training Time** | ~20s | ~31s | ~49s |
| **Avg Epoch Time** | 0.39s | 0.61s | 0.97s |

Note: VibeTensor includes a warmup phase to trigger Triton kernel compilation before timed training.

### Performance Notes

- PyTorch uses highly optimized cuBLAS/cuDNN kernels
- VibeTensor uses Triton kernels from kernel_factory
- First run of VibeTensor incurs ~5s Triton compilation overhead (cached afterward)
- Both achieve similar accuracy with identical hyperparameters

## Dependencies

### PyTorch Version
```
torch
numpy
```

### VibeTensor Version
```
vibetensor (with CUDA)
kernel_factory (Triton kernels)
numpy
```

## Key Differences

| Feature | PyTorch | VibeTensor (Manual) | VibeTensor (Autograd) |
|---------|---------|--------------------|-----------------------|
| Tensor ops | torch.* | vibetensor.torch + ops.vt.* | vibetensor.torch + ops.vt.* |
| GEMM | cuBLAS | Triton (kernel_factory) | Triton (kernel_factory) |
| Softmax | torch.nn.functional | kernel_factory.softmax | kernel_factory.softmax |
| Cross Entropy | torch.nn.CrossEntropyLoss | kernel_factory.loss | kernel_factory.loss |
| Backward | Autograd | Manual implementation | `vag.register()` + `loss.backward()` |
| Data loading | NumPy arrays | NumPy arrays | NumPy arrays |

## Files

```
examples/mnist/
├── README.md                # This file
├── mnist_ops.md             # Detailed kernel-level operations reference
├── train_pytorch.py         # PyTorch implementation
├── train_vbt.py             # VibeTensor manual gradient implementation
└── train_vbt_autograd.py    # VibeTensor autograd implementation

tmp/data/                    # MNIST dataset (gitignored)
└── MNIST/raw/
    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz
```

See also: `design/cuda-autograd-mnist-training.md` for CUDA autograd implementation details.
