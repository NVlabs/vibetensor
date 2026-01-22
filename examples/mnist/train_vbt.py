#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MNIST Training with Pure VibeTensor + vibe_kernels (NO PyTorch)

This script trains a 2-layer MLP on MNIST using:
- VibeTensor tensors for data management
- vibe_kernels VibeTensor-native kernels for compute (GEMM, softmax, cross_entropy)
- Manual backward pass using VibeTensor ops
"""

import sys
import os
import gzip
import struct
import time
import numpy as np
import urllib.request
from pathlib import Path

import vibetensor
from vibetensor import torch as vt  # VibeTensor's torch-like API
from vibetensor.torch.ops import ops  # Dispatcher-backed ops namespace

# Import VibeTensor-native kernels from vibe_kernels
from vibe_kernels.gemm.vbt_native import matmul as vbt_matmul
from vibe_kernels.softmax.vbt_native import softmax as vbt_softmax, log_softmax as vbt_log_softmax
from vibe_kernels.loss.vbt_native import cross_entropy as vbt_cross_entropy
from vibe_kernels.indexing.vbt_native import argmax as vbt_argmax


# =============================================================================
# Data Loading (Pure Python + NumPy)
# =============================================================================

def load_mnist_images(path):
    """Load MNIST image file (IDX format)"""
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def load_mnist_labels(path):
    """Load MNIST label file (IDX format)"""
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir):
    """Load MNIST dataset"""
    data_dir = Path(data_dir)
    
    train_images = load_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = load_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = load_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    
    return train_images, train_labels, test_images, test_labels


_MNIST_BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)
_MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    req = urllib.request.Request(url, headers={"User-Agent": "vibetorch-mnist"})
    with urllib.request.urlopen(req) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, out_path)


def ensure_mnist_raw(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in _MNIST_FILES:
        p = data_dir / fname
        if p.is_file() and p.stat().st_size > 0:
            continue
        last_err: Exception | None = None
        for base in _MNIST_BASE_URLS:
            try:
                _download(base + fname, p)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"Failed to download {fname}: {last_err}")


# =============================================================================
# VibeTensor Tensor Utilities
# =============================================================================

def numpy_to_vbt(arr, device="cuda:0"):
    """Convert NumPy array to VibeTensor tensor on device"""
    # Ensure float arrays are float32 (VibeTensor doesn't support float64)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    # First create CPU tensor from numpy
    cpu_tensor = vt.from_numpy(arr)
    # Move to CUDA using .cuda() method
    if device.startswith("cuda"):
        return cpu_tensor.cuda()
    return cpu_tensor


def vbt_to_numpy(tensor):
    """Convert VibeTensor tensor to NumPy array"""
    return tensor.cpu().numpy()


def vbt_zeros(shape, device="cuda:0"):
    """Create zero tensor"""
    # Create on CPU first, then move to device
    cpu_tensor = vt.zeros(shape, device="cpu")
    if device.startswith("cuda"):
        return cpu_tensor.cuda()
    return cpu_tensor


def vbt_randn(shape, device="cuda:0", std=0.01):
    """Create random normal tensor (via NumPy)"""
    arr = np.random.randn(*shape).astype(np.float32) * std
    return numpy_to_vbt(arr, device)


def vbt_transpose(x):
    """Transpose last two dimensions"""
    return x.transpose(0, 1)


# =============================================================================
# 2-Layer MLP Model (Manual Implementation)
# =============================================================================

class MLP:
    """Simple 2-layer MLP: 784 -> 256 -> 10"""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, device="cuda:0"):
        self.device = device
        
        # Xavier initialization
        std1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        std2 = np.sqrt(2.0 / (hidden_dim + output_dim))
        
        # Layer 1: 784 -> 256
        self.W1 = vbt_randn((input_dim, hidden_dim), device, std=std1)
        self.b1 = vbt_zeros((hidden_dim,), device)
        
        # Layer 2: 256 -> 10
        self.W2 = vbt_randn((hidden_dim, output_dim), device, std=std2)
        self.b2 = vbt_zeros((output_dim,), device)
        
        # Gradients
        self.dW1 = None
        self.dW2 = None
        self.db1 = None
        self.db2 = None
        
        # Cached activations for backward
        self.cache = {}
    
    def forward(self, x):
        """Forward pass: x -> fc1 -> relu -> fc2 -> logits"""
        # Cache input
        self.cache['x'] = x
        
        # Layer 1: Linear
        z1 = vbt_matmul(x, self.W1)
        # Add bias (broadcast)
        z1 = ops.vt.add(z1, self.b1)
        self.cache['z1'] = z1
        
        # ReLU activation
        h1 = ops.vt.relu(z1)
        self.cache['h1'] = h1
        
        # Layer 2: Linear
        z2 = vbt_matmul(h1, self.W2)
        z2 = ops.vt.add(z2, self.b2)
        self.cache['z2'] = z2
        
        return z2  # logits
    
    def backward(self, logits, targets):
        """
        Backward pass with cross-entropy loss.
        Returns loss value.
        
        For cross-entropy, grad_logits = softmax(logits) - one_hot(targets)
        """
        batch_size = logits.sizes[0]
        num_classes = logits.sizes[1]
        
        # Compute loss
        loss = vbt_cross_entropy(logits, targets, reduction='mean')
        
        # Gradient of cross-entropy w.r.t logits: softmax(logits) - one_hot(targets)
        probs = vbt_softmax(logits)  # [B, C]
        
        # Create one-hot from targets and subtract
        # one_hot[i, targets[i]] = 1.0
        # grad_logits = probs - one_hot
        targets_np = vbt_to_numpy(targets)
        one_hot_np = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot_np[np.arange(batch_size), targets_np] = 1.0
        one_hot = numpy_to_vbt(one_hot_np, self.device)
        
        # dL/dz2 = (probs - one_hot) / batch_size
        dz2 = ops.vt.sub(probs, one_hot)
        dz2 = ops.vt.div(dz2, float(batch_size))
        
        # Layer 2 gradients
        h1 = self.cache['h1']
        h1_T = vbt_transpose(h1)  # [hidden, B]
        self.dW2 = vbt_matmul(h1_T, dz2)  # [hidden, num_classes]
        self.db2 = dz2.sum(dim=0)  # [num_classes]
        
        # Backprop to layer 1
        W2_T = vbt_transpose(self.W2)  # [num_classes, hidden]
        dh1 = vbt_matmul(dz2, W2_T)  # [B, hidden]
        
        # ReLU backward: dz1 = dh1 * (z1 > 0) using where
        z1 = self.cache['z1']
        relu_mask = ops.vt.gt(z1, 0.0)  # [B, hidden], dtype=bool
        # Create zeros for where
        zeros = vbt_zeros(dh1.sizes, self.device)
        # where(mask, dh1, 0) = dh1 if z1 > 0 else 0
        dz1 = ops.vt.where(relu_mask, dh1, zeros)
        
        # Layer 1 gradients
        x = self.cache['x']
        x_T = vbt_transpose(x)  # [input_dim, B]
        self.dW1 = vbt_matmul(x_T, dz1)  # [input_dim, hidden]
        self.db1 = dz1.sum(dim=0)  # [hidden]
        
        return loss
    
    def parameters(self):
        """Return list of (param, grad) tuples"""
        return [
            (self.W1, self.dW1),
            (self.b1, self.db1),
            (self.W2, self.dW2),
            (self.b2, self.db2),
        ]


# =============================================================================
# SGD Optimizer (Pure VibeTensor)
# =============================================================================

class SGD:
    """Simple SGD optimizer"""
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, model):
        """Update parameters: param = param - lr * grad"""
        for param, grad in model.parameters():
            if grad is not None:
                # param -= lr * grad using in-place add_
                # First compute -lr * grad
                neg_scaled_grad = ops.vt.mul(grad, -self.lr)
                # In-place update
                param.add_(neg_scaled_grad)


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, optimizer, images, labels, batch_size=128):
    """Train for one epoch"""
    num_samples = images.shape[0]
    num_batches = num_samples // batch_size
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Shuffle indices
    indices = np.random.permutation(num_samples)
    
    for i in range(num_batches):
        # Get batch
        start = i * batch_size
        end = start + batch_size
        batch_idx = indices[start:end]
        
        batch_images = images[batch_idx]
        batch_labels = labels[batch_idx]
        
        # Convert to VibeTensor tensors
        x = numpy_to_vbt(batch_images, model.device)
        y = numpy_to_vbt(batch_labels.astype(np.int64), model.device)
        
        # Forward
        logits = model.forward(x)
        
        # Backward (computes gradients and returns loss)
        loss = model.backward(logits, y)
        
        # Optimizer step
        optimizer.step(model)
        
        # Track metrics
        loss_val = vbt_to_numpy(loss.cpu()).item()
        total_loss += loss_val
        
        # Accuracy
        preds = vbt_argmax(logits, dim=1)
        preds_np = vbt_to_numpy(preds)
        correct += (preds_np == batch_labels).sum()
        total += batch_size
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{num_batches}, Loss: {loss_val:.4f}, Acc: {correct/total:.4f}")
    
    return total_loss / num_batches, correct / total


def evaluate(model, images, labels, batch_size=256):
    """Evaluate on test set"""
    num_samples = images.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    correct = 0
    total = 0
    
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_samples)
        
        batch_images = images[start:end]
        batch_labels = labels[start:end]
        
        x = numpy_to_vbt(batch_images, model.device)
        
        # Forward only (no backward)
        with vt.no_grad():
            logits = model.forward(x)
        
        # Accuracy
        preds = vbt_argmax(logits, dim=1)
        preds_np = vbt_to_numpy(preds)
        correct += (preds_np == batch_labels).sum()
        total += len(batch_labels)
    
    return correct / total


def main():
    print("=" * 60)
    print("MNIST Training with Pure VibeTensor (NO PyTorch)")
    print("=" * 60)
    
    # Check CUDA
    if not vibetensor._C._has_cuda or vibetensor._C._cuda_device_count() == 0:
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    device = "cuda:0"
    print(f"Using device: {device}")
    
    # Load data (from tmp/ folder at repo root)
    data_dir = Path(__file__).parent.parent.parent / "tmp" / "data" / "MNIST" / "raw"
    ensure_mnist_raw(data_dir)
    print(f"\nLoading MNIST from {data_dir}...")
    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Create model
    print("\nCreating model...")
    model = MLP(input_dim=784, hidden_dim=256, output_dim=10, device=device)
    
    # Create optimizer
    optimizer = SGD(lr=0.1)
    
    # Training config
    num_epochs = 50
    batch_size = 128
    
    # Warmup: run a few batches to trigger Triton kernel compilation
    print("\nWarmup (compiling Triton kernels)...")
    warmup_start = time.time()
    warmup_batches = 5
    indices = np.random.permutation(len(train_images))
    for i in range(warmup_batches):
        start = i * batch_size
        end = start + batch_size
        batch_idx = indices[start:end]
        x = numpy_to_vbt(train_images[batch_idx], device)
        y = numpy_to_vbt(train_labels[batch_idx].astype(np.int64), device)
        logits = model.forward(x)
        loss = model.backward(logits, y)
        optimizer.step(model)
    warmup_time = time.time() - warmup_start
    print(f"  Warmup complete in {warmup_time:.2f}s ({warmup_batches} batches)")
    
    # Reset model for fair training
    model = MLP(input_dim=784, hidden_dim=256, output_dim=10, device=device)
    optimizer = SGD(lr=0.1)
    
    # Training loop (timed)
    print(f"\nTraining for {num_epochs} epochs...")
    train_start = time.time()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, optimizer, train_images, train_labels, batch_size
        )
        
        test_acc = evaluate(model, test_images, test_labels)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    train_time = time.time() - train_start
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Warmup time: {warmup_time:.2f}s (Triton compilation)")
    print(f"Training time: {train_time:.2f}s ({num_epochs} epochs)")
    print(f"Avg epoch time: {train_time/num_epochs:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
