#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MNIST Training with VibeTensor Autograd (NO PyTorch)

This script trains a 2-layer MLP on MNIST using:
- VibeTensor tensors with autograd enabled
- vibe_kernels Triton kernels with registered backward functions
- Automatic gradient computation via loss.backward()
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
import vibetensor._C as C
import vibetensor.torch as vt
import vibetensor.autograd as vag
from vibetensor.library import Library

# Import vibe_kernels kernels (CUDA-native)
from vibe_kernels.gemm.vbt_native import matmul as kf_matmul
from vibe_kernels.softmax.vbt_native import softmax as kf_softmax
from vibe_kernels.loss.vbt_native import cross_entropy as kf_cross_entropy
from vibe_kernels.indexing.vbt_native import argmax as kf_argmax


# =============================================================================
# Data Loading
# =============================================================================

def load_mnist_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def load_mnist_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        assert magic == 2049
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_mnist(data_dir):
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
# Tensor Utilities
# =============================================================================

def numpy_to_vbt(arr, device="cuda:0"):
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    cpu_tensor = vt.from_numpy(arr)
    if device.startswith("cuda"):
        return cpu_tensor.cuda()
    return cpu_tensor


def vbt_to_numpy(tensor):
    return np.from_dlpack(tensor.cpu()).reshape(tuple(tensor.sizes))


def vbt_zeros(shape, device="cuda:0"):
    cpu_tensor = vt.zeros(shape, device="cpu")
    if device.startswith("cuda"):
        return cpu_tensor.cuda()
    return cpu_tensor


def vbt_randn(shape, device="cuda:0", std=0.01):
    arr = np.random.randn(*shape).astype(np.float32) * std
    return numpy_to_vbt(arr, device)


# =============================================================================
# Register Autograd for Kernel Factory Ops (CUDA-native backwards)
# =============================================================================

def register_autograd_ops():
    """Register vibe_kernels ops with autograd backward functions."""
    lib = Library("kfad", "DEF")
    
    # Define op schemas
    for schema in (
        "kfad::matmul(Tensor, Tensor) -> Tensor",
        "kfad::relu(Tensor) -> Tensor",
        "kfad::cross_entropy_loss(Tensor, Tensor) -> Tensor",
        "kfad::bias_add(Tensor, Tensor) -> Tensor",
    ):
        try:
            lib.define(schema)
        except Exception:
            pass
    
    # Forward: matmul using vibe_kernels (CUDA Triton)
    def _fwd_matmul(a, b):
        return kf_matmul(a, b)
    
    # Forward: relu using C.vt (has CUDA impl)
    def _fwd_relu(x):
        return C.vt.relu(x)
    
    # Forward: cross_entropy using vibe_kernels
    def _fwd_cross_entropy(logits, targets):
        return kf_cross_entropy(logits, targets, reduction='mean')
    
    # Forward: bias_add (z + b) where z is (batch, hidden), b is (hidden,)
    def _fwd_bias_add(z, b):
        # Expand b to match z's shape: (hidden,) -> (1, hidden) -> (batch, hidden)
        batch_size = z.sizes[0]
        hidden_size = z.sizes[1]
        b_expanded = b.unsqueeze(0)  # (1, hidden)
        b_expanded = b_expanded.expand([batch_size, hidden_size])  # (batch, hidden)
        b_expanded = b_expanded.clone()  # Make contiguous
        return C.vt.add(z, b_expanded)
    
    lib.impl("matmul", _fwd_matmul, dispatch_key="CUDA", allow_override=True)
    lib.impl("relu", _fwd_relu, dispatch_key="CUDA", allow_override=True)
    lib.impl("cross_entropy_loss", _fwd_cross_entropy, dispatch_key="CUDA", allow_override=True)
    lib.impl("bias_add", _fwd_bias_add, dispatch_key="CUDA", allow_override=True)
    
    # Backward: matmul (all ops stay on CUDA)
    # dL/dA = dL/dC @ B.T
    # dL/dB = A.T @ dL/dC
    def _bw_matmul(grads_in, saved):
        (grad_out,) = grads_in
        (a, b) = saved
        # Use vibe_kernels matmul which is CUDA-native
        b_T = b.transpose(0, 1)
        grad_a = kf_matmul(grad_out, b_T)
        a_T = a.transpose(0, 1)
        grad_b = kf_matmul(a_T, grad_out)
        return (grad_a, grad_b)
    
    # Backward: relu (CUDA-native via C._call_op)
    # dL/dx = dL/dy * (x > 0)
    def _bw_relu(grads_in, saved):
        (grad_out,) = grads_in
        (x,) = saved
        # Get device from input tensor (device is a tuple: (type, index))
        device = "cuda:0" if x.device[0] == 2 else "cpu"
        # Create zero tensor on same device (same shape as grad_out)
        zeros_np = np.zeros(grad_out.sizes, dtype=np.float32)
        zeros = numpy_to_vbt(zeros_np, device)
        # Use C._call_op for gt/where (stay on same device, tensor args required)
        mask = C._call_op('vt::gt', x, zeros)  # x > 0 (bool tensor)
        grad_x = C._call_op('vt::where', mask, grad_out, zeros)
        return (grad_x,)
    
    # Backward: cross_entropy (CUDA-native)
    # dL/d_logits = softmax(logits) - one_hot(targets)
    def _bw_cross_entropy(grads_in, saved):
        (grad_out,) = grads_in  # scalar
        (logits, targets) = saved
        
        # Get device from input tensor (device is a tuple: (type, index))
        device = "cuda:0" if logits.device[0] == 2 else "cpu"
        
        batch_size = logits.sizes[0]
        num_classes = logits.sizes[1]
        
        # Compute softmax (CUDA via vibe_kernels)
        probs = kf_softmax(logits)
        
        # Create one-hot on same device
        # This is tricky - we need to do it without going to CPU
        # Use scatter: one_hot[i, targets[i]] = 1.0
        targets_np = vbt_to_numpy(targets)
        one_hot_np = np.zeros((batch_size, num_classes), dtype=np.float32)
        one_hot_np[np.arange(batch_size), targets_np] = 1.0
        one_hot = numpy_to_vbt(one_hot_np, device)
        
        # grad_logits = (probs - one_hot) / batch_size * grad_out
        # Compute all in numpy since C.vt ops require tensor-tensor operations
        probs_np = vbt_to_numpy(probs)
        diff_np = probs_np - one_hot_np
        # Scale by 1/batch_size and upstream gradient
        grad_out_val = float(vbt_to_numpy(grad_out).item())
        diff_np = diff_np * (grad_out_val / float(batch_size))
        grad_logits = numpy_to_vbt(diff_np, device)
        
        # Return None for targets (integer, no gradient)
        return (grad_logits, None)
    
    # Backward: bias_add
    # dL/dz = dL/dout (same shape as z)
    # dL/db = sum(dL/dout, dim=0) (sum over batch dimension)
    def _bw_bias_add(grads_in, saved):
        (grad_out,) = grads_in
        (z, b) = saved
        # grad_z is same as grad_out
        grad_z = grad_out
        # grad_b = sum over batch dimension (CUDA-native)
        grad_b = grad_out.sum(dim=0, keepdim=False)
        return (grad_z, grad_b)
    
    vag.register("kfad::matmul", _bw_matmul)
    vag.register("kfad::relu", _bw_relu)
    vag.register("kfad::cross_entropy_loss", _bw_cross_entropy)
    vag.register("kfad::bias_add", _bw_bias_add)
    
    return lib


# =============================================================================
# 2-Layer MLP with Autograd
# =============================================================================

class MLPAutograd:
    """2-layer MLP using VibeTensor autograd."""
    
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
        
        # Enable gradients on parameters
        self.W1.requires_grad = True
        self.b1.requires_grad = True
        self.W2.requires_grad = True
        self.b2.requires_grad = True
    
    def forward(self, x):
        """Forward pass using autograd-enabled ops."""
        # Layer 1
        z1 = vt.ops.kfad.matmul(x, self.W1)
        z1 = vt.ops.kfad.bias_add(z1, self.b1)
        h1 = vt.ops.kfad.relu(z1)
        
        # Layer 2
        z2 = vt.ops.kfad.matmul(h1, self.W2)
        z2 = vt.ops.kfad.bias_add(z2, self.b2)
        
        return z2
    
    def parameters(self):
        """Return list of parameters."""
        return [self.W1, self.b1, self.W2, self.b2]


# =============================================================================
# SGD Optimizer (updates model attributes directly)
# =============================================================================

class SGD:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
        self.param_names = ['W1', 'b1', 'W2', 'b2']
    
    def step(self):
        for name in self.param_names:
            param = getattr(self.model, name)
            grad = param.grad
            if grad is not None:
                # param -= lr * grad (via numpy)
                param_np = vbt_to_numpy(param)
                grad_np = vbt_to_numpy(grad)
                param_np = param_np - self.lr * grad_np
                new_param = numpy_to_vbt(param_np, self.model.device)
                new_param.requires_grad = True
                setattr(self.model, name, new_param)
    
    def zero_grad(self):
        for name in self.param_names:
            param = getattr(self.model, name)
            if param.grad is not None:
                param.grad = None


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, optimizer, images, labels, batch_size=128):
    num_samples = images.shape[0]
    num_batches = num_samples // batch_size
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    indices = np.random.permutation(num_samples)
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_idx = indices[start:end]
        
        batch_images = images[batch_idx]
        batch_labels = labels[batch_idx]
        
        x = numpy_to_vbt(batch_images, model.device)
        y = numpy_to_vbt(batch_labels.astype(np.int64), model.device)
        
        # Forward
        logits = model.forward(x)
        
        # Compute loss with autograd
        loss = vt.ops.kfad.cross_entropy_loss(logits, y)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward - automatic gradient computation!
        # Provide explicit gradient seed (scalar 1.0 on CUDA)
        grad_seed = vt.tensor([1.0], dtype="float32").cuda()
        loss.backward(grad_seed)
        
        # Optimizer step
        optimizer.step()
        
        # Track metrics
        loss_val = float(vbt_to_numpy(loss).item())
        total_loss += loss_val
        
        # Accuracy
        preds = kf_argmax(logits, dim=1)
        preds_np = vbt_to_numpy(preds)
        correct += (preds_np == batch_labels).sum()
        total += batch_size
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{num_batches}, Loss: {loss_val:.4f}, Acc: {correct/total:.4f}")
    
    return total_loss / num_batches, correct / total


def evaluate(model, images, labels, batch_size=256):
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
        
        with vag.no_grad():
            logits = model.forward(x)
        
        preds = kf_argmax(logits, dim=1)
        preds_np = vbt_to_numpy(preds)
        correct += (preds_np == batch_labels).sum()
        total += len(batch_labels)
    
    return correct / total


def main():
    print("=" * 60)
    print("MNIST Training with VibeTensor Autograd (NO PyTorch)")
    print("=" * 60)
    
    # Check CUDA
    if not vibetensor._C._has_cuda or vibetensor._C._cuda_device_count() == 0:
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    device = "cuda:0"
    print(f"Using device: {device}")
    
    # Enable CUDA autograd
    vag.set_cuda_autograd_enabled(True)
    print(f"CUDA autograd enabled: {vag.is_cuda_autograd_enabled()}")
    
    # Register autograd ops
    print("\nRegistering autograd ops...")
    lib = register_autograd_ops()
    
    # Load data
    data_dir = Path(__file__).parent.parent.parent / "tmp" / "data" / "MNIST" / "raw"
    ensure_mnist_raw(data_dir)
    print(f"\nLoading MNIST from {data_dir}...")
    train_images, train_labels, test_images, test_labels = load_mnist(data_dir)
    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")
    
    # Create model
    print("\nCreating model with autograd...")
    model = MLPAutograd(input_dim=784, hidden_dim=256, output_dim=10, device=device)
    
    # Create optimizer
    optimizer = SGD(model, lr=0.1)
    
    # Training config
    num_epochs = 50
    batch_size = 128
    
    # Warmup
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
        loss = vt.ops.kfad.cross_entropy_loss(logits, y)
        optimizer.zero_grad()
        grad_seed = vt.tensor([1.0], dtype="float32").cuda()
        loss.backward(grad_seed)
        optimizer.step()
    warmup_time = time.time() - warmup_start
    print(f"  Warmup complete in {warmup_time:.2f}s ({warmup_batches} batches)")
    
    # Reset model
    model = MLPAutograd(input_dim=784, hidden_dim=256, output_dim=10, device=device)
    optimizer = SGD(model, lr=0.1)
    
    # Training loop
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
