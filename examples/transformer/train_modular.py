# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-Free Transformer Training with Modular Ops.

Demonstrates transformer training WITHOUT PyTorch or NumPy:
- Uses vibetensor.torch (VibeTensor's PyTorch-compatible API, NOT PyTorch)
- Uses vibe_kernels.ops (Triton-based forward/backward implementations)
- All computation runs on GPU via Triton kernels

Usage:
    python train_modular.py
"""

import os
import sys
import time

import vibetensor.torch as vt

# Import modular ops
from vibe_kernels.ops import Transformer
from vibe_kernels.tensor_ops import vbt_native as tensor_ops


class Config:
    vocab_size = 256
    hidden_dim = 128
    n_heads = 4
    n_layers = 2
    max_seq_len = 64
    
    def __init__(self):
        assert self.hidden_dim % self.n_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
        head_dim = self.hidden_dim // self.n_heads
        assert head_dim % 2 == 0, \
            f"head_dim ({head_dim}) must be even for RoPE"


def train():
    print("=" * 70)
    print("PyTorch-Free Transformer Training (VibeTensor + Triton)")
    print("=" * 70)
    
    cfg = Config()
    print(f"\nConfig: vocab={cfg.vocab_size}, d={cfg.hidden_dim}, h={cfg.n_heads}, L={cfg.n_layers}")
    
    # Create model
    vt.manual_seed(42)
    model = Transformer(
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_seq_len=cfg.max_seq_len
    )
    
    batch_size, seq_len, n_steps, lr = 4, 32, 100, 0.1
    print(f"Training: batch={batch_size}, seq={seq_len}, steps={n_steps}, lr={lr}")
    print("Task: Learn (token + 1) % vocab")
    print("Using: VibeTensor + Triton (NO PyTorch, NO NumPy)\n")
    
    losses = []
    for step in range(n_steps):
        # Data gen using VBT randint - fully on GPU
        inputs = vt.randint(0, cfg.vocab_size, [batch_size, seq_len]).cuda()
        targets = tensor_ops.add_one_mod(inputs, cfg.vocab_size)
        
        start = time.time()
        
        # Forward
        logits = model.fwd(inputs)
        loss = model.compute_loss(logits, targets)
        
        # Backward
        model.bwd()
        
        # Update
        model.update(lr)
        
        step_time = (time.time() - start) * 1000
        
        losses.append(loss)
        if step % 10 == 0 or step == n_steps - 1:
            print(f"Step {step:3d} | Loss: {loss:.4f} | Time: {step_time:.0f}ms")
    
    print(f"\nInitial: {losses[0]:.4f} -> Final: {losses[-1]:.4f} (reduced {losses[0]-losses[-1]:.4f})")
    print("SUCCESS!" if losses[-1] < losses[0] else "WARNING: Loss not decreasing")


if __name__ == "__main__":
    train()
