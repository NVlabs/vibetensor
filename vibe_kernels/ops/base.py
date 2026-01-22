# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base Op class for modular transformer ops."""

import vibetensor.torch as vt


class Op:
    """Base class for transformer ops.
    
    Each op has:
    - weights: dict of weight tensors (initialized in __init__)
    - grads: dict of gradient tensors (populated by bwd)
    - _cache: dict of cached tensors for backward pass
    """
    
    def __init__(self):
        self.weights = {}
        self.grads = {}
        self._cache = {}
    
    def fwd(self, *args, **kwargs):
        """Forward pass. Override in subclass."""
        raise NotImplementedError
    
    def bwd(self, grad_out):
        """Backward pass. Override in subclass.
        
        Should:
        1. Compute gradients for weights and store in self.grads
        2. Return gradient w.r.t. input
        """
        raise NotImplementedError
    
    def zero_grad(self):
        """Clear all gradients."""
        self.grads.clear()
    
    def clear_cache(self):
        """Clear cached tensors."""
        self._cache.clear()
    
    def parameters(self):
        """Return iterator of (name, weight) tuples."""
        return self.weights.items()
    
    def gradients(self):
        """Return iterator of (name, grad) tuples."""
        return self.grads.items()
    
    def update(self, lr):
        """SGD update: w = w - lr * grad (with clipping)."""
        neg_lr = vt.full([1], -lr).cuda()
        for name, w in self.weights.items():
            if name in self.grads:
                g_clipped = self.grads[name].clamp(-1.0, 1.0)
                self.weights[name] = w + g_clipped * neg_lr
