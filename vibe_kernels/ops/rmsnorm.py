# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm Op - standalone with fwd/bwd."""

import vibetensor.torch as vt
from .base import Op

# Import kernel
from vibe_kernels.rmsnorm import vbt_native as rmsnorm_kernel


class RMSNorm(Op):
    """RMS Normalization.
    
    weights:
        weight: [dim] scale parameter
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Initialize weight to ones
        self.weights['weight'] = vt.ones([dim]).cuda()
    
    def fwd(self, x):
        """Forward: RMS normalize input.
        
        Args:
            x: [..., dim] input tensor
        Returns:
            out: [..., dim] normalized output
        """
        self._cache['x'] = x
        out, inv_rms = rmsnorm_kernel.rmsnorm(x, self.weights['weight'])
        self._cache['inv_rms'] = inv_rms
        return out
    
    def bwd(self, grad_out):
        """Backward: compute gradients.
        
        Args:
            grad_out: [..., dim] gradient from upstream
        Returns:
            grad_x: [..., dim] gradient w.r.t. input
        """
        x = self._cache['x']
        inv_rms = self._cache['inv_rms']
        
        grad_x, grad_weight = rmsnorm_kernel.rmsnorm_backward(
            grad_out, x, inv_rms, self.weights['weight']
        )
        self.grads['weight'] = grad_weight
        return grad_x
