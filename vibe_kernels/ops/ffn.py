# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FFN Op - standalone with fwd/bwd."""

import vibetensor.torch as vt
from .base import Op

# Import kernels
from vibe_kernels.gemm import vbt_native as gemm_kernel
from vibe_kernels.activation import vbt_native as act_kernel


class FFN(Op):
    """Feed-Forward Network with GELU activation.
    
    FFN(x) = GELU(x @ W1) @ W2
    
    weights:
        w1: [dim, hidden_dim] first projection
        w2: [hidden_dim, dim] second projection
    """
    
    def __init__(self, dim: int, hidden_dim: int, init_scale: float = 0.02):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights (seed should be set externally before creating model)
        self.weights['w1'] = (vt.randn([dim, hidden_dim]) * init_scale).cuda()
        self.weights['w2'] = (vt.randn([hidden_dim, dim]) * init_scale).cuda()
    
    def fwd(self, x):
        """Forward: FFN computation.
        
        Args:
            x: [*, dim] input tensor
        Returns:
            out: [*, dim] output tensor
        """
        self._cache['x'] = x
        
        # h1 = x @ w1
        h1_pre = gemm_kernel.matmul(x, self.weights['w1'])
        self._cache['h1_pre'] = h1_pre
        
        # h1_act = GELU(h1)
        h1_act = act_kernel.gelu(h1_pre)
        self._cache['h1_act'] = h1_act
        
        # out = h1_act @ w2
        out = gemm_kernel.matmul(h1_act, self.weights['w2'])
        return out
    
    def bwd(self, grad_out):
        """Backward: compute gradients.
        
        Args:
            grad_out: [*, dim] gradient from upstream
        Returns:
            grad_x: [*, dim] gradient w.r.t. input
        """
        x = self._cache['x']
        h1_pre = self._cache['h1_pre']
        h1_act = self._cache['h1_act']
        
        grad_h1_act, grad_w2 = gemm_kernel.matmul_backward(
            grad_out,
            h1_act,
            self.weights['w2'],
            compute_grad_a=True,
            compute_grad_b=True,
        )
        self.grads['w2'] = grad_w2
        
        grad_h1 = act_kernel.gelu_backward(grad_h1_act, h1_pre)
        
        grad_x, grad_w1 = gemm_kernel.matmul_backward(
            grad_h1,
            x,
            self.weights['w1'],
            compute_grad_a=True,
            compute_grad_b=True,
        )
        self.grads['w1'] = grad_w1
        
        return grad_x
