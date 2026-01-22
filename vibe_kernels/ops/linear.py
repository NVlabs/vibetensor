# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linear Op - standalone with fwd/bwd using optimized GEMM backward kernels."""

import vibetensor.torch as vt
from .base import Op

# Import kernel
from vibe_kernels.gemm import vbt_native as gemm_kernel


class Linear(Op):
    """Linear transformation using optimized Triton GEMM kernels.
    
    weights:
        weight: [in_dim, out_dim] weight matrix
    """
    
    def __init__(self, in_dim: int, out_dim: int, init_scale: float = 0.02):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize weight (seed should be set externally before creating model)
        self.weights['weight'] = (vt.randn([in_dim, out_dim]) * init_scale).cuda()
    
    def fwd(self, x):
        """Forward: y = x @ weight.
        
        Args:
            x: [*, in_dim] input tensor
        Returns:
            out: [*, out_dim] output tensor
        """
        self._cache['x'] = x
        return gemm_kernel.matmul(x, self.weights['weight'])
    
    def bwd(self, grad_out):
        """Backward: compute gradients using optimized GEMM backward kernels.
        
        Args:
            grad_out: [*, out_dim] gradient from upstream
        Returns:
            grad_x: [*, in_dim] gradient w.r.t. input
        """
        x = self._cache['x']
        w = self.weights['weight']
        
        # Use optimized backward kernels
        # grad_x = grad_out @ weight^T  (dgrad)
        # grad_weight = x^T @ grad_out  (wgrad)
        grad_x, grad_w = gemm_kernel.matmul_backward(
            grad_out, x, w,
            compute_grad_a=True,
            compute_grad_b=True,
        )
        
        self.grads['weight'] = grad_w
        
        return grad_x
