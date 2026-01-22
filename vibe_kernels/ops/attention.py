# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention Op - standalone with fwd/bwd using Triton kernels."""

import vibetensor.torch as vt
from .base import Op

# Import kernels
from vibe_kernels.attention import vbt_native as attn_kernel
from vibe_kernels.gemm import vbt_native as gemm_kernel
from vibe_kernels.tensor_ops import vbt_native as tensor_ops


def _attention_backward_manual(q, k, v, grad_out, scale, *, causal: bool = False):
    """Manual attention backward matching original train_vbt_transformer.py."""
    B, H, S, D = tuple(int(s) for s in q.sizes)
    q_bh = q.reshape([B*H, S, D])
    k_bh = k.reshape([B*H, S, D])
    v_bh = v.reshape([B*H, S, D])
    do_bh = grad_out.reshape([B*H, S, D])
    
    dq_list, dk_list, dv_list = [], [], []
    scale_t = vt.full([1], scale).cuda()
    neg_one = vt.full([1], -1.0).cuda()
    neg_inf = vt.full([1], -1.0e9).cuda()
    one = vt.full([1], 1.0).cuda()

    causal_mask = None
    if bool(causal):
        i = vt.arange(int(S), dtype="float32").cuda().reshape([int(S), 1])
        j = vt.arange(int(S), dtype="float32").cuda().reshape([1, int(S)])
        causal_mask = ((i + j * neg_one).sign() + one).clamp(0.0, 1.0)
    
    for bh in range(B * H):
        q_h = q_bh.select(0, bh)
        k_h = k_bh.select(0, bh)
        v_h = v_bh.select(0, bh)
        do_h = do_bh.select(0, bh)
        
        scores = gemm_kernel.matmul(q_h, k_h.permute([1, 0])) * scale_t
        if causal_mask is not None:
            scores = scores + (one + causal_mask * neg_one) * neg_inf
        scores_max = scores.amax(dim=1, keepdim=True)
        scores_shifted = scores + scores_max * neg_one
        exp_scores = scores_shifted.exp()
        sum_exp = exp_scores.sum(dim=1, keepdim=True)
        attn = exp_scores * sum_exp.reciprocal()
        
        dv_h = gemm_kernel.matmul(attn.permute([1, 0]), do_h)
        d_attn = gemm_kernel.matmul(do_h, v_h.permute([1, 0]))
        sum_da = (d_attn * attn).sum(dim=1, keepdim=True)
        d_scores = attn * (d_attn + sum_da * neg_one) * scale_t
        dq_h = gemm_kernel.matmul(d_scores, k_h)
        dk_h = gemm_kernel.matmul(d_scores.permute([1, 0]), q_h)
        
        dq_list.append(dq_h)
        dk_list.append(dk_h)
        dv_list.append(dv_h)
    
    dq = tensor_ops.stack(dq_list, dim=0).reshape([B, H, S, D])
    dk = tensor_ops.stack(dk_list, dim=0).reshape([B, H, S, D])
    dv = tensor_ops.stack(dv_list, dim=0).reshape([B, H, S, D])
    return dq, dk, dv


class Attention(Op):
    """Multi-head Self-Attention using Flash Attention Triton kernels.
    
    This op handles the attention computation only (Q*K^T*V).
    QKV projections should be done with separate Linear ops.
    
    Uses Flash Attention Triton kernels for both forward and backward.
    No trainable weights - just computation.
    """
    
    def __init__(self, n_heads: int, head_dim: int, causal: bool = False, use_triton_bwd: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.causal = causal
        self.use_triton_bwd = use_triton_bwd
    
    def fwd(self, q, k, v):
        """Forward: scaled dot-product attention using Flash Attention.
        
        Args:
            q, k, v: [B, H, S, D] query, key, value tensors
        Returns:
            out: [B, H, S, D] attention output
        """
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        self._cache['q'] = q
        self._cache['k'] = k
        self._cache['v'] = v
        # Use forward with LSE to get the logsumexp needed for backward
        out, lse = attn_kernel.attention_with_lse(q, k, v, causal=self.causal, sm_scale=self.scale)
        self._cache['out'] = out
        self._cache['lse'] = lse
        return out
    
    def bwd(self, grad_out):
        """Backward: compute gradients for q, k, v.
        
        Args:
            grad_out: [B, H, S, D] gradient from upstream
        Returns:
            (dq, dk, dv): gradients w.r.t. q, k, v
        """
        grad_out = grad_out.contiguous()
        q = self._cache['q']
        k = self._cache['k']
        v = self._cache['v']
        
        if self.use_triton_bwd:
            out = self._cache['out']
            lse = self._cache['lse']
            try:
                dq, dk, dv = attn_kernel.attention_backward(
                    grad_out, q, k, v, out, lse,
                    causal=self.causal, sm_scale=self.scale
                )
            except Exception:
                self.use_triton_bwd = False
                dq, dk, dv = _attention_backward_manual(q, k, v, grad_out, self.scale, causal=bool(self.causal))
        else:
            # Use manual backward (matches original train_vbt_transformer.py)
            dq, dk, dv = _attention_backward_manual(q, k, v, grad_out, self.scale, causal=bool(self.causal))
        
        return dq, dk, dv
