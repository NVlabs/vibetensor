# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoPE Op - standalone with fwd/bwd."""

import vibetensor.torch as vt
from .base import Op

# Import kernels
from vibe_kernels.rotary import vbt_native as rope_kernel
from vibe_kernels.tensor_ops import vbt_native as tensor_ops


class RoPE(Op):
    """Rotary Position Embedding.
    
    weights:
        cos: [max_seq, head_dim//2] cosine table
        sin: [max_seq, head_dim//2] sine table
    """
    
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # Build RoPE tables on GPU
        hd2 = head_dim // 2
        
        # inv_freq = 1 / (10000 ^ (arange(0, hd2) / hd2))
        freq_indices = tensor_ops.arange(0, hd2)
        freq_exponents = freq_indices * vt.full([1], 1.0 / hd2).cuda()
        base = vt.full([1], 10000.0).cuda()
        inv_freq = (base.log() * freq_exponents).exp().reciprocal().reshape([1, hd2])
        
        # positions: [0, 1, 2, ..., max_seq_len-1]
        pos = tensor_ops.arange(0, max_seq_len).reshape([max_seq_len, 1])
        angles = pos * inv_freq
        
        self.weights['cos'] = angles.cos()
        self.weights['sin'] = angles.sin()
    
    def fwd(self, q, k):
        """Forward: apply rotary embeddings.
        
        Args:
            q, k: [B, H, S, D] query and key tensors
        Returns:
            q_rot, k_rot: [B, H, S, D] rotated tensors
        """
        self._cache['q'] = q
        self._cache['k'] = k
        return rope_kernel.apply_rotary_embedding(
            q, k, self.weights['cos'], self.weights['sin']
        )
    
    def bwd(self, dq_rot, dk_rot):
        """Backward: inverse rotation.
        
        Args:
            dq_rot, dk_rot: [B, H, S, D] gradients
        Returns:
            dq, dk: [B, H, S, D] gradients w.r.t. pre-rotation q, k
        """
        return rope_kernel.apply_rotary_embedding_backward(
            dq_rot, dk_rot, self.weights['cos'], self.weights['sin']
        )
