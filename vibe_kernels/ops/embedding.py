# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding Op - standalone with fwd/bwd."""

import vibetensor.torch as vt
from .base import Op

# Import kernel
from vibe_kernels.embedding import vbt_native as embed_kernel


class Embedding(Op):
    """Token embedding lookup.
    
    weights:
        weight: [vocab_size, dim] embedding table
    """
    
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Initialize weight (seed should be set externally before creating model)
        self.weights['weight'] = (vt.randn([vocab_size, dim]) * 0.02).cuda()
    
    def fwd(self, token_ids):
        """Forward: lookup embeddings.
        
        Args:
            token_ids: [B, S] int64 token indices
        Returns:
            out: [B, S, dim] embeddings
        """
        self._cache['token_ids'] = token_ids
        return embed_kernel.embedding_forward(self.weights['weight'], token_ids)
    
    def bwd(self, grad_out):
        """Backward: accumulate gradients to embedding table.
        
        Args:
            grad_out: [B*S, dim] gradient from upstream
        Returns:
            None (no input gradient for discrete tokens)
        """
        token_ids = self._cache['token_ids']
        self.grads['weight'] = embed_kernel.embedding_backward(
            grad_out, token_ids, self.vocab_size
        )
        return None  # No gradient for token indices
