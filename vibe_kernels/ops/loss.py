# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CrossEntropy Loss Op - standalone with fwd/bwd using optimized Triton kernels."""

import vibetensor.torch as vt
from .base import Op

# Import kernel
from vibe_kernels.loss import vbt_native as loss_kernel


class CrossEntropyLoss(Op):
    """Cross-entropy loss using optimized Triton kernels.
    
    No trainable weights.
    """
    
    def __init__(self, vocab_size: int, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
    
    def fwd(self, logits, targets):
        """Forward: compute cross-entropy loss using Triton kernel.
        
        Args:
            logits: [N, vocab_size] unnormalized logits
            targets: [N] int64 target indices
        Returns:
            loss: scalar loss value (float)
        """
        # Use forward with cache for backward
        loss, cache = loss_kernel.cross_entropy_with_cache(
            logits, targets, ignore_index=self.ignore_index
        )
        self._cache['ce_cache'] = cache
        return loss
    
    def bwd(self, grad_out=1.0):
        """Backward: gradient of loss w.r.t. logits using Triton kernel.
        
        Args:
            grad_out: upstream gradient (default 1.0 for loss)
        Returns:
            grad_logits: [N, vocab_size] gradient
        """
        cache = self._cache['ce_cache']
        grad_logits = loss_kernel.cross_entropy_backward(cache, grad_out)
        return grad_logits
