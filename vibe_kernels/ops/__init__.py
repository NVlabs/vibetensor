# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modular Ops for VibeTensor Transformer Training.

Each Op is a standalone class with:
- weights: trainable parameters
- fwd(): forward pass
- bwd(): backward pass (populates self.grads)
- update(lr): SGD weight update
"""

from .base import Op
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .linear import Linear
from .rope import RoPE
from .attention import Attention
from .ffn import FFN
from .loss import CrossEntropyLoss
from .transformer import TransformerBlock, Transformer

__all__ = [
    'Op',
    'Embedding',
    'RMSNorm', 
    'Linear',
    'RoPE',
    'Attention',
    'FFN',
    'CrossEntropyLoss',
    'TransformerBlock',
    'Transformer',
]
