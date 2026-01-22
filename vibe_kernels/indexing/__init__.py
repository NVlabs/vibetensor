# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexing operations: gather and scatter_add.

This module provides two implementations:

1. PyTorch-based (default):
   - `gather`, `scatter_add`, `scatter_add_`, `gather_with_grad`
   - Uses PyTorch tensors and autograd
   - Import from: `kernel_factory.indexing`

2. VibeTensor-native (NO PyTorch):
   - Pure VibeTensor implementation using Triton PTX compilation
   - Import from: `kernel_factory.indexing.vbt_native`
   
Example (VibeTensor-native):
    >>> import vibetensor.torch as vt
    >>> from vibe_kernels.indexing import vbt_native as idx_ops
    >>> 
    >>> src = vt.cuda.to_device(np.arange(32, dtype=np.float32).reshape(4, 8))
    >>> idx = vt.cuda.to_device(np.array([1, 3, 0], dtype=np.int64))
    >>> out = idx_ops.gather(src, 0, idx)
"""

from .kernel import gather, gather_with_grad, scatter_add, scatter_add_

# Note: vbt_native is available as a submodule for PyTorch-free usage
# Import explicitly: from vibe_kernels.indexing import vbt_native

__all__ = [
    "gather",
    "gather_with_grad",
    "scatter_add",
    "scatter_add_",
]
