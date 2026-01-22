# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Torch-free Triton helpers for the vibetensor.torch overlay.

Historically this module provided a compatibility wrapper that executed Torch/Triton
callables on VibeTensor tensors via DLPack conversion and
``torch.cuda.ExternalStream``.

VibeTensor must not depend on ``torch``. The Torch-interop path has been removed in
favor of the native, torch-free Triton bridge in :pymod:`vibetensor.triton`.

Use :func:`vibetensor.triton.register` to compile and launch ``@triton.jit`` kernels
on VibeTensor CUDA tensors.
"""

from typing import Any, Callable

import vibetensor.triton as _vbt_triton

# Re-export the native Triton surface under the vibetensor.torch namespace.
register = _vbt_triton.register
cache_stats = _vbt_triton.cache_stats
reset_cache_stats = _vbt_triton.reset_cache_stats
clear_cache = _vbt_triton.clear_cache


def make_cuda_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Backward-compat shim for ``Library(..., use_triton=True)``.

    Older revisions wrapped Torch callables so they could accept VibeTensor tensors.
    That required importing ``torch`` at runtime, which is no longer supported.

    The wrapper is now a no-op. Callables registered with ``use_triton=True`` must
    accept and return VibeTensor tensors directly (typically by using
    :func:`vibetensor.triton.register`).
    """

    if not callable(fn):
        raise TypeError("fn must be callable")
    return fn
