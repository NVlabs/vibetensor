# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# VibeTensor adapter for existing Triton kernel in triton_impl.py

try:
    from vibetensor.library import register_triton_op

    # Import the original implementation
    from .triton_impl import layernorm

    # Wrapper to match return value expectation (single Tensor)
    def kf_layernorm_wrapper(x, weight, bias, eps):
        y = layernorm(x, weight, bias, eps=eps)
        return y

    # Register using the high-level helper
    # Automatically handles:
    # 1. Schema definition
    # 2. Scalar unwrapping (eps)
    # 3. Zero-copy bridging (use_triton=True)
    register_triton_op(
        schema="kf::layernorm(Tensor, Tensor, Tensor, float) -> Tensor",
        impl=kf_layernorm_wrapper,
        unwrap_scalars=True,
    )

except ImportError:
    # Silent fallback if VibeTensor is not installed
    pass
