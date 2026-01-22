# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# VibeTensor adapter for RMSNorm

try:
    from vibetensor.library import register_triton_op

    from .triton_impl import _RMSNormFn

    # Adapter Function
    def vibetensor_rmsnorm_fwd(x, gamma, eps):
        # Note: eps is already unwrapped by register_triton_op if unwrap_scalars=True
        # Call the autograd function
        return _RMSNormFn.apply(x, gamma, eps)

    # Register
    register_triton_op(
        schema="kf::rmsnorm_fwd(Tensor, Tensor, float) -> Tensor",
        impl=vibetensor_rmsnorm_fwd,
        unwrap_scalars=True,
    )

except ImportError:
    pass
