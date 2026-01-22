# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike


@triton.jit
def _layernorm_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = x - mean
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Store mean/rstd if requested (pointers not null)
    # Note: Triton kernel args must be tensors/pointers.
    # In this simple impl we assume Mean/Rstd are valid pointers if provided.
    # But here we just assume they are output buffers.
    if Mean is not None:
        tl.store(Mean + row, mean)
    if Rstd is not None:
        tl.store(Rstd + row, rstd)

    # Normalize and write output
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        b = tl.load(B + cols, mask=mask).to(tl.float32) if B is not None else 0.0
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def _stride0(x: Any) -> int:
    stride = getattr(x, "stride", None)
    if callable(stride):
        return int(stride(0))
    strides = getattr(x, "strides", None)
    if isinstance(strides, (tuple, list)) and len(strides) > 0:
        return int(strides[0])
    raise TypeError("expected a tensor-like with stride(0) or .strides[0]")


def layernorm(
    x: TensorLike,
    weight: TensorLike,
    bias: TensorLike | None = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
):
    sizes = getattr(x, "shape", None)
    if sizes is None:
        sizes = getattr(x, "sizes", None)
    if not isinstance(sizes, (tuple, list)) or len(sizes) != 2:
        raise TypeError("expected x to be rank-2 tensor-like")
    M, N = int(sizes[0]), int(sizes[1])
    # heuristics for number of warps
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError("This LayerNorm implementation only supports N <= 65536")
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    y = alloc.empty_like(x)

    # Alloc mean/rstd if needed, or just dummy
    # The benchmark expects (out, rstd, mean)
    # If we don't need them for backprop in inference, we might not allocate?
    # But the benchmark signature requires them.
    # We'll allocate them.
    mean = alloc.empty((M,), like=x, dtype="float32")
    rstd = alloc.empty((M,), like=x, dtype="float32")

    # Enqueue kernel
    _layernorm_kernel[(M,)](
        x,
        y,
        weight,
        bias,
        mean,
        rstd,
        _stride0(x),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    if return_rstd and return_mean:
        return y, rstd, mean
    if return_rstd:
        return y, rstd
    if return_mean:
        return y, mean
    return y


# --- VibeTensor Adapter (Add to end of file) ---
try:
    from vibetensor.library import Library

    # 1. Define Op Schema (Input x, w, b, eps; Returns y)
    # Using a distinct namespace 'kf' for kernel factory ops
    lib = Library("kf", "DEF")
    try:
        lib.define("kf::layernorm(Tensor, Tensor, Tensor, float) -> Tensor")
    except Exception:
        pass

    # 2. Define Adapter Function
    # VibeTensor Dispatcher passes positional args; scalars (like eps) might be wrapped in Tensors if passed via _call_op
    def vibetensor_wrapper(x, weight, bias, eps):
        # Unwrap Scalar if eps is a Tensor (VibeTensor currently wraps scalars for dispatch)
        if hasattr(eps, "item"):
            eps = eps.item()

        # Call original layernorm function
        # Original signature: layernorm(x, weight, bias=None, eps=1e-6, ...)
        # We pass explicit args from VibeTensor call
        y = layernorm(x, weight, bias, eps=eps)

        # VibeTensor current dispatcher convention expects a single Tensor return for this schema
        return y

    # 3. Register Implementation (Key: use_triton=True)
    # use_triton=True enables automatic VBT <-> Torch conversion (Zero-Copy)
    lib.impl(
        "layernorm",
        vibetensor_wrapper,
        dispatch_key="CUDA",
        use_triton=True,
        allow_override=True,
    )

except ImportError:
    # Silent fallback if VibeTensor is not installed
    pass
