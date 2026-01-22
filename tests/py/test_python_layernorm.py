# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

triton = pytest.importorskip("triton")
import triton.language as tl

import vibetensor.torch as vt
import vibetensor.triton as vt_triton
from vibetensor import _C as C


@triton.jit
def _layernorm_apply_kernel(
    X,
    W,
    B,
    Mean,
    Rstd,
    Y,
    M,
    N,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Per-row stats
    mean = tl.load(Mean + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + offs_m, mask=m_mask, other=1.0).to(tl.float32)

    # Per-column affine params
    w = tl.load(W + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs_n, mask=n_mask, other=0.0).to(tl.float32)

    x_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    y = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]

    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=m_mask[:, None] & n_mask[None, :])


def _args_fn(st, inputs, meta):
    x = inputs[0]
    M = int(getattr(x, "sizes")[0])
    N = int(getattr(x, "sizes")[1])
    sxm, sxn = (int(v) for v in getattr(x, "strides"))
    # Output is allocated by VBT as dense contiguous row-major.
    sym, syn = N, 1
    return [M, N, sxm, sxn, sym, syn]


def _grid(st, inputs, meta):
    x = inputs[0]
    M = int(getattr(x, "sizes")[0])
    N = int(getattr(x, "sizes")[1])
    bm = int(meta["BLOCK_M"])
    bn = int(meta["BLOCK_N"])
    return ((M + bm - 1) // bm, (N + bn - 1) // bn)


# Define and register op once at import time.
_schema = "pyext_ln::layernorm_apply(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor"
try:
    getattr(C, "def")(_schema)  # type: ignore[attr-defined]
except Exception:
    pass

vt_triton.register(
    "pyext_ln::layernorm_apply",
    _layernorm_apply_kernel,
    signature="*fp32,*fp32,*fp32,*fp32,*fp32,*fp32,i32,i32,i32,i32,i32,i32",
    meta={"BLOCK_M": 4, "BLOCK_N": 128},
    num_warps=4,
    allow_hetero_shapes=True,
    args_fn=_args_fn,
    grid=_grid,
)


@pytest.mark.cuda
def test_layernorm_op():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA available")

    M, N = 4, 1024
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal(size=(M, N)).astype(np.float32)
    w_np = rng.standard_normal(size=(N,)).astype(np.float32)
    b_np = rng.standard_normal(size=(N,)).astype(np.float32)
    eps = np.float32(1e-5)

    # CPU reference (numpy)
    mean = x_np.mean(axis=1).astype(np.float32)
    var = ((x_np - mean[:, None]) * (x_np - mean[:, None])).mean(axis=1).astype(np.float32)
    rstd = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    y_ref = (x_np - mean[:, None]) * rstd[:, None] * w_np + b_np

    dev = 0
    x_vt = vt.cuda.to_device(x_np, device=dev)
    w_vt = vt.cuda.to_device(w_np, device=dev)
    b_vt = vt.cuda.to_device(b_np, device=dev)
    mean_vt = vt.cuda.to_device(mean, device=dev)
    rstd_vt = vt.cuda.to_device(rstd, device=dev)

    try:
        y_vt = C._call_op("pyext_ln::layernorm_apply", x_vt, w_vt, b_vt, mean_vt, rstd_vt)
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        msg = str(e)
        if "gpu-name" in msg and "is not defined for option 'gpu-name'" in msg:
            pytest.skip(
                "Triton/ptxas does not support the current GPU architecture; skipping layernorm test",
                allow_module_level=False,
            )
        raise

    y_out = vt.cuda.from_device(y_vt)
    np.testing.assert_allclose(y_out, y_ref, atol=1e-3, rtol=1e-3)
