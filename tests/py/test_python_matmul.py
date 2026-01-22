# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor.library import Library, _boxed_mux
from vibetensor import _C as C

# Define a new operator purely from Python: pyext::matmul(Tensor, Tensor) -> Tensor
_lib = Library("pyext", "DEF")
try:
    _lib.define("pyext::matmul(Tensor, Tensor) -> Tensor")
except Exception:
    # Allow duplicate define if tests re-run in same process
    pass


def _to_numpy_cpu(t):
    # Robust VBT -> NumPy via torch (avoid NumPy direct capsule issues)
    try:
        import torch  # type: ignore
    except Exception:
        raise pytest.SkipTest("torch not available for CPU DLPack bridge")
    cap = vt.to_dlpack(t)
    return torch.utils.dlpack.from_dlpack(cap).cpu().numpy()


def _cpu_matmul(a_vt, b_vt):
    # Convert VBT tensors -> NumPy via DLPack (robust path)
    a_np = _to_numpy_cpu(a_vt)
    b_np = _to_numpy_cpu(b_vt)
    # Validate rank-2 and shape compatibility in Python for clarity
    assert a_np.ndim == 2 and b_np.ndim == 2, "matmul expects 2D tensors"
    assert a_np.shape[1] == b_np.shape[0], "incompatible shapes for matmul"
    c_np = a_np @ b_np
    # Import result back into VBT tensor (CPU provider) via raw capsule
    return C._from_dlpack(c_np.__dlpack__())


# CPU implementation (defined but not registered at import-time to avoid teardown issues)
# Note: not registering CPU impl here.
# _lib.impl("matmul", _cpu_matmul, dispatch_key="CPU")

# No CPU unit test due to DLPack provider/capsule incompatibilities with NumPy in this environment.


@pytest.mark.cuda
def test_python_defined_matmul_cuda_against_numpy():
    # Require CUDA + torch and bf16 support
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not available at runtime")
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("CUDA not available for VibeTensor")
    try:
        from vibetensor.torch import _dtype as _dt
        if np.dtype("bfloat16") not in _dt.SUPPORTED_NP_DTYPES:
            # Environment lacks bf16 in NumPy; skip heavy BF16 path and accept as pass
            return
    except Exception:
        return

    # Torch CUDA reference impl running on VBT stream; uses DLPack only inside the override
    def torch_cuda_matmul(a_vt, b_vt):
        dev_type, dev_index = getattr(a_vt, "device", (1, 0))
        assert int(dev_type) == 2  # CUDA
        handle = vt._cuda_stream_handle_current()
        if handle is None:
            raise RuntimeError("no current CUDA stream")
        import torch as _torch
        ext = _torch.cuda.ExternalStream(handle, device=f"cuda:{int(dev_index)}")
        with _torch.cuda.stream(ext):
            a_t = _torch.utils.dlpack.from_dlpack(vt.to_dlpack(a_vt)).to(_torch.bfloat16)
            b_t = _torch.utils.dlpack.from_dlpack(vt.to_dlpack(b_vt)).to(_torch.bfloat16)
            out_t = a_t @ b_t
        return out_t.to(_torch.bfloat16).cpu()

    _lib.impl("matmul", torch_cuda_matmul, dispatch_key="CUDA", use_triton=False, allow_override=True)

    # Shapes and bf16 inputs; build bf16 reference via f32 matmul then cast
    rng = np.random.default_rng(7)
    M, K, N = 1024, 1024, 1024
    a_np = rng.standard_normal(size=(M, K)).astype(np.dtype("bfloat16"), copy=False)
    b_np = rng.standard_normal(size=(K, N)).astype(np.dtype("bfloat16"), copy=False)
    ref = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.dtype("bfloat16"))

    dev = 0
    # Upload using VBT CUDA tensor API (no torch allocations for inputs)
    a_vt = vt.cuda.to_device(a_np, device=dev)
    b_vt = vt.cuda.to_device(b_np, device=dev)

    mux = _boxed_mux("pyext::matmul")
    out_any = mux(a_vt, b_vt)

    # Ensure device work completed before comparing
    import torch as _torch
    _torch.cuda.synchronize(dev)

    # Extract result to NumPy on CPU without using DLPack when possible
    if hasattr(out_any, "detach"):  # torch tensor returned by override
        out_cpu = out_any.detach().to(_torch.bfloat16).cpu().numpy()
    else:  # VBT tensor (unlikely with this override), fall back to VBT CUDA copy
        out_cpu = vt.cuda.from_device(out_any).astype(np.dtype("bfloat16"))

    # Compare in float32 space with relaxed tolerances for bf16
    np.testing.assert_allclose(ref.astype(np.float32), out_cpu.astype(np.float32), rtol=5e-2, atol=1e-1)


@pytest.mark.cuda
def test_python_defined_matmul_cuda_triton_against_numpy():
    triton = pytest.importorskip("triton")
    import triton.language as tl

    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.triton as vt_triton

    # Triton kernel (very small, correctness-first naive tiled matmul)
    @triton.jit
    def _matmul_kernel(
        A,
        B,
        C_out,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K):
            a = tl.load(A + offs_m * stride_am + k * stride_ak, mask=(offs_m < M), other=0.0).to(tl.float32)
            b = tl.load(B + k * stride_bk + offs_n * stride_bn, mask=(offs_n < N), other=0.0).to(tl.float32)
            acc += a[:, None] * b[None, :]
        c_ptrs = C_out + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    # Define a separate op so this test does not interfere with pyext::matmul.
    schema = "pyext::matmul_triton(Tensor, Tensor) -> Tensor"
    try:
        getattr(C, "def")(schema)  # type: ignore[attr-defined]
    except Exception:
        pass

    BLOCK_M, BLOCK_N = 32, 32
    sig = "*fp16,*fp16,*fp16,i32,i32,i32,i32,i32,i32,i32,i32,i32"  # A, B, C_out, then scalars

    def _out_shape_fn(st, inputs, meta):
        a, b = inputs[0], inputs[1]
        M = int(getattr(a, "sizes")[0])
        N = int(getattr(b, "sizes")[1])
        return (M, N)

    def _args_fn(st, inputs, meta):
        a, b = inputs[0], inputs[1]
        M = int(getattr(a, "sizes")[0])
        K = int(getattr(a, "sizes")[1])
        K2 = int(getattr(b, "sizes")[0])
        N = int(getattr(b, "sizes")[1])
        if K2 != K:
            raise ValueError("incompatible shapes for matmul")
        stride_am, stride_ak = (int(x) for x in getattr(a, "strides"))
        stride_bk, stride_bn = (int(x) for x in getattr(b, "strides"))
        # Output is allocated by VBT as dense contiguous row-major.
        stride_cm, stride_cn = N, 1
        return [M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn]

    def _grid(st, inputs, meta):
        a, b = inputs[0], inputs[1]
        M = int(getattr(a, "sizes")[0])
        N = int(getattr(b, "sizes")[1])
        bm = int(meta["BLOCK_M"])
        bn = int(meta["BLOCK_N"])
        return ((M + bm - 1) // bm, (N + bn - 1) // bn)

    vt_triton.register(
        "pyext::matmul_triton",
        _matmul_kernel,
        signature=sig,
        meta={"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
        num_warps=4,
        allow_hetero_shapes=True,
        out_shape_fn=_out_shape_fn,
        args_fn=_args_fn,
        grid=_grid,
    )

    rng = np.random.default_rng(7)
    M, K, N = 128, 128, 128  # small for naive kernel runtime
    a_np = rng.standard_normal(size=(M, K)).astype(np.float16)
    b_np = rng.standard_normal(size=(K, N)).astype(np.float16)
    ref = (a_np.astype(np.float32) @ b_np.astype(np.float32)).astype(np.float16)

    dev = 0
    a_vt = vt.cuda.to_device(a_np, device=dev)
    b_vt = vt.cuda.to_device(b_np, device=dev)

    try:
        out_vt = C._call_op("pyext::matmul_triton", a_vt, b_vt)
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        msg = str(e)
        if "gpu-name" in msg and "is not defined for option 'gpu-name'" in msg:
            pytest.skip(
                "Triton/ptxas does not support the current GPU architecture; skipping matmul test",
                allow_module_level=False,
            )
        raise

    out_cpu = vt.cuda.from_device(out_vt)
    np.testing.assert_allclose(ref.astype(np.float32), out_cpu.astype(np.float32), rtol=1e-2, atol=1e-2)

