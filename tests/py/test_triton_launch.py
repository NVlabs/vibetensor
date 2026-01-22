# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:
    pytest.skip("triton not available at runtime", allow_module_level=True)

from vibetensor import _C as C
import vibetensor.triton as vt_triton
import vibetensor.torch as vt


@pytest.mark.cuda
def test_triton_launch_add_end_to_end():
    # Skip if CUDA not present in VibeTensor
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    # 1D elementwise add kernel
    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    # Register torch-free Triton override for vt::add
    sig = "*fp32,*fp32,*fp32,i32"  # last pointer is out

    def grid_fn(state, inputs, meta):
        n = int(inputs[0].sizes[0])
        BS = int(meta["BLOCK_SIZE"]) if "BLOCK_SIZE" in meta else 128
        return ((n + BS - 1) // BS, 1, 1)

    vt_triton.register(
        "vt::add",
        add_kernel,
        signature=sig,
        grid_fn=grid_fn,
        meta={"BLOCK_SIZE": 128},
        num_warps=4,
    )

    # Build inputs on CUDA
    N = 2048
    a = C._make_cuda_tensor([N], "float32", 1.5)  # type: ignore[attr-defined]
    b = C._make_cuda_tensor([N], "float32", 2.5)  # type: ignore[attr-defined]

    # Call via dispatcher wrapper; override should route to Triton
    out = C.vt.add(a, b)

    # Validate numerics by copying to host via torch-free CUDA overlay
    arr = vt.cuda.from_device(out)  # type: ignore[attr-defined]
    import numpy as np
    expected = np.full((N,), 4.0, dtype=np.float32)
    assert arr.shape == (N,)
    assert np.allclose(arr, expected)


@pytest.mark.cuda
def test_triton_launch_zero_size_fast_path():
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b, mask=mask)

    sig = "*fp32,*fp32,*fp32,i32"

    def grid_fn(state, inputs, meta):
        n = int(inputs[0].sizes[0])
        BS = int(meta["BLOCK_SIZE"]) if "BLOCK_SIZE" in meta else 128
        return ((n + BS - 1) // BS, 1, 1)

    vt_triton.register(
        "vt::add",
        add_kernel,
        signature=sig,
        grid_fn=grid_fn,
        meta={"BLOCK_SIZE": 128},
        num_warps=4,
    )

    # Zero-size inputs
    a = C._cuda_empty([0], "float32", 0)  # type: ignore[attr-defined]
    b = C._cuda_empty([0], "float32", 0)  # type: ignore[attr-defined]

    out = C.vt.add(a, b)
    assert tuple(out.sizes) == (0,)
