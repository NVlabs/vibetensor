# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C
import vibetensor.triton as vt_triton
import vibetensor.torch as vt


def _skip_if_triton_arch_unsupported(exc: Exception, what: str) -> None:
    msg = str(exc)
    if "gpu-name" in msg and "is not defined for option 'gpu-name'" in msg:
        pytest.skip(
            "Triton/ptxas does not support the current GPU architecture; "
            f"skipping {what}",
            allow_module_level=False,
        )
    if "PTX .version" in msg and "does not support .target" in msg:
        pytest.skip(
            "Triton/ptxas PTX version does not support the current GPU architecture; "
            f"skipping {what}",
            allow_module_level=False,
        )

@pytest.mark.cuda
def test_triton_add3_cuda_python_defined_op():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        pytest.skip("triton not available at runtime")

    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    # Define a Triton kernel that sums 3 inputs
    @triton.jit
    def add3_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        c = tl.load(c_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a + b + c, mask=mask)

    sig = "*fp32,*fp32,*fp32,*fp32,i32"  # 3 inputs + 1 output + n

    def grid_fn(state, inputs, meta):
        n = int(inputs[0].sizes[0])
        BS = int(meta["BLOCK_SIZE"]) if "BLOCK_SIZE" in meta else 128
        return ((n + BS - 1) // BS, 1, 1)

    # Define the schema for a new Python-only operator and register override
    schema = "vt::add3(Tensor, Tensor, Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]
    vt_triton.register("vt::add3", add3_kernel, signature=sig, grid_fn=grid_fn, meta={"BLOCK_SIZE": 128}, num_warps=4)

    # Build inputs
    N = 1024
    a = C._make_cuda_tensor([N], "float32", 1.0)  # type: ignore[attr-defined]
    b = C._make_cuda_tensor([N], "float32", 2.0)  # type: ignore[attr-defined]
    c = C._make_cuda_tensor([N], "float32", 3.0)  # type: ignore[attr-defined]

    # Call via generic dispatcher call binding
    try:
        out = C._call_op("vt::add3", a, b, c)  # type: ignore[attr-defined]
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton add3 integration test")
        raise

    # Validate result equals a+b+c
    arr = vt.cuda.from_device(out)  # type: ignore[attr-defined]
    import numpy as np
    expected = np.full((N,), 6.0, dtype=np.float32)
    assert arr.shape == (N,)
    assert np.allclose(arr, expected)
