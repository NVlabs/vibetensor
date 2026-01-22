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
def test_triton_add_cuda_basic():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        pytest.skip("triton not available at runtime")

    # Skip if CUDA not present in VibeTensor
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    import triton
    import triton.language as tl

    # Define a Triton kernel that does NOT match the base vt::add behavior
    # out = 2 * a (ignores b) so we can prove the override was taken.
    @triton.jit
    def add_like_but_scaled(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask)
        # b is intentionally ignored to differentiate from base add
        tl.store(out_ptr + offs, a + a, mask=mask)

    # Register torch-free Python Triton override for a new op vt::add_scaled
    sig = "*fp32,*fp32,*fp32,i32"

    def grid_fn(state, inputs, meta):
        n = int(inputs[0].sizes[0])
        BS = int(meta["BLOCK_SIZE"]) if "BLOCK_SIZE" in meta else 128
        return ((n + BS - 1) // BS, 1, 1)

    schema = "vt::add_scaled(Tensor, Tensor) -> Tensor"
    getattr(C, "def")(schema)  # type: ignore[attr-defined]
    vt_triton.register("vt::add_scaled", add_like_but_scaled, signature=sig, grid_fn=grid_fn, meta={"BLOCK_SIZE": 128}, num_warps=4)

    # Create CUDA tensors in VibeTensor; 1-D for simplicity
    N = 2048
    a = C._make_cuda_tensor([N], "float32", 1.5)  # type: ignore[attr-defined]
    b = C._make_cuda_tensor([N], "float32", 2.5)  # type: ignore[attr-defined]

    # Call via generic dispatcher binding on our new op
    try:
        out = C._call_op("vt::add_scaled", a, b)  # type: ignore[attr-defined]
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton add integration test")
        raise

    # Validate that override executed: expected = 2 * a (3.0)
    arr = vt.cuda.from_device(out)  # type: ignore[attr-defined]
    import numpy as np
    expected = np.full((N,), 3.0, dtype=np.float32)
    assert arr.shape == (N,)
    assert np.allclose(arr, expected)
