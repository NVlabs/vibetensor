# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import vibetensor.torch as vt
from vibetensor import _C as C

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

import vibetensor.triton as vt_triton


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
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
def test_triton_add_fp16():
    # Skip if CUDA not present or no devices
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() <= 0:
        pytest.skip("No CUDA device available")

    # 1. Define Triton kernel for FP16 addition
    @triton.jit
    def add_fp16_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load inputs (pointer arithmetic handles types if ptr is cast, 
        # but here we rely on the signature to interpret pointers)
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        
        tl.store(output_ptr + offsets, output, mask=mask)

    # 2. Define grid function
    def grid_fn(state, inputs, meta):
        n_elements = inputs[0].sizes[0]
        BLOCK_SIZE = meta.get("BLOCK_SIZE", 128)
        return ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # 3. Register the new operation
    op_name = "vt::add_fp16"
    signature = "*fp16,*fp16,*fp16,i32"
    
    # Ensure the op schema is defined
    try:
        getattr(C, "def")(f"{op_name}(Tensor, Tensor) -> Tensor")
    except Exception:
        pass # Might be already defined if run multiple times
        
    vt_triton.register(
        op_name,
        add_fp16_kernel,
        signature=signature,
        grid_fn=grid_fn,
        meta={"BLOCK_SIZE": 128},
        num_warps=4
    )

    # 4. Create FP16 tensors
    N = 1024
    # Using _make_cuda_dlpack_1d_dtype to create uninitialized tensors, then we can copy to them? 
    # Or actually C._make_cuda_tensor might not support fp16 directly based on tests.
    # But we can use C._cuda_h2d_alloc_copy which we just patched.
    
    a_np = np.full((N,), 1.5, dtype=np.float16)
    b_np = np.full((N,), 2.5, dtype=np.float16)
    
    # Use the lower-level h2d copy which supports 'float16' string now
    # Use device 0 explicitly
    a = C._cuda_h2d_alloc_copy(a_np, "float16", 0, False)
    b = C._cuda_h2d_alloc_copy(b_np, "float16", 0, False)
    
    assert a.dtype == "float16"
    assert b.dtype == "float16"

    # 5. Invoke the custom op
    # C._call_op invokes the dispatcher which should pick up our registered triton kernel
    try:
        out = C._call_op(op_name, a, b)
    except triton.runtime.errors.PTXASError as e:  # type: ignore[attr-defined]
        _skip_if_triton_arch_unsupported(e, "Triton FP16 add integration test")
        raise

    assert out.dtype == "float16"
    
    # 6. Verify result
    out_np = vt.cuda.from_device(out)
    
    expected = a_np + b_np
    np.testing.assert_allclose(out_np, expected, atol=1e-3)
    print("Triton FP16 add successful!")

if __name__ == "__main__":
    # Manual run helper
    try:
        test_triton_add_fp16()
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
