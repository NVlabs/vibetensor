# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C


@pytest.mark.cuda
def test_cuda_device_cc_tuple_when_available():
    # Skip when CUDA not present or no devices
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")
    major, minor = C._cuda_device_cc(-1)  # type: ignore[attr-defined]
    assert isinstance(major, int) and isinstance(minor, int)
    assert major >= 1


@pytest.mark.cuda
def test_cuda_launch_ptx_writes_value():
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    # Minimal PTX kernel that stores a scalar into *out_ptr
    ptx = b"""
.version 7.0
.target sm_30
.address_size 64

.visible .entry write_const(
    .param .u64 out_ptr,
    .param .f32 val
)
{
    .reg .u64 %rd1;
    .reg .f32 %f1;
    ld.param.u64 %rd1, [out_ptr];
    ld.param.f32 %f1, [val];
    st.global.f32 [%rd1], %f1;
    ret;
}
"""

    dev = 0
    # Current stream handle for device
    h = C._cuda_stream_handle_current_for_device(dev)  # type: ignore[attr-defined]
    assert isinstance(h, int)

    # Allocate a 1-element float32 tensor on device
    t = C._cuda_empty([1], "float32", dev)  # type: ignore[attr-defined]

    # Launch with 1 block, 1 thread
    C._cuda_launch_ptx(ptx, "write_const", (1, 1, 1), (1, 1, 1), 0, h, [t, 7.25])  # type: ignore[attr-defined]

    # Copy back and verify
    import numpy as np
    arr = C._cuda_d2h_copy_numpy_sync(t)  # type: ignore[attr-defined]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert np.allclose(arr, np.array([7.25], dtype=np.float32))


@pytest.mark.cuda
def test_cuda_launch_grid0_returns_early_without_validating_args():
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    dev = 0
    h = C._cuda_stream_handle_current_for_device(dev)  # type: ignore[attr-defined]
    assert isinstance(h, int)

    # _cuda_launch: grid==0 returns before validating dims or args.
    C._cuda_launch(0, (0, -1, 1), (0, 0, 0), -1, h, [object()])  # type: ignore[attr-defined]

    # _cuda_launch_ptx: same semantics, but still loads/unloads the module.
    ptx = b"""
.version 7.0
.target sm_30
.address_size 64

.visible .entry write_const(
    .param .u64 out_ptr,
    .param .f32 val
)
{
    .reg .u64 %rd1;
    .reg .f32 %f1;
    ld.param.u64 %rd1, [out_ptr];
    ld.param.f32 %f1, [val];
    st.global.f32 [%rd1], %f1;
    ret;
}
"""

    C._cuda_launch_ptx(ptx, "write_const", (0, -1, 1), (0, 0, 0), -1, h, [object()])  # type: ignore[attr-defined]


@pytest.mark.cuda
def test_cuda_launch_packs_small_python_int_as_i64():
    if not getattr(C, "_has_cuda", False):
        pytest.skip("VibeTensor built without CUDA")
    if C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("No CUDA device available")

    # Kernel expects an 8-byte integer param; passing a small Python int must be safe.
    ptx = b"""
.version 7.0
.target sm_30
.address_size 64

.visible .entry write_u64(
    .param .u64 out_ptr,
    .param .u64 val
)
{
    .reg .u64 %rd1;
    .reg .u64 %rd2;
    ld.param.u64 %rd1, [out_ptr];
    ld.param.u64 %rd2, [val];
    st.global.u64 [%rd1], %rd2;
    ret;
}
"""

    dev = 0
    h = C._cuda_stream_handle_current_for_device(dev)  # type: ignore[attr-defined]
    assert isinstance(h, int)

    mod = C._cuda_module_load_ptx(ptx)  # type: ignore[attr-defined]
    fn = C._cuda_module_get_function(mod, "write_u64")  # type: ignore[attr-defined]

    t = C._cuda_empty([1], "int64", dev)  # type: ignore[attr-defined]
    try:
        C._cuda_launch(fn, (1, 1, 1), (1, 1, 1), 0, h, [t, 1])  # type: ignore[attr-defined]
    finally:
        C._cuda_module_unload(mod)  # type: ignore[attr-defined]

    import numpy as np

    arr = C._cuda_d2h_copy_numpy_sync(t)  # type: ignore[attr-defined]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert int(arr[0]) == 1
