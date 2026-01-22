# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest

import vibetensor._C as C


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


@pytest.mark.cuda
def test_cuda_launch_checked_grid0_returns_early_without_validating_expected_sizes_or_args_launch():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    assert hasattr(C, "_cuda_launch_checked"), "_cuda_launch_checked missing in CUDA build"

    h = C._cuda_stream_handle_current_for_device(0)

    # grid==0 returns before validating dims, args, or expected_param_sizes.
    C._cuda_launch_checked(
        0,
        (0, -1, 1),
        (0, 0, 0),
        -1,
        h,
        [object()],
        expected_param_sizes=object(),
        strict=True,
    )


@pytest.mark.cuda
def test_cuda_launch_checked_rejects_param_underflow_launch():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    assert hasattr(C, "_cuda_arg_f32"), "_cuda_arg_f32 missing in CUDA build"

    arg = C._cuda_arg_f32(1.0)
    with pytest.raises(ValueError, match=r"underflow"):
        C._cuda_launch_checked(
            0,
            (1, 1, 1),
            (1, 1, 1),
            0,
            0,
            [arg],
            expected_param_sizes=[8],
            strict=True,
        )


@pytest.mark.cuda
def test_cuda_launch_checked_errors_do_not_leak_pointer_values_launch():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    arg = C._cuda_arg_f32(1.0)
    with pytest.raises(ValueError) as excinfo:
        C._cuda_launch_checked(
            0,
            (1, 1, 1),
            (1, 1, 1),
            0,
            0,
            [arg],
            expected_param_sizes=[8],
            strict=True,
        )

    assert re.search(r"0x[0-9a-fA-F]+", str(excinfo.value)) is None


@pytest.mark.cuda
def test_cuda_launch_checked_enforces_strict_v2_sizes_launch():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    assert hasattr(C, "_cuda_arg_i64"), "_cuda_arg_i64 missing in CUDA build"

    arg = C._cuda_arg_i64(1)
    with pytest.raises(ValueError, match=r"size mismatch"):
        C._cuda_launch_checked(
            0,
            (1, 1, 1),
            (1, 1, 1),
            0,
            0,
            [arg],
            expected_param_sizes=[4],
            strict=True,
        )


@pytest.mark.cuda
def test_cuda_arg_memref_layout_and_view_pointer_semantics_launch():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    assert hasattr(C, "_cuda_arg_memref"), "_cuda_arg_memref missing in CUDA build"

    dev = 0
    h = C._cuda_stream_handle_current_for_device(dev)

    # Kernel signature: (out_ptr, view_ptr, memref_desc)
    # Writes: [eq, offset, size0, size1, stride0, stride1]
    ptx = b"""
.version 7.0
.target sm_30
.address_size 64

.visible .entry check_memref(
    .param .u64 out_ptr,
    .param .u64 view_ptr,
    .param .align 8 .b8 desc[56]
)
{
    .reg .u64 %out;
    .reg .u64 %view;
    .reg .u64 %aligned;
    .reg .u64 %offset;
    .reg .u64 %tmp;
    .reg .u64 %one;
    .reg .u64 %zero;
    .reg .u64 %eq;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %view, [view_ptr];

    // aligned pointer and offset from the memref descriptor
    ld.param.u64 %aligned, [desc+8];
    ld.param.u64 %offset, [desc+16];

    // computed = aligned + offset * 4 (float32)
    mul.lo.u64 %tmp, %offset, 4;
    add.u64 %tmp, %aligned, %tmp;

    setp.eq.u64 %p, %tmp, %view;
    mov.u64 %one, 1;
    mov.u64 %zero, 0;
    selp.u64 %eq, %one, %zero, %p;

    // out[0] = eq
    st.global.u64 [%out], %eq;

    // out[1] = offset
    st.global.u64 [%out+8], %offset;

    // out[2] = size0
    ld.param.u64 %tmp, [desc+24];
    st.global.u64 [%out+16], %tmp;

    // out[3] = size1
    ld.param.u64 %tmp, [desc+32];
    st.global.u64 [%out+24], %tmp;

    // out[4] = stride0
    ld.param.u64 %tmp, [desc+40];
    st.global.u64 [%out+32], %tmp;

    // out[5] = stride1
    ld.param.u64 %tmp, [desc+48];
    st.global.u64 [%out+40], %tmp;

    ret;
}
"""

    # Create a CUDA context by allocating before cuModuleLoadData.
    base = C._cuda_empty([3, 4], "float32", dev)
    t = base.as_strided([2, 3], [4, 1], 2)

    out = C._cuda_empty([6], "int64", dev)
    desc = C._cuda_arg_memref(t, rank=2)

    mod = C._cuda_module_load_ptx(ptx)
    fn = C._cuda_module_get_function(mod, "check_memref")

    try:
        C._cuda_launch_checked(
            fn,
            (1, 1, 1),
            (1, 1, 1),
            0,
            h,
            [out, t, desc],
            expected_param_sizes=[8, 8, 56],
            strict=True,
        )
    finally:
        C._cuda_module_unload(mod)

    arr = C._cuda_d2h_copy_numpy_sync(out)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6,)

    assert int(arr[0]) == 1
    assert int(arr[1]) == 2
    assert int(arr[2]) == 2
    assert int(arr[3]) == 3
    assert int(arr[4]) == 4
    assert int(arr[5]) == 1
