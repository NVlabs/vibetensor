# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest

import vibetensor._C as C

import numpy as np

from vibetensor.cute.runtime import CuteKernel, CuteKernelArtifact, CuteParamSpec, clear_cache


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


def test_cute_expected_param_sizes():
    ptr = struct.calcsize("P")

    art = CuteKernelArtifact(
        cubin=b"x",
        kernel="k",
        params=(
            CuteParamSpec("tensor_ptr"),
            CuteParamSpec("scalar", dtype="i32"),
            CuteParamSpec("scalar", dtype="f64"),
            CuteParamSpec("memref", rank=2),
            CuteParamSpec("bytes", size=16),
            CuteParamSpec("device_ptr"),
        ),
    )

    assert art.expected_param_sizes == (
        ptr,
        4,
        8,
        24 + 16 * 2,
        16,
        ptr,
    )
@pytest.mark.cuda
def test_cute_kernel_launch_and_caches_modules(monkeypatch):
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    # Reset any stale cache entries.
    clear_cache()

    # Kernel signature: (out_ptr, val)
    ptx = b"""
.version 7.0
.target sm_30
.address_size 64

.visible .entry write_u64(
    .param .u64 out_ptr,
    .param .u64 val
)
{
    .reg .u64 %out;
    .reg .u64 %v;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %v, [val];

    st.global.u64 [%out], %v;
    ret;
}
"""

    art = CuteKernelArtifact(
        cubin=ptx,
        kernel="write_u64",
        params=(
            CuteParamSpec("tensor_ptr"),
            CuteParamSpec("scalar", dtype="i64"),
        ),
    )
    k = CuteKernel(art)

    # Create a CUDA context before cuModuleLoadData.
    out = C._cuda_empty([1], "int64", 0)

    # Wrap module load helpers to validate caching behavior.
    load_calls = {"n": 0}
    orig_load = C._cuda_module_load_ptx

    def _wrap_load(data):
        load_calls["n"] += 1
        return orig_load(data)

    monkeypatch.setattr(C, "_cuda_module_load_ptx", _wrap_load)

    get_calls = {"n": 0}
    orig_get = C._cuda_module_get_function

    def _wrap_get(mod, name):
        get_calls["n"] += 1
        return orig_get(mod, name)

    monkeypatch.setattr(C, "_cuda_module_get_function", _wrap_get)

    # Record-stream integration should run once per launch for the single tensor arg.
    C._cuda_debug_reset_record_stream_call_count()

    k.launch(out, 7, grid=(1, 1, 1), block=(1, 1, 1))
    k.launch(out, 9, grid=(1, 1, 1), block=(1, 1, 1))

    assert load_calls["n"] == 1
    assert get_calls["n"] == 1
    assert C._cuda_debug_record_stream_call_count() == 2

    arr = C._cuda_d2h_copy_numpy_sync(out)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1,)
    assert int(arr[0]) == 9

    clear_cache()


@pytest.mark.cuda
def test_cute_cache_key_includes_kernel_name(monkeypatch):
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    clear_cache()

    # Two entry points in the same PTX image.
    ptx = b"""
.version 7.0
.target sm_30
.address_size 64

.visible .entry write_a(
    .param .u64 out_ptr,
    .param .u64 val
)
{
    .reg .u64 %out;
    .reg .u64 %v;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %v, [val];
    st.global.u64 [%out], %v;
    ret;
}

.visible .entry write_b(
    .param .u64 out_ptr,
    .param .u64 val
)
{
    .reg .u64 %out;
    .reg .u64 %v;

    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %v, [val];
    add.u64 %v, %v, 1;
    st.global.u64 [%out], %v;
    ret;
}
"""

    art_a = CuteKernelArtifact(
        cubin=ptx,
        kernel="write_a",
        params=(
            CuteParamSpec("tensor_ptr"),
            CuteParamSpec("scalar", dtype="i64"),
        ),
    )
    art_b = CuteKernelArtifact(
        cubin=ptx,
        kernel="write_b",
        params=(
            CuteParamSpec("tensor_ptr"),
            CuteParamSpec("scalar", dtype="i64"),
        ),
    )

    ka = CuteKernel(art_a)
    kb = CuteKernel(art_b)

    out = C._cuda_empty([1], "int64", 0)

    load_calls = {"n": 0}
    orig_load = C._cuda_module_load_ptx

    def _wrap_load(data):
        load_calls["n"] += 1
        return orig_load(data)

    monkeypatch.setattr(C, "_cuda_module_load_ptx", _wrap_load)

    get_calls = {"n": 0}
    orig_get = C._cuda_module_get_function

    def _wrap_get(mod, name):
        get_calls["n"] += 1
        return orig_get(mod, name)

    monkeypatch.setattr(C, "_cuda_module_get_function", _wrap_get)

    ka.launch(out, 7, grid=(1, 1, 1), block=(1, 1, 1), record_stream=False)
    arr = C._cuda_d2h_copy_numpy_sync(out)
    assert int(arr[0]) == 7

    kb.launch(out, 7, grid=(1, 1, 1), block=(1, 1, 1), record_stream=False)
    arr = C._cuda_d2h_copy_numpy_sync(out)
    assert int(arr[0]) == 8

    assert load_calls["n"] == 1
    assert get_calls["n"] == 2

    clear_cache()


@pytest.mark.cuda
def test_cute_stream_handle_selection_prefers_current_stream(monkeypatch):
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    clear_cache()

    # Reuse the same minimal PTX kernel; we run with grid==0 so no CUDA work is enqueued.
    ptx = b"""
.version 7.0
.target sm_30
.address_size 64

.visible .entry noop(
    .param .u64 out_ptr,
    .param .u64 val
)
{
    ret;
}
"""

    art = CuteKernelArtifact(
        cubin=ptx,
        kernel="noop",
        params=(
            CuteParamSpec("tensor_ptr"),
            CuteParamSpec("scalar", dtype="i64"),
        ),
    )
    k = CuteKernel(art)

    out = C._cuda_empty([1], "int64", 0)

    from vibetensor.torch.cuda import Stream
    import vibetensor.torch as vt

    seen = {"h": None}
    orig_launch_checked = C._cuda_launch_checked

    def _wrap_launch_checked(*a, **kw):
        # Signature: (func, grid, block, shmem, stream, args, *, expected_param_sizes, strict)
        seen["h"] = int(a[4])
        return orig_launch_checked(*a, **kw)

    monkeypatch.setattr(C, "_cuda_launch_checked", _wrap_launch_checked)

    with Stream() as s:
        expected = vt._cuda_stream_handle_current()
        assert expected is not None and int(expected) != 0

        # stream=None should prefer vt._cuda_stream_handle_current() when non-default.
        k.launch(out, 1, grid=(0, 1, 1), block=(1, 1, 1), record_stream=False)
        assert seen["h"] == int(expected)

        # Explicit stream handle overrides current stream resolution.
        k.launch(out, 1, grid=(0, 1, 1), block=(1, 1, 1), stream=0, record_stream=False)
        assert seen["h"] == 0

    clear_cache()
