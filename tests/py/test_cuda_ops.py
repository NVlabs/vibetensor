# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C


def _cuda_only():
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


def test_cuda_ops_out_of_place_and_bool_policy():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    # f32 add/mul/relu
    a = C._make_cuda_tensor([8], "float32", 1.0)
    b = C._make_cuda_tensor([8], "float32", 2.0)
    out_add = C.vt.add(a, b)
    out_mul = C.vt.mul(a, b)
    out_relu = C.vt.relu(a)
    assert out_add.sizes == (8,) and out_mul.sizes == (8,) and out_relu.sizes == (8,)
    # bool relu is identity; add/mul on bool should be unsupported
    bb = C._make_cuda_tensor([8], "bool", 1.0)
    out_br = C.vt.relu(bb)
    assert out_br.sizes == (8,)
    with pytest.raises(Exception):
        _ = C.vt.add(bb, bb)
    with pytest.raises(Exception):
        _ = C.vt.mul(bb, bb)


def test_cuda_ops_in_place_version_bump():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    a = C._make_cuda_tensor([4], "float32", 1.0)
    b = C._make_cuda_tensor([4], "float32", 1.0)
    v0 = a.version()
    a.add_(b)
    assert a.version() == v0 + 1
    v1 = a.version()
    a.relu_()
    assert a.version() == v1 + 1


def test_cuda_ops_broadcast_cap_and_shape_errors():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    # 26D (25D cap + 1) should throw
    a = C._make_cuda_tensor([1] * 26, "float32", 1.0)
    b = C._make_cuda_tensor([1], "float32", 2.0)
    with pytest.raises(Exception) as ei:
        _ = C.vt.add(a, b)
    assert "25D" in str(ei.value)
