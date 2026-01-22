# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_dtype_string_and_cpu_roundtrip_fp16():
    cap = C._make_cpu_dlpack_1d_dtype(8, "float16")
    t = vt.from_dlpack(cap)
    assert t.sizes == (8,)
    assert t.dtype == "float16"
    assert t.device == (1, 0)  # kDLCPU


@pytest.mark.skipif(not getattr(C, "_has_dlpack_bf16", False), reason="BF16 not supported by DLPack headers")
def test_dtype_string_and_cpu_roundtrip_bf16_or_skip():
    cap = C._make_cpu_dlpack_1d_dtype(4, "bfloat16")
    t = vt.from_dlpack(cap)
    assert t.sizes == (4,)
    assert t.dtype == "bfloat16"
    assert t.device == (1, 0)


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available for VibeTensor")
def test_cuda_copy_import_roundtrip_fp16_bf16():
    # Float16
    cap16 = C._make_cuda_dlpack_1d_dtype(16, "float16")
    t16 = vt.from_dlpack(cap16)
    assert t16.sizes == (16,)
    assert t16.dtype == "float16"
    assert t16.device[0] == 2  # kDLCUDA
    # One-shot semantics: second import should fail
    with pytest.raises(Exception):
        _ = vt.from_dlpack(cap16)

    # BFloat16 (if available)
    if not getattr(C, "_has_dlpack_bf16", False):
        return
    capb = C._make_cuda_dlpack_1d_dtype(8, "bfloat16")
    tb = vt.from_dlpack(capb)
    assert tb.sizes == (8,)
    assert tb.dtype == "bfloat16"
    assert tb.device[0] == 2
