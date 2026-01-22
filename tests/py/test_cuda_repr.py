# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

import vibetensor.torch as vt
from vibetensor import _C as C


CUDA_AVAILABLE = bool(getattr(C, "_has_cuda", False)) and hasattr(C, "_cuda_device_count") and C._cuda_device_count() > 0
HAS_BF16 = bool(getattr(C, "_has_dlpack_bf16", False))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_repr_bool_nonempty_and_empty():
    # Non-empty bool CUDA tensor should suppress dtype in repr
    t = C._make_cuda_tensor([2], "bool", 1.0)
    s = repr(t)
    assert "device='cuda:" in s
    assert "dtype=bool" not in s

    # Empty bool CUDA tensor should include dtype in repr
    t_empty = C._make_cuda_tensor([0], "bool", 0.0)
    s2 = repr(t_empty)
    assert "device='cuda:" in s2
    assert ", dtype=bool" in s2


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_repr_unsupported_dtypes_raise():
    if HAS_BF16:
        cap_bf16 = C._make_cuda_dlpack_1d_dtype(4, "bfloat16")
        x_bf16 = vt.from_dlpack(cap_bf16)
        with pytest.raises(RuntimeError) as e2:
            _ = repr(x_bf16)
        assert "unsupported dtype for D2H copy" in str(e2.value)
