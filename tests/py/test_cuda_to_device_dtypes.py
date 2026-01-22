# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.int64, np.bool_, np.float16])
def test_cuda_to_device_allowlist_basic(dtype):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")
    # NumPy >=2.0 disallows arange with dtype=bool for length > 2; build input robustly
    if dtype is np.bool_:
        a = np.arange(4, dtype=np.int8).astype(np.bool_)
    else:
        a = np.arange(4, dtype=dtype)
    t = vt.cuda.to_device(a)
    assert t.device[0] == 2  # kDLCUDA
    assert t.dtype == np.dtype(dtype).name


def test_cuda_to_device_bfloat16_gate():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")
    # Try to obtain NumPy bfloat16 dtype; if unavailable, treat as a no-op pass
    try:
        bf16 = np.dtype("bfloat16")
    except TypeError:
        # Environment lacks NumPy bfloat16; nothing to assert beyond availability check
        return
    a = np.arange(3, dtype=bf16)
    if hasattr(C, "_has_bf16") and C._has_bf16():
        t = vt.cuda.to_device(a)
        assert t.dtype == "bfloat16"
    else:
        with pytest.raises(TypeError):
            _ = vt.cuda.to_device(a)
