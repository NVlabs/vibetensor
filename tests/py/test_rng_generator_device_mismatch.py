# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from vibetensor import _C
import vibetensor.torch as vt


def test_generator_device_mismatch_cpu_to_cuda():
    g = vt.Generator("cpu")
    with pytest.raises(ValueError, match=r"generator device mismatch: expected cuda:0, got cpu"):
        vt.randn([2], device="cuda:0", generator=g)


def test_generator_device_mismatch_cuda_to_cpu():
    if not getattr(_C, "_has_cuda", False) or int(getattr(_C, "_cuda_device_count", lambda: 0)()) <= 0:  # type: ignore[attr-defined]
        return
    g = vt.Generator("cuda:0")
    with pytest.raises(ValueError, match=r"generator device mismatch: expected cpu, got cuda:0"):
        vt.randn([2], device="cpu", generator=g)
