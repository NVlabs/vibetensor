# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pytest

from vibetensor import _C as C


def test_cuda_cpu_only_imports_behavior():
    # This test should pass on both CUDA and non-CUDA builds, but asserts behavior when CUDA is off.
    mod = importlib.import_module("vibetensor.torch.cuda")
    pr = mod.priority_range()
    assert isinstance(pr, tuple) and len(pr) == 2
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        assert pr == (0, 0)
        with pytest.raises(RuntimeError):
            _ = mod.Stream()
        with pytest.raises(RuntimeError):
            _ = mod.Event()
    else:
        # Basic constructors should work when CUDA is present
        s = mod.Stream()
        e = mod.Event()
        assert s is not None and e is not None
