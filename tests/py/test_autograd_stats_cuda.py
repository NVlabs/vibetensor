# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

import vibetensor._C as C

pytestmark = pytest.mark.cuda

def test_autograd_stats_cuda_bump():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:  # type: ignore[attr-defined]
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    ag = C.autograd
    # Best-effort: reset and then run a CUDA vt op to see wrapper bump
    ag.reset_stats()
    # Construct CUDA tensors via test helper (positional args)
    a = C._make_cuda_tensor([1, 2], "float32", 1.0)
    b = C._make_cuda_tensor([1, 2], "float32", 2.0)
    C._call_op("vt::add", a, b)
    s = ag.stats()
    assert s["wrapper_invocations"] >= 1
