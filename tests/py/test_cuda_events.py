# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C

import vibetensor.torch.cuda as vc


def test_events_behavior():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    e = vc.Event()
    assert e.query() is True

    s = vc.Stream()
    e.record(s)
    e.wait(s)
    e.synchronize()
    assert e.query() is True
