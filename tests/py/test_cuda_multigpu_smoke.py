# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C

import vibetensor.torch.cuda as vc


def test_multigpu_event_wait_smoke():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    ndev = C._cuda_device_count()
    if ndev >= 2:
        s0 = vc.Stream(device=0)
        s1 = vc.Stream(device=1)
    else:
        # Fallback to same-device streams when only one GPU is available
        s0 = vc.Stream(device=0)
        s1 = vc.Stream(device=0)

    ev = s0.record_event()
    s1.wait_event(ev)
    s1.synchronize()

    if ndev < 2:
        assert ev.query() is True
