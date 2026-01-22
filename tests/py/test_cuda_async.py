# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C

import vibetensor.torch.cuda as vc


def test_no_implicit_sync_smoke():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    # Create a non-default stream and enqueue simple event ops; ensure calls return and are queryable
    s1 = vc.Stream(priority=0)
    s2 = vc.Stream(priority=0)

    # Use context manager to set current stream, then record/wait across streams
    with s1:
        ev = s1.record_event()
        # s2 waits on event enqueued on s1; this should not block the host
        s2.wait_event(ev)
        # Basic sanity on repr and protocol
        r = repr(s1)
        assert "<vibetensor.cuda.Stream" in r and "cuda_stream=0x" in r
        assert s1.__cuda_stream__()[0] == 0

    # Queries are non-throwing and return a bool
    assert isinstance(ev.query(), bool)
