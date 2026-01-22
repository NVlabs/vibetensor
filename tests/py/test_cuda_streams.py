# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C

import vibetensor.torch.cuda as vc


def test_streams_priority_and_context():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    least, greatest = vc.priority_range()
    assert isinstance(least, int) and isinstance(greatest, int)

    s = vc.Stream(priority=0)
    assert isinstance(s.query(), bool)
    s.synchronize()

    s2 = vc.Stream(priority=0)
    s2.wait_stream(s)
    ev = s.record_event()
    assert isinstance(ev, vc.Event)
    s2.wait_event(ev)

    r = repr(s)
    assert "<vibetensor.cuda.Stream" in r and "device=cuda:" in r and "cuda_stream=0x" in r
    assert s.__cuda_stream__()[0] == 0


def test_nested_stream_context_managers():
    """Test that nested stream context managers restore the previous stream correctly.

    This verifies the fix for stream context manager behavior where exiting a nested
    context should restore the outer stream, not the default stream.
    """
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    s1 = vc.Stream(priority=0)
    s2 = vc.Stream(priority=0)

    base_cls = getattr(C, "_CudaStreamBase", None)
    if base_cls is None:
        pytest.skip("_CudaStreamBase not available")

    initial_stream = base_cls.current()

    with s1:
        after_s1_enter = base_cls.current()
        assert after_s1_enter is not None

        with s2:
            after_s2_enter = base_cls.current()
            assert after_s2_enter is not None

        after_s2_exit = base_cls.current()
        assert after_s2_exit.cuda_stream == after_s1_enter.cuda_stream, (
            "Exiting inner stream context should restore outer stream"
        )

    after_s1_exit = base_cls.current()
    assert after_s1_exit.cuda_stream == initial_stream.cuda_stream, (
        "Exiting outer stream context should restore initial stream"
    )
