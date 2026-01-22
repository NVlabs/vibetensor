# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


@pytest.mark.cuda
def test_cuda_record_stream_increments_counter():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    assert hasattr(C, "_cuda_record_stream"), "_cuda_record_stream missing in CUDA build"
    assert hasattr(
        C, "_cuda_debug_record_stream_call_count"
    ), "debug record_stream counters missing in CUDA build"
    assert hasattr(
        C, "_cuda_debug_reset_record_stream_call_count"
    ), "debug record_stream reset missing in CUDA build"

    C._cuda_debug_reset_record_stream_call_count()

    t = C._make_cuda_tensor([16], "float32", 1.0)

    with pytest.raises(TypeError):
        C._cuda_record_stream(t, True)

    with pytest.raises(ValueError):
        C._cuda_record_stream(t, -1)

    C._cuda_record_stream(t, 0)
    C._cuda_record_stream(t, 0)

    assert C._cuda_debug_record_stream_call_count() == 2
