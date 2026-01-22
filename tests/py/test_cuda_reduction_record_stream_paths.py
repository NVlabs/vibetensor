# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
import vibetensor.torch as vbt


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


if not _cuda_only():
    pytest.skip("CUDA not available for VibeTensor", allow_module_level=True)

if not hasattr(C, "_cuda_debug_record_stream_call_count"):
    pytest.skip("debug record_stream counters not available", allow_module_level=True)


def _reset() -> None:
    C._cuda_debug_reset_record_stream_call_count()


def _count() -> int:
    return int(C._cuda_debug_record_stream_call_count())


@pytest.mark.cuda
def test_sum_records_stream_once_when_out_numel_zero():
    t = vbt.empty((0, 5), dtype=vbt.float32).cuda()
    _reset()
    _ = t.sum(dim=1)
    assert _count() == 1


@pytest.mark.cuda
def test_sum_records_stream_once_when_slice_len_zero():
    t = vbt.empty((5, 0), dtype=vbt.float32).cuda()
    _reset()
    _ = t.sum(dim=1)
    assert _count() == 1


@pytest.mark.cuda
def test_mean_records_stream_once_non_empty():
    t = vbt.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=vbt.float32).cuda()
    _reset()
    _ = t.mean()
    assert _count() == 1


@pytest.mark.cuda
def test_amin_records_stream_once_non_empty():
    t = vbt.tensor([3.0, 1.0, 2.0], dtype=vbt.float32).cuda()
    _reset()
    _ = t.amin()
    assert _count() == 1
