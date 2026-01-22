# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C

CUDA_AVAILABLE = bool(getattr(C, "_has_cuda", False)) and hasattr(C, "_cuda_device_count") and C._cuda_device_count() > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_from_device_async_lifetime():
    # Large enough to ensure a real transfer
    t = C._make_cuda_tensor([1 << 18], "float32", 1.0)
    arr, ev = vt.cuda.from_device_async(t)

    # Drop original reference and force GC before syncing
    tmp = arr
    del arr
    gc.collect()

    # Event should fence the pinned buffer lifetime
    ev.synchronize()
    s = vt.cuda.from_device(t)

    np.testing.assert_allclose(tmp, s)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_from_device_async_source_tensor_dropped_before_sync():
    t = C._make_cuda_tensor([1 << 18], "float32", 3.0)
    arr, ev = vt.cuda.from_device_async(t)

    del t
    gc.collect()

    ev.synchronize()
    np.testing.assert_allclose(arr, 3.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_from_device_async_zero_size():
    t0 = C._make_cuda_tensor([0], "float32", 0.0)
    arr, ev = vt.cuda.from_device_async(t0)
    assert arr.size == 0
    ev.synchronize()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cuda_from_device_async_gc_event_only():
    t = C._make_cuda_tensor([1 << 18], "float32", 2.0)
    arr, ev = vt.cuda.from_device_async(t)
    # Drop all ndarray references before waiting; lifetime is fenced in C++ deleter
    del arr
    gc.collect()
    ev.synchronize()  # should not crash
