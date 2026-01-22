# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor._C as C
import vibetensor.torch as vbt


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and vbt.cuda.is_available() and C._cuda_device_count() > 0


if not _cuda_only():
    pytest.skip("CUDA not available", allow_module_level=True)


def _has_reduction_test_hooks() -> bool:
    return (
        hasattr(C, "_cuda_reduction_last_stats_for_tests")
        and hasattr(C, "_cuda_reduction_reset_last_stats_for_tests")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_OUT_NUMEL")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_SLICE")
    )


if not _has_reduction_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


def _to_numpy_cpu(t) -> np.ndarray:
    cap = vbt.to_dlpack(t)
    return np.from_dlpack(cap)


@pytest.mark.cuda
def test_sum_empty_out_numel_writes_empty_out_numel_reason():
    t = vbt.empty((0, 5), dtype=vbt.float32).cuda()

    C._cuda_reduction_reset_last_stats_for_tests()
    out = t.sum(dim=1)

    stats = C._cuda_reduction_last_stats_for_tests()
    assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_OUT_NUMEL)

    out_cpu = out.cpu()
    arr = _to_numpy_cpu(out_cpu)
    assert arr.size == 0


@pytest.mark.cuda
def test_sum_empty_slice_writes_empty_slice_reason_and_returns_zeros():
    t = vbt.empty((5, 0), dtype=vbt.float32).cuda()

    C._cuda_reduction_reset_last_stats_for_tests()
    out = t.sum(dim=1)

    stats = C._cuda_reduction_last_stats_for_tests()
    assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_SLICE)

    out_cpu = out.cpu()
    arr = _to_numpy_cpu(out_cpu)
    assert arr.shape == (5,)
    assert (arr == 0).all()
