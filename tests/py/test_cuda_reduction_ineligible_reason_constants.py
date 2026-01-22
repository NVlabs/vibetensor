# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
import vibetensor.torch as vbt


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and vbt.cuda.is_available() and C._cuda_device_count() > 0


if not _cuda_only():
    pytest.skip("CUDA not available", allow_module_level=True)


def _has_reduction_test_hooks() -> bool:
    return hasattr(C, "_cuda_reduction_last_stats_for_tests") and hasattr(
        C, "_CUDA_REDUCTION_INELIGIBLE_REASON_NONE"
    )


if not _has_reduction_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


@pytest.mark.cuda
def test_ineligible_reason_constants_match_contract():
    # ABI-locked numeric mapping (design/reduction/README.md §5.5)
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE) == 0
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_OUT_NUMEL) == 1
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_EMPTY_SLICE) == 2
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_OVERFLOW) == 3
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_STRIDE_ZERO) == 4
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE) == 5
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_MULTI_DIM_NEGATIVE_STRIDE) == 6
    assert int(C._CUDA_REDUCTION_INELIGIBLE_REASON_KEPT_NEGATIVE_STRIDE) == 7
