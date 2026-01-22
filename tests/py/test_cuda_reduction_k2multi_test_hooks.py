# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C


def _has_cuda_build() -> bool:
    return bool(getattr(C, "_has_cuda", False))


if not _has_cuda_build():
    pytest.skip("CUDA not built", allow_module_level=True)


def _has_reduction_test_hooks() -> bool:
    return hasattr(C, "_cuda_reduction_last_stats_for_tests") and hasattr(
        C, "_cuda_reduction_reset_last_stats_for_tests"
    )


if not _has_reduction_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


@pytest.mark.cuda
def test_cuda_reduction_last_stats_schema_has_k2multi_fields():
    C._cuda_reduction_reset_last_stats_for_tests()
    stats = C._cuda_reduction_last_stats_for_tests()

    assert int(stats["k2multi_ctas_per_output"]) == 0
    assert int(stats["k2multi_workspace_partials_bytes"]) == 0
    assert int(stats["k2multi_workspace_sema_off"]) == 0
    assert int(stats["k2multi_workspace_total_bytes"]) == 0
    assert int(stats["launch_stream_id"]) == 0


@pytest.mark.cuda
def test_cuda_reduction_k2multi_ctas_per_output_override_roundtrip():
    C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(None)
    assert C._cuda_reduction_get_k2multi_ctas_per_output_for_tests() is None

    C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(7)
    assert int(C._cuda_reduction_get_k2multi_ctas_per_output_for_tests()) == 7

    C._cuda_reduction_clear_k2multi_ctas_per_output_override_for_tests()
    assert C._cuda_reduction_get_k2multi_ctas_per_output_for_tests() is None

    with pytest.raises(ValueError, match=r"k2multi_ctas_per_output_for_tests"):
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(0)


@pytest.mark.cuda
def test_cuda_reduction_k2multi_fault_mode_override_roundtrip():
    C._cuda_reduction_clear_k2multi_fault_mode_override_for_tests()
    assert bool(C._cuda_reduction_k2multi_fault_mode_override_is_active_for_tests()) is False

    assert int(C._cuda_reduction_get_k2multi_fault_mode_for_tests()) == int(
        C._CUDA_REDUCTION_K2MULTI_FAULT_MODE_NONE
    )

    C._cuda_reduction_set_k2multi_fault_mode_for_tests(
        int(C._CUDA_REDUCTION_K2MULTI_FAULT_MODE_SIGNAL_BUT_SKIP_PARTIAL_WRITE)
    )
    assert bool(C._cuda_reduction_k2multi_fault_mode_override_is_active_for_tests()) is True
    assert int(C._cuda_reduction_get_k2multi_fault_mode_for_tests()) == int(
        C._CUDA_REDUCTION_K2MULTI_FAULT_MODE_SIGNAL_BUT_SKIP_PARTIAL_WRITE
    )

    C._cuda_reduction_clear_k2multi_fault_mode_override_for_tests()
    assert bool(C._cuda_reduction_k2multi_fault_mode_override_is_active_for_tests()) is False

    with pytest.raises(ValueError, match=r"invalid k2multi fault mode"):
        C._cuda_reduction_set_k2multi_fault_mode_for_tests(2)
