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
    return hasattr(C, "_cuda_reduction_set_grid_x_cap_for_tests") and hasattr(
        C, "_cuda_reduction_last_stats_for_tests"
    )


if not _has_reduction_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


@pytest.mark.cuda
def test_grid_x_cap_for_tests_clamps_launch_grid_x_to_one():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    t = vbt.ones((1024,), dtype=vbt.float32).cuda()

    C._cuda_reduction_set_grid_x_cap_for_tests(1)
    try:
        C._cuda_reduction_reset_last_stats_for_tests()
        out = t.sum()
        stats = C._cuda_reduction_last_stats_for_tests()
        assert int(stats["grid"][0]) == 1
        out_cpu = out.cpu()
        assert float(out_cpu.item()) == pytest.approx(1024.0)
    finally:
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_grid_x_cap_for_tests_zero_raises_value_error():
    with pytest.raises(ValueError, match=r"grid_x_cap_for_tests"):
        C._cuda_reduction_set_grid_x_cap_for_tests(0)


@pytest.mark.cuda
def test_force_k2_strict_policy_runs_k2_for_sum_f32():
    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        t = vbt.ones((1024,), dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = t.sum()
        out_cpu = out.cpu()
        assert float(out_cpu.item()) == pytest.approx(1024.0)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True
        assert bool(stats["want_plan"]) is True
        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)
        assert int(stats["out_numel"]) == 1
        assert int(stats["slice_len"]) == 1024
        assert int(stats["grid"][0]) == 1
        assert int(stats["k2_smem_bytes"]) == int(stats["block"][0]) * 4
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
