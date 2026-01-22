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


def _has_k3_test_hooks() -> bool:
    return (
        hasattr(C, "_cuda_reduction_last_stats_for_tests")
        and hasattr(C, "_cuda_reduction_reset_last_stats_for_tests")
        and hasattr(C, "_cuda_reduction_set_kernel_policy_for_tests")
        and hasattr(C, "_cuda_reduction_clear_kernel_policy_override_for_tests")
        and hasattr(C, "_cuda_reduction_set_grid_x_cap_for_tests")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_IF_ELIGIBLE")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_STRICT")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_NONE")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE")
    )


if not _has_k3_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


def _clear_overrides() -> None:
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)


def _to_numpy_cpu(t) -> np.ndarray:
    cap = vbt.to_dlpack(t)
    return np.from_dlpack(cap)


@pytest.mark.cuda
def test_force_k3_sum_f32_grid_x_cap_one_tile_loop():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_IF_ELIGIBLE)
    try:
        C._cuda_reduction_set_kernel_policy_for_tests(policy)
        C._cuda_reduction_set_grid_x_cap_for_tests(1)

        x = vbt.arange(513 * 8, dtype=vbt.float32).reshape((513, 8)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K3"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True
        assert bool(stats["want_plan"]) is True
        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)
        assert int(stats["grid"][0]) == 1
        assert int(stats["out_numel"]) == 513
        assert int(stats["block"][0]) < int(stats["out_numel"])

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(513 * 8, dtype=np.float32).reshape(513, 8).sum(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k3_prod_i64_grid_x_cap_one_tile_loop():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_IF_ELIGIBLE)
    try:
        C._cuda_reduction_set_kernel_policy_for_tests(policy)
        C._cuda_reduction_set_grid_x_cap_for_tests(1)

        src = np.ones((513, 8), dtype=np.int64)
        src[:, 0] = np.arange(513, dtype=np.int64) + 2
        x = vbt.from_numpy(src).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K3"
        assert int(stats["requested_policy_id"]) == policy
        assert int(stats["grid"][0]) == 1
        assert int(stats["out_numel"]) == 513
        assert int(stats["block"][0]) < int(stats["out_numel"])

        arr = _to_numpy_cpu(out.cpu())
        expected = src.prod(axis=1)
        assert arr.shape == expected.shape
        assert np.array_equal(arr, expected)
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k3_mean_f32_grid_x_cap_one_tile_loop():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_IF_ELIGIBLE)
    try:
        C._cuda_reduction_set_kernel_policy_for_tests(policy)
        C._cuda_reduction_set_grid_x_cap_for_tests(1)

        x = vbt.arange(513 * 8, dtype=vbt.float32).reshape((513, 8)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.mean(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K3"
        assert int(stats["requested_policy_id"]) == policy
        assert int(stats["grid"][0]) == 1
        assert int(stats["out_numel"]) == 513
        assert int(stats["block"][0]) < int(stats["out_numel"])

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(513 * 8, dtype=np.float32).reshape(513, 8).mean(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k3_strict_ineligible_publishes_stats_and_message():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_STRICT)
    try:
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        t = vbt.ones((2, 3, 4), dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        with pytest.raises(ValueError, match=r"^vt::sum: forced kernel K3 ineligible$"):
            _ = t.sum(dim=(0, 2))

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "None"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True
        assert bool(stats["want_plan"]) is True
        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE)
        assert tuple(stats["grid"]) == (0, 0, 0)
        assert tuple(stats["block"]) == (0, 0, 0)
    finally:
        _clear_overrides()
