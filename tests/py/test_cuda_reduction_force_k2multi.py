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


def _has_reduction_k2multi_test_hooks() -> bool:
    return (
        hasattr(C, "_cuda_reduction_last_stats_for_tests")
        and hasattr(C, "_cuda_reduction_reset_last_stats_for_tests")
        and hasattr(C, "_cuda_reduction_set_kernel_policy_for_tests")
        and hasattr(C, "_cuda_reduction_clear_kernel_policy_override_for_tests")
        and hasattr(C, "_cuda_reduction_set_grid_x_cap_for_tests")
        and hasattr(C, "_cuda_reduction_set_env_config_for_tests")
        and hasattr(C, "_cuda_reduction_clear_env_config_override_for_tests")
        and hasattr(C, "_cuda_reduction_set_k2multi_ctas_per_output_for_tests")
        and hasattr(C, "_cuda_reduction_clear_k2multi_ctas_per_output_override_for_tests")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_IF_ELIGIBLE")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_NONE")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_OVERFLOW")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE")
    )


def _has_k2multi_fault_injection_test_hooks() -> bool:
    return (
        hasattr(C, "_cuda_reduction_set_k2multi_fault_mode_for_tests")
        and hasattr(C, "_cuda_reduction_clear_k2multi_fault_mode_override_for_tests")
        and hasattr(C, "_CUDA_REDUCTION_K2MULTI_FAULT_MODE_SIGNAL_BUT_SKIP_PARTIAL_WRITE")
    )


if not _has_reduction_k2multi_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


def _clear_overrides() -> None:
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)
    C._cuda_reduction_clear_env_config_override_for_tests()
    C._cuda_reduction_clear_k2multi_ctas_per_output_override_for_tests()
    if hasattr(C, "_cuda_reduction_clear_k2multi_fault_mode_override_for_tests"):
        C._cuda_reduction_clear_k2multi_fault_mode_override_for_tests()


def _to_numpy_cpu(t) -> np.ndarray:
    cap = vbt.to_dlpack(t)
    return np.from_dlpack(cap)


@pytest.mark.cuda
def test_force_k2multi_strict_sum_f32_runs_k2multi_and_publishes_workspace_stats():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        # Keep this test hermetic even if the process env clamps grid.x.
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        # Use a non-default stream so launch_stream_id is nonzero.
        s = vbt.cuda.Stream()
        with s:
            x = vbt.arange(4 * 513, dtype=vbt.float32).reshape((4, 513)).cuda()
            C._cuda_reduction_reset_last_stats_for_tests()
            out = x.sum(dim=1)
        s.synchronize()

        stats = C._cuda_reduction_last_stats_for_tests()

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(4 * 513, dtype=np.float32).reshape(4, 513).sum(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)


        assert stats["selected_kernel"] == "K2Multi"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True
        assert bool(stats["want_plan"]) is True

        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)
        assert int(stats["out_numel"]) == 4
        assert int(stats["slice_len"]) == 513

        grid = tuple(stats["grid"])
        block = tuple(stats["block"])
        assert int(stats["slice_len"]) > int(block[0])
        assert grid[0] >= 1
        assert grid[1:] == (2, 1)
        assert block[0] >= 1
        assert block[1:] == (1, 1)

        assert int(stats["k2multi_ctas_per_output"]) == 2
        assert int(stats["k2multi_workspace_partials_bytes"]) == 32
        assert int(stats["k2multi_workspace_sema_off"]) == 256
        assert int(stats["k2multi_workspace_total_bytes"]) == 512

        assert int(stats["launch_stream_id"]) != 0
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_sum_i64_matches_numpy():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        x = vbt.arange(4 * 513, dtype=vbt.int64).reshape((4, 513)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(4 * 513, dtype=np.int64).reshape(4, 513).sum(axis=1)
        assert arr.shape == expected.shape
        assert (arr == expected).all()
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_prod_f32_matches_numpy():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        src = np.ones((4, 513), dtype=np.float32)
        src[:, 300] = np.float32(2.0)
        src[:, 400] = np.float32(3.0)
        x = vbt.from_numpy(src).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        arr = _to_numpy_cpu(out.cpu())
        expected = src.prod(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_prod_i64_matches_numpy():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        src = np.ones((4, 513), dtype=np.int64)
        src[:, 300] = 2
        src[:, 400] = 3
        x = vbt.from_numpy(src).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        arr = _to_numpy_cpu(out.cpu())
        expected = src.prod(axis=1)
        assert arr.shape == expected.shape
        assert (arr == expected).all()
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_mean_f32_matches_numpy():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        x = vbt.arange(4 * 513, dtype=vbt.float32).reshape((4, 513)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.mean(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        arr = _to_numpy_cpu(out.cpu())
        expected = (
            np.arange(4 * 513, dtype=np.float32).reshape(4, 513).sum(axis=1) / np.float32(513)
        )
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_fault_injection_signal_but_skip_partial_write_sum_f32():
    if not _has_k2multi_fault_injection_test_hooks():
        pytest.skip("K2-multi fault injection hooks not available")

    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    fault_mode = int(C._CUDA_REDUCTION_K2MULTI_FAULT_MODE_SIGNAL_BUT_SKIP_PARTIAL_WRITE)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_k2multi_fault_mode_for_tests(fault_mode)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        x = vbt.ones((8, 4096), dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        arr = _to_numpy_cpu(out.cpu())
        assert arr.shape == (8,)
        assert np.isnan(arr[0])
        assert (arr[1:] == np.float32(4096)).all()
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_fault_injection_signal_but_skip_partial_write_prod_f32():
    if not _has_k2multi_fault_injection_test_hooks():
        pytest.skip("K2-multi fault injection hooks not available")

    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    fault_mode = int(C._CUDA_REDUCTION_K2MULTI_FAULT_MODE_SIGNAL_BUT_SKIP_PARTIAL_WRITE)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_k2multi_fault_mode_for_tests(fault_mode)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        src = np.ones((8, 4096), dtype=np.float32)
        src[:, 300] = np.float32(2.0)
        src[:, 400] = np.float32(3.0)
        x = vbt.from_numpy(src).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        arr = _to_numpy_cpu(out.cpu())
        assert arr.shape == (8,)
        assert arr[0] == np.float32(0.0)
        assert (arr[1:] == np.float32(6.0)).all()
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_fault_injection_signal_but_skip_partial_write_mean_f32():
    if not _has_k2multi_fault_injection_test_hooks():
        pytest.skip("K2-multi fault injection hooks not available")

    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    fault_mode = int(C._CUDA_REDUCTION_K2MULTI_FAULT_MODE_SIGNAL_BUT_SKIP_PARTIAL_WRITE)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_k2multi_fault_mode_for_tests(fault_mode)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        x = vbt.ones((8, 4096), dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.mean(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        arr = _to_numpy_cpu(out.cpu())
        assert arr.shape == (8,)
        assert np.isnan(arr[0])
        assert (arr[1:] == np.float32(1.0)).all()
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_grid_x_cap_one_nonuniform_catches_missing_out_idx_loop():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)
        C._cuda_reduction_set_grid_x_cap_for_tests(1)

        x = vbt.arange(128, dtype=vbt.float32).reshape((128, 1)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        grid = tuple(stats["grid"])
        assert grid[0] == 1
        assert grid[1:] == (2, 1)

        arr = _to_numpy_cpu(out.cpu())
        assert arr.shape == (128,)
        assert (arr == np.arange(128, dtype=np.float32)).all()
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_grid_x_cap_one_prod_f32_catches_missing_out_idx_loop():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)
        C._cuda_reduction_set_grid_x_cap_for_tests(1)

        src = (np.arange(128, dtype=np.float32) + np.float32(2.0)).reshape((128, 1))
        x = vbt.from_numpy(src).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        grid = tuple(stats["grid"])
        assert grid[0] == 1
        assert grid[1:] == (2, 1)

        arr = _to_numpy_cpu(out.cpu())
        expected = src.prod(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_grid_x_cap_one_prod_i64_catches_missing_out_idx_loop():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)
        C._cuda_reduction_set_grid_x_cap_for_tests(1)

        src = (np.arange(128, dtype=np.int64) + 2).reshape((128, 1))
        x = vbt.from_numpy(src).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2Multi"

        grid = tuple(stats["grid"])
        assert grid[0] == 1
        assert grid[1:] == (2, 1)

        arr = _to_numpy_cpu(out.cpu())
        expected = src.prod(axis=1)
        assert arr.shape == expected.shape
        assert (arr == expected).all()
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_ineligible_plan_reason_publishes_stats_and_message():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        t = vbt.ones((2, 3, 4), dtype=vbt.float32).cuda()

        s = vbt.cuda.Stream()
        with s:
            C._cuda_reduction_reset_last_stats_for_tests()
            with pytest.raises(
                ValueError,
                match=r"^vt::sum: forced kernel K2Multi ineligible$",
            ):
                _ = t.sum(dim=(0, 2))

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "None"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True
        assert bool(stats["want_plan"]) is True

        assert int(stats["ineligible_reason"]) == int(
            C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE
        )
        assert int(stats["out_numel"]) == 3
        assert int(stats["slice_len"]) == 8
        assert tuple(stats["grid"]) == (0, 0, 0)
        assert tuple(stats["block"]) == (0, 0, 0)

        # Best-effort observability: strict failures still record the launch stream.
        assert int(stats["launch_stream_id"]) != 0
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_strict_negative_red_stride_is_ineligible_overflow():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_STRICT)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        base = vbt.arange(8, dtype=vbt.float32).cuda().reshape((2, 4))
        # Reverse each row: start at the last element of row 0.
        t = base.as_strided((2, 4), (4, -1), 3)

        s = vbt.cuda.Stream()
        with s:
            C._cuda_reduction_reset_last_stats_for_tests()
            with pytest.raises(
                ValueError,
                match=r"^vt::sum: forced kernel K2Multi ineligible$",
            ):
                _ = t.sum(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "None"
        assert bool(stats["want_plan"]) is True
        # Negative reduced stride is K2-only; K2-multi must reject it.
        assert int(stats["plan_red_linear_stride"]) == -1
        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_OVERFLOW)
        assert int(stats["out_numel"]) == 2
        assert int(stats["slice_len"]) == 4

        assert int(stats["k2multi_ctas_per_output"]) == 2
        assert int(stats["k2multi_workspace_total_bytes"]) == 0
        assert int(stats["launch_stream_id"]) != 0
    finally:
        _clear_overrides()


@pytest.mark.cuda
def test_force_k2multi_if_eligible_falls_back_to_k2():
    _clear_overrides()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_MULTI_IF_ELIGIBLE)
    try:
        C._cuda_reduction_set_env_config_for_tests(False, 0)
        C._cuda_reduction_set_k2multi_ctas_per_output_for_tests(2)
        C._cuda_reduction_set_kernel_policy_for_tests(policy)

        x = vbt.arange(4 * 8, dtype=vbt.float32).reshape((4, 8)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(4 * 8, dtype=np.float32).reshape(4, 8).sum(axis=1)
        assert np.allclose(arr, expected)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True
        assert bool(stats["want_plan"]) is True
    finally:
        _clear_overrides()
