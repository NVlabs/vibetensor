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
        and hasattr(C, "_cuda_reduction_set_kernel_policy_for_tests")
        and hasattr(C, "_cuda_reduction_clear_kernel_policy_override_for_tests")
        and hasattr(C, "_cuda_reduction_set_env_config_for_tests")
        and hasattr(C, "_cuda_reduction_get_env_config_for_tests")
        and hasattr(C, "_cuda_reduction_clear_env_config_override_for_tests")
        and hasattr(C, "_cuda_reduction_env_config_override_is_active_for_tests")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_AUTO")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_NONE")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE")
    )


if not _has_reduction_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


def _set_env_staged_default(staged_default: bool) -> None:
    C._cuda_reduction_set_env_config_for_tests(bool(staged_default), 0)
    assert C._cuda_reduction_env_config_override_is_active_for_tests()

    cfg = C._cuda_reduction_get_env_config_for_tests()
    assert bool(cfg["override_active"]) is True
    assert bool(cfg["staged_default"]) is bool(staged_default)


def _to_numpy_cpu(t) -> np.ndarray:
    cap = vbt.to_dlpack(t)
    return np.from_dlpack(cap)


def _run_sum_0_2_and_get_reason() -> int:
    t = vbt.ones((2, 3, 4), dtype=vbt.float32).cuda()
    C._cuda_reduction_reset_last_stats_for_tests()
    out = t.sum(dim=(0, 2))

    stats = C._cuda_reduction_last_stats_for_tests()
    reason = int(stats["ineligible_reason"])

    out_cpu = out.cpu()
    arr = _to_numpy_cpu(out_cpu)
    assert arr.shape == (3,)
    # Each output element sums over 2*4 ones.
    assert (arr == 8).all()
    return reason


@pytest.mark.cuda
def test_nonempty_ineligible_reason_force_k2_if_eligible_enables_plan_even_when_env_off():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_clear_env_config_override_for_tests()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    _set_env_staged_default(False)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        reason = _run_sum_0_2_and_get_reason()
        assert reason == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE)
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_clear_env_config_override_for_tests()


@pytest.mark.cuda
def test_nonempty_ineligible_reason_force_k1_disables_plan_when_env_off():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_clear_env_config_override_for_tests()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1)
    _set_env_staged_default(False)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        reason = _run_sum_0_2_and_get_reason()
        assert reason == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_clear_env_config_override_for_tests()


@pytest.mark.cuda
def test_nonempty_ineligible_reason_force_k1_disables_plan_even_when_env_on():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_clear_env_config_override_for_tests()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1)
    _set_env_staged_default(True)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        reason = _run_sum_0_2_and_get_reason()
        assert reason == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_clear_env_config_override_for_tests()


@pytest.mark.cuda
def test_nonempty_ineligible_reason_no_policy_override_defers_to_env_default():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_clear_env_config_override_for_tests()

    try:
        _set_env_staged_default(False)
        reason = _run_sum_0_2_and_get_reason()
        assert reason == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)

        _set_env_staged_default(True)
        reason = _run_sum_0_2_and_get_reason()
        assert reason == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE)
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_clear_env_config_override_for_tests()


@pytest.mark.cuda
def test_nonempty_ineligible_reason_auto_defers_to_env_on_and_enables_plan():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_clear_env_config_override_for_tests()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_AUTO)
    _set_env_staged_default(True)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        reason = _run_sum_0_2_and_get_reason()
        assert reason == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE)
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_clear_env_config_override_for_tests()
