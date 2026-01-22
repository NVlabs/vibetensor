# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

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
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_STRICT")
    )


if not _has_reduction_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


def _clear_overrides() -> None:
    C._cuda_reduction_clear_kernel_policy_override_for_tests()


_POLICIES = [
    (int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1), "K1"),
    (int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT), "K2"),
    (int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_STRICT), "K3"),
]

# As of today, CUDA reduction forced-kernel policies K2/K3 are only implemented
# for a subset of reductions (e.g. sum). Min/max still uses K1 only; keep tests
# honest by validating the current behavior explicitly.
_MINMAX_POLICIES_SUPPORTED = [
    (int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1), "K1"),
]

_MINMAX_POLICIES_UNSUPPORTED = [
    (int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT), "K2"),
    (int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K3_STRICT), "K3"),
]


@pytest.mark.cuda
@pytest.mark.parametrize("policy, expected_kernel", _MINMAX_POLICIES_SUPPORTED)
def test_amin_inf_identity_matches_across_kernels(policy: int, expected_kernel: str):
    _clear_overrides()
    C._cuda_reduction_set_kernel_policy_for_tests(int(policy))
    try:
        x = vbt.tensor([math.inf, math.inf], dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.amin()

        assert float(out.cpu().item()) == math.inf

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == expected_kernel
    finally:
        _clear_overrides()


@pytest.mark.cuda
@pytest.mark.parametrize("policy, expected_kernel", _MINMAX_POLICIES_SUPPORTED)
def test_amax_neg_inf_identity_matches_across_kernels(policy: int, expected_kernel: str):
    _clear_overrides()
    C._cuda_reduction_set_kernel_policy_for_tests(int(policy))
    try:
        x = vbt.tensor([-math.inf, -math.inf], dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.amax()

        assert float(out.cpu().item()) == -math.inf

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == expected_kernel
    finally:
        _clear_overrides()


@pytest.mark.cuda
@pytest.mark.parametrize("policy, expected_kernel", _MINMAX_POLICIES_UNSUPPORTED)
def test_amin_forced_kernel_not_implemented_raises(policy: int, expected_kernel: str):
    _clear_overrides()
    C._cuda_reduction_set_kernel_policy_for_tests(int(policy))
    try:
        x = vbt.tensor([1.0, 2.0], dtype=vbt.float32).cuda()
        with pytest.raises(
            ValueError,
            match=rf"^vt::min: forced kernel {expected_kernel} not implemented$",
        ):
            _ = x.amin()
    finally:
        _clear_overrides()


@pytest.mark.cuda
@pytest.mark.parametrize("policy, expected_kernel", _MINMAX_POLICIES_UNSUPPORTED)
def test_amax_forced_kernel_not_implemented_raises(policy: int, expected_kernel: str):
    _clear_overrides()
    C._cuda_reduction_set_kernel_policy_for_tests(int(policy))
    try:
        x = vbt.tensor([1.0, 2.0], dtype=vbt.float32).cuda()
        with pytest.raises(
            ValueError,
            match=rf"^vt::max: forced kernel {expected_kernel} not implemented$",
        ):
            _ = x.amax()
    finally:
        _clear_overrides()


@pytest.mark.cuda
@pytest.mark.parametrize("policy, _expected_kernel", _POLICIES)
def test_amin_empty_slice_raises_even_under_forced_kernels(policy: int, _expected_kernel: str):
    _clear_overrides()
    C._cuda_reduction_set_kernel_policy_for_tests(int(policy))
    try:
        x = vbt.empty((2, 0), dtype=vbt.float32).cuda()
        with pytest.raises(RuntimeError, match=r"^amin: empty$"):
            _ = x.amin(dim=1)
    finally:
        _clear_overrides()
