# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from typing import Dict, Optional

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
        and hasattr(C, "_cuda_reduction_set_grid_x_cap_for_tests")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE")
        and hasattr(C, "_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_NONE")
        and hasattr(C, "_CUDA_REDUCTION_INELIGIBLE_REASON_RED_NOT_LINEARIZABLE")
        and hasattr(C, "_cuda_reduction_set_env_config_for_tests")
        and hasattr(C, "_cuda_reduction_clear_env_config_override_for_tests")
    )


if not _has_reduction_test_hooks():
    pytest.skip("CUDA reduction test hooks not available", allow_module_level=True)


def _run_py(code: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env2,
            timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"subprocess timed out: stdout={e.stdout!r} stderr={e.stderr!r}"
        ) from e


def _to_numpy_cpu(t) -> np.ndarray:
    cap = vbt.to_dlpack(t)
    return np.from_dlpack(cap)


@pytest.mark.cuda
def test_cuda_mean_int64_dtype_error_publishes_stats_forcek1():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        t = vbt.ones((2, 3), dtype=vbt.int64).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        with pytest.raises(ValueError, match=r"^mean: expected dtype=float32$"):
            _ = t.mean()

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "None"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True
        assert bool(stats["want_plan"]) is False
        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()


@pytest.mark.cuda
def test_amax_force_k2_if_eligible_nonempty_does_not_build_plan():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        t = vbt.tensor([3.0, 1.0, 2.0], dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = t.amax()
        assert float(out.cpu().item()) == pytest.approx(3.0)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K1"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True

        assert bool(stats["want_plan"]) is False
        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)

        assert int(stats["plan_iter_ndim"]) == 0
        assert int(stats["plan_kept_ndim"]) == 0
        assert int(stats["plan_red_ndim"]) == 0
        assert int(stats["plan_red_linear_stride"]) == 0
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()


@pytest.mark.cuda
def test_scalar_sum_publishes_selected_kernel_none():
    # NOTE: CPU -> CUDA currently materializes 0-d tensors as a 1D (len=1) tensor.
    # Create an actual rank-0 CUDA view so the dispatcher takes the scalar-clone path.
    t = vbt.tensor([7], dtype=vbt.float32).cuda().view([])

    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_reset_last_stats_for_tests()

    out = t.sum()
    assert float(out.cpu().item()) == 7.0

    stats = C._cuda_reduction_last_stats_for_tests()

    assert stats["selected_kernel"] == "None"
    assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)
    assert int(stats["out_numel"]) == 1
    assert int(stats["slice_len"]) == 1
    assert tuple(stats["grid"]) == (0, 0, 0)
    assert tuple(stats["block"]) == (0, 0, 0)

    assert bool(stats["want_plan"]) is False
    assert int(stats["plan_iter_ndim"]) == 0
    assert int(stats["plan_kept_ndim"]) == 0
    assert int(stats["plan_red_ndim"]) == 0
    assert int(stats["plan_red_linear_stride"]) == 0
    assert int(stats["k2_smem_bytes"]) == 0


@pytest.mark.cuda
def test_cub_reduce_all_sum_publishes_selected_kernel_none_subprocess():
    code = (
        "import sys\n"
        "import vibetensor.torch as vt\n"
        "from vibetensor import _C as C\n"
        "\n"
        "def _has_cuda() -> bool:\n"
        "    try:\n"
        "        return bool(getattr(C, '_has_cuda', False)) and int(C._cuda_device_count()) > 0\n"
        "    except Exception:\n"
        "        return False\n"
        "\n"
        "if not _has_cuda():\n"
        "    raise SystemExit(1)\n"
        "\n"
        "if not hasattr(C, '_cuda_reduction_last_stats_for_tests'):\n"
        "    raise SystemExit(1)\n"
        "\n"
        "n = 1024\n"
        "x = vt.arange(n, dtype='float32').cuda()\n"
        "C._cuda_reduction_reset_last_stats_for_tests()\n"
        "out = x.sum()\n"
        "stats = C._cuda_reduction_last_stats_for_tests()\n"
        "expected = (n - 1) * n // 2\n"
        "ok = True\n"
        "ok = ok and abs(float(out.item()) - float(expected)) < 1e-3\n"
        "ok = ok and stats['selected_kernel'] == 'None'\n"
        "ok = ok and int(stats['out_numel']) == 1\n"
        "ok = ok and int(stats['slice_len']) == n\n"
        "ok = ok and tuple(stats['grid']) == (0, 0, 0)\n"
        "ok = ok and tuple(stats['block']) == (0, 0, 0)\n"
        "sys.exit(0 if ok else 1)\n"
    )

    env = {
        # Enable the internal reduce-all sum fast path.
        "VBT_INTERNAL_CUDA_CUB_REDUCE_ALL_SUM": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


@pytest.mark.cuda
def test_cub_disabled_when_policy_forcek1_subprocess():
    code = (
        "import sys\n"
        "import vibetensor.torch as vt\n"
        "from vibetensor import _C as C\n"
        "\n"
        "def _has_cuda() -> bool:\n"
        "    try:\n"
        "        return bool(getattr(C, '_has_cuda', False)) and int(C._cuda_device_count()) > 0\n"
        "    except Exception:\n"
        "        return False\n"
        "\n"
        "if not _has_cuda():\n"
        "    raise SystemExit(1)\n"
        "\n"
        "if not hasattr(C, '_cuda_reduction_last_stats_for_tests'):\n"
        "    raise SystemExit(1)\n"
        "if not hasattr(C, '_cuda_reduction_set_kernel_policy_for_tests'):\n"
        "    raise SystemExit(1)\n"
        "if not hasattr(C, '_cuda_reduction_clear_kernel_policy_override_for_tests'):\n"
        "    raise SystemExit(1)\n"
        "if not hasattr(C, '_CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1'):\n"
        "    raise SystemExit(1)\n"
        "\n"
        "policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K1)\n"
        "C._cuda_reduction_set_kernel_policy_for_tests(policy)\n"
        "try:\n"
        "    n = 1024\n"
        "    x = vt.arange(n, dtype='float32').cuda()\n"
        "    C._cuda_reduction_reset_last_stats_for_tests()\n"
        "    out = x.sum()\n"
        "    stats = C._cuda_reduction_last_stats_for_tests()\n"
        "    expected = (n - 1) * n // 2\n"
        "    ok = True\n"
        "    ok = ok and abs(float(out.item()) - float(expected)) < 1e-3\n"
        "    # CUB reduce-all fast path must be bypassed under ForceK1.\n"
        "    ok = ok and stats['selected_kernel'] == 'K1'\n"
        "    ok = ok and tuple(stats['grid']) != (0, 0, 0)\n"
        "    ok = ok and tuple(stats['block']) != (0, 0, 0)\n"
        "    ok = ok and int(stats['requested_policy_id']) == policy\n"
        "    ok = ok and bool(stats['policy_override_active']) is True\n"
        "    ok = ok and bool(stats['want_plan']) is False\n"
        "    sys.exit(0 if ok else 1)\n"
        "finally:\n"
        "    C._cuda_reduction_clear_kernel_policy_override_for_tests()\n"
    )

    env = {
        # Enable the internal reduce-all sum fast path.
        "VBT_INTERNAL_CUDA_CUB_REDUCE_ALL_SUM": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


@pytest.mark.cuda
def test_amin_force_k2_if_eligible_nonempty_does_not_build_plan():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        t = vbt.tensor([3.0, 1.0, 2.0], dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = t.amin()
        assert float(out.cpu().item()) == pytest.approx(1.0)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K1"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["policy_override_active"]) is True

        assert bool(stats["want_plan"]) is False
        assert int(stats["ineligible_reason"]) == int(C._CUDA_REDUCTION_INELIGIBLE_REASON_NONE)

        assert int(stats["plan_iter_ndim"]) == 0
        assert int(stats["plan_kept_ndim"]) == 0
        assert int(stats["plan_red_ndim"]) == 0
        assert int(stats["plan_red_linear_stride"]) == 0
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()


@pytest.mark.cuda
def test_force_k2_sum_strided_cuda_view_red_linear_stride_non1():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        base = vbt.arange(8, dtype=vbt.float32).cuda().reshape((2, 4))
        t = base.as_strided((2, 2), (4, 2), 0)

        C._cuda_reduction_reset_last_stats_for_tests()
        out = t.sum(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        assert arr.shape == (2,)
        assert float(arr[0]) == pytest.approx(2.0)
        assert float(arr[1]) == pytest.approx(10.0)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert bool(stats["want_plan"]) is True
        assert int(stats["plan_red_linear_stride"]) == 2
        assert int(stats["k2_smem_bytes"]) == int(stats["block"][0]) * 4
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_force_k2_mean_stats_selects_k2_and_reports_smem_bytes():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)
    C._cuda_reduction_clear_env_config_override_for_tests()

    # Keep this test hermetic even if the process env clamps K2 grid.x.
    C._cuda_reduction_set_env_config_for_tests(False, 0)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        x = vbt.arange(4 * 8, dtype=vbt.float32).reshape((4, 8)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.mean(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(4 * 8, dtype=np.float32).reshape(4, 8).mean(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["want_plan"]) is True
        assert int(stats["out_numel"]) == 4
        assert int(stats["slice_len"]) == 8
        assert int(stats["grid"][0]) == int(stats["out_numel"])
        assert int(stats["k2_smem_bytes"]) == int(stats["block"][0]) * 4
    finally:
        C._cuda_reduction_clear_env_config_override_for_tests()
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_force_k2_sum_int64_uses_k2_and_reports_smem_bytes():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        x = vbt.arange(8, dtype=vbt.int64).reshape((2, 4)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(8, dtype=np.int64).reshape(2, 4).sum(axis=1)
        assert arr.shape == expected.shape
        assert np.array_equal(arr, expected)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["want_plan"]) is True
        assert int(stats["out_numel"]) == 2
        assert int(stats["slice_len"]) == 4
        assert int(stats["k2_smem_bytes"]) == int(stats["block"][0]) * 8
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_force_k2_prod_float32_uses_k2():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        x = vbt.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        expected = np.array([6.0, 120.0], dtype=np.float32)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["want_plan"]) is True
        assert int(stats["out_numel"]) == 2
        assert int(stats["slice_len"]) == 3
        assert int(stats["k2_smem_bytes"]) == int(stats["block"][0]) * 4
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_force_k2_prod_int64_uses_k2_and_reports_smem_bytes():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        x = vbt.tensor([[1, 2, 3], [4, 5, 6]], dtype=vbt.int64).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.prod(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        expected = np.array([6, 120], dtype=np.int64)
        assert arr.shape == expected.shape
        assert np.array_equal(arr, expected)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["requested_policy_id"]) == policy
        assert bool(stats["want_plan"]) is True
        assert int(stats["out_numel"]) == 2
        assert int(stats["slice_len"]) == 3
        assert int(stats["k2_smem_bytes"]) == int(stats["block"][0]) * 8
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)



@pytest.mark.cuda
def test_force_k2_sum_negative_red_linear_stride_cuda_view():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        base = vbt.arange(8, dtype=vbt.float32).cuda().reshape((2, 4))
        # Reverse each row: start at the last element of row 0.
        t = base.as_strided((2, 4), (4, -1), 3)

        C._cuda_reduction_reset_last_stats_for_tests()
        out = t.sum(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        assert arr.shape == (2,)
        assert float(arr[0]) == pytest.approx(6.0)
        assert float(arr[1]) == pytest.approx(22.0)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert bool(stats["want_plan"]) is True
        assert int(stats["plan_red_linear_stride"]) == -1
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_force_k2_sum_slice_len_gt_blockdim_multi_iteration():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        x = vbt.arange(8 * 1024, dtype=vbt.float32).reshape((8, 1024)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(8 * 1024, dtype=np.float32).reshape(8, 1024).sum(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["slice_len"]) == 1024
        assert int(stats["block"][0]) < int(stats["slice_len"])
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_force_k2_grid_x_cap_one_nonuniform_catches_missing_out_idx_loop():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    C._cuda_reduction_set_grid_x_cap_for_tests(1)
    try:
        x = vbt.arange(128, dtype=vbt.float32).reshape((128, 1)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert int(stats["grid"][0]) == 1

        arr = _to_numpy_cpu(out.cpu())
        assert arr.shape == (128,)
        assert (arr == np.arange(128, dtype=np.float32)).all()
    finally:
        C._cuda_reduction_set_grid_x_cap_for_tests(None)
        C._cuda_reduction_clear_kernel_policy_override_for_tests()


@pytest.mark.cuda
def test_force_k2_sum_reduce_middle_dim_decode_regression():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_IF_ELIGIBLE)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        x = vbt.arange(2 * 3 * 4, dtype=vbt.float32).reshape((2, 3, 4)).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        out = x.sum(dim=1)

        arr = _to_numpy_cpu(out.cpu())
        expected = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4).sum(axis=1)
        assert arr.shape == expected.shape
        assert np.allclose(arr, expected)

        stats = C._cuda_reduction_last_stats_for_tests()
        assert stats["selected_kernel"] == "K2"
        assert bool(stats["want_plan"]) is True
        assert int(stats["plan_kept_ndim"]) == 2
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)


@pytest.mark.cuda
def test_force_k2_strict_ineligible_publishes_stats_and_message():
    C._cuda_reduction_clear_kernel_policy_override_for_tests()
    C._cuda_reduction_set_grid_x_cap_for_tests(None)

    policy = int(C._CUDA_REDUCTION_KERNEL_POLICY_FORCE_K2_STRICT)
    C._cuda_reduction_set_kernel_policy_for_tests(policy)
    try:
        t = vbt.ones((2, 3, 4), dtype=vbt.float32).cuda()

        C._cuda_reduction_reset_last_stats_for_tests()
        with pytest.raises(ValueError, match=r"^vt::sum: forced kernel K2 ineligible$"):
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
    finally:
        C._cuda_reduction_clear_kernel_policy_override_for_tests()
        C._cuda_reduction_set_grid_x_cap_for_tests(None)
