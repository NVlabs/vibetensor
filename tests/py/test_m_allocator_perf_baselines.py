# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PERF_DIR = _REPO_ROOT / "tools" / "perf"

if not (_PERF_DIR / "check_m_allocator_baselines.py").is_file():
    pytest.skip("tools/perf baseline checker not present", allow_module_level=True)


def _import_checker():
    """Import the baseline checker module from tools/perf.

    This mirrors the import pattern used by other Allocator perf tests.
    """

    import sys

    if str(_PERF_DIR) not in sys.path:
        sys.path.insert(0, str(_PERF_DIR))
    import check_m_allocator_baselines as checker  # type: ignore[import]

    return checker


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _make_minimal_perf_result(
    *,
    scenario_id: str,
    runner: str,
    device_name: str,
    backend: str,
    config_signature: str,
    median_ms: float,
    peak_reserved: int,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "component": "m_allocator",
        "scenario_id": scenario_id,
        "runner": runner,
        "device": {
            "name": device_name,
            "backend": backend,
        },
        "allocator": {
            "config_signature": config_signature,
        },
        "metrics": {
            "median_ms_per_iter": float(median_ms),
            "peak_reserved_bytes": int(peak_reserved),
        },
    }


def _make_baselines_with_entry(
    *,
    scenario_id: str,
    runner: str,
    device_name: str,
    backend: str,
    config_signature: str,
    median_ms: float,
    peak_reserved: int,
    epsilon_time: float,
    epsilon_mem: float,
) -> Dict[str, Any]:
    entry = {
        "scenario_id": scenario_id,
        "runner": runner,
        "device_name": device_name,
        "backend": backend,
        "config_signature": config_signature,
        "metrics": {
            "median_ms_per_iter": float(median_ms),
            "peak_reserved_bytes": int(peak_reserved),
        },
        "tolerances": {
            "epsilon_time": float(epsilon_time),
            "epsilon_mem": float(epsilon_mem),
        },
        "baseline_reason": "unit-test-baseline",
        "baseline_updated_at": "2025-01-01T00:00:00Z",
    }
    return {
        "schema_version": 1,
        "component": "m_allocator",
        "profiles": [
            {
                "profile_id": "ci_native_full",
                "entries": [entry],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_baseline_checker_non_gating_returns_zero(tmp_path: Path) -> None:
    checker = _import_checker()

    current = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.05,
        peak_reserved=1100,
    )
    baselines = _make_baselines_with_entry(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.00,
        peak_reserved=1000,
        epsilon_time=0.10,
        epsilon_mem=0.20,
    )

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    _write_json(cur_path, current)
    _write_json(base_path, baselines)

    rc = checker.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
        ]
    )
    assert rc == 0


def test_baseline_checker_gating_within_tolerances_passes(tmp_path: Path) -> None:
    checker = _import_checker()

    # Current run is slightly slower and uses slightly more memory but remains
    # within the default tolerances (epsilon_time=0.10, epsilon_mem=0.20).
    current = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.05,  # +5%
        peak_reserved=1100,  # +10%
    )
    baselines = _make_baselines_with_entry(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.00,
        peak_reserved=1000,
        epsilon_time=0.10,
        epsilon_mem=0.20,
    )

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    _write_json(cur_path, current)
    _write_json(base_path, baselines)

    rc = checker.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
            "--fail-on-regression",
        ]
    )
    assert rc == 0


def test_baseline_checker_gating_exceeds_tolerances_fails(tmp_path: Path) -> None:
    checker = _import_checker()

    # Current run violates the time tolerance (20% slower vs 10% limit).
    current = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.20,  # +20%
        peak_reserved=1100,
    )
    baselines = _make_baselines_with_entry(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.00,
        peak_reserved=1000,
        epsilon_time=0.10,
        epsilon_mem=0.20,
    )

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    _write_json(cur_path, current)
    _write_json(base_path, baselines)

    rc = checker.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
            "--fail-on-regression",
        ]
    )
    assert rc == 2


def test_baseline_checker_gating_without_matching_baseline_is_non_fatal(tmp_path: Path) -> None:
    checker = _import_checker()

    current = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.20,
        peak_reserved=1200,
    )
    baselines = {
        "schema_version": 1,
        "component": "m_allocator",
        "profiles": [
            {
                "profile_id": "some_other_profile",
                "entries": [],
            }
        ],
    }

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    _write_json(cur_path, current)
    _write_json(base_path, baselines)

    rc = checker.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
            "--fail-on-regression",
        ]
    )
    # Lack of baseline should be treated as "nothing to compare" even in
    # gating mode so that new hardware / profiles can be brought up safely.
    assert rc == 0


def test_baseline_checker_zero_tolerances_honoured(tmp_path: Path) -> None:
    """A baseline with epsilon_time/epsilon_mem == 0.0 should be respected.

    With zero tolerances, any slowdown or extra memory beyond the baseline
    should be treated as a regression when gating is enabled.
    """

    checker = _import_checker()

    baselines = _make_baselines_with_entry(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.00,
        peak_reserved=1000,
        epsilon_time=0.0,
        epsilon_mem=0.0,
    )

    current_ok = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.00,
        peak_reserved=1000,
    )
    current_bad = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.01,
        peak_reserved=1000,
    )

    base_path = tmp_path / "baselines.json"
    _write_json(base_path, baselines)

    cur_path_ok = tmp_path / "current_ok.json"
    _write_json(cur_path_ok, current_ok)
    rc_ok = checker.main(
        [
            "--current",
            str(cur_path_ok),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
            "--fail-on-regression",
        ]
    )
    assert rc_ok == 0

    cur_path_bad = tmp_path / "current_bad.json"
    _write_json(cur_path_bad, current_bad)
    rc_bad = checker.main(
        [
            "--current",
            str(cur_path_bad),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
            "--fail-on-regression",
        ]
    )
    assert rc_bad == 2


def test_baseline_checker_gating_errors_on_nonpositive_baseline_metrics(tmp_path: Path) -> None:
    """Gating should fail with a schema-style error when baselines are invalid."""

    checker = _import_checker()

    current = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.00,
        peak_reserved=1000,
    )
    # Baseline missing a valid median_ms_per_iter (set to 0.0) should be treated
    # as a configuration error when used for gating.
    baselines = _make_baselines_with_entry(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=0.0,
        peak_reserved=1000,
        epsilon_time=0.10,
        epsilon_mem=0.20,
    )

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    _write_json(cur_path, current)
    _write_json(base_path, baselines)

    rc = checker.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
            "--fail-on-regression",
        ]
    )
    assert rc == 1


def test_baseline_checker_gating_errors_on_nonpositive_current_metrics(tmp_path: Path) -> None:
    """Gating should fail when the current metrics are invalid."""

    checker = _import_checker()

    # Baseline is valid.
    baselines = _make_baselines_with_entry(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=1.00,
        peak_reserved=1000,
        epsilon_time=0.10,
        epsilon_mem=0.20,
    )
    # Current has a non-positive median_ms_per_iter, which should be rejected
    # as an invalid metric for gating.
    current = _make_minimal_perf_result(
        scenario_id="B1",
        runner="cpp_native",
        device_name="Mock GPU",
        backend="native",
        config_signature="v1:deadbeef00000001",
        median_ms=0.0,
        peak_reserved=1000,
    )

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    _write_json(cur_path, current)
    _write_json(base_path, baselines)

    rc = checker.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
            "--fail-on-regression",
        ]
    )
    assert rc == 1
