# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Unit tests for allocator perf helpers.

These tests cover small, self-contained pieces of the allocator+graphs perf
harness that do not require CUDA at runtime:

* Python run-count resolution logic (resolve_run_counts_py).
* Config signature generation (_compute_config_signature).
* Baseline checker behaviour (check_m_allocator_baselines.main).
"""

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import perf harness modules from tools/perf
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_PERF_DIR = _REPO_ROOT / "tools" / "perf"
if str(_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_DIR))

if not (_PERF_DIR / "bench_m_allocator_graphs.py").is_file() or not (
    _PERF_DIR / "check_m_allocator_baselines.py"
).is_file():
    pytest.skip("tools/perf allocator harness not present", allow_module_level=True)

import bench_m_allocator_graphs as bench  # type: ignore[import]
import check_m_allocator_baselines as check_mod  # type: ignore[import]


# ---------------------------------------------------------------------------
# resolve_run_counts_py
# ---------------------------------------------------------------------------


def test_resolve_run_counts_py_p_scenario_defaults() -> None:
    """P*-scenario defaults should follow the Allocator design doc.

    Smoke mode is uniform across scenarios; normal/heavy differ for P2 only.
    """

    RC = bench.RunCounts

    # Smoke: identical for all scenarios regardless of overrides.
    counts_smoke = bench.resolve_run_counts_py("P1", "smoke", RC(999, 999, 999))
    assert counts_smoke == RC(warmup_iters=1, measure_iters=3, repeats=1)

    # Normal mode for P1/P2/P3.
    for scenario in ("P1", "P2", "P3"):
        counts = bench.resolve_run_counts_py(scenario, "normal", RC(0, 0, 0))
        assert counts == RC(warmup_iters=3, measure_iters=20, repeats=3)

    # Heavy mode for P2 uses larger defaults.
    counts_p2_heavy = bench.resolve_run_counts_py("P2", "heavy", RC(0, 0, 0))
    assert counts_p2_heavy == RC(warmup_iters=3, measure_iters=40, repeats=5)

    # Heavy mode for P1/P3 matches their normal defaults.
    for scenario in ("P1", "P3"):
        counts = bench.resolve_run_counts_py(scenario, "heavy", RC(0, 0, 0))
        assert counts == RC(warmup_iters=3, measure_iters=20, repeats=3)


def test_resolve_run_counts_py_b_scenario_defaults_and_guardrail() -> None:
    """B*-scenario defaults mirror the C++ harness and enforce guardrails."""

    RC = bench.RunCounts

    # B1 normal uses the C++ defaults (5, 50, 3).
    counts_b1 = bench.resolve_run_counts_py("B1", "normal", RC(0, 0, 0))
    assert counts_b1 == RC(warmup_iters=5, measure_iters=50, repeats=3)

    # B2 heavy uses the C++ heavy defaults (5, 100, 5).
    counts_b2_heavy = bench.resolve_run_counts_py("B2", "heavy", RC(0, 0, 0))
    assert counts_b2_heavy == RC(warmup_iters=5, measure_iters=100, repeats=5)

    # Guardrail: an excessive total iteration count should raise.
    with pytest.raises(ValueError):
        bench.resolve_run_counts_py("B1", "normal", RC(0, 10000, 10000))


# ---------------------------------------------------------------------------
# _compute_config_signature
# ---------------------------------------------------------------------------


def test_compute_config_signature_stable_and_sensitive() -> None:
    """Config signature should be stable and change with any input field."""

    cfg = {
        "per_process_memory_fraction": 1.0,
        "max_split_size_bytes": 64 * 1024 * 1024,
        "max_non_split_rounding_bytes": 32 * 1024 * 1024,
        "roundup_tolerance_bytes": 2 * 1024 * 1024,
    }

    sig0 = bench._compute_config_signature("native", cfg)  # type: ignore[attr-defined]
    # Shape: "v1:" prefix plus 16 hex digits.
    assert sig0.startswith("v1:")
    assert len(sig0) == 3 + 16

    # Identical inputs produce identical signatures.
    assert bench._compute_config_signature("native", cfg) == sig0  # type: ignore[attr-defined]

    # Changing backend changes the signature.
    sig_backend = bench._compute_config_signature("async", cfg)  # type: ignore[attr-defined]
    assert sig_backend != sig0

    # Changing any numeric field changes the signature.
    for key in cfg.keys():
        cfg2 = dict(cfg)
        cfg2[key] = cfg2[key] + 1  # type: ignore[operator]
        sig2 = bench._compute_config_signature("native", cfg2)  # type: ignore[attr-defined]
        assert sig2 != sig0


# ---------------------------------------------------------------------------
# check_m_allocator_baselines behaviour
# ---------------------------------------------------------------------------


def test_check_baselines_no_matching_entry(tmp_path: Path) -> None:
    """When no matching baseline exists, the checker should exit 0."""

    current = {
        "scenario_id": "P1",
        "runner": "python",
        "device": {"name": "GPU-TEST", "backend": "native"},
        "allocator": {"config_signature": "v1:deadbeefdeadbeef"},
        "metrics": {
            "median_ms_per_iter": 0.1,
            "peak_reserved_bytes": 1234,
        },
    }

    baselines = {
        "schema_version": 1,
        "component": "m_allocator",
        "profiles": [],
    }

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    cur_path.write_text(json.dumps(current), encoding="utf-8")
    base_path.write_text(json.dumps(baselines), encoding="utf-8")

    exit_code = check_mod.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
        ]
    )
    assert exit_code == 0


def test_check_baselines_matching_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Matching baseline should print ratios and exit 0."""

    current = {
        "scenario_id": "P1",
        "runner": "python",
        "device": {"name": "GPU-TEST", "backend": "native"},
        "allocator": {"config_signature": "v1:aaaaaaaaaaaaaaaa"},
        "metrics": {
            "median_ms_per_iter": 0.2,
            "peak_reserved_bytes": 200,
        },
    }

    baselines = {
        "schema_version": 1,
        "component": "m_allocator",
        "profiles": [
            {
                "profile_id": "ci_native_full",
                "host_fingerprint": {
                    "hostname": "host",
                    "gpu_name": "GPU-TEST",
                    "cuda_driver_version": "0",
                },
                "entries": [
                    {
                        "scenario_id": "P1",
                        "runner": "python",
                        "device_name": "GPU-TEST",
                        "backend": "native",
                        "config_signature": "v1:aaaaaaaaaaaaaaaa",
                        "metrics": {
                            "median_ms_per_iter": 0.1,
                            "peak_reserved_bytes": 100,
                        },
                        "tolerances": {
                            "epsilon_time": 0.1,
                            "epsilon_mem": 0.2,
                        },
                        "baseline_reason": "test-baseline",
                        "baseline_updated_at": "2025-01-01T00:00:00Z",
                    }
                ],
            }
        ],
    }

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    cur_path.write_text(json.dumps(current), encoding="utf-8")
    base_path.write_text(json.dumps(baselines), encoding="utf-8")

    exit_code = check_mod.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
            "--profile-id",
            "ci_native_full",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr().out
    # Basic sanity check on printed ratios.
    assert "Perf baseline comparison:" in captured
    assert "median_ms_per_iter" in captured
    assert "peak_reserved_bytes" in captured


def test_check_baselines_invalid_json_returns_error(tmp_path: Path) -> None:
    """Malformed JSON should cause the checker to return non-zero."""

    cur_path = tmp_path / "current.json"
    base_path = tmp_path / "baselines.json"
    cur_path.write_text("{}", encoding="utf-8")
    base_path.write_text("not-json", encoding="utf-8")

    exit_code = check_mod.main(
        [
            "--current",
            str(cur_path),
            "--baselines",
            str(base_path),
        ]
    )
    assert exit_code == 1


# ---------------------------------------------------------------------------
# bench_m_allocator_graphs CLI skip behaviour
# ---------------------------------------------------------------------------


def test_bench_cli_handles_pytest_skip(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI should treat pytest.SkipException as a structured skip (exit 77).

    This simulates a scenario where underlying helpers (e.g. _graph_workload_utils)
    raise pytest.SkipException instead of RuntimeError to signal a skip.
    """

    # Force the harness to believe CUDA is available so we do not take the
    # CPU-only shortcut in main().
    monkeypatch.setattr(bench, "_cuda_available", lambda: True, raising=False)

    class _DummyGW:
        def run_eager(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

        def run_graphed(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            pytest.skip("backend_unavailable: test", allow_module_level=False)

        def snapshot_allocator_and_graphs(
            self, *args, **kwargs
        ):  # type: ignore[no-untyped-def]
            return {
                "memory_stats": {},
                "memory_snapshot": [],
                "cuda_graphs_stats": {"graphs": {}, "pools": {}},
                "graph_pool_stats": [],
            }

    monkeypatch.setattr(bench, "gw", _DummyGW(), raising=False)

    # Invoke the CLI entrypoint with json-out=- so any accidental JSON
    # emission would appear on stdout.
    exit_code = bench.main(
        [
            "--scenario",
            "P1",
            "--device",
            "0",
            "--run-mode",
            "smoke",
            "--json-out",
            "-",
        ]
    )

    captured = capsys.readouterr()
    # Skip should map to exit code 77 with no JSON emitted.
    assert exit_code == 77
    assert captured.out.strip() == ""
