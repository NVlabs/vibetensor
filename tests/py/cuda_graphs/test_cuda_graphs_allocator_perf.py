# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Allocator+graphs perf harness smoke tests.

These tests exercise the Python perf harness used for allocator+graphs
micro-benchmarks and validate that it emits a well-shaped PerfResult
JSON record for the ``P1`` scenario in smoke mode.

The tests are opt-in and are skipped unless the environment variable
``VBT_RUN_ALLOCATOR_GRAPHS_PERF=1`` is set.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Opt-in gating
# ---------------------------------------------------------------------------

if os.environ.get("VBT_RUN_ALLOCATOR_GRAPHS_PERF") != "1":
    pytest.skip("allocator+graphs perf wrappers disabled", allow_module_level=True)


# Reuse perf test helpers for CUDA/graphs availability checks.
_THIS_DIR = Path(__file__).resolve().parent
_GRAPH_UTILS_DIR = _THIS_DIR
if str(_GRAPH_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_UTILS_DIR))

try:
    import _graph_workload_utils as gw  # type: ignore[import]
except Exception:  # pragma: no cover - defensive
    gw = None  # type: ignore[assignment]

if gw is None:
    pytest.skip("_graph_workload_utils not importable", allow_module_level=True)


# Import the perf harness module from tools/perf.
_REPO_ROOT = _THIS_DIR.parent.parent.parent
_PERF_DIR = _REPO_ROOT / "tools" / "perf"
if str(_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_DIR))

if not (_PERF_DIR / "bench_m_allocator_graphs.py").is_file():
    pytest.skip("tools/perf allocator harness not present", allow_module_level=True)

import bench_m_allocator_graphs as bench  # type: ignore[import]


@pytest.mark.perf
@pytest.mark.cuda_graphs_perf
def test_cuda_graphs_allocator_perf_python_smoke_schema() -> None:
    """Smoke test: P1 scenario emits a well-formed PerfResult dict.

    This does not assert on the actual numeric values, only on the
    presence and basic types of the top-level fields.
    """

    gw.require_cuda_or_skip("CUDA required for allocator+graphs perf wrappers")

    result: Dict[str, Any] = bench.run_scenario_for_test(
        "P1", device=0, run_mode="smoke", num_replays=1, fraction=1.0, notes="pytest-smoke"
    )

    # Top-level shape
    assert isinstance(result, dict)
    assert result.get("schema_version") == 1
    assert result.get("component") == "m_allocator"
    assert result.get("scenario_id") == "P1"
    assert result.get("runner") == "python"

    for key in ("host", "device", "allocator", "run", "metrics", "notes"):
        assert key in result, f"missing key: {key}"

    assert isinstance(result["host"], dict)
    assert isinstance(result["device"], dict)
    assert isinstance(result["allocator"], dict)
    assert isinstance(result["run"], dict)
    assert isinstance(result["metrics"], dict)
    assert isinstance(result["notes"], list)

    run = result["run"]
    assert run["warmup_iters"] == 1
    assert run["measure_iters"] == 3
    assert run["repeats"] == 1
    assert run["total_iters"] == 3

    metrics = result["metrics"]
    for key in ("median_ms_per_iter", "p95_ms_per_iter"):
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


@pytest.mark.perf
@pytest.mark.cuda_graphs_perf
def test_cuda_graphs_allocator_perf_python_cli_smoke_stdout(tmp_path: Path) -> None:
    """End-to-end smoke test for the CLI with json-out=-.

    The test spawns the script in a subprocess and asserts that it emits
    at least one non-empty JSON object on stdout.
    """

    gw.require_cuda_or_skip("CUDA required for allocator+graphs perf wrappers")

    script = _PERF_DIR / "bench_m_allocator_graphs.py"
    assert script.is_file()

    # Use the same Python executable running the tests.
    import subprocess

    cmd = [
        sys.executable,
        str(script),
        "--scenario",
        "P1",
        "--device",
        "0",
        "--run-mode",
        "smoke",
        "--json-out",
        "-",
    ]

    env = os.environ.copy()
    # Ensure perf is enabled inside the subprocess as well.
    env.setdefault("VBT_RUN_ALLOCATOR_GRAPHS_PERF", "1")

    proc = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    stdout = proc.stdout.strip().splitlines()
    assert stdout, "no output from perf script"

    first = stdout[0]
    obj = json.loads(first)
    assert isinstance(obj, dict)
    assert obj.get("schema_version") == 1
    assert obj.get("scenario_id") == "P1"
