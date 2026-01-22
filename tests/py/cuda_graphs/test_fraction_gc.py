# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Allocator Python tests: fraction-cap and GC observability.

These tests focus on the Python views of the allocator
counters and their interaction with CUDA Graphs observability. They are
lightweight complements to the more detailed C++ and subprocess-heavy
fraction/GC tests in ``tests/py/test_cuda_memory_stats.py``.

Scenarios covered here (adapted from PY‑F* in
the allocator design documentation):

* Verify that the scalar counters exposed by ``_cuda_memoryStats``
  round‑trip cleanly through both ``memory_stats_as_nested_dict`` and
  ``memory_stats``.
* Smoke‑test the combined snapshot helper used by other Allocator tests
  (``snapshot_allocator_and_graphs``).
"""

from typing import Any, Dict

import pytest

import vibetensor.torch.cuda as cuda

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _graph_workload_utils import (
    require_cuda_or_skip,
    require_native_allocator_or_skip,
    snapshot_allocator_and_graphs,
)


def _get_device() -> int:
    try:
        from vibetensor import _C as C  # type: ignore[import]
    except Exception:  # pragma: no cover - extremely defensive
        return 0

    fn = getattr(C, "_cuda_current_device", None)
    if fn is None:
        return 0
    try:
        return int(fn())  # type: ignore[misc]
    except Exception:
        return 0


def test_fraction_counters_roundtrip_views() -> None:
    """PY‑F1 – fraction/GC counters agree between nested and flat views.

    When the underlying ``_cuda_memoryStats`` binding exposes scalar
    counters such as ``fraction_cap_breaches`` and ``gc_passes``, they
    should appear both in the ``device_stats.aggregated`` family of
    ``memory_stats_as_nested_dict`` and under the corresponding
    ``device_stats.aggregated.*`` keys of the flat ``memory_stats`` view
    with identical integer values.
    """

    require_cuda_or_skip()
    require_native_allocator_or_skip()

    dev = _get_device()

    nested = cuda.memory_stats_as_nested_dict(dev)
    flat = cuda.memory_stats(dev)

    dev_stats: Dict[str, Any] = nested.get("device_stats", {}).get("aggregated", {})  # type: ignore[assignment]

    names = [
        "fraction_cap_breaches",
        "fraction_cap_misfires",
        "gc_passes",
        "gc_reclaimed_bytes",
    ]

    # It is permissible for some builds to omit these counters (e.g. when
    # the underlying binding is not available). In that case this test is
    # effectively a no-op rather than a failure.
    missing_all = True
    for name in names:
        if name not in dev_stats:
            continue
        missing_all = False
        val = int(dev_stats[name])
        key = f"device_stats.aggregated.{name}"
        assert key in flat
        assert int(flat[key]) == val

    if missing_all:
        pytest.skip("device_stats.aggregated counters not available in this configuration")


def test_snapshot_allocator_and_graphs_shape() -> None:
    """PY‑F2 – basic shape checks for combined allocator/graphs snapshot.

    This is a smoke test that ensures ``snapshot_allocator_and_graphs``
    returns a dict with the expected top‑level keys and that the nested
    structures have the right container types. Detailed semantics (e.g.
    fraction‑cap OOM advancement, capture denial) are exercised by
    dedicated C++ and subprocess tests.
    """

    require_cuda_or_skip()

    dev = _get_device()
    snap = snapshot_allocator_and_graphs(dev)

    assert isinstance(snap, dict)
    for key in ("memory_stats", "memory_snapshot", "cuda_graphs_stats", "graph_pool_stats"):
        assert key in snap

    assert isinstance(snap["memory_stats"], dict)
    assert isinstance(snap["memory_snapshot"], list)

    graphs_stats = snap["cuda_graphs_stats"]
    assert isinstance(graphs_stats, dict)
    assert "graphs" in graphs_stats and "pools" in graphs_stats

    assert isinstance(graphs_stats["graphs"], dict)
    assert isinstance(graphs_stats["pools"], dict)

    assert isinstance(snap["graph_pool_stats"], list)
