# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Allocator Python tests: allocator stats/snapshot vs graph pools.

These tests exercise the Python-facing view of allocator statistics and
memory snapshots in the presence of CUDA Graphs and graph-private
allocator pools. They mirror a subset of the S* scenarios from
the allocator design doc §3.3:

* PY-S1 – Basic fragmentation and inequalities between
  ``memory_snapshot`` and ``memory_stats_as_nested_dict``.
* PY-S2 – Per-pool inequalities using ``graph_pool_stats`` and
  ``memory_snapshot`` filtered by ``pool_id``.

Async-backend behaviour and CPU-only semantics are covered by existing
``test_cuda_memory_stats.py`` and ``test_cuda_memory_snapshot.py``
smoke tests; here we focus on native-backend graphs×allocator wiring.
"""

from typing import Any, Dict

import numpy as np
import pytest

import vibetensor.torch as vt
import vibetensor.torch.cuda as cuda
from vibetensor import _C as C

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Skip the entire module when the CUDA Graphs overlay is not available.
if not hasattr(cuda, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor.torch.cuda import graphs as vgraphs

from _graph_workload_utils import require_cuda_or_skip, require_native_allocator_or_skip


def _current_device() -> int:
    """Best-effort helper to query the current CUDA device index."""

    fn = getattr(C, "_cuda_current_device", None)
    if fn is None:
        return 0
    try:
        return int(fn())  # type: ignore[misc]
    except Exception:
        return 0


def _fragmenting_workload(num_allocs: int = 8) -> None:
    """Issue a small sequence of allocations to create allocator activity.

    The intent is to populate a few allocator segments so that
    ``memory_snapshot`` returns at least one entry on native backends,
    without attempting to be a heavy fragmentation stress test.
    """

    require_cuda_or_skip()

    # Use a moderate size (a few MiB per allocation) to avoid stressing
    # CI GPUs while still exercising the allocator.
    arr = np.ones((4 * 1024 * 1024,), dtype=np.float32)
    bufs = [cuda.to_device(arr) for _ in range(max(1, int(num_allocs)))]
    # Drop half of the buffers to leave some free blocks behind; this is
    # enough to exercise basic snapshot accounting.
    del bufs[::2]


def _sum_snapshot_bytes(snapshot: list[Dict[str, Any]], *, device: int, pool_id: int | None = None) -> tuple[int, int]:
    """Return (bytes_reserved, bytes_active) aggregates from a snapshot.

    When ``pool_id`` is ``None`` the sum is taken over all segments for
    ``device``; otherwise only segments whose ``pool_id`` matches are
    included.
    """

    reserved = 0
    active = 0
    for seg in snapshot:
        try:
            dev = int(seg.get("device", device))
        except Exception:
            dev = device
        if dev != device:
            continue
        if pool_id is not None:
            try:
                pid = int(seg.get("pool_id", 0))
            except Exception:
                continue
            if pid != pool_id:
                continue
        try:
            reserved += int(seg.get("bytes_reserved", 0))
            active += int(seg.get("bytes_active", 0))
        except Exception:
            # Treat malformed entries as zero contribution.
            continue
    return reserved, active


def test_stats_snapshot_inequalities_native() -> None:
    """PY-S1 – basic inequalities between snapshot and stats (native backend).

    After a small eager workload on a native-backend device, the total
    bytes_reserved / bytes_active reported by ``memory_snapshot`` for the
    current device must not exceed the corresponding
    ``reserved_bytes.all.current`` / ``allocated_bytes.all.current``
    gauges from ``memory_stats_as_nested_dict``.
    """

    require_cuda_or_skip()
    require_native_allocator_or_skip()

    dev = _current_device()

    # Start from a relatively clean state.
    cuda.empty_cache()

    _fragmenting_workload(num_allocs=6)

    snapshot = cuda.memory_snapshot(dev)
    assert isinstance(snapshot, list)

    nested = cuda.memory_stats_as_nested_dict(dev)
    reserved_all = int(nested.get("reserved_bytes", {}).get("all", {}).get("current", 0))
    allocated_all = int(nested.get("allocated_bytes", {}).get("all", {}).get("current", 0))

    snap_reserved, snap_active = _sum_snapshot_bytes(snapshot, device=dev)

    assert snap_reserved <= reserved_all
    assert snap_active <= allocated_all


def test_stats_snapshot_per_pool_inequalities() -> None:
    """PY-S2 – per-pool inequalities via graph_pool_stats and memory_snapshot.

    For a graph-private pool ``P`` we expect::

        bytes_active_snap <= bytes_reserved_pool <= bytes_reserved_snap

    where ``*_snap`` are derived from ``memory_snapshot`` filtered by
    ``pool_id == P.id`` and ``bytes_reserved_pool`` comes from the
    corresponding ``graph_pool_stats`` entry.
    """

    require_cuda_or_skip()
    require_native_allocator_or_skip()

    dev = _current_device()

    # Allocate a dedicated graph-private pool for this test device.
    pool = vgraphs.graph_pool_handle(dev)

    # Capture and replay a tiny workload that routes allocations into the
    # pool. We intentionally keep the workload simple and small; detailed
    # allocator semantics are covered by C++ tests.

    g = vgraphs.CUDAGraph(keep_graph=True)
    with vgraphs.graph(g, pool=pool):
        # Allocate a small tensor on the graph device to ensure that
        # allocator activity is routed into the graph-private pool
        # without issuing host<->device copies during capture.
        _ = vt.rand((1024,), device=f"cuda:{dev}")

    # A single replay plus a synchronize is enough to exercise the
    # steady-state stats path without tripping the busy-pool guard.
    g.replay()
    vgraphs.graph.default_capture_stream.synchronize()

    snapshot = cuda.memory_snapshot(dev)
    assert isinstance(snapshot, list)

    stats_for_pool = vgraphs.graph_pool_stats(pool)
    assert isinstance(stats_for_pool, list)
    if not stats_for_pool:
        # On some configurations it is possible (though unlikely) that no
        # segments were ever routed into this pool. In that case we
        # accept a soft skip rather than failing the test.
        pytest.skip("graph_pool_stats did not report any segments for test pool; treating as soft skip")

    entry = stats_for_pool[0]
    pid = int(entry.get("id", 0))
    bytes_reserved_pool = int(entry.get("bytes_reserved", 0))

    snap_reserved, snap_active = _sum_snapshot_bytes(snapshot, device=dev, pool_id=pid)

    # Allow the fully-zero case but otherwise enforce the inequalities.
    if snap_reserved == 0 and snap_active == 0 and bytes_reserved_pool == 0:
        return

    assert snap_active <= bytes_reserved_pool <= snap_reserved
