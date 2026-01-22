# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as _np
import pytest

from vibetensor import _C as C
import vibetensor.torch.cuda as vc

if not hasattr(vc, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor.torch.cuda import graphs as vgraphs


def _cuda_unavailable() -> bool:
    has_cuda = getattr(C, "_has_cuda", False)
    get_count = getattr(C, "_cuda_device_count", None)
    if not has_cuda or get_count is None:
        return True
    try:
        return int(get_count()) == 0
    except Exception:
        return True


def test_graph_pool_stats_tracks_implicit_pool_and_reset() -> None:
    """End-to-end smoke test for implicit pools and reset().

    This test exercises the Python bindings for cuda_graphs_stats(),
    graph_pool_stats(), and CUDAGraph.reset() without asserting on detailed
    allocator behaviour (which is covered by C++ tests).
    """

    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    graphs_before = vgraphs.cuda_graphs_stats()
    graphs_counters_before = graphs_before["graphs"]

    g = vgraphs.CUDAGraph(keep_graph=True)

    # Capture an empty region on a non-default stream; this still allocates an
    # implicit graph-private pool and exercises the pool lifecycle wiring.
    s = vc.Stream()
    with s:
        with vgraphs.graph(g, stream=s):
            pass
        g.instantiate()
        g.replay()
        s.synchronize()

    pool_handle = g.pool()
    dev, pid = pool_handle.to_tuple()
    assert pid != 0

    # Pool-specific stats should report at most one entry corresponding to
    # this (device, id) pair. bytes_reserved may legitimately be zero if no
    # allocations were routed into the pool.
    stats_for_pool = pool_handle.stats()
    assert isinstance(stats_for_pool, list)
    assert len(stats_for_pool) <= 1
    if stats_for_pool:
        entry = stats_for_pool[0]
        assert entry["device"] == dev
        assert entry["id"] == pid
        assert entry["bytes_reserved"] >= 0

    # reset() should free the underlying CUDA graph/exec and release the
    # graph's reference to the private pool. The graphs_reset counter should
    # increase by exactly 1 for this call.
    g.reset()

    graphs_after_reset = vgraphs.cuda_graphs_stats()
    graphs_counters_after = graphs_after_reset["graphs"]
    assert (
        graphs_counters_after["graphs_reset"]
        == graphs_counters_before["graphs_reset"] + 1
    )

    # After reset(), the original pool handle becomes a dangling identifier.
    # Its stats() view should be empty because the allocator has erased the
    # corresponding pool entry and reclassified any blocks as global.
    stats_after = pool_handle.stats()
    assert isinstance(stats_after, list)
    assert stats_after == []
