# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as _np

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


def test_reset_invalid_state_increments_counters() -> None:
    """Calling CUDAGraph.reset() in an invalid state bumps the counter.

    This is a minimal Python-level smoke test that the new reset API is wired
    through to C++, and that the reset_invalid_state counter is observable via
    cuda_graphs_stats(). The full happy-path semantics are exercised in C++
    tests.
    """

    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    stats_before = vgraphs.cuda_graphs_stats()
    graphs_before = stats_before["graphs"]

    g = vgraphs.CUDAGraph(keep_graph=True)

    with pytest.raises(RuntimeError, match="reset called in invalid state"):
        g.reset()

    stats_after = vgraphs.cuda_graphs_stats()
    graphs_after = stats_after["graphs"]

    assert (
        graphs_after["reset_invalid_state"]
        == graphs_before["reset_invalid_state"] + 1
    )


def test_reset_happy_path_releases_pool_and_allows_recapture() -> None:
    """Happy-path reset() releases the pool and permits recapture.

    This test captures an empty region into a CUDAGraph with an implicit
    pool, instantiates and replays it, then calls reset() and verifies that:

    * graphs_reset increases by exactly 1.
    * the graph's original pool handle becomes dangling (stats() is empty).
    * the same CUDAGraph instance can be reused for a fresh capture.
    """

    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    stats_before = vgraphs.cuda_graphs_stats()
    graphs_before = stats_before["graphs"]

    g = vgraphs.CUDAGraph(keep_graph=True)

    s = vc.Stream()
    with s:
        with vgraphs.graph(g, stream=s):
            pass
        g.instantiate()
        g.replay()
        s.synchronize()

    pool_handle = g.pool()

    # Before reset, stats() for the implicit pool should be a list (possibly
    # empty if no allocations were routed into it), but must not raise.
    before_reset_stats = pool_handle.stats()
    assert isinstance(before_reset_stats, list)

    g.reset()

    stats_after = vgraphs.cuda_graphs_stats()
    graphs_after = stats_after["graphs"]
    assert (
        graphs_after["graphs_reset"]
        == graphs_before["graphs_reset"] + 1
    )

    # After reset, the original pool handle should be dangling and expose an
    # empty stats view.
    assert pool_handle.stats() == []

    # The same CUDAGraph instance can be reused for a fresh capture.
    s2 = vc.Stream()
    with s2:
        with vgraphs.graph(g, stream=s2):
            pass
        g.instantiate()
        g.replay()
        s2.synchronize()
