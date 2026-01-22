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


def test_graph_pool_handle_raises_without_cuda():
    """graph_pool_handle behavior on CPU-only vs CUDA builds.

    On CPU-only builds, `graph_pool_handle` should raise a clear error.
    On CUDA builds, it should succeed and return a handle whose underlying
    device id matches the current CUDA device.
    """

    if _cuda_unavailable():
        with pytest.raises(RuntimeError, match="CUDA unavailable: no devices"):
            vgraphs.graph_pool_handle()
    else:
        h = vgraphs.graph_pool_handle()
        dev, _ = h.to_tuple()
        assert isinstance(dev, int)
        assert dev == C._cuda_current_device()  # type: ignore[attr-defined]


def test_graph_pool_handle_device_none_uses_current_device():
    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    h = vgraphs.graph_pool_handle()
    dev, _ = h.to_tuple()
    assert dev == C._cuda_current_device()  # type: ignore[attr-defined]


def test_graph_pool_handle_invalid_device_errors():
    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    with pytest.raises(RuntimeError, match="device must be >= 0 or None"):
        vgraphs.graph_pool_handle(-1)  # type: ignore[arg-type]

    n = C._cuda_device_count()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="device index out of range"):
        vgraphs.graph_pool_handle(int(n) + 5)


def test_graph_pool_handle_shared_across_graphs_sequential_replays() -> None:
    """Graphs can safely share a pool when replays are sequential.

    This is a Python-level smoke test mirroring the intent of
    torch.cuda.graphs test_graph_share_mem: two graphs capture disjoint
    regions while sharing a GraphPoolHandle and can replay sequentially
    without errors. The stricter busy-pool predicate (which forbids
    overlapping replays on the same pool) is validated in C++ tests.
    """

    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    pool = vgraphs.graph_pool_handle()
    g1 = vgraphs.CUDAGraph(keep_graph=True)
    g2 = vgraphs.CUDAGraph(keep_graph=True)

    s = vc.Stream()
    with s:
        # First graph captures an empty region using the shared pool.
        with vgraphs.graph(g1, pool=pool, stream=s):
            pass
        g1.instantiate()

        # Second graph captures a disjoint (also empty) region using the same
        # pool. The key property is that both graphs select the same underlying
        # allocator pool via the shared handle.
        with vgraphs.graph(g2, pool=pool, stream=s):
            pass
        g2.instantiate()

        # Both graphs should report the same underlying pool id.
        h_dev, h_id = pool.to_tuple()
        g1_dev, g1_id = g1.pool().to_tuple()
        g2_dev, g2_id = g2.pool().to_tuple()

        assert (g1_dev, g1_id) == (h_dev, h_id)
        assert (g2_dev, g2_id) == (h_dev, h_id)

        # Sequential replays on the shared pool must succeed.
        g1.replay()
        s.synchronize()
        g2.replay()
        s.synchronize()
