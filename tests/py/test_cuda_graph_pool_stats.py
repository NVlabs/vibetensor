# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import vibetensor.torch.cuda as vcuda

if not hasattr(vcuda, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor import _C as C
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


def test_graph_pool_stats_cpu_returns_empty_list() -> None:
    """graph_pool_stats should be safe on both CPU-only and CUDA builds.

    When CUDA is unavailable or no devices exist, the graphs overlay is
    expected to degrade gracefully and return an empty list instead of
    raising. On CUDA builds, the helper should still be callable and
    return a list of dicts (possibly empty).
    """

    stats = vgraphs.graph_pool_stats()
    assert isinstance(stats, list)

    if _cuda_unavailable():
        # CPU-only behavior: no pools, empty list.
        assert stats == []


def test_graph_pool_stats_includes_new_pool_when_cuda_available() -> None:
    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    # Allocate a new graph-private pool handle for the current device.
    handle = vgraphs.graph_pool_handle()
    dev, pid = handle.to_tuple()

    # Unfiltered stats should include an entry for this handle.
    all_stats = vgraphs.graph_pool_stats()
    assert isinstance(all_stats, list)

    keys = {"device", "id", "segments", "blocks", "bytes_reserved", "bytes_active"}

    matching = [s for s in all_stats if s.get("device") == dev and s.get("id") == pid]
    assert matching, "expected graph_pool_stats() to include newly created pool handle"

    entry = matching[0]
    for k in keys:
        assert k in entry
        assert isinstance(entry[k], int)

    # The handle-specific helper should return a compatible, filtered view
    # that includes only this pool id.
    handle_stats = handle.stats()
    assert isinstance(handle_stats, list)

    # handle.stats() must be equivalent to graph_pool_stats(handle).
    assert handle_stats == vgraphs.graph_pool_stats(handle)

    # Every entry should correspond exactly to this handle's (device, id).
    assert all(
        (s.get("device"), s.get("id")) == (dev, pid)
        for s in handle_stats
    )
