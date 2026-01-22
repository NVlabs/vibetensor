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


def _graph_pool_release_available() -> bool:
    return getattr(C, "_graph_pool_release", None) is not None


def test_dangling_graph_pool_handle_raises_on_capture() -> None:
    """Using a dangling GraphPoolHandle for capture_begin should fail.

    The allocator erases a pool when its refcount and activity counters reach
    zero. Any existing GraphPoolHandle identifiers for that (device, id)
    become dangling and must trigger an "unknown mempool id" error when used
    for new captures.
    """

    if _cuda_unavailable() or not _graph_pool_release_available():
        pytest.skip(
            "CUDA or _graph_pool_release helper not available",
            allow_module_level=False,
        )

    # Create a fresh pool handle on the current device and immediately drop
    # the allocator's reference via the internal helper. This simulates the
    # handle becoming dangling after all owners (graphs/handles) have
    # released the pool.
    handle = vgraphs.graph_pool_handle()
    dev, pid = handle.to_tuple()

    C._graph_pool_release((dev, pid))  # type: ignore[attr-defined]

    g = vgraphs.CUDAGraph()

    # Using the dangling handle for capture should surface the allocator's
    # "unknown mempool id" error from retain_pool/begin_allocate_to_pool.
    with pytest.raises(RuntimeError, match="unknown mempool id"):
        with vgraphs.graph(g, pool=handle):
            pass

    # Dangling handles must also present an empty stats view.
    assert handle.stats() == []
