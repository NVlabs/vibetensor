# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C
import vibetensor.torch.cuda as vc

if not hasattr(vc, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor.torch.cuda import graphs as vgraphs


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0  # type: ignore[attr-defined]


def test_graph_context_accepts_explicit_pool_handle():
    """Smoke test that graph() accepts an explicit GraphPoolHandle.

    Allocator routing semantics (including denial substrings) are covered by
    dedicated C++ tests; here we only validate that Python wiring for
    graph_pool_handle and the graph context is sound.
    """

    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    pool = vgraphs.graph_pool_handle()
    g = vgraphs.CUDAGraph()

    with vgraphs.graph(g, pool=pool):
        assert vgraphs.is_current_stream_capturing() is True

    assert vgraphs.is_current_stream_capturing() is False
