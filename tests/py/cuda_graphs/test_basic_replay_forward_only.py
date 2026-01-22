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


def test_basic_instantiate_and_replay_forward_only():
    """Smoke test that CUDAGraph.instantiate/replay bindings are usable.

    The detailed semantics of graph contents and allocator routing are
    validated in C++ tests. Here we only ensure that a minimal begin/end
    sequence followed by instantiate() and replay() does not raise when
    capturing on a non-default stream.
    """

    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    g = vgraphs.CUDAGraph(keep_graph=True)

    # Capture on a non-default stream to respect the default-stream ban.
    s = vc.Stream()
    with s:
        g.capture_begin()
        g.capture_end()

    g.instantiate()
    g.replay()
