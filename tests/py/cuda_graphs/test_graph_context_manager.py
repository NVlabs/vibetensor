# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C
import vibetensor.torch.cuda as vcuda

if not hasattr(vcuda, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor.torch.cuda import graphs as vgraphs


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0  # type: ignore[attr-defined]


def test_graph_context_manager_basic_capture_and_replay():
    """Smoke test that the graph context enters and exits cleanly.

    We intentionally avoid allocating new CUDA tensors or asserting on the
    underlying CUDAGraph state here; C++ tests cover instantiate/replay
    semantics. From Python we only validate capture status toggling.
    """

    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    g = vgraphs.CUDAGraph()

    with vgraphs.graph(g):
        # Inside the context we should be capturing on the current stream.
        assert vgraphs.is_current_stream_capturing() is True

    # After exiting the context, capture has ended.
    assert vgraphs.is_current_stream_capturing() is False


def test_capture_error_mode_rejects_non_thread_local():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    g = vgraphs.CUDAGraph()
    # Modes other than "thread_local" should be rejected with the pinned
    # kErrUnsupportedCaptureMode substring.
    with pytest.raises(RuntimeError, match="only ThreadLocal capture mode is supported"):
        g.capture_begin(capture_error_mode="global")


def test_nested_capture_banned():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    outer = vgraphs.CUDAGraph()
    inner = vgraphs.CUDAGraph()

    # Start an outer capture via the high-level context manager and ensure that
    # a second capture attempt on the same thread/device is rejected.
    with vgraphs.graph(outer):
        with pytest.raises(RuntimeError, match="nested CUDA graph capture is not allowed"):
            inner.capture_begin()
