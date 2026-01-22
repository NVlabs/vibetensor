# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C
import vibetensor.torch.cuda as vc

if not hasattr(vc, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor.torch.cuda import graphs as vgraphs


def _cuda_unavailable() -> bool:
    return (not getattr(C, "_has_cuda", False)) or C._cuda_device_count() == 0  # type: ignore[attr-defined]


def test_is_current_stream_capturing_defaults_to_false():
    """is_current_stream_capturing should return False when not capturing."""
    if _cuda_unavailable():
        assert vgraphs.is_current_stream_capturing() is False
        return

    assert vgraphs.is_current_stream_capturing() is False


def test_is_current_stream_capturing_true_during_graph_capture():
    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    g = vgraphs.CUDAGraph()
    with vgraphs.graph(g):
        assert vgraphs.is_current_stream_capturing() is True
    # After leaving the context, capture has ended.
    assert vgraphs.is_current_stream_capturing() is False


def test_is_current_stream_capturing_on_stream_argument():
    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    s = vc.Stream()
    # No capture in progress on this stream; should be False.
    assert vgraphs.is_current_stream_capturing(stream=s) is False


def test_is_current_stream_capturing_type_error_on_non_stream():
    if _cuda_unavailable():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    with pytest.raises(TypeError, match="stream must be a vibetensor.torch.cuda.Stream"):
        vgraphs.is_current_stream_capturing(stream=42)  # type: ignore[arg-type]
