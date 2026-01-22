# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
import vibetensor.torch as vt
import vibetensor.torch.cuda as vcuda


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0  # type: ignore[attr-defined]


@pytest.mark.cuda
def test_pcap1_cuda_backward_forbidden_under_graph_capture():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    if not hasattr(vcuda, "graphs"):
        pytest.skip("CUDA Graphs Python overlay not available", allow_module_level=False)

    from vibetensor.torch.cuda import graphs as vgraphs

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]

        # Build a small CUDA autograd graph *before* entering capture.
        x = C._make_cuda_tensor([2], "float32", 1.0)
        x.set_requires_grad(True)
        y = C._call_op("vt::add", x, x)
        grad = C._make_cuda_tensor([2], "float32", 1.0)

        g = vgraphs.CUDAGraph(keep_graph=True)
        s = vcuda.Stream()

        with s:
            g.capture_begin()
            assert vgraphs.is_current_stream_capturing() is True
            try:
                with pytest.raises(
                    RuntimeError,
                    match="not supported under CUDA Graph capture",
                ):
                    y.backward(grad)
            finally:
                # Always end capture even if backward fails.
                g.capture_end()
            assert vgraphs.is_current_stream_capturing() is False
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]


@pytest.mark.cuda
def test_pcap2_cpu_only_backward_allowed_under_graph_capture():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    if not hasattr(vcuda, "graphs"):
        pytest.skip("CUDA Graphs Python overlay not available", allow_module_level=False)

    from vibetensor.torch.cuda import graphs as vgraphs

    g = vgraphs.CUDAGraph(keep_graph=True)
    s = vcuda.Stream()

    with s:
        g.capture_begin()
        assert vgraphs.is_current_stream_capturing() is True
        try:
            x = vt.tensor([2.0, 3.0], dtype="float32")
            x.requires_grad = True
            y = x * x
            y.backward(vt.ones_like(y))
        finally:
            g.capture_end()
        assert vgraphs.is_current_stream_capturing() is False
