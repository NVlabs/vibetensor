# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C

import vibetensor.torch.cuda as vc


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


@pytest.mark.cuda
def test_grad_records_stream_on_access_across_streams():
    if not _cuda_only():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    if not hasattr(C, "_cuda_debug_record_stream_call_count"):
        pytest.skip("debug record_stream counters not available", allow_module_level=False)

    ag = C.autograd
    prev = bool(ag.is_cuda_autograd_enabled())  # type: ignore[attr-defined]
    try:
        ag.set_cuda_autograd_enabled(True)  # type: ignore[attr-defined]
        vc.empty_cache()

        x = C._make_cuda_tensor([1024], "float32", 1.0)
        x.set_requires_grad(True)

        y = C._call_op("vt::add", x, x)
        grad = C._make_cuda_tensor([1024], "float32", 1.0)
        y.backward(grad)

        # Reset after backward so only grad() accesses are counted.
        C._cuda_debug_reset_record_stream_call_count()

        s1 = vc.Stream(priority=0)
        s2 = vc.Stream(priority=0)

        with s1:
            g1 = x.grad()
            assert g1 is not None

        with s2:
            g2 = x.grad()
            assert g2 is not None

        assert C._cuda_debug_record_stream_call_count() == 2
    finally:
        ag.set_cuda_autograd_enabled(prev)  # type: ignore[attr-defined]
