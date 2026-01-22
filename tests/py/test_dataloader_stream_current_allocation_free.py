# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


def test_dataloader_stream_current_allocation_free(monkeypatch: pytest.MonkeyPatch):
    import vibetensor.torch.cuda as vcuda
    from vibetensor import _C as C

    base_cls = getattr(C, "_CudaStreamBase", None)
    if base_cls is None:
        pytest.skip("_CudaStreamBase not available")

    def _boom(self, *args, **kwargs):
        raise AssertionError("Stream.__init__ should not be called by Stream.current")

    monkeypatch.setattr(vcuda.Stream, "__init__", _boom, raising=True)

    # Must not allocate a new pooled stream.
    s_default = vcuda.Stream.current()
    s0 = vcuda.Stream.current(device=0)

    assert getattr(s_default, "_base", None) is not None
    assert getattr(s0, "_base", None) is not None
    assert int(s0._base.device_index) == 0

    # If we have ≥2 devices, validate that device-index selection returns the
    # per-device current stream (TLS), not just the current-device stream.
    try:
        n = int(C._cuda_device_count())
    except Exception:
        n = 0

    if n >= 2:
        prev1 = vcuda.Stream.current(device=1)
        base1 = base_cls(0, 1)
        try:
            vcuda.Stream.set_current(vcuda.Stream._wrap_base(base1))
            s1 = vcuda.Stream.current(device=1)
            assert int(s1._base.device_index) == 1
            assert int(s1._base.cuda_stream) == int(base1.cuda_stream)
        finally:
            vcuda.Stream.set_current(prev1)
