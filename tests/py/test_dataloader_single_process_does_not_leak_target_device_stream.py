# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest


def _has_cuda() -> bool:
    from vibetensor import _C as C

    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_dataloader_single_process_does_not_leak_target_device_stream():
    from vibetensor import _C as C

    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")
    if int(C._cuda_device_count()) < 2:
        pytest.skip("need >= 2 CUDA devices")

    import vibetensor.torch.cuda as vcuda
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return np.asarray([1, 2], dtype=np.float32)

    orig = vcuda.Stream.current(device=1)
    try:
        # Set a known current stream for device 1.
        s = vcuda.Stream(device=1)
        vcuda.Stream.set_current(s)
        before = vcuda.Stream.current(device=1).__cuda_stream__()

        dl = vtd.DataLoader(_DS(), batch_size=None, prefetch_to_device=True, device=1)
        out = next(iter(dl))
        assert int(out.device[1]) == 1

        after = vcuda.Stream.current(device=1).__cuda_stream__()
        assert after == before

    finally:
        vcuda.Stream.set_current(orig)
