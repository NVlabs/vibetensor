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


def test_dataloader_single_process_partial_failure_synchronizes(monkeypatch):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch.cuda as vcuda
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            a = np.asarray([1, 2], dtype=np.float32)
            base = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
            b = base[:, 0]
            assert not b.flags.c_contiguous
            return (a, b)

    calls = {"n": 0}
    real_sync = vcuda.Stream.synchronize

    def _sync_spy(self):
        calls["n"] += 1
        return real_sync(self)

    monkeypatch.setattr(vcuda.Stream, "synchronize", _sync_spy)

    dl = vtd.DataLoader(
        _DS(),
        batch_size=None,
        collate_fn=lambda x: x,
        prefetch_to_device=True,
        device=0,
        non_blocking=True,
    )

    with pytest.raises(ValueError) as exc:
        _ = next(iter(dl))

    assert "to_device: expected a C-contiguous NumPy array" in str(exc.value)
    assert calls["n"] >= 1
