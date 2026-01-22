# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time

import numpy as np
import pytest


def _has_cuda() -> bool:
    from vibetensor import _C as C

    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def _wait_until(pred, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if bool(pred()):
            return
        time.sleep(float(interval))
    raise AssertionError("Timed out waiting for condition")


def test_dataloader_non_blocking_close_synchronizes_device_stream(monkeypatch: pytest.MonkeyPatch):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch.cuda as vcuda
    import vibetensor.torch.utils.data as vtd

    idx0_started = threading.Event()
    allow_idx0_return = threading.Event()

    class _DS:
        def __len__(self):
            return 2

        def __getitem__(self, idx: int):
            idx = int(idx)
            if idx == 0:
                idx0_started.set()
                assert allow_idx0_return.wait(timeout=5.0)
                return {"x": np.asarray([0], dtype=np.float32)}
            if idx == 1:
                assert idx0_started.wait(timeout=5.0)
                return {"x": np.asarray([1], dtype=np.float32)}
            raise IndexError(idx)

    # NOTE: in_order=True so idx=1 gets buffered in _reorder.
    dl = vtd.DataLoader(
        _DS(),
        batch_size=None,
        num_workers=2,
        in_order=True,
        prefetch_to_device=True,
        device=0,
        non_blocking=True,
    )

    it = iter(dl)

    sync_called = threading.Event()
    allow_sync_return = threading.Event()

    real_sync = vcuda.Stream.synchronize

    def _sync_spy(self):
        # Only block the DataLoader device-prefetch thread, so other potential
        # synchronize calls (if any) do not hang the test runner.
        if threading.current_thread().name == "vbt_dataloader_prefetch_to_device":
            sync_called.set()
            assert allow_sync_return.wait(timeout=5.0)
        return real_sync(self)

    monkeypatch.setattr(vcuda.Stream, "synchronize", _sync_spy, raising=True)

    next_done = threading.Event()

    def _t_next0() -> None:
        try:
            _ = next(it)
        except BaseException:
            pass
        finally:
            next_done.set()

    t0 = threading.Thread(target=_t_next0, daemon=True)
    t0.start()

    try:
        assert idx0_started.wait(timeout=2.0)
        _wait_until(lambda: 1 in it._reorder, timeout=2.0)

        it.close()

        assert sync_called.wait(timeout=2.0)

    finally:
        allow_idx0_return.set()
        allow_sync_return.set()
        assert next_done.wait(timeout=2.0)
