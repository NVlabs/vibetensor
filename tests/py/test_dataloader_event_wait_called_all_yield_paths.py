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


def run_with_timeout(fn, timeout: float = 2.0):
    done = threading.Event()
    out: dict[str, object] = {}

    def _t() -> None:
        try:
            out["value"] = fn()
        except BaseException as e:
            out["exc"] = e
        finally:
            done.set()

    t = threading.Thread(target=_t, daemon=True)
    t.start()

    assert done.wait(timeout=timeout)

    if "exc" in out:
        raise out["exc"]  # type: ignore[misc]

    return out.get("value")


def test_dataloader_event_wait_called_all_yield_paths(monkeypatch: pytest.MonkeyPatch):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch.cuda as vcuda
    import vibetensor.torch.utils.data as vtd

    wait_calls: dict[str, int] = {"n": 0}
    real_wait = vcuda.Event.wait

    def _counting_wait(self, stream):
        wait_calls["n"] += 1
        return real_wait(self, stream)

    monkeypatch.setattr(vcuda.Event, "wait", _counting_wait, raising=True)

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

    dl = vtd.DataLoader(
        _DS(),
        batch_size=None,
        num_workers=2,
        in_order=True,
        prefetch_to_device=True,
        device=0,
    )

    it = iter(dl)
    try:
        out0_holder: dict[str, object] = {}
        out0_done = threading.Event()

        def _t_next0() -> None:
            try:
                out0_holder["value"] = next(it)
            except BaseException as e:
                out0_holder["exc"] = e
            finally:
                out0_done.set()

        t0 = threading.Thread(target=_t_next0, daemon=True)
        t0.start()

        assert idx0_started.wait(timeout=2.0)

        # Wait until the out-of-order item is buffered.
        _wait_until(lambda: 1 in it._reorder, timeout=2.0)

        # Buffering must not call Event.wait.
        assert wait_calls["n"] == 0

        # Unblock idx0 and let the first yield complete.
        allow_idx0_return.set()
        assert out0_done.wait(timeout=5.0)
        if "exc" in out0_holder:
            raise out0_holder["exc"]  # type: ignore[misc]

        out0 = out0_holder.get("value")
        assert isinstance(out0, dict)
        assert wait_calls["n"] == 1

        # Yield from reorder fast path.
        out1 = run_with_timeout(lambda: next(it), timeout=2.0)
        assert isinstance(out1, dict)
        assert wait_calls["n"] == 2
    finally:
        allow_idx0_return.set()
        it.close()


def test_dataloader_event_wait_called_in_order_false_yield_path(monkeypatch: pytest.MonkeyPatch):
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch.cuda as vcuda
    import vibetensor.torch.utils.data as vtd

    wait_calls: dict[str, int] = {"n": 0}
    real_wait = vcuda.Event.wait

    def _counting_wait(self, stream):
        wait_calls["n"] += 1
        return real_wait(self, stream)

    monkeypatch.setattr(vcuda.Event, "wait", _counting_wait, raising=True)

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return {"x": np.asarray([1], dtype=np.float32)}

    dl = vtd.DataLoader(
        _DS(),
        batch_size=None,
        num_workers=2,
        in_order=False,
        prefetch_to_device=True,
        device=0,
    )

    it = iter(dl)
    try:
        out = run_with_timeout(lambda: next(it))
        assert isinstance(out, dict)
        assert wait_calls["n"] == 1
    finally:
        it.close()
