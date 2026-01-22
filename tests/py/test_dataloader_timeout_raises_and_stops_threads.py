# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import pytest


def run_with_timeout(fn, timeout: float = 1.0):
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


class _BlockingDS:
    def __init__(self):
        self.started0 = threading.Event()
        self.release0 = threading.Event()

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx != 0:
            raise IndexError(idx)
        self.started0.set()
        self.release0.wait()
        return 0


class _TimeoutDS:
    def __init__(self):
        self.started0 = threading.Event()
        self.release0 = threading.Event()

    def __len__(self):
        return 2

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx == 0:
            self.started0.set()
            self.release0.wait()
            return 0
        if idx == 1:
            self.started0.wait()
            return 1
        raise IndexError(idx)


def _wait_workers_stop(workers, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while any(w.is_alive() for w in workers) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert all(not w.is_alive() for w in workers)


def test_dataloader_timeout_out_of_order_raises_and_is_terminal():
    import vibetensor.torch.utils.data as vtd

    ds = _TimeoutDS()
    timeout_s = 0.5

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=False,
        timeout=timeout_s,
        collate_fn=lambda x: x,
    )

    it = iter(dl)
    try:
        out1 = run_with_timeout(lambda: next(it), timeout=2.0)
        assert out1 == 1

        with pytest.raises(RuntimeError) as exc1:
            run_with_timeout(lambda: next(it), timeout=2.0)
        assert str(exc1.value) == f"DataLoader timed out after {timeout_s} seconds"

        with pytest.raises(RuntimeError) as exc2:
            run_with_timeout(lambda: next(it), timeout=0.5)
        assert exc2.value is exc1.value

    finally:
        ds.release0.set()
        it.close()
        _wait_workers_stop(it._workers)


def test_dataloader_timeout_in_order_buffers_but_times_out():
    import vibetensor.torch.utils.data as vtd

    ds = _TimeoutDS()
    timeout_s = 0.5

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=True,
        timeout=timeout_s,
        collate_fn=lambda x: x,
    )

    it = iter(dl)
    try:
        with pytest.raises(RuntimeError) as exc1:
            run_with_timeout(lambda: next(it), timeout=2.0)
        assert str(exc1.value) == f"DataLoader timed out after {timeout_s} seconds"

        with pytest.raises(RuntimeError) as exc2:
            run_with_timeout(lambda: next(it), timeout=0.5)
        assert exc2.value is exc1.value

    finally:
        ds.release0.set()
        it.close()
        _wait_workers_stop(it._workers)


def test_dataloader_close_unblocks_blocked_next_and_is_terminal():
    import vibetensor.torch.utils.data as vtd

    ds = _BlockingDS()

    dl = vtd.DataLoader(
        ds,
        num_workers=1,
        batch_size=None,
        in_order=True,
        collate_fn=lambda x: x,
    )

    it = iter(dl)

    done = threading.Event()
    out: dict[str, object] = {}

    def _t_next() -> None:
        try:
            out["value"] = next(it)
        except BaseException as e:
            out["exc"] = e
        finally:
            done.set()

    t = threading.Thread(target=_t_next, daemon=True)
    t.start()

    try:
        assert ds.started0.wait(timeout=2.0)
        assert not done.is_set()

        it.close()

        assert done.wait(timeout=2.0)

        assert "value" not in out
        exc = out.get("exc")
        assert isinstance(exc, StopIteration)

        # close() is terminal: subsequent next() must stop immediately.
        with pytest.raises(StopIteration):
            run_with_timeout(lambda: next(it), timeout=0.5)

    finally:
        ds.release0.set()
        it.close()
        _wait_workers_stop(it._workers)
