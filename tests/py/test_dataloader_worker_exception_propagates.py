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


def _wait_workers_stop(workers, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while any(w.is_alive() for w in workers) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert all(not w.is_alive() for w in workers)


class _ErrDS:
    def __init__(self):
        self._started0 = threading.Event()

    def __len__(self):
        return 2

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx == 0:
            # Let idx=1 proceed only after idx=0 has started, to make the
            # completion order deterministic (idx=1 may complete first).
            self._started0.set()
            raise ValueError("boom")
        if idx == 1:
            if not self._started0.wait(timeout=2.0):
                raise RuntimeError("test setup failed: idx=0 never started")
            return 1
        raise IndexError(idx)


def test_dataloader_worker_exception_propagates_and_is_terminal():
    import vibetensor.torch.utils.data as vtd

    ds = _ErrDS()

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=True,
        timeout=0.0,
        collate_fn=lambda x: x,
    )

    it = iter(dl)
    try:
        with pytest.raises(RuntimeError) as exc1:
            run_with_timeout(lambda: next(it), timeout=2.0)

        msg = str(exc1.value)
        assert "DataLoader worker" in msg
        assert "ValueError" in msg
        assert "boom" in msg

        with pytest.raises(RuntimeError) as exc2:
            run_with_timeout(lambda: next(it), timeout=0.5)

        # Terminal replay: the same stored RuntimeError instance is re-raised.
        assert exc2.value is exc1.value

    finally:
        it.close()
        _wait_workers_stop(it._workers)


class BadReprExc(Exception):
    def __repr__(self) -> str:
        raise RuntimeError("nope")


class _BadReprDS:
    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        raise BadReprExc("boom")


def test_dataloader_worker_exception_bad_repr_is_safe():
    import vibetensor.torch.utils.data as vtd

    dl = vtd.DataLoader(
        _BadReprDS(),
        num_workers=1,
        batch_size=None,
        in_order=True,
        timeout=0.0,
        collate_fn=lambda x: x,
    )

    it = iter(dl)
    try:
        with pytest.raises(RuntimeError) as exc1:
            run_with_timeout(lambda: next(it), timeout=2.0)

        msg = str(exc1.value)
        assert "BadReprExc" in msg
        assert "<repr failed>" in msg

        with pytest.raises(RuntimeError) as exc2:
            run_with_timeout(lambda: next(it), timeout=0.5)
        assert exc2.value is exc1.value

    finally:
        it.close()
        _wait_workers_stop(it._workers)


class _SingleErrDS:
    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        raise ValueError("boom")


def test_dataloader_close_does_not_overwrite_existing_terminal_error():
    import vibetensor.torch.utils.data as vtd

    dl = vtd.DataLoader(
        _SingleErrDS(),
        num_workers=1,
        batch_size=None,
        in_order=True,
        timeout=0.0,
        collate_fn=lambda x: x,
    )

    it = iter(dl)
    try:
        with pytest.raises(RuntimeError) as exc1:
            run_with_timeout(lambda: next(it), timeout=2.0)

        it.close()

        with pytest.raises(RuntimeError) as exc2:
            run_with_timeout(lambda: next(it), timeout=0.5)

        assert exc2.value is exc1.value

    finally:
        it.close()
        _wait_workers_stop(it._workers)
