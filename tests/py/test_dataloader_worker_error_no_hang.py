# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import threading

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
            self._started0.wait()
            return 1
        raise IndexError(idx)


def test_dataloader_worker_error_propagates_and_is_terminal_and_non_blocking():
    import vibetensor.torch.utils.data as vtd

    ds = _ErrDS()

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=True,
        collate_fn=lambda x: x,
    )

    it = iter(dl)
    try:
        with pytest.raises(RuntimeError) as exc1:
            run_with_timeout(lambda: next(it), timeout=2.0)

        msg = str(exc1.value)
        assert "DataLoader worker" in msg
        assert re.search(r"worker\s+\d+", msg)
        assert "ValueError" in msg
        assert "boom" in msg

        with pytest.raises(RuntimeError) as exc2:
            run_with_timeout(lambda: next(it), timeout=0.5)

        # Terminal replay: the same stored RuntimeError instance is re-raised.
        assert exc2.value is exc1.value

    finally:
        it.close()
