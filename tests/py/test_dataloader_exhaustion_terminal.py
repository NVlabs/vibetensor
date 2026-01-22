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


class _DS:
    def __init__(self, n: int):
        self._n = int(n)

    def __len__(self):
        return int(self._n)

    def __getitem__(self, idx: int):
        return int(idx)


def test_dataloader_exhaustion_is_terminal_and_non_blocking_and_joins_workers():
    import vibetensor.torch.utils.data as vtd

    N = 16
    ds = _DS(N)

    dl = vtd.DataLoader(
        ds,
        num_workers=2,
        batch_size=None,
        in_order=False,
        collate_fn=lambda x: x,
    )

    it = iter(dl)

    out = run_with_timeout(lambda: list(it), timeout=4.0)
    assert sorted(out) == list(range(N))

    with pytest.raises(StopIteration):
        run_with_timeout(lambda: next(it), timeout=0.5)

    deadline = time.monotonic() + 1.0
    while any(t.is_alive() for t in it._workers) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert all(not t.is_alive() for t in it._workers)
