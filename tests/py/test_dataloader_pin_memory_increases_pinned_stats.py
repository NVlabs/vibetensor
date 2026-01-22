# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import numpy as np


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


class _DS:
    def __len__(self):
        return 4

    def __getitem__(self, idx: int):
        idx = int(idx)
        return np.asarray([idx, idx + 1, idx + 2, idx + 3], dtype=np.float32)


def _max_pinned_allocated_bytes() -> int:
    from vibetensor import _C as C

    return int(C._cpu_getHostPinnedStats()[2])


def test_dataloader_pin_memory_increases_pinned_stats_single_process():
    import vibetensor.torch.utils.data as vtd
    from vibetensor import _C as C

    C._cpu_resetPeakHostPinnedStats()
    base = _max_pinned_allocated_bytes()

    dl = vtd.DataLoader(_DS(), batch_size=None, pin_memory=True)
    out = next(iter(dl))
    np.testing.assert_allclose(out.numpy(), np.asarray([0, 1, 2, 3], dtype=np.float32))

    assert _max_pinned_allocated_bytes() > base


def test_dataloader_pin_memory_increases_pinned_stats_threaded():
    import vibetensor.torch.utils.data as vtd
    from vibetensor import _C as C

    C._cpu_resetPeakHostPinnedStats()
    base = _max_pinned_allocated_bytes()

    dl = vtd.DataLoader(
        _DS(),
        num_workers=2,
        batch_size=None,
        pin_memory=True,
        timeout=0.0,
    )

    it = iter(dl)
    try:
        out = run_with_timeout(lambda: next(it), timeout=2.0)
        assert out is not None
        np.testing.assert_allclose(out.numpy(), np.asarray([0, 1, 2, 3], dtype=np.float32))
        assert _max_pinned_allocated_bytes() > base
    finally:
        it.close()
