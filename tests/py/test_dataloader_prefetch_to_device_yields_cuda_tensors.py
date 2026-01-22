# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import numpy as np
import pytest


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


def _has_cuda() -> bool:
    from vibetensor import _C as C

    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_dataloader_prefetch_to_device_yields_cuda_tensors_single_process():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return {
                "x": np.asarray([1, 2], dtype=np.float32),
                "y": [np.asarray([3], dtype=np.int64)],
            }

    dl = vtd.DataLoader(_DS(), batch_size=None, prefetch_to_device=True, device=0)
    out = next(iter(dl))

    assert isinstance(out, dict)
    x = out["x"]
    y0 = out["y"][0]

    assert int(x.device[0]) in (2, 13)
    assert int(x.device[1]) == 0
    assert int(y0.device[0]) in (2, 13)
    assert int(y0.device[1]) == 0


def test_dataloader_prefetch_to_device_yields_cuda_tensors_threaded():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return {
                "x": np.asarray([1, 2], dtype=np.float32),
                "y": [np.asarray([3], dtype=np.int64)],
            }

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
    finally:
        it.close()

    assert isinstance(out, dict)
    x = out["x"]
    y0 = out["y"][0]

    assert int(x.device[0]) in (2, 13)
    assert int(x.device[1]) == 0
    assert int(y0.device[0]) in (2, 13)
    assert int(y0.device[1]) == 0
