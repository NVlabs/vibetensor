# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections import namedtuple

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


def test_dataloader_pin_memory_structure_preserved():
    import vibetensor.torch.utils.data as vtd
    from vibetensor import _C as C

    Pair = namedtuple("Pair", ["x", "y"])

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return {
                "a": np.asarray([1, 2], dtype=np.float32),
                "b": [np.asarray([3], dtype=np.int64), (np.asarray([4, 5], dtype=np.float32),)],
                "c": Pair(np.asarray([6], dtype=np.float32), np.asarray([7], dtype=np.float32)),
                "d": "keep_me",
            }

    C._cpu_resetPeakHostPinnedStats()
    base = int(C._cpu_getHostPinnedStats()[2])

    dl = vtd.DataLoader(_DS(), batch_size=None, pin_memory=True)
    out = next(iter(dl))

    assert type(out) is dict
    assert set(out.keys()) == {"a", "b", "c", "d"}

    assert type(out["b"]) is list
    assert type(out["b"][1]) is tuple

    assert type(out["c"]) is Pair
    assert out["d"] == "keep_me"

    np.testing.assert_allclose(out["a"].numpy(), np.asarray([1, 2], dtype=np.float32))
    np.testing.assert_array_equal(out["b"][0].numpy(), np.asarray([3], dtype=np.int64))
    np.testing.assert_allclose(out["b"][1][0].numpy(), np.asarray([4, 5], dtype=np.float32))
    np.testing.assert_allclose(out["c"].x.numpy(), np.asarray([6], dtype=np.float32))
    np.testing.assert_allclose(out["c"].y.numpy(), np.asarray([7], dtype=np.float32))

    assert int(C._cpu_getHostPinnedStats()[2]) > base


def test_dataloader_pin_memory_cuda_tensor_leaf_raises_typeerror_single_process():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch as vt
    import vibetensor.torch.utils.data as vtd

    t_cuda = vt.cuda.to_device(np.asarray([1, 2], dtype=np.float32))

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return t_cuda

    dl = vtd.DataLoader(_DS(), batch_size=None, pin_memory=True, collate_fn=lambda x: x)

    with pytest.raises(TypeError) as exc:
        _ = next(iter(dl))

    assert "DataLoader: pin_memory expects CPU tensors" in str(exc.value)


def test_dataloader_pin_memory_cuda_tensor_leaf_raises_typeerror_threaded():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch as vt
    import vibetensor.torch.utils.data as vtd

    t_cuda = vt.cuda.to_device(np.asarray([1, 2], dtype=np.float32))

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return t_cuda

    dl = vtd.DataLoader(
        _DS(),
        num_workers=1,
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
        timeout=0.0,
    )

    it = iter(dl)
    try:
        with pytest.raises(TypeError) as exc:
            run_with_timeout(lambda: next(it), timeout=2.0)

        assert "DataLoader: pin_memory expects CPU tensors" in str(exc.value)

    finally:
        it.close()
