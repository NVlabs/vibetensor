# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

def test_cpu_repr_nd_empty_size_and_dtype():
    # Create (2,1) then narrow to length 0 along dim=1 to get (2,0)
    base = _make_cpu_tensor(2, "int64").view((2, 1))
    t = base.narrow(1, 0, 0)
    r = repr(t)
    assert "size=(2, 0)" in r
    assert ", dtype=int64" in r


def test_cpu_repr_non_contiguous_view_prints_correctly():
    # Build (2,3) from 1D then transpose to make it non-contiguous
    base = _make_cpu_tensor(6, "int64").view((2, 3))
    b = base.transpose(0, 1)
    # Should still not include dtype (non-empty int64 suppressed) and no device on cpu default
    with device_default_env("cpu"):
        r = repr(b)
        assert r.startswith("tensor([") and r.endswith(")")
        assert "dtype=" not in r


def test_printoptions_affect_repr_threshold_and_precision():
    # Large array to trigger summarization when threshold is low
    vt.set_printoptions(threshold=5, edgeitems=2, precision=1)
    t = _make_cpu_tensor(10, "float32")
    s = repr(t)
    # Expect ellipsis due to summarization and low precision formatting
    assert "..." in s
    # restore defaults for other tests
    vt.set_printoptions(precision=4, threshold=1000, edgeitems=3)
import re

import vibetensor.torch as vt
from vibetensor import _C as C


def device_default_env(value: str):
    class _Env:
        def __enter__(self):
            self._prev = os.environ.get("VBT_DEFAULT_DEVICE_TYPE")
            os.environ["VBT_DEFAULT_DEVICE_TYPE"] = value
            return self
        def __exit__(self, exc_type, exc, tb):
            if self._prev is None:
                os.environ.pop("VBT_DEFAULT_DEVICE_TYPE", None)
            else:
                os.environ["VBT_DEFAULT_DEVICE_TYPE"] = self._prev
    return _Env()


def test_printoptions_get_set_defaults():
    opts = vt.get_printoptions()
    assert set(opts.keys()) == {"precision", "threshold", "edgeitems", "linewidth", "sci_mode"}
    # Change a couple options and verify roundtrip
    vt.set_printoptions(precision=2, edgeitems=1)
    opts2 = vt.get_printoptions()
    assert opts2["precision"] == 2
    assert opts2["edgeitems"] == 1


def _make_cpu_tensor(n: int, dtype: str):
    cap = C._make_cpu_dlpack_1d_dtype(n, dtype)
    return vt.from_dlpack(cap)


def test_cpu_repr_non_empty_dtype_suppression():
    # float32 non-empty: suppress dtype
    t = _make_cpu_tensor(3, "float32")
    r = repr(t)
    assert r.startswith("tensor([") and r.endswith(")")
    assert "dtype=" not in r
    # int64 non-empty: suppress dtype
    t2 = _make_cpu_tensor(2, "int64")
    r2 = repr(t2)
    assert "dtype=" not in r2
    # bool non-empty: suppress dtype
    t3 = _make_cpu_tensor(2, "bool")
    r3 = repr(t3)
    assert "dtype=" not in r3


def test_cpu_repr_empty_rules():
    # Empty 1D int64: dtype printed, no size
    t = _make_cpu_tensor(0, "int64")
    r = repr(t)
    assert r.startswith("tensor([]")
    assert "size=" not in r
    assert ", dtype=int64" in r
    # Empty 1D float32: dtype suppressed
    t2 = _make_cpu_tensor(0, "float32")
    r2 = repr(t2)
    assert r2.startswith("tensor([]")
    assert ", dtype=" not in r2


def test_cpu_repr_device_suffix_flips_with_default():
    t = _make_cpu_tensor(2, "int64")
    with device_default_env("cpu"):
        assert C._get_default_device_type() == "cpu"
        r = repr(t)
        assert "device='cpu'" not in r
    with device_default_env("cuda"):
        assert C._get_default_device_type() == "cuda"
        r = repr(t)
        assert "device='cpu'" in r
