# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

import vibetensor.torch as vt
from vibetensor.autograd import forward_ad as fad
import vibetensor.autograd.functional as F


def _make_input() -> vt.Tensor:  # type: ignore[name-defined]
    return vt.tensor([1.0, 2.0, 3.0], dtype="float32")


def _to_numpy_cpu(t):
    cap = vt.to_dlpack(t)
    try:
        arr = np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover - tiny adapter
                return self._inner

        arr = np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]
    return arr.reshape(tuple(int(s) for s in t.sizes))


def _assert_allclose(a, b):
    np.testing.assert_allclose(_to_numpy_cpu(a), _to_numpy_cpu(b))


def _fn(x):
    return vt.ops.vt.mul(x, x).sum()


def test_vjp_ignores_forward_tangents():
    x = _make_input()

    # Baseline without tangents.
    y0, vjp0 = F.vjp(_fn, x)

    with fad.dual_level() as lvl:
        x2 = _make_input()
        t = vt.ones_like(x2)
        x2_dual = fad.make_dual(x2, t, level=lvl)
        y1_dual, vjp1 = F.vjp(_fn, x2_dual)
        y1, y1_t = fad.unpack_dual(y1_dual, level=lvl)

    # Functional helper should not have propagated tangents.
    assert y1_t is None
    _assert_allclose(y0, y1)
    _assert_allclose(vjp0, vjp1)


def test_jvp_ignores_forward_tangents():
    x = _make_input()
    t_dir = vt.ones_like(x)

    # Baseline without tangents.
    y0, jvp0 = F.jvp(_fn, x, t_dir)

    with fad.dual_level() as lvl:
        x2 = _make_input()
        t = vt.ones_like(x2)
        x2_dual = fad.make_dual(x2, t, level=lvl)
        y1_dual, jvp1 = F.jvp(_fn, x2_dual, t_dir)
        y1, y1_t = fad.unpack_dual(y1_dual, level=lvl)

    assert y1_t is None
    _assert_allclose(y0, y1)
    _assert_allclose(jvp0, jvp1)
