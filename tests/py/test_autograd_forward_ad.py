# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import numpy as np

import vibetensor.torch as vt
from vibetensor.autograd import forward_ad as fad


def _cpu_float32_vector(vals):
    return vt.tensor(vals, dtype="float32")


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


def test_basic_dual_add_clears_on_exit():
    x = _cpu_float32_vector([1.0, 2.0, 3.0])
    t = vt.ones_like(x)

    with fad.dual_level() as lvl:
        x_dual = fad.make_dual(x, t, level=lvl)
        y_dual = vt.ops.vt.add(x_dual, x_dual)
        y, y_t = fad.unpack_dual(y_dual, level=lvl)

    assert y_t is not None
    _assert_allclose(y, _cpu_float32_vector([2.0, 4.0, 6.0]))
    _assert_allclose(y_t, _cpu_float32_vector([2.0, 2.0, 2.0]))

    # After exit, tangents are cleared for this level.
    y2, y2_t = fad.unpack_dual(y_dual)
    assert y2_t is None
    _assert_allclose(y2, y)


def test_nested_dual_level_is_rejected():
    with fad.dual_level():
        with pytest.raises(RuntimeError) as excinfo:
            with fad.dual_level():  # type: ignore[call-arg]
                pass
        msg = str(excinfo.value)
        assert "nested forward-mode" in msg or "nested" in msg


def test_make_dual_requires_active_level_and_cpu_float32():
    x = _cpu_float32_vector([1.0])
    t = vt.ones_like(x)

    # No active level.
    with pytest.raises(RuntimeError):
        fad.make_dual(x, t)

    with fad.dual_level() as lvl:
        # Non-tensor primals/tangents.
        with pytest.raises(TypeError):
            fad.make_dual(1.0, t, level=lvl)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            fad.make_dual(x, 1.0, level=lvl)  # type: ignore[arg-type]

        # Device/dtype/shape mismatches.
        y = vt.tensor([1.0, 2.0], dtype="float32")
        with pytest.raises(RuntimeError):
            fad.make_dual(y, t, level=lvl)

        z = vt.tensor([1.0], dtype="int32")
        with pytest.raises(RuntimeError):
            fad.make_dual(z, t, level=lvl)


def test_dual_level_disallowed_inside_inference_mode():
    x = _cpu_float32_vector([1.0])
    t = vt.ones_like(x)

    with vt.inference_mode():
        with pytest.raises(RuntimeError):
            with fad.dual_level():
                fad.make_dual(x, t)


def test_make_dual_disallowed_while_inference_mode_enabled():
    x = _cpu_float32_vector([1.0])
    t = vt.ones_like(x)

    with fad.dual_level() as lvl:
        with vt.inference_mode():
            with pytest.raises(RuntimeError) as excinfo:
                fad.make_dual(x, t, level=lvl)
        # After inference_mode block, no tangent should have been attached.
        x_detached, t_attached = fad.unpack_dual(x, level=lvl)
        assert t_attached is None
        _assert_allclose(x_detached, x)



def test_mixed_forward_and_reverse_for_square():
    from vibetensor.torch import autograd as vt_autograd

    with fad.dual_level() as lvl:
        x = _cpu_float32_vector([1.0, 2.0])
        x.requires_grad = True
        t = vt.ones_like(x)
        x_dual = fad.make_dual(x, t, level=lvl)
        y_dual = vt.ops.vt.add(x_dual, x_dual)
        y, y_t = fad.unpack_dual(y_dual, level=lvl)

    grad_in = vt.ones_like(y_dual)
    (gx,) = vt_autograd.grad(y_dual, (x,), grad_outputs=grad_in)

    two = _cpu_float32_vector([2.0, 2.0])
    expected = two
    _assert_allclose(y_t, expected)
    _assert_allclose(gx, expected)
