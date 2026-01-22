# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


if not vt.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def _require_op(name: str) -> None:
    if not getattr(vt, "_has_vt_op", lambda _n: True)(name):
        pytest.skip(f"vt.{name} not available")


def _to_numpy_cuda(t, *, non_blocking: bool = False):
    return vt.cuda.from_device(t, non_blocking=non_blocking)


def test_cuda_complex64_add_mul_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_op("add")
    _require_op("mul")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    a = np.array([[1 + 2j], [3 + 4j]], dtype=np.complex64)  # (2,1)
    b = np.array([[5 + 6j, 7 + 8j]], dtype=np.complex64)  # (1,2)

    ta = vt.cuda.to_device(a)
    tb = vt.cuda.to_device(b)

    out_add = vt.add(ta, tb)
    out_mul = vt.mul(ta, tb)

    out_add_nb = _to_numpy_cuda(out_add, non_blocking=True)
    np.testing.assert_allclose(out_add_nb, a + b)
    assert out_add_nb.base is not None
    assert not out_add_nb.flags.owndata

    np.testing.assert_allclose(_to_numpy_cuda(out_mul), a * b)


def test_cuda_complex128_add_mul(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_op("add")
    _require_op("mul")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    a = np.array([1 + 2j, -3 + 4j, 5 - 6j], dtype=np.complex128)
    b = np.array([7 - 8j, 9 + 10j, -11 + 12j], dtype=np.complex128)

    ta = vt.cuda.to_device(a)
    tb = vt.cuda.to_device(b)

    out_add = vt.add(ta, tb)
    out_mul = vt.mul(ta, tb)

    np.testing.assert_allclose(_to_numpy_cuda(out_add), a + b)

    out_mul_nb = _to_numpy_cuda(out_mul, non_blocking=True)
    np.testing.assert_allclose(out_mul_nb, a * b)
    assert out_mul_nb.base is not None
    assert not out_mul_nb.flags.owndata


def test_cuda_complex_conj_resolve_and_clone(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_op("add")
    _require_op("mul")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    x = np.array([1 + 2j, 3 - 4j, -5 + 6j], dtype=np.complex64)
    t = vt.cuda.to_device(x)

    # clone() should work for complex CUDA now that clone_cuda supports it.
    t2 = t.clone()
    np.testing.assert_allclose(_to_numpy_cuda(t2), x)

    # conj() is lazy (metadata-only); resolve_conj() must materialize.
    tc = t.conj()
    tr = tc.resolve_conj()
    np.testing.assert_allclose(_to_numpy_cuda(tr), np.conj(x))

    tc_clone = tc.clone()
    np.testing.assert_allclose(_to_numpy_cuda(tc_clone), np.conj(x))

    # Elementwise ops should resolve conj inputs implicitly.
    out_add = vt.add(tc, tc)
    out_mul = vt.mul(tc, tc)
    np.testing.assert_allclose(_to_numpy_cuda(out_add), np.conj(x) + np.conj(x))
    np.testing.assert_allclose(_to_numpy_cuda(out_mul), np.conj(x) * np.conj(x))
