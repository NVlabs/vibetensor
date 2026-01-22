# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _np(t):
    return np.from_dlpack(t)


def _item(t):
    return np.from_dlpack(vt.to_dlpack(t)).item()


def test_complex_requires_grad_rejected_when_autograd_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX_AUTOGRAD", "0")

    x = vt.tensor([1.0 + 2.0j], dtype="complex64")
    with pytest.raises(RuntimeError) as exc:
        x.requires_grad = True
    assert "complex autograd is disabled" in str(exc.value)


def test_autograd_add_complex64_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX_AUTOGRAD", "1")

    a = vt.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype="complex64")
    b = vt.tensor([-0.5 + 0.25j, 2.0 + 0.0j], dtype="complex64")
    a.requires_grad = True
    b.requires_grad = True

    y = C.vt.add(a, b)
    grad = vt.full_like(y, 1.0 + 0.0j)
    y.backward(grad)

    assert a.grad is not None
    assert b.grad is not None

    np.testing.assert_allclose(_np(a.grad), _np(grad))
    np.testing.assert_allclose(_np(b.grad), _np(grad))


def test_autograd_mul_complex64_cpu_conjugate_wirtinger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX_AUTOGRAD", "1")

    a = vt.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype="complex64")
    b = vt.tensor([-0.5 + 0.25j, 2.0 + 0.0j], dtype="complex64")
    a.requires_grad = True
    b.requires_grad = True

    y = C.vt.mul(a, b)
    grad = vt.full_like(y, 0.75 - 0.125j)
    y.backward(grad)

    assert a.grad is not None
    assert b.grad is not None

    grad_np = _np(grad)
    a_np = _np(a)
    b_np = _np(b)

    np.testing.assert_allclose(_np(a.grad), grad_np * np.conj(b_np))
    np.testing.assert_allclose(_np(b.grad), grad_np * np.conj(a_np))


def test_autograd_mul_complex128_cpu_conjugate_wirtinger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX_AUTOGRAD", "1")

    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal(4) + 1j * rng.standard_normal(4)).astype(np.complex128)
    b_np = (rng.standard_normal(4) + 1j * rng.standard_normal(4)).astype(np.complex128)

    a = vt.tensor(a_np, dtype="complex128")
    b = vt.tensor(b_np, dtype="complex128")
    a.requires_grad = True
    b.requires_grad = True

    y = C.vt.mul(a, b)
    grad = vt.full_like(y, 0.5 + 0.25j)
    y.backward(grad)

    assert a.grad is not None
    assert b.grad is not None

    grad_np = _np(grad)
    a_np2 = _np(a)
    b_np2 = _np(b)

    np.testing.assert_allclose(_np(a.grad), grad_np * np.conj(b_np2))
    np.testing.assert_allclose(_np(b.grad), grad_np * np.conj(a_np2))


def test_autograd_mul_complex_duplicate_leaf_accumulates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX_AUTOGRAD", "1")

    x = vt.tensor([1.5 - 0.25j, -0.5 + 2.0j], dtype="complex64")
    x.requires_grad = True

    y = C.vt.mul(x, x)
    grad = vt.full_like(y, 1.0 + 0.0j)
    y.backward(grad)

    assert x.grad is not None

    grad_np = _np(grad)
    x_np = _np(x)

    expected = 2.0 * grad_np * np.conj(x_np)
    np.testing.assert_allclose(_np(x.grad), expected)


def test_backward_implicit_grad_does_not_mutate_scalar_root_float32() -> None:
    x = vt.tensor([2.0], dtype="float32")
    x.requires_grad = True

    y = C.vt.mul(x, x)
    loss = y.sum()

    before = _item(loss)
    loss.backward()
    after = _item(loss)

    assert before == after


def test_backward_implicit_grad_does_not_mutate_scalar_root_complex64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")
    monkeypatch.setenv("VBT_ENABLE_COMPLEX_AUTOGRAD", "1")

    x = vt.tensor(2.0 + 3.0j, dtype="complex64")
    x.requires_grad = True

    y = C.vt.mul(x, x)

    before = _item(y)
    y.backward()
    after = _item(y)

    assert before == after
