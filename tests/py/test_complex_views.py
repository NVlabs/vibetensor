# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


def _to_numpy_cpu(t):
    cap = vt.to_dlpack(t)
    try:
        return np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        # Older NumPy expects a provider with __dlpack__.
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover
                return self._inner

        return np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]


def test_view_as_real_and_back_roundtrip_complex64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    x_np = np.array([[1.0 + 2.0j, 3.0 - 4.0j]], dtype=np.complex64)
    x = vt.from_numpy(x_np)

    xr = vt.view_as_real(x)
    xr_np = _to_numpy_cpu(xr)
    assert xr_np.shape == x_np.shape + (2,)

    expected = np.stack([x_np.real, x_np.imag], axis=-1)
    np.testing.assert_allclose(xr_np, expected)

    x2 = vt.view_as_complex(xr)
    x2_np = _to_numpy_cpu(x2)
    assert x2_np.shape == x_np.shape

    np.testing.assert_allclose(x2_np, x_np)


def test_view_as_real_is_zero_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    x = vt.from_numpy(np.array([[1.0 + 2.0j]], dtype=np.complex64))
    r = x.view_as_real()

    # Mutate through the real view; should reflect in the original complex tensor.
    r[0, 0, 0] = 7.0
    r[0, 0, 1] = -8.0

    x_np2 = _to_numpy_cpu(x)
    np.testing.assert_allclose(x_np2[0, 0], 7.0 - 8.0j)

    c = r.view_as_complex()
    c_np = _to_numpy_cpu(c)
    np.testing.assert_allclose(c_np[0, 0], 7.0 - 8.0j)


def test_view_as_complex_is_zero_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    a = vt.from_numpy(np.array([[[0.0, 0.0]]], dtype=np.float32))
    c = a.view_as_complex()

    # Mutate through the real[...,2] input; should reflect in the complex view.
    a[0, 0, 0] = 9.0
    a[0, 0, 1] = 10.0

    c_np = _to_numpy_cpu(c)
    assert c_np.shape == (1, 1)
    assert c_np.dtype == np.complex64
    np.testing.assert_allclose(c_np[0, 0], 9.0 + 10.0j)


def test_view_as_complex_allows_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    a = vt.from_numpy(np.empty((0, 2), dtype=np.float32))
    c = a.view_as_complex()
    c_np = _to_numpy_cpu(c)

    assert c_np.shape == (0,)
    assert c_np.dtype == np.complex64


def test_view_as_real_allows_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    x = vt.from_numpy(np.empty((0,), dtype=np.complex64))
    xr = x.view_as_real()
    xr_np = _to_numpy_cpu(xr)

    assert xr_np.shape == (0, 2)
    assert xr_np.dtype == np.float32


def test_conj_toggle_resolve_and_dlpack_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    rng = np.random.default_rng(0)
    x_np = (rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3))).astype(
        np.complex128
    )
    x = vt.from_numpy(x_np)

    xc = x.conj()

    with pytest.raises(
        RuntimeError,
        match=r"to_dlpack: cannot export conjugated complex tensor; call resolve_conj\(\)",
    ):
        vt.to_dlpack(xc)

    # Toggle back.
    xcc = xc.conj()
    _ = vt.to_dlpack(xcc)

    # resolve_conj() materializes the conjugation and clears the conj bit.
    x_res = xc.resolve_conj()
    x_res_np = _to_numpy_cpu(x_res)
    assert x_res_np.shape == x_np.shape

    np.testing.assert_allclose(x_res_np, np.conj(x_np))


def test_view_as_real_rejects_conj(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    x_np = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    x = vt.from_numpy(x_np)

    with pytest.raises(Exception, match=r"call resolve_conj\(\)"):
        _ = x.conj().view_as_real()
