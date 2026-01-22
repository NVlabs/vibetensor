# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt


def _to_numpy_cpu(t):
    cap = vt.to_dlpack(t)
    try:
        arr = np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        # Older NumPy expects a provider with __dlpack__.
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover
                return self._inner

        arr = np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]
    return arr.reshape(tuple(int(s) for s in t.sizes))


def test_vt_add_complex64_cpu_matches_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    a_np = np.array([[1.0 + 2.0j], [3.5 - 4.25j]], dtype=np.complex64)
    b_np = np.array([[5.0 - 1.0j, 6.0 + 0.0j, -1.0 + 2.0j]], dtype=np.complex64)

    a = vt.from_numpy(a_np)
    b = vt.from_numpy(b_np)

    out = vt.add(a, b)
    out_np = _to_numpy_cpu(out)

    np.testing.assert_allclose(out_np, a_np + b_np)


def test_vt_mul_complex128_cpu_matches_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((2, 1)) + 1j * rng.standard_normal((2, 1))).astype(np.complex128)
    b_np = (rng.standard_normal((1, 3)) + 1j * rng.standard_normal((1, 3))).astype(np.complex128)

    a = vt.from_numpy(a_np)
    b = vt.from_numpy(b_np)

    out = vt.mul(a, b)
    out_np = _to_numpy_cpu(out)

    np.testing.assert_allclose(out_np, a_np * b_np)
