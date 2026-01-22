# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
import vibetensor.torch.ops as ops


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


def test_default_kernels_no_identity_for_complex_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    a = vt.tensor([1.0 + 2.0j, 3.0 - 4.0j], dtype="complex64")
    b = vt.tensor([0.5 - 1.0j, -2.0 + 0.25j], dtype="complex64")

    a_np = _to_numpy_cpu(a)
    b_np = _to_numpy_cpu(b)

    # These should have real implementations on CPU for complex dtypes.
    out_add = ops.vt.add(a, b)
    out_mul = ops.vt.mul(a, b)

    np.testing.assert_allclose(_to_numpy_cpu(out_add), a_np + b_np)
    np.testing.assert_allclose(_to_numpy_cpu(out_mul), a_np * b_np)

    # Everything else should raise rather than silently returning an identity.
    with pytest.raises(ValueError, match=r"unsupported dtype"):
        _ = ops.vt.sub(a, b)
    with pytest.raises(ValueError, match=r"unsupported dtype"):
        _ = ops.vt.div(a, b)
    with pytest.raises(ValueError, match=r"unsupported dtype"):
        _ = ops.vt.abs(a)
    with pytest.raises(ValueError, match=r"unsupported dtype"):
        _ = ops.vt.neg(a)
    with pytest.raises(ValueError, match=r"unsupported dtype"):
        _ = ops.vt.reciprocal(a)
    with pytest.raises(ValueError, match=r"unsupported dtype"):
        _ = ops.vt.relu(a)
