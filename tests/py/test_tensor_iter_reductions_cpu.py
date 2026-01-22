# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C

try:
    import torch
except Exception:  # pragma: no cover - torch is an optional test dependency
    torch = None


def _skip_if_no_torch() -> None:
    if torch is None:
        pytest.skip("torch not available for tensor_iter reduction parity tests")


def _to_numpy_vt(t) -> np.ndarray:
    """Convert a VibeTensor tensor to a NumPy array via DLPack.

    Uses torch when available (supports CUDA tensors) and falls back to
    numpy.from_dlpack for CPU-only environments.
    """
    cap = vt.to_dlpack(t)
    if torch is not None:
        return torch.utils.dlpack.from_dlpack(cap).cpu().numpy()  # type: ignore[union-attr]
    return np.from_dlpack(cap)


def _make_inputs(shape, dtype: str):
    rng = np.random.default_rng(0)
    if dtype == "float32":
        arr = rng.standard_normal(shape).astype(np.float32)
        t_torch = torch.from_numpy(arr)
    elif dtype == "int64":
        arr = rng.integers(-5, 5, size=shape, dtype=np.int64)
        t_torch = torch.from_numpy(arr)
    else:  # pragma: no cover - defensive
        raise AssertionError("unsupported dtype")
    t_vt = vt.from_numpy(arr)
    return t_torch, t_vt


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (0, 3), (2, 0, 3)])
def test_sum_mean_dim_none_and_dim_int_float32(shape):
    _skip_if_no_torch()
    t_torch, t_vt = _make_inputs(shape, dtype="float32")

    # dim=None, keepdim=False
    out_t = t_torch.sum()
    out_v = t_vt.sum()
    np.testing.assert_allclose(_to_numpy_vt(out_v), out_t.numpy(), rtol=1e-6, atol=1e-6)

    out_t = t_torch.mean()
    out_v = t_vt.mean()
    np.testing.assert_allclose(
        _to_numpy_vt(out_v), out_t.numpy(), rtol=1e-6, atol=1e-6, equal_nan=True
    )

    with pytest.raises(ValueError):
        t_vt.sum(dim=None, keepdim=True)
    with pytest.raises(ValueError):
        t_vt.mean(dim=None, keepdim=True)

    if t_torch.dim() > 0:
        D = t_torch.dim()
        for dim in range(-D, D):
            out_t = t_torch.sum(dim=dim, keepdim=False)
            out_v = t_vt.sum(dim=dim, keepdim=False)
            np.testing.assert_allclose(
                _to_numpy_vt(out_v), out_t.numpy(), rtol=1e-6, atol=1e-6
            )

            out_t = t_torch.sum(dim=dim, keepdim=True)
            out_v = t_vt.sum(dim=dim, keepdim=True)
            np.testing.assert_allclose(
                _to_numpy_vt(out_v), out_t.numpy(), rtol=1e-6, atol=1e-6
            )

            out_t = t_torch.mean(dim=dim, keepdim=False)
            out_v = t_vt.mean(dim=dim, keepdim=False)
            np.testing.assert_allclose(
                _to_numpy_vt(out_v), out_t.numpy(), rtol=1e-6, atol=1e-6, equal_nan=True
            )


def test_mean_unsupported_dtype_raises_value_error():
    _skip_if_no_torch()
    t_torch, t_vt = _make_inputs((4, 3), dtype="int64")
    with pytest.raises(ValueError):
        _ = t_vt.mean()


@pytest.mark.parametrize("dtype", ["float32", "int64"])
@pytest.mark.parametrize("shape, dim", [((4,), 0), ((2, 3), 0), ((2, 3), 1)])
def test_amin_amax_dim_int_and_none(dtype, shape, dim):
    _skip_if_no_torch()
    t_torch, t_vt = _make_inputs(shape, dtype=dtype)

    # dim=int, keepdim variants
    out_t = t_torch.amin(dim=dim, keepdim=False)
    out_v = t_vt.amin(dim=dim, keepdim=False)
    np.testing.assert_array_equal(_to_numpy_vt(out_v), out_t.numpy())

    out_t = t_torch.amin(dim=dim, keepdim=True)
    out_v = t_vt.amin(dim=dim, keepdim=True)
    np.testing.assert_array_equal(_to_numpy_vt(out_v), out_t.numpy())

    out_t = t_torch.amax(dim=dim, keepdim=False)
    out_v = t_vt.amax(dim=dim, keepdim=False)
    np.testing.assert_array_equal(_to_numpy_vt(out_v), out_t.numpy())

    out_t = t_torch.amax(dim=dim, keepdim=True)
    out_v = t_vt.amax(dim=dim, keepdim=True)
    np.testing.assert_array_equal(_to_numpy_vt(out_v), out_t.numpy())

    # dim=None full reductions
    out_t = t_torch.amin()
    out_v = t_vt.amin()
    np.testing.assert_array_equal(_to_numpy_vt(out_v), out_t.numpy())

    out_t = t_torch.amax()
    out_v = t_vt.amax()
    np.testing.assert_array_equal(_to_numpy_vt(out_v), out_t.numpy())


@pytest.mark.parametrize("op_name", ["amin", "amax"])
@pytest.mark.parametrize("shape", [(0,), (0, 3)])
def test_amin_amax_empty_raises_runtime_error(op_name, shape):
    _skip_if_no_torch()
    t_torch, t_vt = _make_inputs(shape, dtype="float32")
    op_vt = getattr(t_vt, op_name)
    with pytest.raises(RuntimeError, match="empty"):
        op_vt()


@pytest.mark.parametrize("dtype", ["float32", "int64"])
@pytest.mark.parametrize("shape, dim", [((4,), 0), ((2, 3), 0), ((2, 3), 1)])
def test_argmin_argmax_and_indices(dtype, shape, dim):
    _skip_if_no_torch()
    t_torch, t_vt = _make_inputs(shape, dtype=dtype)

    # argmax / argmin indices
    idx_t = t_torch.argmax(dim=dim)
    idx_v = t_vt.argmax(dim=dim)
    np.testing.assert_array_equal(_to_numpy_vt(idx_v), idx_t.numpy())

    idx_t = t_torch.argmin(dim=dim)
    idx_v = t_vt.argmin(dim=dim)
    np.testing.assert_array_equal(_to_numpy_vt(idx_v), idx_t.numpy())

    # max / min pair-returning forms
    val_t, idx_t = t_torch.max(dim=dim, keepdim=False)
    val_v, idx_v = t_vt.max(dim=dim, keepdim=False)
    np.testing.assert_array_equal(_to_numpy_vt(val_v), val_t.numpy())
    np.testing.assert_array_equal(_to_numpy_vt(idx_v), idx_t.numpy())

    val_t, idx_t = t_torch.min(dim=dim, keepdim=False)
    val_v, idx_v = t_vt.min(dim=dim, keepdim=False)
    np.testing.assert_array_equal(_to_numpy_vt(val_v), val_t.numpy())
    np.testing.assert_array_equal(_to_numpy_vt(idx_v), idx_t.numpy())


def test_argmax_argmin_dim_none_behavior_matches_torch():
    _skip_if_no_torch()
    t_torch, t_vt = _make_inputs((5,), dtype="float32")

    idx_t = t_torch.argmax()
    idx_v = t_vt.argmax()
    np.testing.assert_array_equal(_to_numpy_vt(idx_v), idx_t.numpy())

    idx_t = t_torch.argmin()
    idx_v = t_vt.argmin()
    np.testing.assert_array_equal(_to_numpy_vt(idx_v), idx_t.numpy())


@pytest.mark.parametrize("shape", [(0,), (0, 4)])
@pytest.mark.parametrize("op_name", ["argmax", "argmin"])
def test_argmin_argmax_empty_raises_runtime_error(op_name, shape):
    _skip_if_no_torch()
    t_torch, t_vt = _make_inputs(shape, dtype="float32")
    op_vt = getattr(t_vt, op_name)
    with pytest.raises(RuntimeError, match="empty"):
        op_vt()


def test_elementwise_multi_output_ti_via_cpp_binding():
    """Smoke test: ensure vt.add still works after TI changes.

    This indirectly exercises the elementwise multi-output iterator path
    used by existing ops.
    """
    a = C.vt.unit()
    b = C.vt.unit()
    out = C.vt.add(a, b)
    np.testing.assert_allclose(_to_numpy_vt(out), _to_numpy_vt(a) + _to_numpy_vt(b))
