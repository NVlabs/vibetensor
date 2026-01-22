# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.autograd as A
import vibetensor.torch as vt
from vibetensor import _C as C


def _reset_engine_stats():
    ag = C.autograd
    ag.reset_stats()
    return ag


def _to_numpy(t):
    arr = np.from_dlpack(t)
    return arr.reshape(tuple(int(s) for s in t.sizes))


def _simple_scalar_fn(x):
    # Small quadratic scalar loss: sum(x ** 2).
    y = C.vt.mul(x, x)
    return y.sum()


def _oracle_grad_for_simple_scalar(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=None,
    create_graph: bool = False,
    allow_unused=None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
):
    """Analytical gradient oracle for ``_simple_scalar_fn``.

    This ignores ``outputs`` and options and returns a gradient computed via a
    separate finite-difference pass. It is used in tests to exercise the
    gradcheck algorithm without depending on engine support for scalar roots.
    """

    if isinstance(inputs, tuple):
        tensors = inputs
    else:
        tensors = (inputs,)

    assert len(tensors) == 1
    (x,) = tensors

    x_np = _to_numpy(x).astype(np.float64)
    eps = 1e-3

    grad_np = np.empty_like(x_np, dtype=np.float64)
    flat = x_np.reshape(-1)
    grad_flat = grad_np.reshape(-1)

    for k in range(flat.size):
        flat_plus = flat.copy()
        flat_minus = flat.copy()
        flat_plus[k] += eps
        flat_minus[k] -= eps

        plus = vt.tensor(flat_plus.reshape(x_np.shape), dtype="float32")
        minus = vt.tensor(flat_minus.reshape(x_np.shape), dtype="float32")

        with vt.no_grad():
            out_plus = _simple_scalar_fn(plus)
            out_minus = _simple_scalar_fn(minus)

        f_plus = float(_to_numpy(out_plus).reshape(-1)[0])
        f_minus = float(_to_numpy(out_minus).reshape(-1)[0])
        grad_flat[k] = (f_plus - f_minus) / (2.0 * eps)

    g = vt.tensor(grad_np.astype(np.float32), dtype="float32")
    return (g,)


# ---- GC1: happy path ---------------------------------------------------------


def test_gradcheck_gc1_happy_path_returns_true(monkeypatch) -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([0.5, -1.0, 2.0], dtype="float32")
    x.requires_grad = True

    # Use a simple oracle for analytical gradients so we can exercise the
    # gradcheck algorithm without relying on engine support for scalar roots.
    monkeypatch.setattr(A, "grad", _oracle_grad_for_simple_scalar)

    ok = A.gradcheck(_simple_scalar_fn, x)
    assert ok is True

    # When using the oracle stub, gradcheck does not touch engine stats or .grad.
    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert x.grad is None


# ---- GC2: input validation ---------------------------------------------------


def test_gradcheck_gc2_non_leaf_or_requires_grad_false_inputs_raise() -> None:
    x_leaf = vt.tensor([1.0, 2.0], dtype="float32")
    x_leaf.requires_grad = True
    # Non-leaf input built from a differentiable op.
    y = C.vt.mul(x_leaf, x_leaf)

    with pytest.raises(A.GradcheckError) as exc1:
        A.gradcheck(_simple_scalar_fn, y)
    assert str(exc1.value) == (
        "gradcheck: all inputs must be non-view leaf tensors with requires_grad=True"
    )

    # Leaf input with requires_grad=False.
    x2 = vt.tensor([1.0, 2.0], dtype="float32")
    assert bool(getattr(x2, "is_leaf", False))
    assert not bool(getattr(x2, "requires_grad", False))

    with pytest.raises(A.GradcheckError) as exc2:
        A.gradcheck(_simple_scalar_fn, x2)
    assert str(exc2.value) == (
        "gradcheck: all inputs must be non-view leaf tensors with requires_grad=True"
    )


def test_gradcheck_gc2_cpu_float32_enforced() -> None:
    x = vt.tensor([1, 2], dtype="int32")
    x.requires_grad = True

    with pytest.raises(A.GradcheckError) as exc:
        A.gradcheck(_simple_scalar_fn, x)
    assert str(exc.value) == (
        "gradcheck: all inputs must be CPU float32 tensors"
    )


def test_gradcheck_gc2_noncontiguous_and_overlapping_inputs_raise() -> None:
    base = vt.arange(6, dtype="float32").reshape((2, 3)).detach()
    base.requires_grad = True

    # Transposed view: non-contiguous but still dense.
    if hasattr(base, "transpose"):
        transposed = base.transpose(0, 1)
        with pytest.raises(A.GradcheckError) as exc_t:
            A.gradcheck(_simple_scalar_fn, transposed)
        assert "non-contiguous or overlapping inputs" in str(exc_t.value)

    # Strided slice: non-contiguous and non-dense.
    sliced = base[:, ::2]
    with pytest.raises(A.GradcheckError) as exc_s:
        A.gradcheck(_simple_scalar_fn, sliced)
    assert "non-contiguous or overlapping inputs" in str(exc_s.value)


def test_gradcheck_gc2_empty_input_sequence_raises() -> None:
    with pytest.raises(A.GradcheckError) as exc:
        A.gradcheck(_simple_scalar_fn, [])
    assert str(exc.value) == (
        "gradcheck: inputs must not be an empty sequence"
    )


# ---- GC3: output validation --------------------------------------------------


def test_gradcheck_gc3_fn_must_return_scalar_tensor() -> None:
    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True

    def returns_non_tensor(t):  # noqa: ANN001
        return 1.0

    with pytest.raises(A.GradcheckError) as exc1:
        A.gradcheck(returns_non_tensor, x)
    assert "expects fn to return a single 0-D scalar Tensor" in str(exc1.value)

    def returns_non_scalar(t):  # noqa: ANN001
        return t  # 1-D tensor

    with pytest.raises(A.GradcheckError) as exc2:
        A.gradcheck(returns_non_scalar, x)
    assert "expects fn to return a single 0-D scalar Tensor" in str(exc2.value)


# ---- GC4: environment misuse -------------------------------------------------


def test_gradcheck_gc4_inference_mode_and_no_grad_guarded() -> None:
    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True

    with vt.inference_mode(True):
        with pytest.raises(A.GradcheckError) as exc_inf:
            A.gradcheck(_simple_scalar_fn, x)
    assert str(exc_inf.value) == (
        "vibetensor.autograd.gradcheck: cannot run gradcheck inside inference_mode; "
        "enable gradients and rebuild the graph"
    )

    with vt.no_grad():
        with pytest.raises(A.GradcheckError) as exc_ng:
            A.gradcheck(_simple_scalar_fn, x)
    assert str(exc_ng.value) == (
        "vibetensor.autograd.gradcheck: cannot run gradcheck inside no_grad; "
        "enable gradients and rebuild the graph"
    )


def test_gradcheck_gc4_grad_precondition_failure_wrapped() -> None:
    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    with pytest.raises(A.GradcheckError) as exc:
        A.gradcheck(_simple_scalar_fn, x)
    assert str(exc.value) == (
        "gradcheck: grad precondition failed: vibetensor.autograd.grad: got unused input with allow_unused=False"
    )


# ---- GC5: numeric mismatch behavior -----------------------------------------


def test_gradcheck_gc5_numeric_mismatch_raise_or_false(monkeypatch) -> None:
    x = vt.tensor([0.5, -1.0], dtype="float32")
    x.requires_grad = True

    def bad_grad(outputs, inputs, **kwargs):  # noqa: ANN001
        (g,) = _oracle_grad_for_simple_scalar(outputs, inputs, **kwargs)
        scale = vt.full_like(g, 2.0)
        return (C.vt.mul(g, scale),)

    monkeypatch.setattr(A, "grad", bad_grad)

    with pytest.raises(A.GradcheckError) as exc_raise:
        A.gradcheck(_simple_scalar_fn, x, raise_exception=True)
    assert "numerical and analytical gradients differ" in str(exc_raise.value)

    ok = A.gradcheck(_simple_scalar_fn, x, raise_exception=False)
    assert ok is False


# ---- GC6: fast_mode handling -------------------------------------------------


@pytest.mark.parametrize("fast_mode", [None, False])
def test_gradcheck_gc6_fast_mode_valid_values_work(fast_mode, monkeypatch) -> None:  # noqa: ANN001
    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    monkeypatch.setattr(A, "grad", _oracle_grad_for_simple_scalar)
    assert A.gradcheck(_simple_scalar_fn, x, fast_mode=fast_mode)


@pytest.mark.parametrize("fast_mode", [True, "yes"])
def test_gradcheck_gc6_fast_mode_invalid_values_error(fast_mode) -> None:  # noqa: ANN001
    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    with pytest.raises(A.GradcheckError) as exc:
        A.gradcheck(_simple_scalar_fn, x, fast_mode=fast_mode)
    msg = str(exc.value)
    if fast_mode is True:
        assert (
            msg
            == "vibetensor.autograd.gradcheck: fast_mode=True is not implemented; "
            "pass fast_mode=False or None"
        )
    else:
        assert msg == (
            "vibetensor.autograd.gradcheck: fast_mode must be a bool or None"
        )


# ---- GC7: numeric-parameter validation ---------------------------------------


@pytest.mark.parametrize(
    "eps, atol, rtol",
    [
        (0.0, 1e-4, 1e-2),
        (-1e-3, 1e-4, 1e-2),
        (1e-3, -1e-4, 1e-2),
        (1e-3, 1e-4, -1e-2),
        (float("nan"), 1e-4, 1e-2),
        (1e-3, float("inf"), 1e-2),
    ],
)
def test_gradcheck_gc7_invalid_numeric_parameters_raise(eps, atol, rtol) -> None:  # noqa: ANN001
    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    with pytest.raises(A.GradcheckError) as exc:
        A.gradcheck(_simple_scalar_fn, x, eps=eps, atol=atol, rtol=rtol)
    assert str(exc.value) == (
        "vibetensor.autograd.gradcheck: eps must be > 0 and atol/rtol must be finite, "
        "non-negative floats"
    )


# ---- GC10: raw-exception boundary --------------------------------------------


def test_gradcheck_gc10_raw_exception_boundary(monkeypatch) -> None:
    # Patch vt.to_dlpack so that internal helper failures surface as raw errors.
    def boom_to_dlpack(t):  # noqa: ANN001
        raise RuntimeError("boom dlpack")

    monkeypatch.setattr(vt, "to_dlpack", boom_to_dlpack)
    monkeypatch.setattr(A, "grad", _oracle_grad_for_simple_scalar)

    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    with pytest.raises(RuntimeError, match="boom dlpack"):
        A.gradcheck(_simple_scalar_fn, x)


# ---- GG1–GG3: gradgradcheck stub behavior ------------------------------------


def test_gradgradcheck_gg1_stub_behavior_and_raise_exception_flag() -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    called = {"fn": False}

    def fn(t):  # noqa: ANN001
        called["fn"] = True
        return _simple_scalar_fn(t)

    # raise_exception=False -> return False without calling fn or touching stats.
    result = A.gradgradcheck(fn, x, raise_exception=False)
    assert result is False
    assert called["fn"] is False

    stats = ag.stats()
    s0 = {k: v for k, v in stats.items()}

    # raise_exception=True -> GradcheckError with higher-order-autograd message.
    with pytest.raises(A.GradcheckError) as exc:
        A.gradgradcheck(fn, x, raise_exception=True)
    assert str(exc.value) == (
        "vibetensor.autograd.gradgradcheck: higher-order autograd is not implemented"
    )

    # Stats remain unchanged across both calls.
    stats2 = ag.stats()
    assert stats2 == s0

    # .grad on inputs is untouched.
    assert x.grad is None


@pytest.mark.parametrize("fast_mode", [True, "yes"])
def test_gradgradcheck_gg2_fast_mode_validation(fast_mode) -> None:  # noqa: ANN001
    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    with pytest.raises(A.GradcheckError) as exc:
        A.gradgradcheck(_simple_scalar_fn, x, fast_mode=fast_mode)
    msg = str(exc.value)
    if fast_mode is True:
        assert (
            msg
            == "vibetensor.autograd.gradgradcheck: fast_mode=True is not implemented; "
            "pass fast_mode=False or None"
        )
    else:
        assert msg == (
            "vibetensor.autograd.gradgradcheck: fast_mode must be a bool or None"
        )


def test_gradgradcheck_gg3_stats_and_side_effects() -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    s0 = ag.stats()

    # Multiple calls with various options should not change stats or .grad.
    assert A.gradgradcheck(_simple_scalar_fn, x, raise_exception=False) is False
    assert A.gradgradcheck(_simple_scalar_fn, x, raise_exception=False, fast_mode=None) is False

    s1 = ag.stats()
    assert s1 == s0
    assert x.grad is None
