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
    # Support both core tensors and GradProxy-style wrappers by preferring the
    # DLPack protocol when available.
    if hasattr(t, "__dlpack__"):
        return np.from_dlpack(t)
    cap = vt.to_dlpack(t)
    return np.from_dlpack(cap)


def _scalar_from_tensor(t) -> float:  # noqa: ANN001
    arr = _to_numpy(t).reshape(-1)
    assert arr.size == 1
    return float(arr[0])


def _simple_scalar_fn(x):  # noqa: ANN001
    """Small quadratic scalar loss used for grad/gradcheck integration tests."""

    y = C.vt.mul(x, x)
    return y.sum()


def _oracle_grad_for_simple_scalar_integration(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=None,
    create_graph: bool = False,
    allow_unused=None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
):
    """Analytical gradient oracle for `_simple_scalar_fn` used in integration tests.

    This mirrors the GC1 stub from the core gradcheck tests so that we can
    exercise the gradcheck driver without depending on engine support for
    scalar roots.
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


# ---- Group A: Torch-overlay autograd integration (UX3, INT1, PERF1) ---------


def test_autograd_integration_a1_vt_autograd_grad_matches_analytic_quadratic() -> None:
    ag = _reset_engine_stats()

    # Simple quadratic scalar loss: sum(x ** 2).
    x = vt.tensor([2.0, -3.0], dtype="float32")
    x.requires_grad = True
    y = C.vt.mul(x, x)
    grad_in = vt.ones_like(y)

    (g,) = vt.autograd.grad(y, (x,), grad_outputs=grad_in)
    stats = ag.stats()

    assert stats["engine_runs"] == 1
    assert x.grad is not None
    # Returned gradient matches the leaf .grad buffer numerically.
    np.testing.assert_allclose(_to_numpy(g), _to_numpy(x.grad_tensor()))

    x_np = _to_numpy(x)
    grad_np = _to_numpy(g)
    np.testing.assert_allclose(grad_np, 2.0 * x_np)


def test_autograd_integration_a2_backward_module_vs_overlay_parity() -> None:
    ag = _reset_engine_stats()

    def _make_graph():
        x = vt.tensor([2.0, -3.0], dtype="float32")
        x.requires_grad = True
        y = C.vt.mul(x, x)
        grad = vt.full_like(y, 1.0)
        return x, y, grad

    # Module-level backward with explicit grad.
    x_mod, y_mod, grad_mod_in = _make_graph()
    A.backward(y_mod, grad_tensors=grad_mod_in)
    stats_mod = ag.stats()
    grad_mod = _to_numpy(x_mod.grad)  # type: ignore[arg-type]

    # Overlay backward on an independent copy of the graph.
    ag.reset_stats()

    x_vt, y_vt, grad_vt_in = _make_graph()
    vt.autograd.backward(y_vt, grad_tensors=grad_vt_in)
    stats_vt = ag.stats()
    grad_vt = _to_numpy(x_vt.grad)  # type: ignore[arg-type]

    assert stats_mod["engine_runs"] == stats_vt["engine_runs"] == 1
    np.testing.assert_allclose(grad_mod, grad_vt)


def test_autograd_integration_a3_overlay_import_aliasing_and_behavior() -> None:
    ag = _reset_engine_stats()

    from vibetensor.torch import autograd as At  # noqa: PLC0415
    from vibetensor.torch.autograd import grad as vt_grad  # noqa: PLC0415

    assert At is A
    assert vt_grad is A.grad

    # Use non-scalar roots with explicit grad_outputs so we exercise the same
    # paths as the core MG1 tests while going through the torch-like overlay.
    x1 = vt.tensor([1.0, -2.0], dtype="float32")
    x1.requires_grad = True
    y1 = C.vt.mul(x1, x1)
    grad_in1 = vt.ones_like(y1)

    (g1,) = At.grad(y1, (x1,), grad_outputs=grad_in1)
    stats1 = ag.stats()

    assert stats1["engine_runs"] == 1
    assert x1.grad is not None
    np.testing.assert_allclose(_to_numpy(g1), _to_numpy(x1.grad_tensor()))

    ag.reset_stats()

    x2 = vt.tensor([1.0, -2.0], dtype="float32")
    x2.requires_grad = True
    y2 = C.vt.mul(x2, x2)
    grad_in2 = vt.ones_like(y2)

    (g2,) = vt_grad(y2, (x2,), grad_outputs=grad_in2)
    stats2 = ag.stats()

    assert stats2["engine_runs"] == 1
    assert x2.grad is not None
    np.testing.assert_allclose(_to_numpy(g2), _to_numpy(x2.grad_tensor()))
    np.testing.assert_allclose(_to_numpy(g1), _to_numpy(g2))


# ---- Group B: Training-style flows with backward (UX1, INT2, PERF1) ---------


def autograd_integration_b1_single_parameter_training_loop_example() -> None:
    ag = _reset_engine_stats()

    # Simple 1D regression y = w * x with L2 loss.
    xs = vt.tensor([1.0, 2.0, 3.0], dtype="float32")
    targets = vt.tensor([2.0, 4.0, 6.0], dtype="float32")
    lr = 0.1

    # Represent parameters as a length-3 vector for simplicity.
    w_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    losses: list[float] = []

    for _ in range(3):
        w = vt.tensor(w_np.tolist(), dtype="float32")
        w.requires_grad = True

        y_pred = C.vt.mul(w, xs)
        neg_targets = C.vt.mul(targets, vt.full_like(targets, -1.0))
        err = C.vt.add(y_pred, neg_targets)
        loss_vec = C.vt.mul(err, err)

        # Scalar loss for UX assertions.
        loss_scalar = loss_vec.sum()
        losses.append(_scalar_from_tensor(loss_scalar))

        grad_outputs = vt.ones_like(loss_vec)
        (g_w,) = vt.autograd.grad(loss_vec, (w,), grad_outputs=grad_outputs)

        g_np = _to_numpy(g_w)
        w_np = w_np - lr * g_np

    # Loss decreases monotonically.
    assert losses[1] < losses[0]
    assert losses[2] < losses[1]

    stats = ag.stats()
    assert stats["engine_runs"] == len(losses)


# ---- Group C: Grad integration and mode interactions (UX2, INT2, PERF1) -----


def test_autograd_integration_c1_grad_under_no_grad_and_inference_mode() -> None:
    ag = _reset_engine_stats()

    # Build differentiable root under default grad-mode.
    x_ref = vt.tensor([2.0, -3.0], dtype="float32")
    x_ref.requires_grad = True
    y_ref = C.vt.mul(x_ref, x_ref)
    grad_in_ref = vt.ones_like(y_ref)
    y_ref.backward(grad_in_ref)
    grad_ref = _to_numpy(x_ref.grad)  # type: ignore[arg-type]

    ag.reset_stats()

    # Same graph pattern, but call vt.autograd.grad under no_grad().
    x_ng = vt.tensor([2.0, -3.0], dtype="float32")
    x_ng.requires_grad = True
    y_ng = C.vt.mul(x_ng, x_ng)
    grad_in_ng = vt.ones_like(y_ng)

    with vt.no_grad():
        (g_ng,) = vt.autograd.grad(y_ng, (x_ng,), grad_outputs=grad_in_ng)

    stats_ng = ag.stats()
    assert stats_ng["engine_runs"] == 1
    np.testing.assert_allclose(_to_numpy(g_ng), grad_ref)

    ag.reset_stats()

    # Same graph pattern, but call vt.autograd.grad under inference_mode(True).
    x_inf = vt.tensor([2.0, -3.0], dtype="float32")
    x_inf.requires_grad = True
    y_inf = C.vt.mul(x_inf, x_inf)
    grad_in_inf = vt.ones_like(y_inf)

    with vt.inference_mode(True):
        (g_inf,) = vt.autograd.grad(y_inf, (x_inf,), grad_outputs=grad_in_inf)

    stats_inf = ag.stats()
    assert stats_inf["engine_runs"] == 1
    np.testing.assert_allclose(_to_numpy(g_inf), grad_ref)


# ---- Group D: Gradcheck and gradgradcheck integration (UX4, SEC3, OBS1) -----


def test_autograd_integration_d1_gradcheck_happy_path_via_vt_autograd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([0.5, -1.0, 2.0], dtype="float32")
    x.requires_grad = True

    # Use an oracle for analytical gradients so we can exercise the gradcheck
    # driver via the vt.autograd alias without depending on engine support for
    # scalar roots.
    monkeypatch.setattr(A, "grad", _oracle_grad_for_simple_scalar_integration)

    ok = vt.autograd.gradcheck(_simple_scalar_fn, x)
    assert ok is True

    # When using the oracle stub, gradcheck does not touch engine stats or .grad.
    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert x.grad is None


def test_autograd_integration_d2_gradcheck_under_no_grad_and_inference_mode() -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True

    with vt.inference_mode(True):
        with pytest.raises(A.GradcheckError) as exc_inf:
            vt.autograd.gradcheck(_simple_scalar_fn, x)
    assert "cannot run gradcheck inside inference_mode" in str(exc_inf.value)

    stats_inf = ag.stats()
    assert stats_inf["engine_runs"] == 0

    ag.reset_stats()

    with vt.no_grad():
        with pytest.raises(A.GradcheckError) as exc_ng:
            vt.autograd.gradcheck(_simple_scalar_fn, x)
    assert "cannot run gradcheck inside no_grad" in str(exc_ng.value)

    stats_ng = ag.stats()
    assert stats_ng["engine_runs"] == 0


def test_autograd_integration_d3_gradgradcheck_stub_via_overlay() -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    called = {"fn": False}

    def fn(t):  # noqa: ANN001
        called["fn"] = True
        return _simple_scalar_fn(t)

    # raise_exception=False -> return False without calling fn or touching stats.
    result = vt.autograd.gradgradcheck(fn, x, raise_exception=False)
    assert result is False
    assert called["fn"] is False

    stats0 = ag.stats()

    # raise_exception=True -> GradcheckError with higher-order-autograd message.
    with pytest.raises(A.GradcheckError) as exc:
        vt.autograd.gradgradcheck(fn, x, raise_exception=True)
    assert "higher-order autograd is not implemented" in str(exc.value)

    # Stats remain unchanged and GradcheckError is catchable as RuntimeError.
    stats1 = ag.stats()
    assert stats1 == stats0
    assert isinstance(exc.value, RuntimeError)


# ---- Group E: Error typing and raw-exception boundaries (SEC3, OBS1) --------


def test_autograd_integration_e2_gradcheck_raw_exception_boundary_via_vt_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch vt.to_dlpack so that internal helper failures surface as raw errors
    # even when called through the vt.autograd overlay.
    def boom_to_dlpack(t):  # noqa: ANN001
        raise RuntimeError("boom dlpack from integration")

    monkeypatch.setattr(vt, "to_dlpack", boom_to_dlpack)

    # Also patch A.grad to the oracle stub so gradcheck reaches the DLPack
    # conversion path instead of failing on grad preconditions.
    monkeypatch.setattr(A, "grad", _oracle_grad_for_simple_scalar_integration)

    x = vt.tensor([0.5], dtype="float32")
    x.requires_grad = True

    with pytest.raises(RuntimeError, match="boom dlpack from integration") as exc:
        vt.autograd.gradcheck(_simple_scalar_fn, x)

    # Ensure the raw RuntimeError is not wrapped in GradcheckError.
    assert not isinstance(exc.value, A.GradcheckError)
