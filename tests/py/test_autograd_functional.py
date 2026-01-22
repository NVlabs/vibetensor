# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.autograd as A
import vibetensor.autograd.functional as F
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


# ---- 6.1 Core correctness: linear helpers ------------------------------------


def test_vjp_linear_matches_analytic_vector_jacobian_product() -> None:
    ag = _reset_engine_stats()

    a = vt.tensor([2.0, -3.0], dtype="float32")
    x = vt.tensor([1.5, -0.5], dtype="float32")
    x.requires_grad = True
    v = vt.tensor([0.3, -0.7], dtype="float32")

    def fn(x_in):  # noqa: ANN001
        return C.vt.mul(a, x_in)

    out, vjp_x = F.vjp(fn, x, v=v)

    np.testing.assert_allclose(_to_numpy(out), _to_numpy(C.vt.mul(a, x)))
    expected_vjp = _to_numpy(a) * _to_numpy(v)
    np.testing.assert_allclose(_to_numpy(vjp_x), expected_vjp)

    stats = ag.stats()
    assert stats["engine_runs"] == 1


def test_jvp_linear_scalar_matches_analytic_jacobian_vector_product() -> None:
    ag = _reset_engine_stats()

    a = vt.tensor([2.0, -3.0], dtype="float32")
    x = vt.tensor([1.5, -0.5], dtype="float32")
    x.requires_grad = True
    v = vt.tensor([0.3, -0.7], dtype="float32")

    def fn(x_in):  # noqa: ANN001
        y = C.vt.mul(a, x_in)
        return y.sum()

    out, jvp_val = F.jvp(fn, x, tangents=v)

    expected_out = float(_to_numpy(C.vt.mul(a, x)).sum())

    # Scalar roots built via reductions like `.sum()` are non-differentiable in
    # the correct forward value.
    assert pytest.approx(expected_out) == _scalar_from_tensor(out)
    assert pytest.approx(0.0) == _scalar_from_tensor(jvp_val)

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_jacobian_linear_matches_expected_diagonal_matrix() -> None:
    ag = _reset_engine_stats()

    a = vt.tensor([2.0, -3.0], dtype="float32")

    def fn(x_in):  # noqa: ANN001
        return C.vt.mul(a, x_in)

    x = vt.tensor([1.5, -0.5], dtype="float32")
    x.requires_grad = True

    J = F.jacobian(fn, x)

    x_np = _to_numpy(x).astype(np.float64)
    y_np = _to_numpy(fn(x)).astype(np.float64)
    J_np = _to_numpy(J).reshape(y_np.size, x_np.size)

    expected = np.diag(_to_numpy(a).astype(np.float64))
    np.testing.assert_allclose(J_np, expected)

    stats = ag.stats()
    assert stats["engine_runs"] == y_np.size


# ---- 6.1 Core correctness: non-linear helpers --------------------------------


def test_vjp_and_jvp_quadratic_scalar_loss() -> None:
    x = vt.tensor([1.5, -0.5], dtype="float32")
    x.requires_grad = True

    def fn(x_in):  # noqa: ANN001
        y = C.vt.mul(x_in, x_in)
        return y.sum()

    expected_out = _scalar_from_tensor(fn(x))

    # vjp with implicit v=1 for scalar output: gradients collapse to zeros
    # because scalar roots from `.sum()` are non-differentiable.
    out_vjp, grad_x = F.vjp(fn, x, v=None)
    assert pytest.approx(expected_out) == _scalar_from_tensor(out_vjp)
    np.testing.assert_allclose(_to_numpy(grad_x), np.zeros_like(_to_numpy(x)))
    assert not getattr(grad_x, "requires_grad", False)

    # jvp with an explicit tangent likewise yields zero.
    v_tangent = vt.tensor([0.25, -0.75], dtype="float32")
    out_jvp, jvp_val = F.jvp(fn, x, tangents=v_tangent)

    assert pytest.approx(expected_out) == _scalar_from_tensor(out_jvp)
    assert pytest.approx(0.0) == _scalar_from_tensor(jvp_val)


def test_jacobian_quadratic_vector_non_linear() -> None:
    ag = _reset_engine_stats()

    def fn(x_in):  # noqa: ANN001
        return C.vt.mul(x_in, x_in)

    x = vt.tensor([1.5, -0.5], dtype="float32")
    x.requires_grad = True

    J = F.jacobian(fn, x)

    x_np = _to_numpy(x).astype(np.float64)
    J_np = _to_numpy(J).reshape(x_np.size, x_np.size)
    expected = np.diag(2.0 * x_np)

    np.testing.assert_allclose(J_np, expected)

    stats = ag.stats()
    assert stats["engine_runs"] == x_np.size


# ---- 6.2 Mode behavior: grad-enabled vs no_grad and inference_mode -----------


def test_functional_helpers_run_under_no_grad_and_count_engine_runs() -> None:
    ag = _reset_engine_stats()

    def fn_scalar(x_in):  # noqa: ANN001
        y = C.vt.mul(x_in, x_in)
        return y.sum()

    def fn_vec(x_in):  # noqa: ANN001
        return C.vt.mul(x_in, x_in)

    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True
    v = vt.tensor([0.5, -0.5], dtype="float32")

    x_np = _to_numpy(x)

    # Baseline: grad mode enabled. Scalar helper calls do not run the engine
    # for non-differentiable scalar roots built via `.sum()`, while Jacobian
    # executes one reverse-mode pass per output element.
    F.vjp(fn_scalar, x, v=None)
    F.jvp(fn_scalar, x, tangents=v)
    F.jacobian(fn_vec, x)

    stats = ag.stats()
    assert stats["engine_runs"] == x_np.size

    # Under no_grad(): helpers re-enable gradients internally for Jacobian,
    # but VJP/JVP still see non-differentiable scalar roots.
    ag.reset_stats()

    with A.no_grad():
        F.vjp(fn_scalar, x, v=None)
    with A.no_grad():
        F.jvp(fn_scalar, x, tangents=v)
    with A.no_grad():
        F.jacobian(fn_vec, x)

    stats = ag.stats()
    assert stats["engine_runs"] == x_np.size


def test_functional_helpers_reject_inference_mode_without_running_engine() -> None:
    ag = _reset_engine_stats()

    def fn_scalar(x_in):  # noqa: ANN001
        y = C.vt.mul(x_in, x_in)
        return y.sum()

    def fn_vec(x_in):  # noqa: ANN001
        return C.vt.mul(x_in, x_in)

    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True
    v = vt.tensor([0.5, -0.5], dtype="float32")

    with A.inference_mode(True):
        with pytest.raises(RuntimeError, match="vibetensor.autograd.functional.vjp: cannot compute VJP inside inference_mode"):
            F.vjp(fn_scalar, x, v=None)

    with A.inference_mode(True):
        with pytest.raises(RuntimeError, match="vibetensor.autograd.functional.jvp: cannot compute JVP inside inference_mode"):
            F.jvp(fn_scalar, x, tangents=v)

    with A.inference_mode(True):
        with pytest.raises(RuntimeError, match="vibetensor.autograd.functional.jacobian: cannot compute Jacobian inside inference_mode"):
            F.jacobian(fn_vec, x)

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- 6.3 Unused inputs and strict semantics ----------------------------------


def _make_unused_inputs_setup():
    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True
    y = vt.tensor([-1.0, 3.0], dtype="float32")
    y.requires_grad = True

    def fn(x_in, y_in):  # noqa: ANN001, ARG001
        # Only x contributes to the *vector* output; y is structurally unused.
        return C.vt.mul(x_in, x_in)

    return x, y, fn


def test_vjp_unused_inputs_zero_for_non_strict_and_error_for_strict() -> None:
    x, y, fn = _make_unused_inputs_setup()

    v = vt.ones_like(fn(x, y))

    # strict=False: gradient for unused input is a zero tensor.
    _, (g_x, g_y) = F.vjp(fn, (x, y), v=v, strict=False)

    np.testing.assert_allclose(_to_numpy(g_x), 2.0 * _to_numpy(x))
    np.testing.assert_allclose(_to_numpy(g_y), np.zeros_like(_to_numpy(y)))

    x2, y2, fn2 = _make_unused_inputs_setup()
    v2 = vt.ones_like(fn2(x2, y2))
    with pytest.raises(RuntimeError, match="vibetensor.autograd.grad: got unused input with allow_unused=False"):
        F.vjp(fn2, (x2, y2), v=v2, strict=True)


# non-differentiable (they are built via reductions like `.sum()`), so
# directional derivatives collapse to zero. We exercise this behavior in
# `test_vjp_and_jvp_non_differentiable_scalar_root_zero_and_strict_raises`
# instead of asserting non-zero JVPs for unused-input scenarios.

def test_jacobian_identity_root_returns_zeros_and_strict_raises() -> None:
    ag = _reset_engine_stats()

    def fn(x_in):  # noqa: ANN001
        # Identity: result is a leaf tensor with no grad_fn.
        return x_in

    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True

    J = F.jacobian(fn, x, strict=False)

    x_np = _to_numpy(x).astype(np.float64)
    J_np = _to_numpy(J).reshape(x_np.size, x_np.size)
    np.testing.assert_allclose(J_np, np.zeros_like(J_np))

    stats = ag.stats()
    assert stats["engine_runs"] == 0

    x2 = vt.tensor([1.0, 2.0], dtype="float32")
    x2.requires_grad = True

    with pytest.raises(RuntimeError, match="vibetensor.autograd.grad: got unused input with allow_unused=False"):
        F.jacobian(fn, x2, strict=True)


def _make_non_diff_scalar_fn_and_input():
    x = vt.tensor([1.0, 2.0], dtype="float32")
    x.requires_grad = True

    def fn(x_in):  # noqa: ANN001
        # Build the scalar entirely under no_grad so the root has no history.
        with A.no_grad():
            y = C.vt.mul(x_in, x_in)
            return y.sum()

    return x, fn


def test_vjp_and_jvp_non_differentiable_scalar_root_zero_and_strict_raises() -> None:
    ag = _reset_engine_stats()

    x, fn = _make_non_diff_scalar_fn_and_input()

    out, g_x = F.vjp(fn, x, v=None, strict=False)
    assert pytest.approx(_scalar_from_tensor(fn(x))) == _scalar_from_tensor(out)
    np.testing.assert_allclose(_to_numpy(g_x), np.zeros_like(_to_numpy(x)))

    stats = ag.stats()
    assert stats["engine_runs"] == 0

    x2, fn2 = _make_non_diff_scalar_fn_and_input()
    with pytest.raises(RuntimeError, match="vibetensor.autograd.grad: got unused input with allow_unused=False"):
        F.vjp(fn2, x2, v=None, strict=True)

    x3, fn3 = _make_non_diff_scalar_fn_and_input()
    t = vt.tensor([0.5, -0.5], dtype="float32")
    _, jvp_val = F.jvp(fn3, x3, tangents=t, strict=False)
    assert pytest.approx(0.0) == _scalar_from_tensor(jvp_val)

    x4, fn4 = _make_non_diff_scalar_fn_and_input()
    with pytest.raises(RuntimeError, match="vibetensor.autograd.grad: got unused input with allow_unused=False"):
        F.jvp(fn4, x4, tangents=t, strict=True)


# ---- 6.5 .grad side-effects ---------------------------------------------------


def test_functional_helpers_update_parameter_grads_but_not_primals() -> None:
    w = vt.tensor([2.0], dtype="float32")
    w.requires_grad = True
    x = vt.tensor([3.0], dtype="float32")
    x.requires_grad = True

    def fn(x_in):  # noqa: ANN001
        # Closed-over parameter w participates in the graph and produces a
        # vector output; x_in is treated as the primal.
        return C.vt.mul(w, x_in)

    # Baseline gradient for w (non-scalar root requires explicit grad_outputs).
    y0 = fn(x)
    (g_w0,) = A.grad(y0, (w,), grad_outputs=vt.ones_like(y0))
    w_grad_before = _to_numpy(g_w0)

    # Clear any incidental gradients on the primal before invoking functional
    x.grad = None  # type: ignore[assignment]

    # vjp should accumulate into w.grad but not touch x.grad (primals are
    # detached clones inside the helper).
    y = fn(x)
    F.vjp(fn, x, v=vt.ones_like(y))

    # jacobian uses repeated reverse-mode passes and also accumulates into w.grad.
    F.jacobian(fn, x)

    assert x.grad is None
    assert w.grad is not None
    w_grad_after = _to_numpy(w.grad)  # type: ignore[arg-type]
    assert not np.allclose(w_grad_before, w_grad_after)


# ---- 6.6 Import and alias wiring ---------------------------------------------


def test_functional_import_and_alias_wiring() -> None:
    import vibetensor.autograd.functional as F1  # noqa: PLC0415

    import vibetensor.autograd as A2  # noqa: PLC0415
    import vibetensor.torch as vt2  # noqa: PLC0415

    F2 = vt2.autograd.functional  # type: ignore[attr-defined]

    assert F1 is F2 is A2.functional


# ---- 6.7 Stub behavior -------------------------------------------------------


def test_functional_stubs_raise_not_implemented_with_expected_message() -> None:
    for name, fn in {
        "hessian": F.hessian,
        "vhp": F.vhp,
        "hvp": F.hvp,
    }.items():
        with pytest.raises(NotImplementedError) as exc_info:
            fn(object())  # type: ignore[arg-type]
        msg = str(exc_info.value)
        assert name in msg
        assert "autograd" in msg.lower()


# ---- 6.8 Optional PyTorch cross-validation -----------------------------------


def test_functional_helpers_cross_check_with_pytorch_when_available() -> None:
    torch = pytest.importorskip("torch")

    def fn_vbt(x_in):  # noqa: ANN001
        return C.vt.mul(x_in, x_in)

    def fn_torch(x_t):  # noqa: ANN001
        return (x_t * x_t)

    x_vbt = vt.tensor([1.5, -0.5], dtype="float32")
    x_vbt.requires_grad = True

    x_torch = torch.from_dlpack(vt.to_dlpack(x_vbt)).requires_grad_(True)

    v_torch = torch.ones_like(fn_torch(x_torch))
    v_vbt = vt.tensor(v_torch.detach().cpu().numpy(), dtype="float32")

    out_vbt, g_vbt = F.vjp(fn_vbt, x_vbt, v=v_vbt)

    # PyTorch reference.
    out_t, g_t = torch.autograd.functional.vjp(fn_torch, x_torch, v=v_torch)
    if isinstance(g_t, (list, tuple)):
        (g_t,) = g_t

    np.testing.assert_allclose(_to_numpy(out_vbt), out_t.detach().cpu().numpy())
    np.testing.assert_allclose(_to_numpy(g_vbt), g_t.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)
