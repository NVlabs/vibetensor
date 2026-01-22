# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.autograd as A
import vibetensor.torch as vt
from vibetensor import _C as C


def _to_list(t):
    return np.from_dlpack(t).tolist()


def _reset_engine_stats():
    ag = C.autograd
    ag.reset_stats()
    return ag



def _make_scalar_loss():
    """Return (x, loss) where loss is a scalar with no history.

    This helper mirrors a root without autograd history: ``loss`` has
    ``requires_grad=False`` and ``grad_fn=None`` even though it is built from
    differentiable operations. It is useful for MB2-style no-op tests.
    """

    x = vt.tensor([2.0, -3.0], dtype="float32")
    x.requires_grad = True
    y = C.vt.mul(x, x)
    loss = y.sum()
    return x, loss


def _make_non_scalar_root_and_grad():
    """Return (x, y, grad) where ``y`` has history and is non-scalar.

    ``y`` is a length-2 float32 tensor with ``requires_grad=True`` and a valid
    ``grad_fn``; ``grad`` is a matching tensor suitable for explicit backward.
    """

    x = vt.tensor([2.0, -3.0], dtype="float32")
    x.requires_grad = True
    y = C.vt.mul(x, x)
    grad = vt.full_like(y, 1.0)
    return x, y, grad


# ---- MB1: happy paths and parity --------------------------------------------


def test_backward_mb1_scalar_root_no_grad_matches_tensor_backward() -> None:
    ag = _reset_engine_stats()

    # For non-scalar roots, omitting an explicit gradient is illegal and both
    # the module-level and Tensor-level APIs must raise the same ValueError
    # from the C++ engine.
    x_mod, y_mod, _ = _make_non_scalar_root_and_grad()
    with pytest.raises(ValueError) as exc_mod:
        A.backward(y_mod)
    stats_mod = ag.stats()

    ag.reset_stats()
    x_t, y_t, _ = _make_non_scalar_root_and_grad()
    with pytest.raises(ValueError) as exc_t:
        y_t.backward()
    stats_t = ag.stats()

    assert str(exc_mod.value) == str(exc_t.value)
    assert stats_mod["engine_runs"] == stats_t["engine_runs"] == 0
    assert x_mod.grad is None
    assert x_t.grad is None


def test_backward_mb1_scalar_root_explicit_grad_matches_tensor_backward() -> None:
    ag = _reset_engine_stats()

    # Module-level backward with explicit grad
    x_mod, y_mod, grad_mod_in = _make_non_scalar_root_and_grad()
    A.backward(y_mod, grad_tensors=grad_mod_in)
    stats_mod = ag.stats()
    grad_mod = _to_list(x_mod.grad)  # type: ignore[arg-type]

    # Tensor-level backward with explicit grad
    ag.reset_stats()
    x_t, y_t, grad_t_in = _make_non_scalar_root_and_grad()
    y_t.backward(grad_t_in)
    stats_t = ag.stats()
    grad_t = _to_list(x_t.grad)  # type: ignore[arg-type]

    assert stats_mod["engine_runs"] == stats_t["engine_runs"] == 1
    assert grad_mod == grad_t


def test_backward_mb1_tuple_forms_match_single_tensor() -> None:
    ag = _reset_engine_stats()

    # Single-tensor form
    x_single, y_single, grad_single_in = _make_non_scalar_root_and_grad()
    A.backward(y_single, grad_tensors=grad_single_in)
    stats_single = ag.stats()
    grad_single = _to_list(x_single.grad)  # type: ignore[arg-type]

    # Tuple forms for tensors and grad_tensors
    ag.reset_stats()
    x_tuple, y_tuple, grad_tuple_in = _make_non_scalar_root_and_grad()
    A.backward((y_tuple,), grad_tensors=(grad_tuple_in,))
    stats_tuple = ag.stats()
    grad_tuple = _to_list(x_tuple.grad)  # type: ignore[arg-type]

    assert stats_single["engine_runs"] == stats_tuple["engine_runs"] == 1
    assert grad_single == grad_tuple


# ---- MB2: non-differentiable roots / no history -----------------------------


def test_backward_mb2_no_history_tensor_is_noop_and_stats_unchanged() -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([1.0], dtype="float32")
    assert x.grad is None
    assert x.grad_fn is None
    assert x.requires_grad is False

    A.backward(x)

    assert x.grad is None
    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb2_inference_mode_loss_is_noop() -> None:
    ag = _reset_engine_stats()

    with vt.inference_mode(True):
        x = vt.tensor([2.0], dtype="float32")
        x.requires_grad = True
        y = C.vt.mul(x, x)
        loss = y.sum()

    # loss was built under inference_mode; C++ treats it as no-history
    assert loss.grad_fn is None
    assert loss.requires_grad is False

    A.backward(loss)

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb2_no_history_with_explicit_grad_is_noop() -> None:
    ag = _reset_engine_stats()

    x = vt.tensor([1.0], dtype="float32")
    assert x.grad is None
    assert x.grad_fn is None
    assert x.requires_grad is False

    grad = vt.ones_like(x)
    A.backward(x, grad_tensors=grad)

    assert x.grad is None
    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- MB3: multiple outputs rejected -----------------------------------------


def test_backward_mb3_multiple_tensors_list_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    x1 = vt.tensor([1.0], dtype="float32")
    x1.requires_grad = True
    x2 = vt.tensor([2.0], dtype="float32")
    x2.requires_grad = True
    y1 = C.vt.mul(x1, x1)
    y2 = C.vt.mul(x2, x2)

    with pytest.raises(RuntimeError) as exc:
        A.backward([y1, y2])
    assert str(exc.value) == (
        "vibetensor.autograd.backward: multiple outputs are not supported"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb3_multiple_tensors_tuple_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    x1 = vt.tensor([1.0], dtype="float32")
    x1.requires_grad = True
    x2 = vt.tensor([2.0], dtype="float32")
    x2.requires_grad = True
    y1 = C.vt.mul(x1, x1)
    y2 = C.vt.mul(x2, x2)

    with pytest.raises(RuntimeError) as exc:
        A.backward((y1, y2))
    assert str(exc.value) == (
        "vibetensor.autograd.backward: multiple outputs are not supported"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- MB4: malformed tensors / grad_tensors ----------------------------------


def test_backward_mb4a_tensors_non_tensor_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    with pytest.raises(TypeError) as exc:
        A.backward(object())
    assert str(exc.value) == "autograd.backward: tensors must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb4b_tensors_sequence_with_non_tensor_element_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()

    for bad in ([loss, object()], [[loss]]):
        with pytest.raises(TypeError) as exc:
            A.backward(bad)
        assert str(exc.value) == "autograd.backward: tensors must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb4f_tensors_empty_sequence_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    for empty in ([], ()):  # type: ignore[list-item]
        with pytest.raises(RuntimeError) as exc:
            A.backward(empty)
        assert str(exc.value) == (
            "autograd.backward: tensors must not be an empty sequence"
        )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb4c_grad_tensors_non_tensor_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()

    with pytest.raises(TypeError) as exc:
        A.backward(loss, grad_tensors=object())
    assert str(exc.value) == "autograd.backward: grad_tensors must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb4d_grad_tensors_sequence_with_non_tensor_element_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()
    g = vt.full_like(loss, 1.0)

    for bad in ([g, object()], [[g]]):
        with pytest.raises(TypeError) as exc:
            A.backward(loss, grad_tensors=bad)
        assert str(exc.value) == "autograd.backward: grad_tensors must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb4e_grad_tensors_length_mismatch_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()
    g1 = vt.full_like(loss, 1.0)
    g2 = vt.full_like(loss, 2.0)

    with pytest.raises(RuntimeError) as exc:
        A.backward(loss, grad_tensors=(g1, g2))
    assert str(exc.value) == (
        "autograd.backward: grad_tensors must match tensors in length"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb4m_grad_tensors_non_tensor_and_mismatch_prefers_typeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()
    g = vt.full_like(loss, 1.0)

    with pytest.raises(TypeError) as exc:
        A.backward(loss, grad_tensors=[g, object()])
    assert str(exc.value) == "autograd.backward: grad_tensors must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- Container policy: non-Sequence iterables -------------------------------


def test_backward_tensors_non_sequence_iterable_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()

    def _gen():
        yield loss

    with pytest.raises(TypeError) as exc:
        A.backward(_gen())
    assert str(exc.value) == "autograd.backward: tensors must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_grad_tensors_non_sequence_iterable_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()
    g = vt.full_like(loss, 1.0)

    def _gen():
        yield g

    with pytest.raises(TypeError) as exc:
        A.backward(loss, grad_tensors=_gen())
    assert str(exc.value) == "autograd.backward: grad_tensors must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- MB5: unsupported options -----------------------------------------------


def test_backward_mb5_create_graph_true_not_implemented() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()

    with pytest.raises(NotImplementedError) as exc:
        A.backward(loss, create_graph=True)
    assert str(exc.value) == (
        "vibetensor.autograd.backward: create_graph=True is not supported "
        "(higher-order autograd is out of scope)"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb5_create_graph_non_bool_typeerror() -> None:
    ag = _reset_engine_stats()

    _, loss = _make_scalar_loss()

    with pytest.raises(TypeError) as exc:
        A.backward(loss, create_graph=1)
    assert str(exc.value) == (
        "vibetensor.autograd.backward: create_graph must be a bool"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb5_inputs_not_none_not_implemented() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss()

    with pytest.raises(NotImplementedError) as exc:
        A.backward(loss, inputs=(x,))
    assert str(exc.value) == (
        "vibetensor.autograd.backward: the inputs argument is not supported"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb5_inputs_precedence_over_mb4() -> None:
    ag = _reset_engine_stats()

    # inputs combined with malformed tensors should still trigger the inputs
    # NotImplementedError before any MB3/MB4 normalization.
    with pytest.raises(NotImplementedError) as exc:
        A.backward(object(), inputs=(object(),))
    assert str(exc.value) == (
        "vibetensor.autograd.backward: the inputs argument is not supported"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb5_create_graph_precedence_over_inputs() -> None:
    ag = _reset_engine_stats()

    # When both create_graph and inputs are invalid, create_graph’s error wins.
    with pytest.raises(NotImplementedError) as exc:
        A.backward(object(), inputs=(object(),), create_graph=True)
    assert "create_graph=True is not supported" in str(exc.value)

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_backward_mb5_precedence_over_mb4() -> None:
    ag = _reset_engine_stats()

    # create_graph=True should win over malformed tensors
    with pytest.raises(NotImplementedError) as exc1:
        A.backward(object(), create_graph=True)
    assert "create_graph=True is not supported" in str(exc1.value)

    # Non-bool create_graph should also win
    with pytest.raises(TypeError) as exc2:
        A.backward(object(), create_graph=1)
    assert "create_graph must be a bool" in str(exc2.value)

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- retain_graph semantics -------------------------------------------------


def test_backward_retain_graph_ignored_and_bool_not_evaluated() -> None:
    class _BoolCounter:
        def __init__(self) -> None:
            self.calls = 0

        def __bool__(self) -> bool:  # pragma: no cover - behavior is asserted indirectly
            self.calls += 1
            return True

    ag = _reset_engine_stats()

    # Module-level backward with custom retain_graph object
    x_mod, y_mod, grad_mod_in = _make_non_scalar_root_and_grad()
    flag = _BoolCounter()
    A.backward(y_mod, grad_tensors=grad_mod_in, retain_graph=flag)
    stats_mod = ag.stats()
    grad_mod = _to_list(x_mod.grad)  # type: ignore[arg-type]

    # Tensor-level baseline
    ag.reset_stats()
    x_t, y_t, grad_t_in = _make_non_scalar_root_and_grad()
    y_t.backward(grad_t_in, retain_graph=False)
    stats_t = ag.stats()
    grad_t = _to_list(x_t.grad)  # type: ignore[arg-type]

    assert flag.calls == 0
    assert stats_mod["engine_runs"] == stats_t["engine_runs"] == 1
    assert grad_mod == grad_t


def test_grad_matches_backward_for_scalar_loss() -> None:
    """Compare A.grad(root, x) with root.backward() for a single root.

    Ensures engine stats and gradient values match between the module-level
    grad wrapper and the Tensor API.
    """

    ag = _reset_engine_stats()

    # Module-level grad on a non-scalar root with explicit gradient.
    x_mod, y_mod, grad_mod_in = _make_non_scalar_root_and_grad()
    (g_mod,) = A.grad(y_mod, x_mod, grad_outputs=grad_mod_in)
    stats_mod = ag.stats()
    grad_buf_mod = _to_list(x_mod.grad)  # type: ignore[arg-type]
    grad_out_mod = _to_list(g_mod)

    # Tensor-level backward baseline.
    ag.reset_stats()
    x_t, y_t, grad_t_in = _make_non_scalar_root_and_grad()
    y_t.backward(grad_t_in)
    stats_t = ag.stats()
    grad_buf_t = _to_list(x_t.grad)  # type: ignore[arg-type]

    assert stats_mod["engine_runs"] == stats_t["engine_runs"] == 1
    assert grad_buf_mod == grad_buf_t == grad_out_mod


# ---- Alias coverage ---------------------------------------------------------


def test_backward_torch_autograd_alias_delegates_to_module_level() -> None:
    ag = _reset_engine_stats()

    # Module-level backward
    x_mod, y_mod, grad_mod_in = _make_non_scalar_root_and_grad()
    A.backward(y_mod, grad_tensors=grad_mod_in)
    stats_mod = ag.stats()
    grad_mod = _to_list(x_mod.grad)  # type: ignore[arg-type]

    # Alias via vibetensor.torch.autograd
    ag.reset_stats()
    x_alias, y_alias, grad_alias_in = _make_non_scalar_root_and_grad()
    vt.autograd.backward(y_alias, grad_tensors=grad_alias_in)
    stats_alias = ag.stats()
    grad_alias = _to_list(x_alias.grad)  # type: ignore[arg-type]

    assert stats_mod["engine_runs"] == stats_alias["engine_runs"] == 1
    assert grad_mod == grad_alias
