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


def _make_scalar_loss_with_history():
    """Return (x, loss) where loss is a scalar with no history.

    This mirrors ``_make_scalar_loss`` from the backward tests: ``loss`` has
    ``requires_grad=False`` and ``grad_fn=None`` even though it is built from
    differentiable operations. It is useful for container/option tests that
    should not run the engine.
    """

    x = vt.tensor([2.0, -3.0], dtype="float32")
    x.requires_grad = True
    y = C.vt.mul(x, x)
    loss = y.sum()
    return x, loss


def _make_non_scalar_root_and_grad():
    """Return (x, y, grad) where ``y`` has history and is non-scalar."""

    x = vt.tensor([2.0, -3.0], dtype="float32")
    x.requires_grad = True
    y = C.vt.mul(x, x)
    grad = vt.full_like(y, 1.0)
    return x, y, grad


def _make_diff_root_with_used_and_unused_input():
    """Return (x_used, x_unused, root) with a differentiable non-scalar root.

    ``root`` depends only on ``x_used``; ``x_unused`` is a leaf that requires
    grad but is structurally unused by the graph.
    """

    x_used = vt.tensor([2.0, -3.0], dtype="float32")
    x_used.requires_grad = True
    x_unused = vt.tensor([1.5, 0.5], dtype="float32")
    x_unused.requires_grad = True

    root = C.vt.mul(x_used, x_used)
    return x_used, x_unused, root


def _make_non_diff_root_and_inputs():
    """Return (x_used, x_unused, root) where root has no history."""

    x_used = vt.tensor([2.0, -3.0], dtype="float32")
    x_used.requires_grad = True
    x_unused = vt.tensor([1.5, 0.5], dtype="float32")
    x_unused.requires_grad = True

    root = vt.tensor([1.0], dtype="float32")
    # By construction, ``root`` has requires_grad=False and grad_fn=None.
    return x_used, x_unused, root


def _make_inference_mode_non_diff_loss_and_input():
    """Return (x, loss) built under inference_mode so loss has no history."""

    with vt.inference_mode(True):
        x = vt.tensor([2.0, -3.0], dtype="float32")
        x.requires_grad = True
        y = C.vt.mul(x, x)
        loss = y.sum()
    return x, loss




# ---- MG1–MG4: core scalar-root behavior -------------------------------------


def _make_diff_scalar_root_single_input():
    """Return (x, root) where root has history and uses one input.

    This is a thin wrapper around ``_make_non_scalar_root_and_grad`` that
    exposes only the input tensor and differentiable root needed for MG1.
    """

    x, root, _ = _make_non_scalar_root_and_grad()
    return x, root



def _make_diff_scalar_root_two_inputs():
    """Return (x1, x2, root) where root has history and uses both inputs.

    ``root`` is a length-2 tensor built from vt::mul/vt::add and is used for
    MG2-style tests.
    """

    x1 = vt.tensor([2.0, -3.0], dtype="float32")
    x1.requires_grad = True
    x2 = vt.tensor([1.5, 0.5], dtype="float32")
    x2.requires_grad = True

    y1 = C.vt.mul(x1, x1)
    y2 = C.vt.mul(x2, x2)
    root = C.vt.add(y1, y2)

    return x1, x2, root



def test_grad_mg1_scalar_root_single_leaf_matches_backward_and_aliases_grad() -> None:
    ag = _reset_engine_stats()

    # Baseline: Tensor.backward on a differentiable root with explicit grad.
    x_ref, loss_ref = _make_diff_scalar_root_single_input()
    grad_in_ref = vt.ones_like(loss_ref)
    loss_ref.backward(grad_in_ref)
    stats_ref = ag.stats()
    grad_ref = _to_list(x_ref.grad)  # type: ignore[arg-type]

    assert stats_ref["engine_runs"] == 1

    # Wrapper: grad() should run the engine once, alias .grad, and
    # numerically match Tensor.backward with the same explicit gradient.
    ag.reset_stats()

    x, loss = _make_diff_scalar_root_single_input()
    grad_in = vt.ones_like(loss)
    (g,) = A.grad(loss, x, grad_outputs=grad_in)

    stats = ag.stats()
    assert stats["engine_runs"] == 1

    # The returned gradient numerically matches Tensor.backward and shares
    # the same values as ``x.grad``.
    assert x.grad is not None
    assert _to_list(g) == grad_ref
    assert _to_list(x.grad) == grad_ref  # type: ignore[arg-type]



def test_grad_mg2_scalar_root_multiple_leaf_inputs_match_backward_and_aliases_grad() -> None:
    ag = _reset_engine_stats()

    # Simple differentiable root depending on two leaf inputs.
    x1 = vt.tensor([2.0, -3.0], dtype="float32")
    x1.requires_grad = True
    x2 = vt.tensor([1.5, 0.5], dtype="float32")
    x2.requires_grad = True

    root = C.vt.add(x1, x2)

    grad_in = vt.ones_like(root)
    g1, g2 = A.grad(root, (x1, x2), grad_outputs=grad_in)

    stats = ag.stats()
    assert stats["engine_runs"] == 1

    # Expected gradients: d/dx (x1 + x2) = 1 for each input.
    expected_g1 = [1.0, 1.0]
    expected_g2 = [1.0, 1.0]

    assert _to_list(g1) == expected_g1
    assert _to_list(g2) == expected_g2

    assert x1.grad is not None and x2.grad is not None
    assert _to_list(x1.grad) == expected_g1  # type: ignore[arg-type]
    assert _to_list(x2.grad) == expected_g2  # type: ignore[arg-type]

    # Duplicated-input variant: both entries see the same gradient values.
    ag.reset_stats()

    x_dup = vt.tensor([2.0, -3.0], dtype="float32")
    x_dup.requires_grad = True
    y_dup = C.vt.add(x_dup, x_dup)
    grad_dup_in = vt.ones_like(y_dup)

    g_dup1, g_dup2 = A.grad(y_dup, (x_dup, x_dup), grad_outputs=grad_dup_in)
    stats_dup = ag.stats()

    assert stats_dup["engine_runs"] == 1
    assert _to_list(g_dup1) == _to_list(g_dup2)



def test_grad_mg4_non_differentiable_scalar_root_default_options_raise_and_preserve_grads() -> None:
    ag = _reset_engine_stats()

    x_used, x_unused, root = _make_non_diff_root_and_inputs()

    # Root is explicitly non-differentiable (MG4 precondition).
    assert root.grad_fn is None
    assert not bool(getattr(root, "requires_grad", False))

    # Inputs are valid leaf tensors that require grad, and .grad starts None.
    assert bool(getattr(x_used, "is_leaf", False))
    assert bool(getattr(x_used, "requires_grad", False))
    assert bool(getattr(x_unused, "is_leaf", False))
    assert bool(getattr(x_unused, "requires_grad", False))
    assert x_used.grad is None
    assert x_unused.grad is None

    with pytest.raises(RuntimeError) as exc:
        A.grad(root, (x_used, x_unused))
    assert str(exc.value) == (
        "vibetensor.autograd.grad: got unused input with allow_unused=False"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert x_used.grad is None
    assert x_unused.grad is None


# ---- MG5: allow_unused=True, materialize_grads=False -------------------------


def test_grad_mg5_differentiable_root_unused_input_returns_none() -> None:
    ag = _reset_engine_stats()

    x_used, x_unused, root = _make_diff_root_with_used_and_unused_input()
    grad_root = vt.full_like(root, 1.0)

    g_used, g_unused = A.grad(
        root,
        (x_used, x_unused),
        grad_outputs=grad_root,
        allow_unused=True,
        materialize_grads=False,
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 1

    assert g_unused is None
    assert x_unused.grad is None

    assert g_used is not None
    assert _to_list(g_used) == _to_list(x_used.grad)  # type: ignore[arg-type]


def test_grad_mg5_non_differentiable_root_returns_none_tuple_and_keeps_stats() -> None:
    ag = _reset_engine_stats()

    x_used, x_unused, root = _make_non_diff_root_and_inputs()
    assert root.grad_fn is None
    assert not bool(getattr(root, "requires_grad", False))

    g_used, g_unused = A.grad(
        root,
        (x_used, x_unused),
        allow_unused=True,
        materialize_grads=False,
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0

    assert g_used is None and g_unused is None
    assert x_used.grad is None and x_unused.grad is None


# ---- MG6: allow_unused=True, materialize_grads=True --------------------------


def test_grad_mg6_differentiable_root_unused_input_materializes_zero() -> None:
    ag = _reset_engine_stats()

    x_used, x_unused, root = _make_diff_root_with_used_and_unused_input()
    grad_root = vt.full_like(root, 1.0)

    g_used, g_unused = A.grad(
        root,
        (x_used, x_unused),
        grad_outputs=grad_root,
        allow_unused=True,
        materialize_grads=True,
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 1

    assert g_used is not None and g_unused is not None
    assert x_unused.grad is None

    # Zero tensor invariants
    assert tuple(g_unused.sizes) == tuple(x_unused.sizes)  # type: ignore[attr-defined]
    assert g_unused.device == x_unused.device
    assert g_unused.dtype == "float32"
    assert not bool(getattr(g_unused, "requires_grad", False))
    assert getattr(g_unused, "grad_fn", None) is None

    arr = np.from_dlpack(g_unused)
    assert np.all(arr == 0.0)


def test_grad_mg6_non_differentiable_root_materializes_zeros_and_keeps_stats() -> None:
    ag = _reset_engine_stats()

    x_used, x_unused, root = _make_non_diff_root_and_inputs()

    g_used, g_unused = A.grad(
        root,
        (x_used, x_unused),
        allow_unused=True,
        materialize_grads=True,
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0

    for x, g in zip((x_used, x_unused), (g_used, g_unused)):
        assert g is not None
        assert x.grad is None
        assert tuple(g.sizes) == tuple(x.sizes)  # type: ignore[attr-defined]
        assert g.device == x.device
        assert g.dtype == "float32"
        assert not bool(getattr(g, "requires_grad", False))
        assert getattr(g, "grad_fn", None) is None
        arr = np.from_dlpack(g)
        assert np.all(arr == 0.0)


# ---- MG7: invalid materialize_grads / allow_unused combinations --------------


def test_grad_mg7_materialize_true_allow_unused_false_valueerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    with pytest.raises(ValueError) as exc:
        A.grad(
            loss,
            x,
            allow_unused=False,
            materialize_grads=True,
        )
    assert str(exc.value) == (
        "vibetensor.autograd.grad: materialize_grads=True requires allow_unused=True or None"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_mg7_allow_unused_and_materialize_type_errors() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    with pytest.raises(TypeError) as exc1:
        A.grad(loss, x, allow_unused="yes")  # type: ignore[arg-type]
    assert str(exc1.value) == (
        "vibetensor.autograd.grad: allow_unused must be a bool or None"
    )

    with pytest.raises(TypeError) as exc2:
        A.grad(loss, x, materialize_grads=None)  # type: ignore[arg-type]
    assert str(exc2.value) == (
        "vibetensor.autograd.grad: materialize_grads must be a bool"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- MG8: is_grads_batched semantics -----------------------------------------


def test_grad_mg8_is_grads_batched_non_bool_typeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    with pytest.raises(TypeError) as exc:
        A.grad(loss, x, is_grads_batched=1)  # type: ignore[arg-type]
    assert str(exc.value) == (
        "vibetensor.autograd.grad: is_grads_batched must be a bool"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_mg8_is_grads_batched_true_not_implemented() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    with pytest.raises(NotImplementedError) as exc:
        A.grad(loss, x, is_grads_batched=True)
    assert str(exc.value).startswith(
        "vibetensor.autograd.grad: is_grads_batched=True is not supported"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- MG9: behavior under no_grad() and inference_mode() ----------------------


def test_grad_mg9_graph_built_under_grad_mode_grad_inside_no_grad_runs_engine() -> None:
    ag = _reset_engine_stats()

    # Baseline gradient via Tensor.backward on a non-scalar root.
    x_ref, y_ref, grad_in_ref = _make_non_scalar_root_and_grad()
    y_ref.backward(grad_in_ref)
    grad_ref = _to_list(x_ref.grad)  # type: ignore[arg-type]

    ag.reset_stats()

    # Same graph pattern, but call grad inside no_grad().
    x, y, grad_in = _make_non_scalar_root_and_grad()
    with vt.no_grad():
        (g,) = A.grad(y, x, grad_outputs=grad_in)

    stats = ag.stats()
    assert stats["engine_runs"] == 1
    assert _to_list(g) == grad_ref


def test_grad_mg9_graph_built_under_grad_mode_grad_inside_inference_mode_runs_engine() -> None:
    ag = _reset_engine_stats()

    x_ref, y_ref, grad_in_ref = _make_non_scalar_root_and_grad()
    y_ref.backward(grad_in_ref)
    grad_ref = _to_list(x_ref.grad)  # type: ignore[arg-type]

    ag.reset_stats()

    x, y, grad_in = _make_non_scalar_root_and_grad()
    with vt.inference_mode(True):
        (g,) = A.grad(y, x, grad_outputs=grad_in)

    stats = ag.stats()
    assert stats["engine_runs"] == 1
    assert _to_list(g) == grad_ref


def test_grad_mg9_graph_built_under_inference_mode_uses_stage4_early_return() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_inference_mode_non_diff_loss_and_input()
    assert loss.grad_fn is None
    assert not bool(getattr(loss, "requires_grad", False))

    # Default options (allow_unused_eff=False, materialize_eff=False) -> RuntimeError
    with pytest.raises(RuntimeError) as exc_default:
        A.grad(loss, x, allow_unused=False)
    assert str(exc_default.value) == (
        "vibetensor.autograd.grad: got unused input with allow_unused=False"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert x.grad is None

    # With allow_unused=True, materialize_grads=False -> None result
    g_none, = A.grad(loss, x, allow_unused=True, materialize_grads=False)
    assert g_none is None
    assert x.grad is None

    # With allow_unused=True, materialize_grads=True -> zero result
    g_zero, = A.grad(loss, x, allow_unused=True, materialize_grads=True)
    assert g_zero is not None
    assert x.grad is None
    assert tuple(g_zero.sizes) == tuple(x.sizes)  # type: ignore[attr-defined]
    assert g_zero.device == x.device
    assert g_zero.dtype == "float32"
    assert not bool(getattr(g_zero, "requires_grad", False))
    assert getattr(g_zero, "grad_fn", None) is None


# ---- Grad-outputs container tests -------------------------------------------


def test_grad_grad_outputs_none_non_scalar_root_matches_tensor_backward() -> None:
    ag = _reset_engine_stats()

    # Non-scalar root: grad_outputs=None should match Tensor.backward behavior
    x_mod, y_mod, _ = _make_non_scalar_root_and_grad()
    with pytest.raises(ValueError) as exc_mod:
        A.grad(y_mod, x_mod, grad_outputs=None)
    stats_mod = ag.stats()

    ag.reset_stats()
    x_t, y_t, _ = _make_non_scalar_root_and_grad()
    with pytest.raises(ValueError) as exc_t:
        y_t.backward()
    stats_t = ag.stats()

    assert str(exc_mod.value) == str(exc_t.value)
    assert stats_mod["engine_runs"] == stats_t["engine_runs"] == 0


def test_grad_grad_outputs_single_tensor_and_tuple_match() -> None:
    ag = _reset_engine_stats()

    x_single, y_single, grad_single_in = _make_non_scalar_root_and_grad()
    g_single, = A.grad(y_single, x_single, grad_outputs=grad_single_in)
    stats_single = ag.stats()

    ag.reset_stats()
    x_tuple, y_tuple, grad_tuple_in = _make_non_scalar_root_and_grad()
    g_tuple, = A.grad(
        y_tuple,
        x_tuple,
        grad_outputs=(grad_tuple_in,),
    )
    stats_tuple = ag.stats()

    assert stats_single["engine_runs"] == stats_tuple["engine_runs"] == 1
    assert _to_list(g_single) == _to_list(g_tuple)


def test_grad_grad_outputs_empty_sequence_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    for empty in ([], ()):  # type: ignore[list-item]
        with pytest.raises(RuntimeError) as exc:
            A.grad(loss, x, grad_outputs=empty)
        assert str(exc.value) == (
            "autograd.grad: grad_outputs must match outputs in length"
        )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_grad_outputs_sequence_with_non_tensor_or_none_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()
    g = vt.full_like(loss, 1.0)

    bad_values = ([g, object()], [None])  # type: ignore[list-item]
    for bad in bad_values:
        with pytest.raises(TypeError) as exc:
            A.grad(loss, x, grad_outputs=bad)
        assert str(exc.value) == (
            "autograd.grad: grad_outputs must be VibeTensor tensors"
        )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_grad_outputs_non_sequence_iterable_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()
    g = vt.full_like(loss, 1.0)

    def _gen():
        yield g

    with pytest.raises(TypeError) as exc:
        A.grad(loss, x, grad_outputs=_gen())
    assert str(exc.value) == (
        "autograd.grad: grad_outputs must be VibeTensor tensors"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- Container and semantic errors for outputs / inputs ---------------------


def test_grad_outputs_non_tensor_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    with pytest.raises(TypeError) as exc:
        A.grad(object(), x)
    assert str(exc.value) == "autograd.grad: outputs must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_outputs_sequence_with_non_tensor_element_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    for bad in ([loss, object()], [[loss]]):
        with pytest.raises(TypeError) as exc:
            A.grad(bad, x)
        assert str(exc.value) == "autograd.grad: outputs must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_outputs_empty_sequence_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    x, _ = _make_scalar_loss_with_history()

    for empty in ([], ()):  # type: ignore[list-item]
        with pytest.raises(RuntimeError) as exc:
            A.grad(empty, x)
        assert str(exc.value) == (
            "autograd.grad: outputs must not be an empty sequence"
        )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_outputs_multiple_tensors_raise_runtimeerror_and_no_engine() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()
    other = vt.tensor([1.0], dtype="float32")

    with pytest.raises(RuntimeError) as exc_list:
        A.grad([loss, other], x)
    assert str(exc_list.value) == (
        "vibetensor.autograd.grad: multiple outputs are not supported"
    )

    with pytest.raises(RuntimeError) as exc_tuple:
        A.grad((loss, other), x)
    assert str(exc_tuple.value) == (
        "vibetensor.autograd.grad: multiple outputs are not supported"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_inputs_non_tensor_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    with pytest.raises(TypeError) as exc:
        A.grad(loss, object())
    assert str(exc.value) == "autograd.grad: inputs must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_inputs_sequence_with_non_tensor_element_raises_typeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    for bad in ([x, object()], [[x]]):
        with pytest.raises(TypeError) as exc:
            A.grad(loss, bad)
        assert str(exc.value) == "autograd.grad: inputs must be VibeTensor tensors"

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_inputs_empty_sequence_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    x, loss = _make_scalar_loss_with_history()

    for empty in ([], ()):  # type: ignore[list-item]
        with pytest.raises(RuntimeError) as exc:
            A.grad(loss, empty)
        assert str(exc.value) == (
            "autograd.grad: inputs must not be an empty sequence"
        )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_inputs_non_leaf_or_requires_grad_false_raises_runtimeerror() -> None:
    ag = _reset_engine_stats()

    # Non-leaf input
    x_leaf, loss = _make_scalar_loss_with_history()
    y = C.vt.mul(x_leaf, x_leaf)

    with pytest.raises(RuntimeError) as exc_non_leaf:
        A.grad(loss, y)
    assert str(exc_non_leaf.value) == (
        "vibetensor.autograd.grad: inputs must be leaf CPU float32 tensors that require grad"
    )

    # requires_grad=False input
    x_nograd = vt.tensor([1.0], dtype="float32")
    x_true, loss_true = _make_scalar_loss_with_history()

    with pytest.raises(RuntimeError) as exc_req_false:
        A.grad(loss_true, (x_true, x_nograd))
    assert str(exc_req_false.value) == (
        "vibetensor.autograd.grad: inputs must be leaf CPU float32 tensors that require grad"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_grad_inputs_non_cpu_rejected_without_engine_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ag = _reset_engine_stats()

    x_real, loss = _make_scalar_loss_with_history()
    assert x_real.grad is None

    orig_is_vbt_tensor = A._is_vbt_tensor

    class _FakeTensor:
        def __init__(self) -> None:
            self.is_leaf = True
            self.requires_grad = True
            self.dtype = "float32"
            self.device = (2, 0)  # non-CPU dev_type in current builds

        def grad_tensor(self):  # type: ignore[no-untyped-def]
            return None

    def _fake_is_vbt_tensor(obj: object) -> bool:
        return isinstance(obj, _FakeTensor) or orig_is_vbt_tensor(obj)

    monkeypatch.setattr(A, "_is_vbt_tensor", _fake_is_vbt_tensor, raising=False)

    fake = _FakeTensor()

    with pytest.raises(RuntimeError) as exc:
        A.grad(loss, (x_real, fake))
    assert str(exc.value) == (
        "vibetensor.autograd.grad: inputs must be leaf CPU float32 tensors that require grad"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert x_real.grad is None



def test_grad_inputs_non_float32_rejected_without_engine_run() -> None:
    ag = _reset_engine_stats()

    x_f32, loss = _make_scalar_loss_with_history()
    x_int = vt.tensor([1, 2], dtype="int32")
    x_int.requires_grad = True

    assert x_f32.grad is None
    assert x_int.grad is None

    with pytest.raises(RuntimeError) as exc:
        A.grad(loss, (x_f32, x_int))
    assert str(exc.value) == (
        "vibetensor.autograd.grad: inputs must be leaf CPU float32 tensors that require grad"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert x_f32.grad is None
    assert x_int.grad is None



# ---- _clear_inputs_grad failure tests ---------------------------------------


def test_grad_clear_inputs_grad_extension_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ag = _reset_engine_stats()

    x_used, x_unused, root = _make_diff_root_with_used_and_unused_input()

    autograd_mod = C.autograd
    real_clear = getattr(autograd_mod, "_clear_tensor_grad", None)
    call_count = {"n": 0}

    def _clear_stub(t):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if real_clear is not None:
            real_clear(t)
        if call_count["n"] == 2:
            raise RuntimeError("clear failed")

    monkeypatch.setattr(autograd_mod, "_clear_tensor_grad", _clear_stub, raising=False)

    with pytest.raises(RuntimeError) as exc:
        A.grad(root, (x_used, x_unused))
    assert str(exc.value) == "clear failed"

    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert call_count["n"] == 2


def test_grad_clear_inputs_grad_descriptor_failure_yields_pinned_runtimeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ag = _reset_engine_stats()

    # Disable extension-level clear so we hit the descriptor fallback path.
    autograd_mod = C.autograd
    monkeypatch.setattr(autograd_mod, "_clear_tensor_grad", None, raising=False)

    orig_is_vbt_tensor = A._is_vbt_tensor

    class _FakeTensor:
        def __init__(self) -> None:
            self.is_leaf = True
            self.requires_grad = True
            self.dtype = "float32"
            self.device = (1, 0)

        @property
        def grad(self):  # type: ignore[no-untyped-def]
            return None

        @grad.setter
        def grad(self, value):  # type: ignore[no-untyped-def]
            raise AttributeError("boom")

        def grad_tensor(self):  # type: ignore[no-untyped-def]
            return None

    def _fake_is_vbt_tensor(obj: object) -> bool:
        return isinstance(obj, _FakeTensor) or orig_is_vbt_tensor(obj)

    monkeypatch.setattr(A, "_is_vbt_tensor", _fake_is_vbt_tensor, raising=False)

    x_real, root, _ = _make_non_scalar_root_and_grad()
    fake = _FakeTensor()

    with pytest.raises(RuntimeError) as exc:
        A.grad(root, (x_real, fake))
    assert str(exc.value) == (
        "vibetensor.autograd.grad: cannot clear .grad for an input tensor; "
        "_clear_tensor_grad is unavailable and Tensor.grad is not settable"
    )

    stats = ag.stats()
    assert stats["engine_runs"] == 0


# ---- Repeated grad tests ----------------------------------------------------


def test_grad_repeated_calls_fresh_graph_per_call() -> None:
    ag = _reset_engine_stats()

    x1, root1, grad1_in = _make_non_scalar_root_and_grad()
    g1, = A.grad(root1, x1, grad_outputs=grad1_in)
    stats1 = ag.stats()

    ag.reset_stats()
    x2, root2, grad2_in = _make_non_scalar_root_and_grad()
    g2, = A.grad(root2, x2, grad_outputs=grad2_in)
    stats2 = ag.stats()

    assert stats1["engine_runs"] == stats2["engine_runs"] == 1
    assert _to_list(g1) == _to_list(g2)


def test_grad_repeated_grad_on_same_graph_second_call_uses_stage4() -> None:
    ag = _reset_engine_stats()

    x, root, grad_in = _make_non_scalar_root_and_grad()

    g1, = A.grad(root, x, grad_outputs=grad_in)
    stats1 = ag.stats()
    grad_after_first = _to_list(x.grad)  # type: ignore[arg-type]

    assert stats1["engine_runs"] == 1

    ag.reset_stats()

    # Second call on the same graph with the same explicit gradient should
    # run the engine again and produce identical gradients.
    g2, = A.grad(root, x, grad_outputs=grad_in)
    stats2 = ag.stats()
    grad_after_second = _to_list(x.grad)  # type: ignore[arg-type]

    assert stats2["engine_runs"] == 1
    assert grad_after_first == grad_after_second
    assert _to_list(g1) == _to_list(g2) == grad_after_first
