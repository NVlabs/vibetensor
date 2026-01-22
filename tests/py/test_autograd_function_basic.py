# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor.autograd as A
import vibetensor.torch as vt
from vibetensor import _C as C
from vibetensor.autograd_graph import get_gradient_edge
from vibetensor.autograd import Function


class Square(Function):
    @staticmethod
    def forward(ctx, x):
        # x is a differentiable VibeTensor tensor
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        grad_x = 2.0 * x * grad_out
        return (grad_x,)


def _reset_stats():
    ag = C.autograd
    ag.reset_stats()
    return ag


def test_function_basic_forward_backward() -> None:
    ag = _reset_stats()

    x = vt.full([], 3.0, dtype="float32")
    x.requires_grad = True

    y = Square.apply(x)
    assert isinstance(y, C.Tensor)
    assert bool(getattr(y, "requires_grad", False))

    # Run backward and check gradient value.
    y.backward()

    grad = x.grad
    assert grad is not None
    assert float(grad.item()) == pytest.approx(6.0)

    stats = ag.stats()
    # One or more engine runs and at least one custom Function node created/applied.
    assert stats["engine_runs"] >= 1
    assert stats["py_function_nodes_created"] >= 1
    assert stats["py_function_nodes_applied"] >= 1
    assert stats["py_function_backward_failures"] == 0

    # Graph inspection should see a PyFunctionNode named "SquareBackward".
    edge = get_gradient_edge(y)
    node = edge.node
    assert "SquareBackward" in node.name


def test_function_no_grad_skips_graph() -> None:
    ag = _reset_stats()

    x = vt.full([], 3.0, dtype="float32")
    x.requires_grad = True

    with A.no_grad():
        y = Square.apply(x)

    # Under no_grad, no autograd history is recorded.
    assert getattr(y, "grad_fn", None) is None

    stats = ag.stats()
    assert stats["engine_runs"] == 0
    assert stats["py_function_nodes_created"] == 0
    assert stats["py_function_nodes_applied"] == 0


def test_function_non_requires_grad_input() -> None:
    ag = _reset_stats()

    x = vt.full([], 3.0, dtype="float32")
    x.requires_grad = False

    y = Square.apply(x)

    # Input does not require grad, so output should not either and no node is created.
    assert not bool(getattr(y, "requires_grad", False))
    assert getattr(y, "grad_fn", None) is None

    stats = ag.stats()
    assert stats["py_function_nodes_created"] == 0
    assert stats["py_function_nodes_applied"] == 0


class ForwardOnly(Function):
    @staticmethod
    def forward(ctx, x):
        return x * x
    # Intentionally no backward override


def test_function_backward_missing_impl_raises_on_apply() -> None:
    x = vt.full([], 2.0, dtype="float32")
    x.requires_grad = True

    msg = (
        "vibetensor.autograd.Function subclass ForwardOnly must override "
        "backward(ctx, grad_output)"
    )
    with pytest.raises(RuntimeError) as exc:
        ForwardOnly.apply(x)
    assert msg in str(exc.value)
