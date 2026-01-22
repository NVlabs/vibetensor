# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _make_leaf_vector(vals):
    t = vt.tensor(list(vals), dtype="float32")
    t.requires_grad = True
    return t


def _cpu_grad_like(t, fill):
    sizes = list(int(s) for s in getattr(t, "sizes"))
    return C._cpu_full(sizes, "float32", float(fill))


def _to_list(t):
    return np.from_dlpack(t).tolist()


def test_grad_lifecycle_basic_and_reset():
    x = _make_leaf_vector([2.0])
    # y = x * x
    y = C.vt.mul(x, x)
    grad_seed = _cpu_grad_like(y, 1.0)
    y.backward(grad_seed)

    g = x.grad
    assert g is not None
    assert g.requires_grad is False
    assert _to_list(g) == [4.0]

    # Reset grads and ensure a second backward repopulates .grad
    x.grad = None
    assert x.grad is None

    y2 = C.vt.mul(x, x)
    grad_seed2 = _cpu_grad_like(y2, 1.0)
    y2.backward(grad_seed2)
    g2 = x.grad
    assert g2 is not None
    assert _to_list(g2) == [4.0]


def test_grad_method_and_grad_tensor_match_property():
    x = _make_leaf_vector([2.0])
    y = C.vt.mul(x, x)
    grad_seed = _cpu_grad_like(y, 1.0)
    y.backward(grad_seed)

    # Property returns a proxy but behaves like a tensor.
    g_attr = x.grad
    assert g_attr is not None
    assert _to_list(g_attr) == [4.0]
    assert g_attr.requires_grad is False

    # Calling x.grad() still works and returns the underlying tensor.
    g_method = x.grad()
    assert _to_list(g_method) == [4.0]
    assert g_method.requires_grad is False

    # grad_tensor() also exposes the same gradient values.
    g_tensor = x.grad_tensor()
    assert _to_list(g_tensor) == [4.0]
    assert g_tensor.requires_grad is False


def test_grad_is_none_for_nonleaf_and_views():
    x = _make_leaf_vector([2.0, 3.0])
    v = x[0:1]  # view of x
    assert v.is_leaf is False  # view is not a leaf

    y = C.vt.mul(v, v)
    grad_seed = _cpu_grad_like(y, 1.0)
    y.backward(grad_seed)

    # Gradients accumulate into the base tensor only
    assert x.grad is not None
    assert _to_list(x.grad) == [4.0, 0.0]
    assert v.grad is None
    assert y.grad is None


def test_requires_grad_leaf_and_nonleaf_invariants():
    x = vt.tensor([1.0, 2.0], dtype="float32")
    assert x.requires_grad is False

    x.requires_grad = True
    assert x.requires_grad is True

    # Non-leaf tensors may not toggle requires_grad directly
    y = C.vt.mul(x, x)
    assert y.is_leaf is False
    with pytest.raises(RuntimeError):
        y.requires_grad = True
    with pytest.raises(RuntimeError):
        y.requires_grad = False

    # Views also reject setting requires_grad
    v = x[0:1]
    assert v.is_leaf is False
    with pytest.raises(RuntimeError):
        v.requires_grad = True


def test_is_leaf_and_grad_fn_properties():
    x = vt.tensor([1.0], dtype="float32")
    assert x.is_leaf is True
    assert x.grad_fn is None

    x.requires_grad = True
    y = C.vt.mul(x, x)
    assert y.is_leaf is False
    assert y.grad_fn is not None
    assert isinstance(y.grad_fn.name, str)
    assert y.grad_fn.name == "MulBackward"

    # Detach creates a fresh leaf without grad_fn
    z = y.detach()
    assert z.is_leaf is True
    assert z.grad_fn is None
    assert z.requires_grad is False


def test_detach_semantics_and_independent_history():
    x = _make_leaf_vector([2.0])
    y = C.vt.mul(x, x)

    z = y.detach()
    assert z.requires_grad is False
    assert z.is_leaf is True
    assert z.grad_fn is None

    # Build a new branch from z; gradients should not flow back to x
    z.requires_grad = True
    w = C.vt.mul(z, z)
    grad_seed = _cpu_grad_like(w, 1.0)
    w.backward(grad_seed)

    # x had no grad_fn in this graph; its .grad remains None
    assert x.grad is None
    # z receives gradients for the detached branch
    assert z.grad is not None
    assert _to_list(z.grad) == [8.0]


def test_detach_inplace_preconditions_and_effects():
    # Valid in-place detach on a leaf without grad_fn
    x = vt.tensor([1.0], dtype="float32")
    x.requires_grad = True
    assert x.is_leaf is True
    assert x.grad_fn is None

    x_det = x.detach_()
    assert x_det is x
    assert x.requires_grad is False
    assert x.is_leaf is True
    assert x.grad_fn is None

    # Non-leaf tensors reject detach_()
    x2 = _make_leaf_vector([2.0])
    y2 = C.vt.mul(x2, x2)
    with pytest.raises(RuntimeError):
        y2.detach_()


def test_retain_grad_and_register_hook_behavior():
    x = _make_leaf_vector([3.0])
    x.retain_grad()  # no-op but should be accepted

    seen = []

    def hook(g):
        arr = np.from_dlpack(g)
        seen.append(arr.copy())
        # Mutate the hook argument; .grad should remain unchanged. NumPy's
        # from_dlpack may return a read-only view, so copy if needed.
        if not arr.flags.writeable:
            arr = arr.copy()
        arr[...] = 0.0
        assert g.requires_grad is False

    handle = x.register_hook(hook)

    y = C.vt.mul(x, x)
    grad_seed = _cpu_grad_like(y, 1.0)
    y.backward(grad_seed)

    assert x.grad is not None
    assert _to_list(x.grad) == [6.0]
    assert len(seen) == 1
    assert seen[0].tolist() == [6.0]

    # .remove() stops future invocations
    handle.remove()
    x.grad = None
    seen.clear()

    y2 = C.vt.mul(x, x)
    grad_seed2 = _cpu_grad_like(y2, 1.0)
    y2.backward(grad_seed2)
    assert x.grad is not None
    assert _to_list(x.grad) == [6.0]
    assert seen == []

    # retain_grad/register_hook on non-leaves or views raise
    v = x[0:1]
    with pytest.raises(RuntimeError):
        v.retain_grad()
    with pytest.raises(RuntimeError):
        v.register_hook(lambda g: g)


def test_backward_no_history_is_noop_and_stats_unchanged():
    ag = C.autograd
    ag.reset_stats()

    x = vt.tensor([1.0], dtype="float32")
    assert x.grad is None

    # x has no grad_fn and does not require grad; backward is a no-op
    grad_seed = _cpu_grad_like(x, 1.0)
    x.backward(grad_seed)

    assert x.grad is None
    stats = ag.stats()
    assert stats["engine_runs"] == 0


def test_nested_backward_guard_from_hook():
    ag = C.autograd
    ag.reset_stats()

    x = _make_leaf_vector([2.0])
    y = C.vt.mul(x, x)

    calls = []

    def hook(g):
        calls.append(_to_list(g))
        with pytest.raises(RuntimeError):
            # Nested backward from inside a hook should hit the guard
            grad_seed_inner = _cpu_grad_like(y, 1.0)
            y.backward(grad_seed_inner)

    x.register_hook(hook)

    grad_seed = _cpu_grad_like(y, 1.0)
    y.backward(grad_seed)

    # Outer backward succeeded exactly once
    assert x.grad is not None
    assert _to_list(x.grad) == [4.0]
    assert len(calls) == 1

    stats = ag.stats()
    assert stats["engine_runs"] == 1

    # Subsequent independent backward calls still work (guard was reset)
    x.grad = None
    grad_seed2 = _cpu_grad_like(y, 1.0)
    y.backward(grad_seed2)
    assert _to_list(x.grad) == [4.0]
