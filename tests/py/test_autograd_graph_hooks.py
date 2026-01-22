# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import vibetensor.torch as vt
import vibetensor.autograd as A
from vibetensor import _C as _C


# ---------------------------------------------------------------------------
# Graph inspection
# ---------------------------------------------------------------------------


def _cpu_float32(x):
    return vt.tensor(x, dtype="float32")


def test_graph_basic_edge_structure():
    x = _cpu_float32([2.0])
    x.requires_grad = True
    y = vt.relu(x * x)

    edge = A.graph.get_gradient_edge(y)
    node = edge.node

    assert isinstance(node, A.graph.Node)
    assert edge.output_nr == 0
    assert isinstance(node.name, str) and node.name

    next_funcs = node.next_functions
    assert isinstance(next_funcs, tuple)
    for child, input_nr in next_funcs:
        assert isinstance(input_nr, int)
        assert child is None or isinstance(child, A.graph.Node)


def test_graph_leaf_view_uses_accumulategrad():
    x = _cpu_float32([2.0, 3.0])
    x.requires_grad = True
    v = x[0:1]

    edge = A.graph.get_gradient_edge(v)
    assert isinstance(edge.node, A.graph.Node)
    # Synthetic AccumulateGrad node used for leaf views.
    assert edge.node.name == "AccumulateGrad"
    assert edge.output_nr == 0


def test_iter_nodes_deduplicates_and_respects_max_depth():
    x = _cpu_float32([1.0])
    x.requires_grad = True
    y = vt.relu(x * x)

    edge = A.graph.get_gradient_edge(y)
    nodes = list(A.graph.iter_nodes(edge))
    # At least two nodes: AccumulateGrad sink and one backward node.
    assert len(nodes) >= 2

    ids = {n.metadata()["debug_id"] for n in nodes}
    assert len(ids) == len(nodes)

    shallow = list(A.graph.iter_nodes(edge, max_depth=1))
    assert len(shallow) <= len(nodes)


# ---------------------------------------------------------------------------
# Saved-tensor hooks
# ---------------------------------------------------------------------------


def test_saved_tensors_hooks_basic_invocation_and_grad_stability():
    x = _cpu_float32([2.0])
    x.requires_grad = True

    seen_pack = []
    seen_unpack = []

    def pack(t):
        # Hooks see cloned, detached tensors; type is implementation-dependent.
        seen_pack.append(t)
        return {"shape": tuple(int(s) for s in getattr(t, "sizes", []))}

    def unpack(payload):  # noqa: D401 - simple recorder
        seen_unpack.append(payload)

    with A.graph.saved_tensors_hooks(pack, unpack):
        y = vt.relu(x * x)
        y.backward()

    assert seen_pack
    assert seen_unpack

    g = x.grad_tensor()
    assert isinstance(g, _C.Tensor)
    assert tuple(int(s) for s in g.sizes) == (1,)


def test_disable_saved_tensors_hooks_blocks_new_contexts():
    x = _cpu_float32([1.0])
    x.requires_grad = True

    def pack(t):  # pragma: no cover - error path
        return None

    def unpack(payload):  # pragma: no cover - error path
        return None

    with A.graph.disable_saved_tensors_hooks("hooks disabled for test"):
        with pytest.raises(RuntimeError) as excinfo:
            with A.graph.saved_tensors_hooks(pack, unpack):
                vt.relu(x * x)
        assert "hooks disabled" in str(excinfo.value)


def test_saved_tensors_hooks_reject_tensor_payload():
    x = _cpu_float32([1.0])
    x.requires_grad = True

    def pack(t):
        return t

    def unpack(payload):  # pragma: no cover - not reached
        return None

    with pytest.raises(RuntimeError):
        with A.graph.saved_tensors_hooks(pack, unpack):
            vt.relu(x * x)


# ---------------------------------------------------------------------------
# Multi-tensor gradient hooks
# ---------------------------------------------------------------------------


def test_multi_grad_hook_mode_all_basic():
    a = _cpu_float32([1.0, 2.0])
    b = _cpu_float32([3.0, 4.0])
    a.requires_grad = True
    b.requires_grad = True

    calls: list[list[object | None]] = []

    def hook(grads):
        calls.append(list(grads))

    handle = A.graph.register_multi_grad_hook((a, b), hook, mode="all")

    y = a * b
    grad_seed = vt.ones_like(y)
    y.backward(grad_seed)

    assert len(calls) == 1
    grads = calls[0]
    assert len(grads) == 2
    # Each entry is a tensor or None; type is implementation-dependent.
    assert all((g is None) or hasattr(g, "shape") for g in grads)

    # Removal stops future calls.
    handle.remove()
    z = a * b
    # Clear grads to avoid accumulation affecting behavior.
    a.grad = None  # type: ignore[assignment]
    b.grad = None  # type: ignore[assignment]
    grad_seed2 = vt.ones_like(z)
    z.backward(grad_seed2)
    assert len(calls) == 1


def test_multi_grad_hook_mode_any_triggers_on_first_grad():
    a = _cpu_float32([1.0])
    b = _cpu_float32([2.0])
    a.requires_grad = True
    b.requires_grad = True

    grads_seen: list[object | None] = []

    def hook(grad):
        grads_seen.append(grad)

    A.graph.register_multi_grad_hook((a, b), hook, mode="any")

    y = a * b
    grad_seed = vt.ones_like(y)
    y.backward(grad_seed)

    assert len(grads_seen) == 1
    g = grads_seen[0]
    assert g is None or hasattr(g, "shape")


def test_multi_grad_hook_no_gradients_means_no_calls():
    a = _cpu_float32([1.0])
    a.requires_grad = True

    calls: list[list[object | None]] = []

    def hook(grads):
        calls.append(list(grads))

    # Register on ``a`` but build a graph that does not use it.
    A.graph.register_multi_grad_hook((a,), hook, mode="all")

    b = _cpu_float32([2.0])
    b.requires_grad = True
    y = b * b
    grad_seed = vt.ones_like(y)
    y.backward(grad_seed)

    assert calls == []


def test_autograd_stats_counters_for_graph_and_hooks():
    ag = _C.autograd
    ag.reset_stats()

    # Graph inspection bumps node/edge counters.
    x = _cpu_float32([1.0])
    x.requires_grad = True
    y = vt.relu(x * x)
    edge = A.graph.get_gradient_edge(y)
    list(A.graph.iter_nodes(edge))

    s = ag.stats()
    assert s["graph_nodes_exposed"] >= 1
    assert s["graph_edges_exposed"] >= 0

    # Saved-tensor hooks bump pack/unpack counters and track violations.
    ag.reset_stats()

    def pack(t):
        # Observe a lightweight payload so unpack is exercised.
        return {"shape": tuple(int(s) for s in getattr(t, "sizes", []))}

    def unpack(payload):
        return None

    with A.graph.saved_tensors_hooks(pack, unpack):
        y2 = vt.relu(x * x)
        y2.backward()

    s = ag.stats()
    assert s["saved_tensors_packed"] >= 1
    assert s["saved_tensors_unpacked"] >= 1
    assert s["saved_tensors_hook_violations"] == 0

    # Tensor-return violation increments the violation counter.
    ag.reset_stats()

    def pack_violation(t):  # pragma: no cover - simple stats path
        return t

    def unpack_violation(payload):  # pragma: no cover - simple stats path
        return None

    with pytest.raises(RuntimeError):
        with A.graph.saved_tensors_hooks(pack_violation, unpack_violation):
            vt.relu(x * x)

    s = ag.stats()
    assert s["saved_tensors_hook_violations"] >= 1

    # Multi-grad hooks bump registration and fire counters.
    ag.reset_stats()
    a = _cpu_float32([1.0])
    b = _cpu_float32([2.0])
    a.requires_grad = True
    b.requires_grad = True

    def hook_all(grads):
        return None

    def hook_any(grad):
        return None

    A.graph.register_multi_grad_hook((a, b), hook_all, mode="all")
    A.graph.register_multi_grad_hook((a, b), hook_any, mode="any")

    y = a * b
    grad_seed = vt.ones_like(y)
    y.backward(grad_seed)

    s = ag.stats()
    assert s["multi_grad_hooks_registered"] >= 2
    assert s["multi_grad_hooks_fired_all"] >= 1
    assert s["multi_grad_hooks_fired_any"] >= 1


def test_multi_grad_hook_validates_fn_and_mode_and_tensors():
    a = _cpu_float32([1.0])
    a.requires_grad = True

    # Non-callable fn is rejected.
    with pytest.raises(TypeError, match="fn must be callable"):
        A.graph.register_multi_grad_hook((a,), None, mode="all")

    # Invalid mode string is rejected.
    with pytest.raises(ValueError, match="mode must be 'all' or 'any'"):
        A.graph.register_multi_grad_hook((a,), lambda g: None, mode="foo")

    # Tensor container type and elements are validated.
    with pytest.raises(TypeError, match="tensors must be VibeTensor `_C.Tensor` objects"):
        A.graph.register_multi_grad_hook((a, 123), lambda g: None, mode="all")


def test_multi_grad_hook_rejects_registration_during_backward():
    a = _cpu_float32([1.0])
    a.requires_grad = True

    def bad_hook(grad):
        # Attempt to register a multi-grad hook from inside backward.
        A.graph.register_multi_grad_hook((a,), lambda g: None, mode="all")
        return None

    h = a.register_hook(bad_hook)

    y = a * a
    grad_seed = vt.ones_like(y)
    with pytest.raises(RuntimeError, match="cannot be called while a backward is in progress"):
        y.backward(grad_seed)

    h.remove()
