# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import typing

import pytest

import vibetensor.torch.cuda as vc

if not hasattr(vc, "graphs"):
    pytest.skip("CUDA Graphs Python overlay not available for this build", allow_module_level=True)

from vibetensor.torch.cuda import graphs as vgraphs


def test_make_graphed_callables_not_implemented_single_callable():
    def f(x):  # pragma: no cover (body is never run)
        return x

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        vgraphs.make_graphed_callables(f, sample_args=(object(),))


def test_make_graphed_callables_not_implemented_tuple_of_callables():
    def f(x):  # pragma: no cover
        return x

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        vgraphs.make_graphed_callables(
            (f, f), sample_args=((object(),), (object(),))
        )


def test_make_graphed_callables_stub_does_not_touch_cuda(monkeypatch):
    if hasattr(vgraphs, "_cuda_available"):
        monkeypatch.setattr(
            vgraphs,
            "_cuda_available",
            lambda: (_ for _ in ()).throw(
                RuntimeError("_cuda_available should not be called")
            ),
            raising=False,
        )

    # If _C exists, replace it with a harmless sentinel; any attribute access
    # that would have used real bindings will now fail noisily in future tests.
    monkeypatch.setattr(vgraphs, "_C", object(), raising=False)

    def f(x):  # pragma: no cover
        return x

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        vgraphs.make_graphed_callables(f, sample_args=(object(),))


def test_make_graphed_callables_does_not_invoke_user_callable():
    seen = []

    def f(x):  # pragma: no cover
        seen.append(x)
        return x

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        vgraphs.make_graphed_callables(f, sample_args=(object(),))

    assert seen == []


def test_make_graphed_callables_export_and_reexport():
    assert hasattr(vgraphs, "make_graphed_callables")

    # Top-level re-export exists and refers to the same function object.
    assert hasattr(vc, "make_graphed_callables")
    assert vc.make_graphed_callables is vgraphs.make_graphed_callables


def test_make_graphed_callables_in_all_lists():
    assert "make_graphed_callables" in getattr(vgraphs, "__all__", [])
    assert "make_graphed_callables" in getattr(vc, "__all__", [])


def test_make_graphed_callables_signature_matches_design():
    sig = inspect.signature(vgraphs.make_graphed_callables)
    params = list(sig.parameters.values())

    names = [p.name for p in params]
    assert names == [
        "callables",
        "sample_args",
        "num_warmup_iters",
        "allow_unused_input",
        "pool",
    ]

    # All parameters are positional-or-keyword, as in upstream.
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)

    assert sig.parameters["num_warmup_iters"].default == 3
    assert sig.parameters["allow_unused_input"].default is False
    assert sig.parameters["pool"].default is None


def test_make_graphed_callables_annotations_are_reasonable():
    hints = typing.get_type_hints(vgraphs.make_graphed_callables)

    # pool annotation should obviously involve GraphPoolHandle and Optional-like
    # semantics, but we only assert on the repr to stay robust.
    pool_ann = hints["pool"]
    pool_repr = repr(pool_ann)
    assert "GraphPoolHandle" in pool_repr

    # Return annotation should obviously involve a callable arm and a tuple arm.
    ret_ann = hints["return"]
    ret_repr = repr(ret_ann)
    assert "Callable" in ret_repr
    assert "tuple" in ret_repr
