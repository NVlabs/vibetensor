# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor  # noqa: F401 - import-time patches
import vibetensor._C as C
import vibetensor.autograd as A
import vibetensor.torch as vt


def _reset_state() -> None:
    ag = C.autograd
    # Best-effort reset of raw grad mode and inference flags.
    if hasattr(ag, "_set_inference_mode_enabled"):
        ag._set_inference_mode_enabled(False)  # type: ignore[attr-defined]
    if hasattr(ag, "set_grad_enabled"):
        ag.set_grad_enabled(True)


def test_no_grad_enable_grad_nesting_and_exceptions():
    _reset_state()
    initial_graph = A.is_grad_enabled()

    # vibetensor.autograd helpers
    with A.no_grad():
        assert A.is_grad_enabled() is False
        with A.enable_grad():
            assert A.is_grad_enabled() is True
        assert A.is_grad_enabled() is False
    assert A.is_grad_enabled() == initial_graph

    # vibetensor.torch aliases
    with vt.no_grad():
        assert vt.is_grad_enabled() is False
        with vt.enable_grad():
            assert vt.is_grad_enabled() is True
        assert vt.is_grad_enabled() is False
    assert vt.is_grad_enabled() == initial_graph

    class Boom(Exception):
        pass

    # Exception paths still restore state
    with pytest.raises(Boom):
        with vt.no_grad():
            assert vt.is_grad_enabled() is False
            raise Boom()
    assert vt.is_grad_enabled() == initial_graph


def test_set_grad_enabled_function_and_queries():
    _reset_state()
    ag = C.autograd

    # Start from True raw grad-mode
    ag.set_grad_enabled(True)
    assert ag._raw_grad_mode_enabled() is True  # type: ignore[attr-defined]

    # Disable via Python helper
    A.set_grad_enabled(False)
    assert ag._raw_grad_mode_enabled() is False  # type: ignore[attr-defined]
    assert A.is_grad_enabled() is False
    assert vt.is_grad_enabled() is False

    # Re-enable
    A.set_grad_enabled(True)
    assert ag._raw_grad_mode_enabled() is True  # type: ignore[attr-defined]
    assert A.is_grad_enabled() is True
    assert vt.is_grad_enabled() is True


def test_set_grad_enabled_misuse_raises():
    _reset_state()

    # Using as a context manager should fail (returns None, not a CM)
    with pytest.raises((AttributeError, TypeError)):
        with vt.set_grad_enabled(True):  # type: ignore[misc]
            pass

    # Using as a decorator should eventually fail when calling the wrapped fn
    with pytest.raises(TypeError):

        @A.set_grad_enabled  # type: ignore[misc]
        def _foo():  # pragma: no cover - simple decorator misuse
            return 1

        _foo()


def test_aliasing_between_surfaces_no_grad_enable_grad():
    ag = C.autograd

    # _C.autograd and vibetensor.autograd share the same helpers
    assert ag.no_grad is A.no_grad  # type: ignore[attr-defined]
    assert ag.enable_grad is A.enable_grad  # type: ignore[attr-defined]

    # vibetensor.torch re-exports the same helpers
    assert vt.no_grad is A.no_grad
    assert vt.enable_grad is A.enable_grad


def test_is_grad_enabled_matches_raw_and_inference_state():
    _reset_state()
    ag = C.autograd

    # Baseline: raw=True, inf=False → graph_on=True
    ag.set_grad_enabled(True)
    if hasattr(ag, "_set_inference_mode_enabled"):
        ag._set_inference_mode_enabled(False)  # type: ignore[attr-defined]
    assert A.is_grad_enabled() is True

    # raw=False, inf=False → graph_on=False
    ag.set_grad_enabled(False)
    assert A.is_grad_enabled() is False

    # raw=True, inf=True → graph_on=False
    ag.set_grad_enabled(True)
    if hasattr(ag, "_set_inference_mode_enabled"):
        ag._set_inference_mode_enabled(True)  # type: ignore[attr-defined]
    assert A.is_grad_enabled() is False
