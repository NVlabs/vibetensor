# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor  # noqa: F401 - import-time patches
import vibetensor._C as C


def _reset_state() -> None:
    ag = C.autograd
    # Best-effort reset of raw grad-mode and inference flags.
    if hasattr(ag, "_set_inference_mode_enabled"):
        ag._set_inference_mode_enabled(False)  # type: ignore[attr-defined]
    if hasattr(ag, "set_grad_enabled"):
        ag.set_grad_enabled(True)


def _raw_or_graph_enabled(ag) -> bool:
    """Return a baseline boolean for grad-mode from _C.autograd.

    Prefer the dedicated raw helper when available, otherwise fall back to the
    graph-enabled query. This matches how older builds exposed only
    ``is_grad_enabled``.
    """

    raw_get = getattr(ag, "_raw_grad_mode_enabled", None)
    if callable(raw_get):
        return bool(raw_get())
    return bool(getattr(ag, "is_grad_enabled", lambda: False)())


def test_C_autograd_no_grad_enable_grad_nesting_and_exceptions():
    """Baseline RAII semantics for `_C.autograd.no_grad` / `.enable_grad`.

    This test captures the current behavior of the low-level context managers
    without going through the higher-level ``vibetensor.autograd`` helpers.
    It focuses on nesting, save/restore, and exception safety.
    """

    _reset_state()
    ag = C.autograd

    initial_raw = _raw_or_graph_enabled(ag)
    initial_graph = bool(ag.is_grad_enabled())

    # Nesting: no_grad outside, enable_grad inside.
    with ag.no_grad():  # type: ignore[attr-defined]
        assert ag.is_grad_enabled() is False
        with ag.enable_grad():  # type: ignore[attr-defined]
            assert ag.is_grad_enabled() is True
        assert ag.is_grad_enabled() is False

    # State after the outer context exits should match the initial state.
    assert bool(ag.is_grad_enabled()) == initial_graph
    assert _raw_or_graph_enabled(ag) == initial_raw

    class Boom(Exception):
        pass

    # Exception paths still restore the original state.
    with pytest.raises(Boom):
        with ag.no_grad():  # type: ignore[attr-defined]
            assert ag.is_grad_enabled() is False
            raise Boom()

    assert bool(ag.is_grad_enabled()) == initial_graph
    assert _raw_or_graph_enabled(ag) == initial_raw
