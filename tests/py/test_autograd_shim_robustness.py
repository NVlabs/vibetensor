# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import vibetensor  # noqa: F401 - import-time patches
import vibetensor._C as C
import vibetensor.autograd as A
import vibetensor.torch as vt


@contextlib.contextmanager
def _swap_autograd(new_ag):
    """Temporarily replace ``vibetensor._C.autograd`` for shim tests.

    This exercises the defensive branches in :mod:`vibetensor.autograd` without
    permanently affecting global state for other tests.
    """

    orig = getattr(C, "autograd", None)
    setattr(C, "autograd", new_ag)
    try:
        yield
    finally:
        setattr(C, "autograd", orig)


def test_helpers_noop_when_C_autograd_is_missing():
    """All helpers degrade to no-op behavior when `_C.autograd` is None."""

    with _swap_autograd(None):
        # Queries fall back to safe defaults.
        assert A.is_grad_enabled() is False
        assert A.is_inference_mode_enabled() is False

        # Context managers must not raise and must be nestable.
        with A.no_grad():
            with A.enable_grad():
                with A.inference_mode():
                    pass

        # The torch overlay helpers are thin aliases and should behave
        # identically.
        assert vt.is_grad_enabled() is False
        assert vt.is_inference_mode_enabled() is False

        with vt.no_grad():
            with vt.enable_grad():
                with vt.inference_mode():
                    pass


class _GradOnlyAutograd:
    """Minimal grad-mode surface with no inference helpers.

    This object intentionally lacks ``_raw_grad_mode_enabled`` and any
    inference-mode APIs so that the Python helpers exercise their fallback
    paths.
    """

    def __init__(self) -> None:
        self._raw = True

    def is_grad_enabled(self) -> bool:  # pragma: no cover - trivial
        return self._raw

    def set_grad_enabled(self, v: bool) -> None:  # pragma: no cover - trivial
        self._raw = bool(v)


def test_no_grad_enable_grad_degrade_without_raw_helper():
    """no_grad/enable_grad fall back to graph-enabled snapshots when needed."""

    dummy = _GradOnlyAutograd()

    with _swap_autograd(dummy):
        assert A.is_grad_enabled() is True

        with A.no_grad():
            assert A.is_grad_enabled() is False
        assert A.is_grad_enabled() is True

        with A.enable_grad():
            assert A.is_grad_enabled() is True
        assert A.is_grad_enabled() is True

        # inference_mode has no helpers and should behave as a no-op.
        with A.inference_mode():
            assert A.is_grad_enabled() is True
        assert A.is_inference_mode_enabled() is False


class _PartialInferenceAutograd:
    """Grad + inference flags without the raw helper.

    This hits the partial-build branch in ``inference_mode`` where we can
    toggle the inference flag but not the raw TLS bit directly.
    """

    def __init__(self) -> None:
        self._raw = True
        self._inf = False

    def is_grad_enabled(self) -> bool:  # pragma: no cover - trivial
        return self._raw and not self._inf

    def set_grad_enabled(self, v: bool) -> None:  # pragma: no cover - trivial
        self._raw = bool(v)

    def is_inference_mode_enabled(self) -> bool:  # pragma: no cover - trivial
        return self._inf

    def _set_inference_mode_enabled(self, v: bool) -> None:  # pragma: no cover - trivial
        self._inf = bool(v)


def test_inference_mode_partial_build_uses_no_grad_and_restores():
    """Partial builds still suppress graphs and restore state correctly."""

    dummy = _PartialInferenceAutograd()

    with _swap_autograd(dummy):
        assert A.is_grad_enabled() is True
        assert A.is_inference_mode_enabled() is False

        with A.inference_mode():
            # inference flag is set and grad is suppressed via no_grad()
            assert A.is_inference_mode_enabled() is True
            assert A.is_grad_enabled() is False

        # Both flags are restored on exit.
        assert A.is_inference_mode_enabled() is False
        assert A.is_grad_enabled() is True
