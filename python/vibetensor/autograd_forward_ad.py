# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from typing import Any, Tuple

from vibetensor import _C as _C

_Tensor = _C.Tensor

# Thread-local forward AD state for Python surface. We keep only a handle to
# the currently active level; C++ owns the real TLS flags and level id.
_state = threading.local()
_state.current_level = None  # type: _DualLevel | None  # type: ignore[name-defined]


class _DualLevel:
    def __init__(self) -> None:
        self._level_id: int | None = None
        self._active: bool = False

    @property
    def level_id(self) -> int:
        if self._level_id is None:
            raise RuntimeError(
                "forward_ad.DualLevel: level is not active"
            )
        return self._level_id

    def __enter__(self) -> "_DualLevel":  # pragma: no cover - exercised via tests
        import vibetensor.autograd as A

        if getattr(_state, "current_level", None) is not None:
            raise RuntimeError(
                "vibetensor.autograd.forward_ad.dual_level: nested forward-mode "
                "levels are not supported"
            )

        if A.is_inference_mode_enabled():
            raise RuntimeError(
                "vibetensor.autograd.forward_ad.dual_level: cannot enter forward-mode "
                "inside inference_mode"
            )

        level_id = _C.autograd._enter_dual_level()
        self._level_id = int(level_id)
        self._active = True
        _state.current_level = self
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - exercised via tests
        try:
            if self._active and self._level_id is not None:
                _C.autograd._exit_dual_level(self._level_id)
        finally:
            self._active = False
            _state.current_level = None


def dual_level() -> _DualLevel:
    """Create a new forward-mode level (single-level only).

    All tangents attached via :func:`make_dual` or created by forward-mode rules
    while the level is active are cleared on exit. Nested use is rejected.
    """

    return _DualLevel()


def _require_tensor(value: Any, *, arg_name: str) -> _Tensor:
    if not isinstance(value, _C.Tensor):
        raise TypeError(
            f"vibetensor.autograd.forward_ad.{arg_name}: expected a VibeTensor "
            f"Tensor, got {type(value)!r}"
        )
    return value


def make_dual(primal: Any, tangent: Any, *, level: _DualLevel | None = None) -> _Tensor:
    """Attach a forward-mode tangent to ``primal`` and return the dual.

    Currently:

    - Only CPU float32 tensors are supported for both arguments.
    - Shapes, dtypes, and devices must match exactly.
    - Exactly one forward AD level per thread is supported. The optional
      ``level`` argument is accepted for forward compatibility:

      * when ``level is None``, the current thread-local level is used;
      * when provided, ``level.level_id`` must match the active level.

    - The returned tensor is the **same object** as ``primal``.
    - Repeated calls overwrite the stored tangent ("last writer wins").
    """

    p = _require_tensor(primal, arg_name="primal")
    t = _require_tensor(tangent, arg_name="tangent")

    # Resolve level
    lvl = level or getattr(_state, "current_level", None)
    if lvl is None or not getattr(lvl, "_active", False):
        raise RuntimeError(
            "vibetensor.autograd.forward_ad.make_dual: no active dual_level; "
            "wrap calls in `with dual_level():`"
        )

    if lvl is not level and level is not None:
        raise RuntimeError(
            "vibetensor.autograd.forward_ad.make_dual: level handle does not "
            "match the current forward AD level"
        )

    # Enforce CPU float32 and shape/device/dtype equality via existing helpers.
    import vibetensor.autograd as A

    if not A._is_cpu_float32_tensor(p) or not A._is_cpu_float32_tensor(t):  # type: ignore[attr-defined]
        raise RuntimeError(
            "vibetensor.autograd.forward_ad.make_dual: primal and tangent must "
            "be CPU float32 tensors"
        )

    if tuple(p.sizes) != tuple(t.sizes) or p.device != t.device:
        raise RuntimeError(
            "vibetensor.autograd.forward_ad.make_dual: primal and tangent must "
            "have matching shape and device"
        )

    is_view_helper = getattr(A, "_is_view_tensor", None)
    if callable(is_view_helper) and is_view_helper(p):  # type: ignore[call-arg]
        raise RuntimeError(
            "vibetensor.autograd.forward_ad.make_dual: views are not supported "
            "as primals"
        )

    if A.is_inference_mode_enabled():
        raise RuntimeError(
            "vibetensor.autograd.forward_ad.make_dual: cannot attach tangents inside "
            "inference_mode"
        )

    _C.autograd._set_forward_grad(p, t, lvl.level_id)
    return p


def unpack_dual(dual: Any, *, level: _DualLevel | None = None) -> Tuple[_Tensor, _Tensor | None]:
    """Return ``(primal, tangent)`` for a dual tensor.

    If no tangent is attached for the active level, returns ``(dual.detach(), None)``.

    The optional ``level`` argument is accepted for forward compatibility but
    does not affect semantics; behavior is always defined in
    terms of the **currently active** forward AD level.
    """

    d = _require_tensor(dual, arg_name="dual")

    # If there is no active level at all, just return a detached primal.
    cur_level = _C.autograd._current_dual_level()
    if int(cur_level) < 0:
        return d.detach(), None

    if level is not None and getattr(level, "level_id", int(cur_level)) != int(cur_level):
        raise RuntimeError(
            "vibetensor.autograd.forward_ad.unpack_dual: level handle does not "
            "match the current forward AD level"
        )

    tangent = _C.autograd._get_forward_grad(d, int(cur_level))
    primal = d.detach()
    return primal, tangent
