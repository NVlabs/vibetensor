# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence

import vibetensor.torch as vt
from vibetensor import _C as _C
import vibetensor.autograd as A

_Tensor = _C.Tensor


def _as_vbt_tensor_sequence(
    value,
    *,
    api_name: str,
    arg_name: str,
) -> tuple[_Tensor, ...]:
    """Normalize ``value`` to a non-empty tuple of ``_C.Tensor``.

    Delegates container/type checks to :func:`vibetensor.autograd._as_tensor_sequence`,
    which enforces that ``value`` is either a single ``_C.Tensor`` or a non-empty
    :class:`collections.abc.Sequence` of ``_C.Tensor`` and rejects other iterables.
    """

    return A._as_tensor_sequence(  # type: ignore[attr-defined]
        value,
        api_name=api_name,
        arg_name=arg_name,
        allow_empty=False,
    )


def _ensure_cpu_float32(t: _Tensor, *, api_name: str, arg_name: str) -> None:
    if not A._is_cpu_float32_tensor(t):  # type: ignore[attr-defined]
        raise RuntimeError(
            f"{api_name}: {arg_name} must be CPU float32 tensors"
        )


def _functional_enable_grad():
    """Context manager alias for enabling grad inside functional helpers.

    This centralizes the choice between :func:`vibetensor.autograd.enable_grad`
    and :func:`vibetensor.torch.enable_grad` and keeps import-time behavior
    side-effect free.
    """

    if hasattr(A, "enable_grad"):
        return A.enable_grad()
    return vt.enable_grad()


def _normalize_functional_options(*, api_name: str, create_graph, strict) -> None:
    """Validate ``create_graph`` / ``strict`` for functional helpers.

    All helpers currently forbid higher-order autograd and require ``strict``
    to be a bool.
    """

    # create_graph
    if create_graph is True:
        raise NotImplementedError(
            f"{api_name}: create_graph=True is not supported "
            "(higher-order autograd is out of scope)"
        )
    if create_graph is not False:
        raise TypeError(
            f"{api_name}: create_graph must be a bool"
        )

    # strict
    if not isinstance(strict, bool):
        raise TypeError(f"{api_name}: strict must be a bool")


def vjp(
    fn: Callable[..., _Tensor],
    primals: _Tensor | Sequence[_Tensor],
    v: _Tensor | None = None,
    *,
    create_graph: bool = False,
    strict: bool = False,
) -> tuple[_Tensor, _Tensor | tuple[_Tensor, ...]]:
    """Compute a vector–Jacobian product for a single-output function.

    Let ``y = fn(*primals)`` and ``J = dy/dx`` (Jacobian w.r.t. ``primals``).
    This helper computes ``vᵀ J`` using a **single reverse-mode pass**.

    Differences vs :func:`torch.autograd.functional.vjp`:

    * Accepts a single vector ``v`` up front and returns gradients directly;
      it does **not** return a closure.
    * Only a single Tensor output is supported.
    * All outputs are non-differentiable (``requires_grad=False``,
      ``grad_fn is None``).
    * Gradients are only defined for CPU float32 tensors; other dtypes/devices
      raise.
    """

    api_name = "vibetensor.autograd.functional.vjp"

    if not callable(fn):
        raise TypeError(f"{api_name}: fn must be callable")

    _normalize_functional_options(api_name=api_name, create_graph=create_graph, strict=strict)
    if A.is_inference_mode_enabled():
        raise RuntimeError(
            "vibetensor.autograd.functional.vjp: cannot compute VJP inside inference_mode; "
            "enable gradients and rebuild the graph",
        )

    primals_tuple = _as_vbt_tensor_sequence(
        primals,
        api_name=api_name,
        arg_name="primals",
    )
    for p in primals_tuple:
        _ensure_cpu_float32(p, api_name=api_name, arg_name="primals")

    primals_is_tensor = isinstance(primals, _Tensor)

    # Prepare cloned primals for differentiation.
    primals_cloned: list[_Tensor] = []
    for p in primals_tuple:
        pc = p.detach()
        pc.requires_grad = True
        primals_cloned.append(pc)

    # Forward pass under enable_grad to ensure history is recorded.
    with _functional_enable_grad():
        y = fn(*primals_cloned)

    if not isinstance(y, _Tensor):
        raise TypeError(
            f"{api_name}: fn(primals) must return a VibeTensor tensor"
        )
    _ensure_cpu_float32(y, api_name=api_name, arg_name="fn(primals)")

    # Normalize v in output space.
    if v is None:
        if tuple(getattr(y, "sizes", ())) != ():
            raise RuntimeError(
                f"{api_name}: v can be None only when fn returns a single scalar tensor"
            )
        v_tensor = vt.ones_like(y)
    else:
        if not isinstance(v, _Tensor):
            raise TypeError(
                f"{api_name}: v must be a VibeTensor tensor"
            )
        _ensure_cpu_float32(v, api_name=api_name, arg_name="v")
        if tuple(getattr(v, "sizes", ())) != tuple(getattr(y, "sizes", ())):
            raise RuntimeError(
                f"{api_name}: v must have the same shape as fn(primals)"
            )
        v_tensor = v

    allow_unused = not strict

    grads_raw = A.grad(
        outputs=y,
        inputs=tuple(primals_cloned),
        grad_outputs=v_tensor,
        create_graph=False,
        allow_unused=allow_unused,
        materialize_grads=False,
    )

    vjp_elems: list[_Tensor] = []
    for p, g in zip(primals_cloned, grads_raw):
        if g is None:
            if strict:
                # A.grad would already have raised for allow_unused=False.
                raise AssertionError("unreachable: strict=True but got None gradient")
            vjp_elems.append(vt.zeros_like(p, dtype="float32"))
        else:
            vjp_elems.append(g.detach())

    with A.no_grad():
        out = y.detach()

    if primals_is_tensor:
        return out, vjp_elems[0]
    return out, tuple(vjp_elems)


def jvp(
    fn: Callable[..., _Tensor],
    primals: _Tensor | Sequence[_Tensor],
    tangents: _Tensor | Sequence[_Tensor],
    *,
    create_graph: bool = False,
    strict: bool = False,
) -> tuple[_Tensor, _Tensor]:
    """Compute a Jacobian–vector product for a scalar-output function.

    This helper computes ``J v`` where ``J = dy/dx`` is the Jacobian of a
    **scalar** output ``y = fn(*primals)`` with respect to ``primals``, and
    ``v`` is a tuple of tangent vectors matching ``primals``.

    Because the output is scalar, we can compute ``J v`` via a single
    reverse-mode pass and a dot product:

        jvp = sum_i <∂y/∂x_i, v_i>

    Differences vs :func:`torch.autograd.functional.jvp`:

    * Only scalar outputs are supported.
    * All outputs are non-differentiable.
    * We never build higher-order graphs or use the double-backward trick.
    """

    api_name = "vibetensor.autograd.functional.jvp"

    if not callable(fn):
        raise TypeError(f"{api_name}: fn must be callable")

    _normalize_functional_options(api_name=api_name, create_graph=create_graph, strict=strict)
    if A.is_inference_mode_enabled():
        raise RuntimeError(
            "vibetensor.autograd.functional.jvp: cannot compute JVP inside inference_mode; "
            "enable gradients and rebuild the graph",
        )

    primals_tuple = _as_vbt_tensor_sequence(
        primals,
        api_name=api_name,
        arg_name="primals",
    )
    for p in primals_tuple:
        _ensure_cpu_float32(p, api_name=api_name, arg_name="primals")

    tangents_tuple = _as_vbt_tensor_sequence(
        tangents,
        api_name=api_name,
        arg_name="tangents",
    )
    if len(tangents_tuple) != len(primals_tuple):
        raise RuntimeError(
            f"{api_name}: tangents must match primals in length"
        )

    for p, t in zip(primals_tuple, tangents_tuple):
        _ensure_cpu_float32(t, api_name=api_name, arg_name="tangents")
        if tuple(getattr(t, "sizes", ())) != tuple(getattr(p, "sizes", ())):
            raise RuntimeError(
                f"{api_name}: each tangent must have the same shape as its corresponding primal"
            )

    primals_cloned: list[_Tensor] = []
    for p in primals_tuple:
        pc = p.detach()
        pc.requires_grad = True
        primals_cloned.append(pc)

    with _functional_enable_grad():
        y = fn(*primals_cloned)

    if not isinstance(y, _Tensor):
        raise TypeError(
            f"{api_name}: fn(primals) must return a VibeTensor tensor"
        )
    _ensure_cpu_float32(y, api_name=api_name, arg_name="fn(primals)")

    if tuple(getattr(y, "sizes", ())) != ():
        raise RuntimeError(
            f"{api_name}: fn(primals) must return a single scalar tensor"
        )

    allow_unused = not strict
    grads_raw = A.grad(
        outputs=y,
        inputs=tuple(primals_cloned),
        grad_outputs=None,
        create_graph=False,
        allow_unused=allow_unused,
        materialize_grads=False,
    )

    with A.no_grad():
        jvp_val: _Tensor | None = None
        for g_i, t_i in zip(grads_raw, tangents_tuple):
            if g_i is None:
                if strict:
                    # A.grad would already have raised for allow_unused=False.
                    continue
                # Unused input contributes zero.
                continue
            contrib = (g_i * t_i).sum()
            jvp_val = contrib if jvp_val is None else jvp_val + contrib

        if jvp_val is None:
            # All inputs unused; Jv is mathematically zero.
            jvp_val = vt.zeros_like(y, dtype="float32")

        out = y.detach()
        jvp_val = jvp_val.detach()

    return out, jvp_val


def jacobian(
    fn: Callable[[_Tensor], _Tensor],
    inputs: _Tensor,
    *,
    create_graph: bool = False,
    strict: bool = False,
) -> _Tensor:
    """Compute the dense Jacobian d(fn(x))/dx for a single input and output.

    This helper is intentionally constrained:

    * ``inputs`` must be a single CPU float32 ``_C.Tensor``.
    * ``fn(inputs)`` must return a single CPU float32 ``_C.Tensor``.
    * The Jacobian is materialized as a Tensor ``J`` with shape
      ``fn(inputs).shape + inputs.shape``.
    * Implementation uses **reverse-mode only** via :func:`vibetensor.autograd.grad`
      with one engine run per output element.

    It is suitable only for **tiny outputs** (e.g., losses or low-dimensional
    diagnostics) and is not intended for large Jacobians.
    """

    api_name = "vibetensor.autograd.functional.jacobian"

    if not callable(fn):
        raise TypeError(f"{api_name}: fn must be callable")

    _normalize_functional_options(api_name=api_name, create_graph=create_graph, strict=strict)
    if A.is_inference_mode_enabled():
        raise RuntimeError(
            "vibetensor.autograd.functional.jacobian: cannot compute Jacobian inside inference_mode; "
            "enable gradients and rebuild the graph",
        )

    if not isinstance(inputs, _Tensor):
        raise TypeError(
            f"{api_name}: inputs must be a single VibeTensor tensor"
        )

    _ensure_cpu_float32(inputs, api_name=api_name, arg_name="inputs")

    x = inputs.detach()
    x.requires_grad = True

    with _functional_enable_grad():
        y = fn(x)

    if not isinstance(y, _Tensor):
        raise TypeError(
            f"{api_name}: fn(inputs) must return a VibeTensor tensor"
        )
    _ensure_cpu_float32(y, api_name=api_name, arg_name="fn(inputs)")

    # Flatten output under no_grad to avoid recording history for shape helpers.
    with A.no_grad():
        y_flat = y.reshape((-1,))
        numel_out = int(tuple(getattr(y_flat, "sizes", (1,)))[0])

    # Allocate J with shape y.shape + x.shape.
    y_sizes = tuple(getattr(y, "sizes", ()))
    x_sizes = tuple(getattr(x, "sizes", ()))
    J = vt.zeros((*y_sizes, *x_sizes), dtype="float32")

    allow_unused = not strict

    for k in range(numel_out):
        # One-hot seed in output space.
        with A.no_grad():
            go = vt.zeros_like(y, dtype="float32")
            go.view((-1,))[k] = 1.0

        (g_x,) = A.grad(
            outputs=y,
            inputs=(x,),
            grad_outputs=go,
            create_graph=False,
            allow_unused=allow_unused,
            materialize_grads=False,
        )

        if g_x is None:
            if strict:
                # A.grad would already have raised for allow_unused=False.
                continue
            # Non-differentiable: row is all zeros; nothing to write.
            continue

        with A.no_grad():
            J_view = J.view((numel_out,) + x_sizes)
            J_view[k] = g_x

    return J


def hessian(*args, **kwargs):  # pragma: no cover - trivial stub
    raise NotImplementedError(
        "vibetensor.autograd.functional.hessian: higher-order autograd is not "
        "implemented; use jacobian or grad instead"
    )


def vhp(*args, **kwargs):  # pragma: no cover - trivial stub
    raise NotImplementedError(
        "vibetensor.autograd.functional.vhp: vector-Hessian products are not "
        "implemented; they require higher-order autograd"
    )


def hvp(*args, **kwargs):  # pragma: no cover - trivial stub
    raise NotImplementedError(
        "vibetensor.autograd.functional.hvp: Hessian-vector products are not "
        "implemented; they require higher-order autograd"
    )
