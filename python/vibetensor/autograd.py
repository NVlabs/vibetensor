# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from collections.abc import Callable, Sequence
import contextlib
import math
import os

import vibetensor.torch as vt
from vibetensor import _C as _C
from . import autograd_graph as graph


def _get_ag():
    """Return `_C.autograd` submodule or None.

    This indirection simplifies testing and gracefully handles builds where
    autograd is unavailable.
    """

    return getattr(_C, "autograd", None)


def _is_vbt_tensor(obj: object) -> bool:
    """Return True iff obj is a VibeTensor `_C.Tensor`."""

    return isinstance(obj, _C.Tensor)


def _as_tensor_sequence(
    value,
    *,
    api_name: str,
    arg_name: str,
    allow_empty: bool,
) -> tuple[_C.Tensor, ...]:
    """Normalize `value` to a tuple of `_C.Tensor`.

    - Accepts a single `_C.Tensor` or a `collections.abc.Sequence` of `_C.Tensor`.
    - Rejects non-sequence iterables (generators, custom iterables) with `TypeError`.
    - Optionally allows or rejects empty sequences.
    - Raises TypeError with an "autograd.backward: ..."-style prefix when types
      are wrong.
    """

    # Single tensor fast-path.
    if _is_vbt_tensor(value):
        return (value,)

    # Sequence of tensors (excluding string-likes).
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = tuple(value)
        if not allow_empty and len(seq) == 0:
            raise RuntimeError(
                f"{api_name}: {arg_name} must not be an empty sequence"
            )

        for x in seq:
            if not _is_vbt_tensor(x):
                raise TypeError(
                    f"{api_name}: {arg_name} must be VibeTensor tensors"
                )
        return seq

    # Non-tensor, non-Sequence, or string-like.
    raise TypeError(f"{api_name}: {arg_name} must be VibeTensor tensors")


def _normalize_backward_tensors(tensors) -> tuple[_C.Tensor, ...]:
    """Normalize `tensors` for `vibetensor.autograd.backward`.

    - Accepts `_C.Tensor` or non-empty `Sequence` of `_C.Tensor`.
    - Rejects empty sequences (MB4.f) and non-tensor entries (MB4.a/b).
    - Returns a non-empty tuple of `_C.Tensor`.
    """

    return _as_tensor_sequence(
        tensors,
        api_name="autograd.backward",
        arg_name="tensors",
        allow_empty=False,
    )


def _normalize_backward_grad_tensors(
    grad_tensors,
    *,
    expected_len: int,
) -> tuple[_C.Tensor, ...] | None:
    """Normalize `grad_tensors` for `vibetensor.autograd.backward`.

    - Accepts None, a single `_C.Tensor`, or a `Sequence` of `_C.Tensor` of
      length `expected_len`.
    - Rejects non-tensor entries with `TypeError` (MB4.c/d/m).
    - Rejects length mismatches with `RuntimeError` (MB4.e).
    - Returns None or a tuple of `_C.Tensor`.
    """

    if grad_tensors is None:
        return None

    if _is_vbt_tensor(grad_tensors):
        return (grad_tensors,)

    if isinstance(grad_tensors, Sequence) and not isinstance(grad_tensors, (str, bytes)):
        seq = tuple(grad_tensors)

        # Element type check first (MB4.c/d/m).
        for x in seq:
            if not _is_vbt_tensor(x):
                raise TypeError(
                    "autograd.backward: grad_tensors must be VibeTensor tensors"
                )

        # Length check second (MB4.e).
        if len(seq) != expected_len:
            raise RuntimeError(
                "autograd.backward: grad_tensors must match tensors in length"
            )

        return seq

    raise TypeError("autograd.backward: grad_tensors must be VibeTensor tensors")


def _normalize_outputs_arg(outputs) -> tuple[_C.Tensor, ...]:
    """Normalize `outputs` for `vibetensor.autograd.grad`.

    This delegates container/type checks to `_as_tensor_sequence`.
    """

    return _as_tensor_sequence(
        outputs,
        api_name="autograd.grad",
        arg_name="outputs",
        allow_empty=False,
    )


def _cpu_device_sentinel() -> object | None:
    """Return the canonical CPU device descriptor for VibeTensor tensors.

    This observes the public `device` attribute of a small CPU float32 tensor
    and intentionally avoids inspecting private DLPack enum values directly.
    """

    try:
        t = vt.tensor([0.0], dtype="float32")
    except Exception:  # pragma: no cover - extremely defensive
        return None
    return getattr(t, "device", None)


_CPU_DEVICE: object | None = None


def _is_cpu_float32_tensor(x: _C.Tensor) -> bool:
    """Return True iff ``x`` is a CPU float32 tensor.

    This helper relies only on the public `_C.Tensor` surface (`dtype` and
    `device`/`device_type` accessors) and avoids depending directly on private
    DLPack enums. Tests validate behavior using real CPU vs non-CPU and
    float32 vs non-float32 tensors, not by asserting on internal encodings.
    """

    if getattr(x, "dtype", None) != "float32":
        return False

    global _CPU_DEVICE
    if _CPU_DEVICE is None:
        _CPU_DEVICE = _cpu_device_sentinel()

    dev = getattr(x, "device", None)
    if _CPU_DEVICE is not None and dev == _CPU_DEVICE:
        return True

    # Future-proof for potential explicit device_type attribute.
    if getattr(x, "device_type", None) == "cpu":
        return True

    return False


def _is_cuda_float32_tensor(x: _C.Tensor) -> bool:
    """Return True iff ``x`` is a CUDA float32 tensor."""
    if getattr(x, "dtype", None) != "float32":
        return False
    dev = getattr(x, "device", None)
    # device is a tuple (type, index); CUDA is type 2 (kDLCUDA)
    if isinstance(dev, tuple) and len(dev) >= 1 and dev[0] == 2:
        return True
    if getattr(x, "device_type", None) == "cuda":
        return True
    return False


def _is_cuda_float16_tensor(x: _C.Tensor) -> bool:
    """Return True iff ``x`` is a CUDA float16 tensor."""
    if getattr(x, "dtype", None) != "float16":
        return False
    dev = getattr(x, "device", None)
    if isinstance(dev, tuple) and len(dev) >= 1 and dev[0] == 2:
        return True
    if getattr(x, "device_type", None) == "cuda":
        return True
    return False


def _is_supported_autograd_tensor(x: _C.Tensor) -> bool:
    """Return True iff ``x`` is a supported autograd tensor."""
    if _is_cpu_float32_tensor(x):
        return True

    # Allow CUDA float32/float16 only when CUDA autograd is enabled.
    if is_cuda_autograd_enabled() and (
        _is_cuda_float32_tensor(x) or _is_cuda_float16_tensor(x)
    ):
        return True

    return False



_env_flag = os.getenv("VBT_AUTOGRAD_ENABLE_FUNCTION", "1")
try:
    _ENABLE_CUSTOM_FUNCTION: bool = bool(int(_env_flag))
except ValueError:  # pragma: no cover - defensive against malformed env
    _ENABLE_CUSTOM_FUNCTION = True


def _assert_function_enabled() -> None:
    if not _ENABLE_CUSTOM_FUNCTION:
        raise RuntimeError(
            "vibetensor.autograd.Function is disabled by "
            "VBT_AUTOGRAD_ENABLE_FUNCTION=0"
        )


class _FunctionMeta(type):
    """Metaclass for custom autograd Functions.

    - Prevents instantiation of Function subclasses.
    - Reserved for future per-class caches (e.g. invocation style).
    """

    def __call__(cls, *args, **kwargs):  # pragma: no cover - simple guard
        raise RuntimeError(
            "vibetensor.autograd.Function subclasses must not be instantiated; "
            "call Class.apply(...) instead."
        )


class _FunctionCtx:
    """Context for a single `Function.apply` call (VibeTensor subset).

    - Created in Python before `forward`.
    - Reconstructed in C++ before `backward` using saved state.
    - Users may attach non-tensor attributes on it in `forward` and read
      them in `backward`.
    """

    __slots__ = (
        "_saved_raw",
        "_saved_unpacked",
        "_saved_called",
        "needs_input_grad",
        "_stage",
        "__dict__",
    )

    def __init__(self, needs_input_grad: tuple[bool, ...], stage: str) -> None:
        self._saved_raw: tuple[_C.Tensor, ...] = ()
        self._saved_unpacked: tuple[_C.Tensor, ...] | None = None
        self._saved_called: bool = False
        self.needs_input_grad = needs_input_grad
        # "forward" or "backward" – used to guard misuse.
        self._stage = stage

    # ---- Saving tensors for backward ---------------------------------

    def save_for_backward(self, *tensors: _C.Tensor) -> None:
        """Record tensors for use in `backward`.

        Rules:
          * May be called at most once per `apply` call.
          * Only valid in `forward` (stage == "forward").
          * All arguments must be `_C.Tensor`; non-tensors raise TypeError.
          * Actual snapshots are taken in C++ (`SavedVariable`) when the node
            is created; Python stores only a tuple of tensors.
        """

        if self._stage != "forward":
            raise RuntimeError(
                "ctx.save_for_backward(...) may only be called in Function.forward"
            )
        if self._saved_called:
            raise RuntimeError(
                "ctx.save_for_backward() may only be called once per Function.forward"
            )
        self._saved_called = True

        raw: list[_C.Tensor] = []
        for t in tensors:
            if not isinstance(t, _C.Tensor):
                raise TypeError(
                    "ctx.save_for_backward(...) arguments must be VibeTensor _C.Tensor "
                    "instances"
                )
            raw.append(t)
        self._saved_raw = tuple(raw)

    @property
    def saved_tensors(self) -> tuple[_C.Tensor, ...]:
        """Tensors saved via `save_for_backward` (available only in backward).

        - Valid only inside `backward`; reading it in `forward` raises.
        - Populated by C++ from `SavedVariable` snapshots when invoking
          `backward`.
        """

        if self._stage != "backward" or self._saved_unpacked is None:
            raise RuntimeError(
                "ctx.saved_tensors is only available inside backward()"
            )
        return self._saved_unpacked


class Function(metaclass=_FunctionMeta):
    """Base class for defining custom autograd Functions in VibeTensor.

    Subclasses must override:
      * `forward(ctx, *args, **kwargs)` – runs the forward pass.
      * `backward(ctx, grad_output)` – computes gradients for inputs.

    Use `MyFn.apply(*args, **kwargs)` to call the operation.
    """

    @classmethod
    def apply(cls, *args, **kwargs):
        return _function_apply(cls, args, kwargs)

    @staticmethod
    def forward(*args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def setup_context(ctx, inputs, output):
        raise RuntimeError(
            "setup_context is not supported; "
            "use forward(ctx, ...) instead."
        )

    @staticmethod
    def jvp(ctx, *grad_inputs):
        raise NotImplementedError("jvp is not supported")

    vjp = backward


def _function_apply(cls: type[Function], args, kwargs):
    """Internal implementation of Function.apply.

    This helper handles grad-mode checks, ctx creation, forward execution,
    and wiring a PyFunctionNode when a graph should be built.
    """

    _assert_function_enabled()

    # Disallow usage under inference_mode.
    if is_inference_mode_enabled():
        raise RuntimeError(
            "vibetensor.autograd.Function.apply cannot be used inside "
            "inference_mode(); recompute outside inference_mode"
        )

    flat_args = tuple(args)

    if kwargs:
        for v in kwargs.values():
            if _is_vbt_tensor(v):
                raise TypeError(
                    "vibetensor.autograd.Function: tensor kwargs are not supported; "
                    "pass tensors as positional arguments instead"
                )

    graph_on = is_grad_enabled()

    # Pre-pass: validate differentiable inputs and build mapping metadata.
    tensor_inputs: list[_C.Tensor] = []
    edge_index_by_arg: list[int] = []
    needs_input_grad_flags: list[bool] = []

    diff_dev: tuple[int, int] | None = None

    def _device_tuple(t: _C.Tensor) -> tuple[int, int] | None:
        dev = getattr(t, "device", None)
        if isinstance(dev, (tuple, list)) and len(dev) >= 2:
            try:
                return (int(dev[0]), int(dev[1]))
            except Exception:
                pass

        # Fall back to the string-based device_type/device_index surface when
        # available.
        dev_type = getattr(t, "device_type", None)
        if dev_type == "cpu":
            return (1, 0)  # kDLCPU
        if dev_type == "cuda":
            dev_index = getattr(t, "device_index", None)
            if dev_index is not None:
                try:
                    return (2, int(dev_index))  # kDLCUDA
                except Exception:
                    return None

        return None

    for arg in flat_args:
        if not _is_vbt_tensor(arg):
            needs_input_grad_flags.append(False)
            edge_index_by_arg.append(-1)
            continue

        requires = bool(getattr(arg, "requires_grad", False))
        participates = graph_on and requires

        if participates and not _is_supported_autograd_tensor(arg):
            raise TypeError(
                "vibetensor.autograd.Function supports autograd only for "
                "CPU float32 tensors (or CUDA float32/float16 when CUDA autograd is enabled); "
                "use requires_grad=False or convert tensors"
            )

        if participates:
            dev_tup = _device_tuple(arg)
            if dev_tup is None:
                raise TypeError(
                    "vibetensor.autograd.Function: unable to parse tensor device"
                )
            if diff_dev is None:
                diff_dev = dev_tup
            elif dev_tup != diff_dev:
                raise TypeError(
                    "vibetensor.autograd.Function: all differentiable tensor inputs must be on the same device"
                )

            edge_index = len(tensor_inputs)
            tensor_inputs.append(arg)
        else:
            edge_index = -1

        edge_index_by_arg.append(edge_index)
        needs_input_grad_flags.append(participates)

    needs_input_grad = tuple(needs_input_grad_flags)
    is_differentiable_call = any(needs_input_grad_flags)

    diff_is_cuda = False
    diff_cuda_index = 0
    if is_differentiable_call:
        if diff_dev is None:
            raise TypeError(
                "vibetensor.autograd.Function: internal error (missing differentiable device)"
            )

        diff_is_cuda = diff_dev[0] == 2  # kDLCUDA
        diff_cuda_index = int(diff_dev[1])

        # Enforce single-device graphs: non-differentiable CUDA tensor args must
        # match the differentiable CUDA device; CPU tensors are allowed.
        for arg in flat_args:
            if not _is_vbt_tensor(arg):
                continue
            dev_tup = _device_tuple(arg)
            if dev_tup is None:
                raise TypeError(
                    "vibetensor.autograd.Function: unable to parse tensor device"
                )
            dev_type, dev_index = dev_tup
            if dev_type == 2:  # kDLCUDA
                if not diff_is_cuda:
                    raise TypeError(
                        "vibetensor.autograd.Function: all tensor arguments must be CPU when differentiable inputs are on CPU"
                    )
                if int(dev_index) != diff_cuda_index:
                    raise TypeError(
                        "vibetensor.autograd.Function: mixed CUDA devices in Function.apply; all CUDA tensor arguments must share the same device index"
                    )
            elif dev_type == 1:  # kDLCPU
                continue
            else:
                raise TypeError(
                    "vibetensor.autograd.Function: unsupported tensor device type"
                )

    cls_dict = getattr(cls, "__dict__", {})
    base_setup = Function.__dict__.get("setup_context")
    if cls_dict.get("setup_context") not in (None, base_setup):
        raise RuntimeError(
            "setup_context is not supported; "
            "use forward(ctx, ...) instead."
        )

    # Ensure backward is overridden for differentiable calls.
    base_backward = Function.__dict__.get("backward")
    cls_backward = cls_dict.get("backward")
    if is_differentiable_call and (cls_backward is None or cls_backward is base_backward):
        raise RuntimeError(
            f"vibetensor.autograd.Function subclass {cls.__name__} must override "
            "backward(ctx, grad_output)"
        )

    # Create ctx for this call.
    ctx = _FunctionCtx(needs_input_grad=needs_input_grad, stage="forward")

    # Run forward; under no_grad when building a differentiable graph.
    if is_differentiable_call:
        with no_grad():
            outputs = cls.forward(ctx, *flat_args, **kwargs)
    else:
        outputs = cls.forward(ctx, *flat_args, **kwargs)

    # Normalize outputs and locate the single tensor output (if any).
    primary_tensor: _C.Tensor | None
    if _is_vbt_tensor(outputs):
        primary_tensor = outputs
    elif isinstance(outputs, (tuple, list)):
        found: _C.Tensor | None = None
        for item in outputs:
            if _is_vbt_tensor(item):
                if found is None:
                    found = item
                else:
                    raise NotImplementedError(
                        "vibetensor.autograd.Function supports at most one tensor output; "
                        "found multiple tensor outputs"
                    )
        primary_tensor = found
    else:
        primary_tensor = None

    # If no tensor output or call not participating in autograd, return outputs as-is.
    if primary_tensor is None or not is_differentiable_call:
        return outputs

    out_dev = _device_tuple(primary_tensor)
    if out_dev is None:
        raise TypeError(
            "vibetensor.autograd.Function: unable to parse output tensor device"
        )
    if diff_dev is None or out_dev != diff_dev:
        raise TypeError(
            "vibetensor.autograd.Function: output tensor must be on the same device as differentiable inputs"
        )

    out_dtype = getattr(primary_tensor, "dtype", None)
    if not diff_is_cuda:
        if out_dtype != "float32":
            raise TypeError(
                "vibetensor.autograd.Function: output tensor must be float32"
            )
    else:
        if out_dtype not in ("float32", "float16"):
            raise TypeError(
                "vibetensor.autograd.Function: CUDA output tensor must be float32/float16"
            )

    ag = _get_ag()
    create_node = getattr(ag, "_create_py_function_node", None) if ag is not None else None
    if not callable(create_node):
        raise RuntimeError(
            "vibetensor.autograd.Function requires an autograd-enabled build "
            "with _C.autograd._create_py_function_node available"
        )

    saved = getattr(ctx, "_saved_raw", ())
    if not isinstance(saved, tuple):
        saved = tuple(saved)

    # Validate saved tensor devices. CPU saved tensors are always allowed, but
    # any CUDA saved tensor must match the differentiable CUDA device.
    for t in saved:
        if not _is_vbt_tensor(t):
            raise TypeError(
                "vibetensor.autograd.Function: save_for_backward expects only tensors"
            )
        dev_tup = _device_tuple(t)
        if dev_tup is None:
            raise TypeError(
                "vibetensor.autograd.Function: unable to parse saved tensor device"
            )
        dev_type, dev_index = dev_tup
        if dev_type == 2:  # kDLCUDA
            if not diff_is_cuda or int(dev_index) != diff_cuda_index:
                raise TypeError(
                    "vibetensor.autograd.Function: CUDA saved tensors must be on the same device as differentiable inputs"
                )
        elif dev_type == 1:  # kDLCPU
            continue
        else:
            raise TypeError(
                "vibetensor.autograd.Function: unsupported saved tensor device type"
            )

    # Break Python references from ctx to tensors now that C++ will snapshot.
    ctx._saved_raw = ()

    # Capture non-tensor ctx attributes; tensor attributes stored directly on
    # ctx are a known footgun and are not supported for lifetime guarantees.
    ctx_state = dict(getattr(ctx, "__dict__", {}))

    create_node(
        cls,
        tuple(tensor_inputs),
        primary_tensor,
        tuple(saved),
        tuple(needs_input_grad),
        tuple(edge_index_by_arg),
        ctx_state,
    )

    return outputs


def _normalize_grad_inputs(inputs) -> tuple[_C.Tensor, ...]:
    """Normalize and validate `inputs` for `vibetensor.autograd.grad`.

    - Accepts a `_C.Tensor` or non-empty `Sequence` of `_C.Tensor`.
    - Requires each tensor to be a leaf CPU float32 tensor with
      ``requires_grad=True``.
    """

    inputs_tuple = _as_tensor_sequence(
        inputs,
        api_name="autograd.grad",
        arg_name="inputs",
        allow_empty=False,
    )

    for x in inputs_tuple:
        if (
            not bool(getattr(x, "is_leaf", False))
            or not bool(getattr(x, "requires_grad", False))
            or not _is_cpu_float32_tensor(x)
        ):
            raise RuntimeError(
                "vibetensor.autograd.grad: inputs must be leaf CPU float32 tensors that require grad"
            )
    return inputs_tuple


def _normalize_grad_outputs(
    grad_outputs,
    *,
    expected_len: int,
) -> tuple[_C.Tensor, ...] | None:
    """Normalize `grad_outputs` for `vibetensor.autograd.grad`.

    - Accepts None, a single `_C.Tensor`, or a `Sequence` of `_C.Tensor`
      of length `expected_len`.
    - Rejects non-tensor entries with `TypeError`.
    - Rejects length mismatches with `RuntimeError`.
    """

    if grad_outputs is None:
        return None

    if _is_vbt_tensor(grad_outputs):
        seq = (grad_outputs,)
    elif isinstance(grad_outputs, Sequence) and not isinstance(
        grad_outputs, (str, bytes)
    ):
        seq = tuple(grad_outputs)
    else:
        raise TypeError("autograd.grad: grad_outputs must be VibeTensor tensors")

    # Element type check first (mirrors `_normalize_backward_grad_tensors`).
    for x in seq:
        if not _is_vbt_tensor(x):
            raise TypeError("autograd.grad: grad_outputs must be VibeTensor tensors")

    # Length check second.
    if len(seq) != expected_len:
        raise RuntimeError(
            "autograd.grad: grad_outputs must match outputs in length"
        )

    return seq


def _normalize_grad_options(
    *,
    create_graph,
    allow_unused,
    is_grads_batched,
    materialize_grads,
) -> tuple[bool, bool]:
    """Normalize options for `vibetensor.autograd.grad`.

    Returns `(allow_unused_eff, materialize_eff)` and raises pinned
    `TypeError` / `NotImplementedError` / `ValueError` for invalid
    combinations.
    """

    # 1. create_graph
    if create_graph is True:
        raise NotImplementedError(
            "vibetensor.autograd.grad: create_graph=True is not supported "
            "(higher-order autograd is out of scope)"
        )
    elif create_graph is False:
        pass
    else:
        raise TypeError(
            "vibetensor.autograd.grad: create_graph must be a bool"
        )

    # 2. allow_unused / materialize_grads types
    if allow_unused is not None and not isinstance(allow_unused, bool):
        raise TypeError(
            "vibetensor.autograd.grad: allow_unused must be a bool or None"
        )
    if not isinstance(materialize_grads, bool):
        raise TypeError(
            "vibetensor.autograd.grad: materialize_grads must be a bool"
        )

    # 3. Effective flags and conflict
    if allow_unused is None:
        allow_unused_eff = materialize_grads
    else:
        allow_unused_eff = allow_unused
    materialize_eff = materialize_grads

    if materialize_eff and not allow_unused_eff:
        raise ValueError(
            "vibetensor.autograd.grad: materialize_grads=True requires allow_unused=True or None"
        )

    # 4. is_grads_batched
    if not isinstance(is_grads_batched, bool):
        raise TypeError(
            "vibetensor.autograd.grad: is_grads_batched must be a bool"
        )
    if is_grads_batched:
        raise NotImplementedError(
            "vibetensor.autograd.grad: is_grads_batched=True is not supported "
            "(batched gradients are out of scope)"
        )

    return allow_unused_eff, materialize_eff


def _make_zero_like_for_grad(x: _C.Tensor) -> _C.Tensor:
    """Create a non-differentiable zero tensor for `x`.

    The result matches `x`'s shape and device, uses `dtype=float32`, and is
    guaranteed to be non-differentiable (`requires_grad=False`, `grad_fn=None`).
    """

    z = vt.zeros_like(x, dtype="float32")
    try:
        z.requires_grad = False
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError(
            "vibetensor.autograd.grad: failed to enforce requires_grad=False on materialized zero gradient"
        ) from e

    grad_fn = getattr(z, "grad_fn", None)
    if bool(getattr(z, "requires_grad", False)) or grad_fn is not None:
        raise RuntimeError(
            "vibetensor.autograd.grad: _make_zero_like_for_grad returned a differentiable tensor"
        )
    return z


def _clear_inputs_grad(inputs: tuple[_C.Tensor, ...]) -> None:
    """Clear `.grad` buffers for all input tensors.

    Uses `_C.autograd._clear_tensor_grad` when available, otherwise falls back
    to setting `x.grad = None`. Descriptor failures surface as a pinned
    `RuntimeError`.
    """

    ag = _get_ag()
    clear_fn = getattr(ag, "_clear_tensor_grad", None) if ag is not None else None

    for x in inputs:
        if callable(clear_fn):
            # Treat extension errors as engine-level; propagate as-is.
            clear_fn(x)
            continue
        try:
            x.grad = None  # type: ignore[assignment]
        except AttributeError:
            raise RuntimeError(
                "vibetensor.autograd.grad: cannot clear .grad for an input tensor; "
                "_clear_tensor_grad is unavailable and Tensor.grad is not settable"
            )



# Simple Python-facing facade for autograd fallback registration.
# For legacy callables, we first try a structured `_C.autograd._register_py_autograd_fallback`
# entry when available, then fall back to the original
# `_C._try_register_boxed_autograd_fallback` helper used in earlier versions.

def register(opname: str, backward: Callable[[tuple, tuple], Any]) -> bool:
    if not isinstance(opname, str):
        raise TypeError("opname must be a string")
    if not callable(backward):
        raise TypeError("backward must be callable")

    # Preferred path: structured registry under _C.autograd when supported.
    ag = _get_ag()
    reg = getattr(ag, "_register_py_autograd_fallback", None) if ag is not None else None
    if callable(reg):
        try:
            return bool(reg(opname, backward))  # type: ignore[misc]
        except Exception:
            return False

    # Fallback: older boxed-only autograd fallback helper.
    try:
        fn = getattr(_C, "_try_register_boxed_autograd_fallback", None)
        if callable(fn):
            return bool(fn(opname, backward))  # type: ignore[misc]
    except Exception:
        return False
    return False


def register_function(opname: str, fn_cls: type[Function]) -> bool:
    """Register a Function subclass as the autograd rule for a vt op.

    Currently this is a thin wrapper over the same underlying C++ registry
    used by :func:`register`. When the structured
    ``_C.autograd._register_py_autograd_fallback`` helper is available,
    it is preferred; otherwise this function returns ``False`` to
    indicate that Function-based registration is unsupported by the
    current build.
    """

    if not isinstance(opname, str):
        raise TypeError("opname must be a string")
    if not isinstance(fn_cls, type) or not issubclass(fn_cls, Function):
        raise TypeError("fn_cls must be a vibetensor.autograd.Function subclass")

    ag = _get_ag()
    reg = getattr(ag, "_register_py_autograd_fallback", None) if ag is not None else None
    if callable(reg):
        try:
            return bool(reg(opname, fn_cls))  # type: ignore[misc]
        except Exception:
            return False

    # No structured registry available; Function-based fallbacks are disabled.
    return False


# --- Grad-mode helpers --------------------------------------------------------

def is_grad_enabled() -> bool:
    """Return True if graphs are currently being recorded in this thread.

    This forwards to `_C.autograd.is_grad_enabled()` when available, which in
    turn reflects the derived graph-enabled state (`GradMode && !InferenceMode`).
    """

    ag = _get_ag()
    if ag is None:
        return False
    try:
        return bool(ag.is_grad_enabled())
    except Exception:
        return False


@contextlib.contextmanager
def no_grad():
    """Disable autograd graph recording in this thread for the duration.

    - Saves the current **raw** grad-mode bit via `_raw_grad_mode_enabled`.
    - Restores it on exit, even if the body raises.
    - Degrades gracefully on older `_C` that lack the raw helper.
    """

    ag = _get_ag()
    if ag is None:
        yield
        return

    raw_get = getattr(ag, "_raw_grad_mode_enabled", None)
    set_grad = getattr(ag, "set_grad_enabled", None)
    if raw_get is None or not callable(set_grad):
        # Fallback: snapshot graph-enabled state when raw bit unavailable.
        prev = bool(getattr(ag, "is_grad_enabled", lambda: False)())
        if callable(set_grad):
            set_grad(False)
        try:
            yield
        finally:
            if callable(set_grad):
                set_grad(prev)
        return

    prev = bool(raw_get())
    set_grad(False)
    try:
        yield
    finally:
        set_grad(prev)


@contextlib.contextmanager
def enable_grad():
    """Temporarily enable autograd graph recording in this thread."""

    ag = _get_ag()
    if ag is None:
        yield
        return

    raw_get = getattr(ag, "_raw_grad_mode_enabled", None)
    set_grad = getattr(ag, "set_grad_enabled", None)
    if raw_get is None or not callable(set_grad):
        prev = bool(getattr(ag, "is_grad_enabled", lambda: False)())
        if callable(set_grad):
            set_grad(True)
        try:
            yield
        finally:
            if callable(set_grad):
                set_grad(prev)
        return

    prev = bool(raw_get())
    set_grad(True)
    try:
        yield
    finally:
        set_grad(prev)


def set_grad_enabled(mode: bool) -> None:
    """Set the raw grad-mode flag for this thread.

    Unlike PyTorch, this is not a context manager or decorator. It is a
    simple function returning None.
    """

    ag = _get_ag()
    if ag is None or not hasattr(ag, "set_grad_enabled"):
        return
    ag.set_grad_enabled(bool(mode))


# --- Inference-mode helpers ---------------------------------------------------

@contextlib.contextmanager
def inference_mode(mode: bool = True):
    """Inference-mode context manager (VibeTensor semantics).

    This is a stronger, graph-suppressing variant of :func:`no_grad`.

    - When ``mode`` is True (default):
      * Saves raw grad-mode and inference-mode state.
      * Disables raw grad-mode.
      * Enables inference mode via ``_set_inference_mode_enabled(True)``.
      * Any tensors created inside behave like tensors created under
        ``no_grad`` (``requires_grad=False``, ``grad_fn=None``), even if
        their inputs required grad.
    - When ``mode`` is False: acts as a no-op context.

    Tensors created inside ``inference_mode(True)`` **may later be reused** in
    grad-enabled regions by toggling ``requires_grad=True``. Gradients then
    flow only through work performed after leaving the inference block; there
    is no recorded history back to the original computation. This is an
    intentional divergence from PyTorch, which generally forbids reusing
    inference-mode tensors and may raise instead of silently dropping those
    gradients.

    If inference helpers are unavailable (older ``_C`` builds), this is a
    best-effort no-op context.
    """

    ag = _get_ag()
    if ag is None:
        yield
        return

    # If helpers are missing, degrade to no-op.
    if not hasattr(ag, "_set_inference_mode_enabled") or not hasattr(ag, "is_inference_mode_enabled"):
        yield
        return

    if not bool(mode):
        # No-op wrapper to preserve decorator signature.
        yield
        return

    raw_get = getattr(ag, "_raw_grad_mode_enabled", None)
    set_grad = getattr(ag, "set_grad_enabled", None)

    if not callable(set_grad):
        # We cannot touch the raw grad bit at all; the best we can do is
        # toggle the inference flag so C++ treats this region as graph-off.
        try:
            prev_inf = bool(ag.is_inference_mode_enabled())
        except Exception:
            prev_inf = False

        ag._set_inference_mode_enabled(True)
        try:
            yield
        finally:
            ag._set_inference_mode_enabled(prev_inf)
        return

    if raw_get is None:
        # Partial build: snapshot and restore *graph-enabled* state around an
        # inference-mode block even though we cannot see the raw TLS bit.
        graph_get = getattr(ag, "is_grad_enabled", None)
        prev_graph = bool(graph_get()) if callable(graph_get) else False
        try:
            prev_inf = bool(ag.is_inference_mode_enabled())
        except Exception:
            prev_inf = False

        # Inside the block we want graphs fully off.
        set_grad(False)
        ag._set_inference_mode_enabled(True)
        try:
            yield
        finally:
            ag._set_inference_mode_enabled(prev_inf)
            if not prev_inf:
                # Restore original graph-enabled state exactly.
                set_grad(prev_graph)
            else:
                # When inference was already enabled, graph_on was necessarily
                # False; keep graphs off to avoid surprising toggles.
                set_grad(False)
        return

    # Full-featured build: raw TLS helper exists; restore exact raw + inf.
    prev_grad = bool(raw_get())
    try:
        prev_inf = bool(ag.is_inference_mode_enabled())
    except Exception:
        prev_inf = False

    set_grad(False)
    ag._set_inference_mode_enabled(True)
    try:
        yield
    finally:
        ag._set_inference_mode_enabled(prev_inf)
        set_grad(prev_grad)


def is_inference_mode_enabled() -> bool:
    """Return True if inference mode is enabled in this thread."""

    ag = _get_ag()
    if ag is None or not hasattr(ag, "is_inference_mode_enabled"):
        return False
    try:
        return bool(ag.is_inference_mode_enabled())
    except Exception:
        return False


# --- Engine toggle stubs ------------------------------------------------------

def is_multithreading_enabled() -> bool:
    ag = _get_ag()
    if ag is None or not hasattr(ag, "is_multithreading_enabled"):
        return False
    return bool(ag.is_multithreading_enabled())


def set_multithreading_enabled(mode: bool) -> None:
    ag = _get_ag()
    if ag is None or not hasattr(ag, "set_multithreading_enabled"):
        return
    ag.set_multithreading_enabled(bool(mode))


def is_view_replay_enabled() -> bool:
    ag = _get_ag()
    if ag is None or not hasattr(ag, "is_view_replay_enabled"):
        return False
    return bool(ag.is_view_replay_enabled())


def set_view_replay_enabled(mode: bool) -> None:
    ag = _get_ag()
    if ag is None or not hasattr(ag, "set_view_replay_enabled"):
        return
    ag.set_view_replay_enabled(bool(mode))


def get_device_mode() -> str:
    """Return the global autograd device mode.

    Values:
      - "single_device" (default)
      - "multi_device_experimental"
    """

    ag = _get_ag()
    if ag is None or not hasattr(ag, "get_device_mode"):
        return "single_device"
    try:
        return str(ag.get_device_mode())
    except AttributeError:  # pragma: no cover
        return "single_device"


def set_device_mode(mode: str) -> None:
    """Set the global autograd device mode for future backward runs."""

    ag = _get_ag()
    if ag is None or not hasattr(ag, "set_device_mode"):
        return
    ag.set_device_mode(str(mode))


def is_cuda_autograd_enabled() -> bool:
    """Return True if CUDA autograd/streaming backward is enabled.

    On CPU-only builds or when the underlying extension does not expose the
    toggle, this returns False.
    """

    ag = _get_ag()
    if ag is None or not hasattr(ag, "is_cuda_autograd_enabled"):
        return False
    try:
        return bool(ag.is_cuda_autograd_enabled())
    except Exception:  # noqa: BLE001
        return False


def set_cuda_autograd_enabled(enabled: bool) -> None:
    """Enable or disable CUDA autograd/streaming backward.

    On CPU-only builds or when the underlying extension does not expose the
    toggle, this is a no-op.
    """

    ag = _get_ag()
    if ag is None or not hasattr(ag, "set_cuda_autograd_enabled"):
        return
    ag.set_cuda_autograd_enabled(bool(enabled))


def backward(
    tensors,
    grad_tensors=None,
    retain_graph=None,
    create_graph: bool = False,
    *,
    inputs=None,
) -> None:
    """Run a backward pass for a single output tensor (VibeTensor wrapper).

    This is a thin, single-root wrapper over `_C.Tensor.backward`. It
    normalizes arguments and surfaces precise Python error messages while
    delegating gradient computation to the C++ autograd engine.
    """

    # 1. MB5 – unsupported options (`create_graph`, `inputs`).
    if create_graph is True:
        raise NotImplementedError(
            "vibetensor.autograd.backward: create_graph=True is not supported "
            "(higher-order autograd is out of scope)"
        )
    elif create_graph is False:
        pass
    else:
        raise TypeError(
            "vibetensor.autograd.backward: create_graph must be a bool"
        )

    if inputs is not None:
        raise NotImplementedError(
            "vibetensor.autograd.backward: the inputs argument is not supported"
        )

    # 2. MB4/MB3 – `tensors` normalization and single-root enforcement.
    roots = _normalize_backward_tensors(tensors)

    if len(roots) != 1:
        raise RuntimeError(
            "vibetensor.autograd.backward: multiple outputs are not supported"
        )

    (root,) = roots

    # 3. MB4 – `grad_tensors` normalization.
    grads_tuple = _normalize_backward_grad_tensors(
        grad_tensors,
        expected_len=len(roots),
    )
    grad = None if grads_tuple is None else grads_tuple[0]

    # 4. Delegation to Tensor API.
    # NOTE: `retain_graph` is accepted for API parity but ignored here.
    # We always pass retain_graph=False to keep semantics explicit and stable.
    root.backward(grad, retain_graph=False)


def grad(
    outputs,
    inputs,
    grad_outputs=None,
    retain_graph=None,
    create_graph: bool = False,
    allow_unused=None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> tuple[_C.Tensor | None, ...]:
    """Compute gradients of ``outputs`` w.r.t. ``inputs`` (VibeTensor wrapper).

    This is a stateful wrapper around a single backward pass. It normalizes
    arguments, enforces Autograd option semantics, and returns tensors that
    alias leaf ``.grad`` buffers for used inputs.
    """

    # Stage 0 – option normalization (highest precedence).
    allow_unused_eff, materialize_eff = _normalize_grad_options(
        create_graph=create_graph,
        allow_unused=allow_unused,
        is_grads_batched=is_grads_batched,
        materialize_grads=materialize_grads,
    )

    # NOTE: `retain_graph` is accepted for API parity but ignored; the engine
    # call below always uses `retain_graph=False`.

    # Stage 1 – normalize `outputs` and enforce single root.
    outs = _normalize_outputs_arg(outputs)
    if len(outs) != 1:
        raise RuntimeError(
            "vibetensor.autograd.grad: multiple outputs are not supported"
        )
    (root,) = outs

    # Stage 2 – normalize and validate `inputs`.
    inputs_tuple = _normalize_grad_inputs(inputs)

    # Stage 3 – normalize `grad_outputs`.
    go_tuple = _normalize_grad_outputs(grad_outputs, expected_len=1)
    grad_output = None if go_tuple is None else go_tuple[0]

    # Stage 4 – non-differentiable root early-return.
    non_diff = (not bool(getattr(root, "requires_grad", False))) or (
        getattr(root, "grad_fn", None) is None
    )
    if non_diff:
        if not allow_unused_eff:
            raise RuntimeError(
                "vibetensor.autograd.grad: got unused input with allow_unused=False"
            )
        if not materialize_eff:
            return tuple(None for _ in inputs_tuple)
        return tuple(_make_zero_like_for_grad(x) for x in inputs_tuple)

    # Stage 5 – clear `.grad` on inputs.
    _clear_inputs_grad(inputs_tuple)

    # Stage 6 – run backward once.
    root.backward(grad_output, retain_graph=False)

    # Stage 7 – collect gradients and apply `allow_unused` / `materialize_grads`.
    result: list[_C.Tensor | None] = []
    for x in inputs_tuple:
        g = x.grad_tensor()
        if g is None:
            if not allow_unused_eff:
                raise RuntimeError(
                    "vibetensor.autograd.grad: got unused input with allow_unused=False"
                )
            if not materialize_eff:
                result.append(None)
            else:
                result.append(_make_zero_like_for_grad(x))
        else:
            result.append(g)
    return tuple(result)


class GradcheckError(RuntimeError):
    """Error raised by :func:`gradcheck` and :func:`gradgradcheck` in VibeTensor.

    Used for semantically invalid gradcheck arguments, environment issues,
    and numeric mismatches between analytical and numerical gradients.
    """


def _require_numpy_for_gradcheck() -> Any:
    """Import and return NumPy for gradcheck.

    Raises GradcheckError with a pinned message when NumPy is unavailable.
    """

    try:
        import numpy as _np  # type: ignore[import]
    except Exception as e:  # noqa: BLE001
        raise GradcheckError(
            "vibetensor.autograd.gradcheck: NumPy is required for numerical gradient "
            "computation",
        ) from e
    return _np


def _dlpack_to_numpy(t: _C.Tensor, *, role: str, _np: Any):
    """Convert a VibeTensor tensor to a NumPy array via DLPack.

    This helper mirrors existing test helpers and keeps the NumPy-from-DLPack
    adapter localized. It intentionally lets unexpected exceptions from
    VibeTensor or NumPy propagate as raw errors to keep internal bugs visible.
    """

    cap = vt.to_dlpack(t)
    try:
        arr = _np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        # Newer NumPy expects an object with a __dlpack__ method; wrap capsule.
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover - tiny adapter
                return self._inner

        arr = _np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]
    return arr


def _call_fn_with_replaced_input(
    fn: Callable[..., _C.Tensor],
    inputs_tuple: tuple[_C.Tensor, ...],
    index: int,
    new_tensor_i: _C.Tensor,
    _np: Any,
) -> _C.Tensor:
    """Call ``fn`` with one input tensor replaced, under ``no_grad``.

    This is used for numerical finite-difference evaluation. It wraps
    user-visible exceptions in GradcheckError but lets existing GradcheckError
    instances propagate unchanged.
    """

    call_args = list(inputs_tuple)
    call_args[index] = new_tensor_i
    try:
        with no_grad():
            out = fn(*call_args)
    except GradcheckError:
        raise
    except Exception as e:  # noqa: BLE001
        raise GradcheckError(
            f"gradcheck: fn raised during numerical forward: {e}",
        ) from e
    return out


def _as_float_scalar(t: _C.Tensor, _np: Any) -> float:
    """Convert a 0-D scalar tensor to a finite Python float.

    Used for numerical finite-difference evaluation.
    """

    if not isinstance(t, _C.Tensor):
        raise GradcheckError(
            "gradcheck: internal error: expected VibeTensor tensor for scalar output",
        )

    arr = _dlpack_to_numpy(t, role="output", _np=_np)
    if arr.size != 1:
        raise GradcheckError(
            "gradcheck: fn must return a single scalar Tensor even for perturbed inputs",
        )
    val = float(arr.reshape(-1)[0])
    if not _np.isfinite(val):
        raise GradcheckError(
            "gradcheck: non-finite scalar output during numerical gradient computation",
        )
    return val


def gradcheck(
    fn: Callable[..., _C.Tensor],
    inputs: _C.Tensor | Sequence[_C.Tensor],
    *,
    eps: float = 1e-3,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    raise_exception: bool = True,
    fast_mode: bool | None = None,
) -> bool:
    """Numerical gradient check for scalar-output functions.

    This implements a slow, dense, central-difference gradcheck for CPU
    float32 leaf inputs using the stateful :func:`grad` helper for analytical
    gradients. It is intentionally narrow but Torch-shaped.
    """

    if not callable(fn):
        raise TypeError("gradcheck: fn must be callable")

    # Stage 0 – numeric-parameter and fast_mode validation, then environment.
    try:
        eps_f = float(eps)
        atol_f = float(atol)
        rtol_f = float(rtol)
    except Exception as e:  # noqa: BLE001
        raise GradcheckError(
            "vibetensor.autograd.gradcheck: eps must be > 0 and atol/rtol must be finite, "
            "non-negative floats",
        ) from e

    if (
        not math.isfinite(eps_f)
        or not math.isfinite(atol_f)
        or not math.isfinite(rtol_f)
        or eps_f <= 0.0
        or atol_f < 0.0
        or rtol_f < 0.0
    ):
        raise GradcheckError(
            "vibetensor.autograd.gradcheck: eps must be > 0 and atol/rtol must be finite, "
            "non-negative floats",
        )

    if fast_mode is not None and not isinstance(fast_mode, bool):
        raise GradcheckError(
            "vibetensor.autograd.gradcheck: fast_mode must be a bool or None",
        )
    if fast_mode is True:
        raise GradcheckError(
            "vibetensor.autograd.gradcheck: fast_mode=True is not implemented; "
            "pass fast_mode=False or None",
        )

    if is_inference_mode_enabled():
        raise GradcheckError(
            "vibetensor.autograd.gradcheck: cannot run gradcheck inside inference_mode; "
            "enable gradients and rebuild the graph",
        )
    if not is_grad_enabled():
        raise GradcheckError(
            "vibetensor.autograd.gradcheck: cannot run gradcheck inside no_grad; "
            "enable gradients and rebuild the graph",
        )

    _np = _require_numpy_for_gradcheck()

    # Stage 1 – input normalization and tensor validation.
    tensors = _as_tensor_sequence(
        inputs,
        api_name="gradcheck",
        arg_name="inputs",
        allow_empty=True,
    )

    if not tensors:
        raise GradcheckError(
            "gradcheck: inputs must not be an empty sequence",
        )

    normalized_inputs: list[_C.Tensor] = []
    for x in tensors:
        # CPU float32 check.
        if getattr(x, "dtype", None) != "float32":
            raise GradcheckError(
                "gradcheck: all inputs must be CPU float32 tensors",
            )
        dev = getattr(x, "device", None)
        dev_type = None
        if isinstance(dev, (tuple, list)) and dev:
            try:
                dev_type = int(dev[0])
            except Exception:  # pragma: no cover - defensive
                dev_type = None
        if dev_type != 1:  # 1 = kDLCPU
            raise GradcheckError(
                "gradcheck: all inputs must be CPU float32 tensors",
            )

        # Layout: contiguous + non-overlapping and dense when available.
        is_contig = bool(getattr(x, "is_contiguous", lambda: False)())
        dense_ok = True
        iod = getattr(x, "is_non_overlapping_and_dense", None)
        if callable(iod):
            dense_ok = bool(iod())
        if not is_contig or not dense_ok:
            raise GradcheckError(
                "gradcheck: non-contiguous or overlapping inputs are unsupported; call .clone() or vt.tensor(..., copy=True) first",
            )

        # Leaf and requires_grad checks.
        is_leaf = bool(getattr(x, "is_leaf", False))
        requires_grad = bool(getattr(x, "requires_grad", False))
        if (not is_leaf) or (not requires_grad):
            raise GradcheckError(
                "gradcheck: all inputs must be non-view leaf tensors with requires_grad=True",
            )

        normalized_inputs.append(x)

    inputs_tuple = tuple(normalized_inputs)

    # Stage 2 – analytical forward and gradients.
    try:
        out = fn(*inputs_tuple)
    except GradcheckError:
        raise
    except Exception as e:  # noqa: BLE001
        raise GradcheckError(f"gradcheck: fn raised during forward: {e}") from e

    if not isinstance(out, _C.Tensor) or tuple(out.sizes) != ():
        raise GradcheckError(
            "gradcheck expects fn to return a single 0-D scalar Tensor",
        )

    try:
        analytical_tensors = grad(out, inputs_tuple, allow_unused=False)
    except GradcheckError:
        raise
    except Exception as e:  # noqa: BLE001
        raise GradcheckError(f"gradcheck: grad precondition failed: {e}") from e

    analytical_np: list[Any] = []
    for i, (x, g) in enumerate(zip(inputs_tuple, analytical_tensors)):
        if not isinstance(g, _C.Tensor) or tuple(g.sizes) != tuple(x.sizes) or getattr(g, "dtype", None) != "float32":
            raise GradcheckError(
                f"gradcheck: analytical gradient shape/dtype/finiteness mismatch for input {i}",
            )
        arr = _dlpack_to_numpy(g, role="analytic_grad", _np=_np)
        if not _np.all(_np.isfinite(arr)):
            raise GradcheckError(
                f"gradcheck: analytical gradient has non-finite values for input {i}",
            )
        analytical_np.append(
            arr.astype(_np.float64, copy=True).reshape(
                tuple(int(s) for s in x.sizes),
            )
        )

    # Stage 3 – numeric finite differences.
    base_np = [
        _dlpack_to_numpy(x, role="clone", _np=_np)
        .astype(_np.float64, copy=True)
        .reshape(tuple(int(s) for s in x.sizes))
        for x in inputs_tuple
    ]

    numeric_np: list[Any] = []
    for i, (x, base_arr) in enumerate(zip(inputs_tuple, base_np)):
        shape_i = base_arr.shape
        flat = base_arr.reshape(-1)
        num_grad_flat = _np.empty_like(flat)
        for k in range(flat.size):
            flat_plus = flat.copy()
            flat_minus = flat.copy()
            flat_plus[k] += eps_f
            flat_minus[k] -= eps_f

            plus_tensor_i = vt.tensor(
                flat_plus.reshape(shape_i),
                dtype="float32",
            )
            minus_tensor_i = vt.tensor(
                flat_minus.reshape(shape_i),
                dtype="float32",
            )

            out_plus = _call_fn_with_replaced_input(fn, inputs_tuple, i, plus_tensor_i, _np)
            out_minus = _call_fn_with_replaced_input(fn, inputs_tuple, i, minus_tensor_i, _np)
            f_plus = _as_float_scalar(out_plus, _np)
            f_minus = _as_float_scalar(out_minus, _np)
            num_grad_flat[k] = (f_plus - f_minus) / (2.0 * eps_f)

        numeric_np.append(num_grad_flat.reshape(shape_i))

    # Stage 4 – comparison and result.
    flag = bool(raise_exception)
    for i, (a_arr, n_arr) in enumerate(zip(analytical_np, numeric_np)):
        a_flat = a_arr.reshape(-1)
        n_flat = n_arr.reshape(-1)
        for k, (a_val, n_val) in enumerate(zip(a_flat, n_flat)):
            a_f = float(a_val)
            n_f = float(n_val)
            if (not math.isfinite(a_f)) or (not math.isfinite(n_f)):
                raise GradcheckError(
                    f"gradcheck: non-finite analytical or numerical gradient for input {i} at flat index {k}: "
                    f"analytical={a_f}, numerical={n_f}",
                )

            tol = atol_f + rtol_f * max(abs(a_f), abs(n_f))
            if abs(a_f - n_f) > tol:
                if flag:
                    raise GradcheckError(
                        f"gradcheck: numerical and analytical gradients differ for input {i} at flat index {k}: "
                        f"analytical={a_f}, numerical={n_f}, tol={tol}",
                    )
                return False

    return True


def gradgradcheck(
    fn: Callable[..., _C.Tensor],
    inputs: _C.Tensor | Sequence[_C.Tensor],
    *,
    eps: float = 1e-3,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    raise_exception: bool = True,
    fast_mode: bool | None = None,
) -> bool:
    """Structured stub for higher-order gradient checking.

    This mirrors the PyTorch API but does not perform any computation
    It validates ``fast_mode`` and then either raises a GradcheckError or
    returns ``False`` depending on ``raise_exception``.
    """

    if fast_mode is not None and not isinstance(fast_mode, bool):
        raise GradcheckError(
            "vibetensor.autograd.gradgradcheck: fast_mode must be a bool or None",
        )
    if fast_mode is True:
        raise GradcheckError(
            "vibetensor.autograd.gradgradcheck: fast_mode=True is not implemented; "
            "pass fast_mode=False or None",
        )

    if bool(raise_exception):
        raise GradcheckError(
            "vibetensor.autograd.gradgradcheck: higher-order autograd is not implemented",
        )

    return False


import sys as _sys

try:  # pragma: no cover - import wiring
    from vibetensor import autograd_functional as _functional
except Exception:  # pragma: no cover
    _functional = None

if _functional is not None:
    _sys.modules[__name__ + ".functional"] = _functional
    # Expose as attribute for ``vibetensor.autograd.functional`` access.
    functional = _functional  # type: ignore[assignment]

try:  # pragma: no cover - import wiring
    from vibetensor import autograd_forward_ad as _forward_ad
except Exception:  # pragma: no cover
    _forward_ad = None

if _forward_ad is not None:
    _sys.modules[__name__ + ".forward_ad"] = _forward_ad
    # Expose as attribute for ``vibetensor.autograd.forward_ad`` access.
    forward_ad = _forward_ad  # type: ignore[assignment]
