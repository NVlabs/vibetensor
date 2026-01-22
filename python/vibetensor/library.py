# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from vibetensor import _C as _C
import atexit as _atexit

# Ensure C++ override map is cleared while the interpreter is still alive
def _reset_overrides() -> None:
    try:
        reset = getattr(_C, "_reset_boxed_python_overrides", None)
        if callable(reset):
            reset()
    except Exception:
        # Best effort: ignore errors during shutdown
        pass

def _reset_autograd_py() -> None:
    try:
        reset = getattr(_C, "_reset_autograd_py", None)
        if callable(reset):
            reset()
    except Exception:
        pass

# Register only safe teardown: clearing Python function overrides must happen
# before interpreter finalization to avoid DECREF after teardown.
_atexit.register(_reset_overrides)
# Note: Autograd registry reset is now handled in vibetensor/__init__.py with
# proper exception handling. The centralized cleanup there covers both
# g_autograd_py and g_overrides registries.

# DLPack device type codes (reuse enums semantics without importing a third-party).
_KDLCPU = 1
_KDLCUDA = 2
_KDLCUDAMANAGED = 13

# Per-op registry entry
class _Entry:
    __slots__ = ("impls", "fallbacks", "wildcard_fallback", "flags")
    def __init__(self) -> None:
        self.impls: Dict[str, Callable[..., Any]] = {}
        self.fallbacks: Dict[str, Callable[..., Any]] = {}
        self.wildcard_fallback: Optional[Callable[..., Any]] = None
        # Flags per dispatch key, reserved for future (e.g., use_triton)
        self.flags: Dict[str, Dict[str, bool]] = {"CPU": {}, "CUDA": {}}

# Global state keyed by fully-qualified op name
_REG: Dict[str, _Entry] = {}
_INSTALLED: set[str] = set()


def _norm_fqname(ns: str, spec: str) -> str:
    return spec if "::" in spec else f"{ns}::{spec}"


def _norm_dispatch_key(key: str) -> str:
    k = key.upper()
    if k not in ("CPU", "CUDA"):
        raise ValueError("dispatch_key must be 'CPU' or 'CUDA'")
    return k


def _boxed_mux(fqname: str) -> Callable[..., Any]:
    # Single closure per fqname; looks up live registry on each call
    def _override(*args: Any, **kwargs: Any) -> Any:
        # 0-arity: let base implementation run (CPU-only for nullary)
        if len(args) == 0:
            return _C._redispatch_boxed_current(*args)
        # Determine device of first tensor
        first = args[0]
        dev: Optional[Tuple[int, int]] = getattr(first, "device", None)
        if not isinstance(dev, (tuple, list)) or len(dev) < 2:
            return _C._redispatch_boxed_current(*args)
        dev_type = int(dev[0])
        if dev_type == _KDLCPU:
            sel = "CPU"
        elif dev_type == _KDLCUDA or dev_type == _KDLCUDAMANAGED:
            sel = "CUDA"
        else:
            return _C._redispatch_boxed_current(*args)
        # Optional: if mixed devices are present, defer to base for canonical error
        for a in args[1:]:
            d = getattr(a, "device", None)
            if d != dev:
                return _C._redispatch_boxed_current(*args)
        # Look up selection in registry
        entry = _REG.get(fqname)
        if entry is None:
            return _C._redispatch_boxed_current(*args)
        target = entry.impls.get(sel) or entry.fallbacks.get(sel) or entry.wildcard_fallback
        if target is None:
            return _C._redispatch_boxed_current(*args)
        try:
            out = target(*args, **kwargs)
        except NotImplementedError:
            return _C._redispatch_boxed_current(*args)
        if out is NotImplemented:
            return _C._redispatch_boxed_current(*args)
        return out

    return _override


def _ensure_installed(fqname: str) -> None:
    if fqname in _INSTALLED:
        return
    fn = _boxed_mux(fqname)
    try_register = getattr(_C, "_try_register_boxed_python_override", None)
    if callable(try_register):
        ret = try_register(fqname, fn)
        # ret is expected to be True (installed) or False (duplicate)
        _INSTALLED.add(fqname)
        return
    try:
        _C._register_boxed_python_override(fqname, fn)
    except Exception as e:
        msg = str(e)
        if msg.startswith("duplicate CPU impl (boxed): ") and fqname in msg:
            # Treat as success when another override already owns this op
            _INSTALLED.add(fqname)
            return
        raise
    _INSTALLED.add(fqname)


class Library:
    """Minimal torch.library-like surface for extension surface.

    Example:
        lib = Library("ext", "DEF")
        lib.define("ext::square(Tensor) -> Tensor")
        lib.impl("square", lambda x: x, dispatch_key="CPU")
    """

    __slots__ = ("_ns", "_kind", "_default_dispatch")

    def __init__(self, ns: str, kind: str, dispatch_key: str = "") -> None:
        self._ns = ns
        self._kind = kind
        self._default_dispatch = dispatch_key

    def define(self, schema: str) -> None:
        # 'def' is a reserved keyword; access via getattr
        c_def = getattr(_C, "def")
        c_def(schema)

    def impl(self, op_name_or_overload: str, fn: Callable[..., Any], *, dispatch_key: str, use_triton: bool = False, allow_override: bool = False) -> None:  # noqa: D401
        fq = _norm_fqname(self._ns, op_name_or_overload)
        key = _norm_dispatch_key(dispatch_key)
        entry = _REG.get(fq)
        if entry is None:
            entry = _Entry()
            _REG[fq] = entry
        if use_triton and key == "CUDA":
            # Legacy hook: allow vibetensor.torch.triton.make_cuda_wrapper to adapt
            # the callable (torch-free). Historically this performed Torch/Triton
            # interop; it is now a no-op shim.
            try:
                import vibetensor.torch.triton as vt_triton  # lazy import
            except Exception:
                # If import fails (e.g., cyclic), keep original fn; mux will redispatch on NotImplementedError
                pass
            else:
                fn = vt_triton.make_cuda_wrapper(fn)
                entry.flags.setdefault("CUDA", {})["use_triton"] = True
        if key in entry.impls:
            existing = entry.impls[key]
            if existing is not fn and not allow_override:
                raise ValueError(f"duplicate {key} impl: {fq}")
        entry.impls[key] = fn
        _ensure_installed(fq)

    def fallback(self, op_name_or_overload: str, fn: Callable[..., Any], *, dispatch_key: Optional[str] = None, allow_override: bool = False) -> None:
        fq = _norm_fqname(self._ns, op_name_or_overload)
        entry = _REG.get(fq)
        if entry is None:
            entry = _Entry()
            _REG[fq] = entry
        if dispatch_key is None:
            if entry.wildcard_fallback is not None and entry.wildcard_fallback is not fn and not allow_override:
                raise ValueError(f"duplicate fallback impl: {fq}")
            entry.wildcard_fallback = fn
        else:
            key = _norm_dispatch_key(dispatch_key)
            if key in entry.fallbacks:
                existing = entry.fallbacks[key]
                if existing is not fn and not allow_override:
                    raise ValueError(f"duplicate {key} fallback: {fq}")
            entry.fallbacks[key] = fn
        _ensure_installed(fq)


def register_triton_op(
    schema: str,
    impl: Callable[..., Any],
    *,
    dispatch_key: str = "CUDA",
    unwrap_scalars: bool = True
) -> None:
    """
    Convenience helper to register a VibeTensor operator backed by a Python implementation
    (typically a Triton kernel wrapper).

    Handles:
    1. Defining the schema (if not already defined)
    2. Unwrapping scalar arguments (if requested), as VibeTensor boxed dispatch currently
       passes all args as Tensors.

    Note:
    - This helper does *not* depend on torch.
    - For native Triton compilation/launch on VBT CUDA tensors, prefer
      :func:`vibetensor.triton.register`.

    Args:
        schema: Full schema string, e.g. "kf::layernorm(Tensor, Tensor, Tensor) -> Tensor"
        impl: The Python function implementing the kernel logic.
        dispatch_key: "CUDA" or "CPU" (default "CUDA")
        unwrap_scalars: If True, automatically unwraps 0-d Tensor arguments to Python scalars
                        before calling `impl`.
    """
    # Parse namespace/opname from schema (simplified parsing)
    # Expected format: "ns::opname(...)"
    if "(" not in schema:
         raise ValueError("Invalid schema format: missing '('")
    
    full_name = schema.split("(")[0].strip()
    if "::" not in full_name:
         raise ValueError("Schema must include namespace, e.g. 'ns::op'")
    
    ns, op_name = full_name.split("::", 1)
    
    # Define Library
    lib = Library(ns, "DEF")
    try:
        lib.define(schema)
    except Exception:
        # Schema might already exist, ignore re-definition errors
        pass
        
    # Wrap implementation if unwrapping is needed
    final_impl = impl
    if unwrap_scalars:
        def _scalar_unwrapper(*args, **kwargs):
            new_args = []
            for a in args:
                # Heuristic: unwrap 0-d tensors that have .item()
                if hasattr(a, "item") and hasattr(a, "numel") and a.numel() == 1:
                    new_args.append(a.item())
                else:
                    new_args.append(a)
            return impl(*new_args, **kwargs)
        final_impl = _scalar_unwrapper

    # Register
    lib.impl(op_name, final_impl, dispatch_key=dispatch_key, use_triton=False, allow_override=True)

