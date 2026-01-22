# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Iterator, List

import importlib as _importlib
import os as _os

from vibetensor import _C as _C
from . import _dtype as _dt


# Env-gated compatibility mapping for loader errors (to OSError)
def _compat_enabled() -> bool:
    return _os.getenv("VBT_OPS_COMPAT", "0") == "1"


def _is_fabric_tensor_marker(obj: Any) -> bool:
    # Use type-level marker checks to avoid triggering instance __getattr__.
    try:
        return getattr(type(obj), "__vbt_fabric_tensor__", False) is True
    except Exception:
        return False


def _raise_fabric_ops_error(fq: str) -> None:
    from vibetensor.fabric import _raise_fabric_error

    _raise_fabric_error(
        f"FabricTensor is not supported by vibetensor.torch.ops ({fq}); "
        "use vibetensor.fabric.* or export a local shard via ft.to_local_shards()"
    )


class _OpWrapper:
    def __init__(self, ns: str, op: str) -> None:
        self._ns = ns
        self._op = op
        fq = f"{ns}::{op}"
        # Introspection attributes for tests
        try:
            object.__setattr__(self, "__name__", op)
            object.__setattr__(self, "__qualname__", f"ops.{ns}.{op}")
            object.__setattr__(self, "__module__", "vibetensor.torch.ops")
            object.__setattr__(self, "__doc__", f"VibeTensor operator wrapper for {fq}")
        except Exception:
            pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fq = f"{self._ns}::{self._op}"

        # Reject FabricTensor early so users see a stable [Fabric] error rather
        # than a raw nanobind TypeError("incompatible function arguments").
        for x in args:
            if _is_fabric_tensor_marker(x):
                _raise_fabric_ops_error(fq)
        for x in kwargs.values():
            if _is_fabric_tensor_marker(x):
                _raise_fabric_ops_error(fq)

        # Auto-wrap scalars to 0-dim tensors for C++ dispatch.
        # This handles cases like polygamma(int, tensor) -> polygamma(Tensor, Tensor).
        # Avoid circular import by importing locally.
        from .factory import tensor

        c_args: List[Any] = []
        has_cuda = False
        cuda_device_idx = 0
        target_dtype = None

        # First pass: pick a target dtype (preferring non-bool for ops like `where`)
        # and (if present) a CUDA device.
        for a in args:
            if hasattr(a, "device") and hasattr(a, "dtype"):
                # Skip bool dtype for target_dtype so `where(cond, x, scalar)` uses x's dtype
                if target_dtype is None and a.dtype != "bool":
                    target_dtype = a.dtype
                if not has_cuda and isinstance(a.device, tuple) and a.device[0] == 2:
                    has_cuda = True
                    cuda_device_idx = a.device[1]
        # Fallback: if all tensors are bool, use the first one
        if target_dtype is None:
            for a in args:
                if hasattr(a, "device") and hasattr(a, "dtype"):
                    target_dtype = a.dtype
                    break

        for a in args:
            if isinstance(a, (int, float, bool)):
                if target_dtype is not None:
                    if has_cuda:
                        try:
                            # Optimization: create scalar directly on device.
                            tok = _dt.normalize_dtype_token(target_dtype, for_full=True)
                            t = _C._cuda_full([], tok, a, cuda_device_idx)
                        except Exception:
                            t = tensor(a, dtype=target_dtype)
                    else:
                        t = tensor(a, dtype=target_dtype)
                else:
                    t = tensor(a)

                # If we have CUDA tensors, move scalar to CUDA to satisfy dispatcher strict check.
                if has_cuda and hasattr(t, "cuda"):
                    # Check if already on correct device (type 2 is CUDA)
                    if not (
                        isinstance(t.device, tuple)
                        and t.device[0] == 2
                        and t.device[1] == cuda_device_idx
                    ):
                        t = t.cuda(cuda_device_idx)
                c_args.append(t)
            else:
                c_args.append(a)

        try:
            if kwargs:
                # Kwargs only supported for Python overrides (not C++ kernels)
                return _C._call_op_kwargs(fq, *c_args, **kwargs)
            return _C._call_op(fq, *c_args)
        except Exception as e:
            # Under compat, map specific messages to PyTorch-like
            if _compat_enabled():
                msg = str(e)
                if msg.startswith("unknown op: "):
                    raise RuntimeError(f"Didn't find operator '{fq}'") from None
                if msg.startswith("no CPU kernel registered: "):
                    raise RuntimeError(
                        f"No kernel found for dispatch key CPU for operator '{fq}'"
                    ) from None
                if msg.startswith("no CUDA kernel registered: "):
                    raise RuntimeError(
                        f"No kernel found for dispatch key CUDA for operator '{fq}'"
                    ) from None
            raise

    def __repr__(self) -> str:
        return f"<op 'ops.{self._ns}.{self._op}'>"


class _OpNamespace:
    __slots__ = ("_ns", "_cache")

    def __init__(self, ns: str) -> None:
        self._ns = ns
        self._cache: Dict[str, _OpWrapper] = {}

    def __getattr__(self, name: str) -> _OpWrapper:
        if name.startswith("_"):
            raise AttributeError(name)
        w = self._cache.get(name)
        if w is None:
            w = _OpWrapper(self._ns, name)
            self._cache[name] = w
        return w

    def __iter__(self) -> Iterator[_OpWrapper]:
        for k in list(self._cache.keys()):
            yield self._cache[k]

    def __dir__(self) -> List[str]:
        return sorted(list(self._cache.keys()))

    def __repr__(self) -> str:
        return f"<vibetensor.torch.ops namespace '{self._ns}'>"


class _Ops:
    __slots__ = ("_ns_cache",)

    def __init__(self) -> None:
        self._ns_cache: Dict[str, _OpNamespace] = {}

    # Loader helpers
    def load_library(self, path: str) -> None:
        try:
            _C._load_library(path)
        except Exception as e:
            if _compat_enabled():
                raise OSError(f"Could not load this library: {path}") from None
            raise

    @property
    def loaded_libraries(self) -> List[str]:
        try:
            return list(_C._loaded_libraries())
        except Exception:
            return []

    def import_module(self, mod_or_name: str | ModuleType) -> ModuleType:
        if isinstance(mod_or_name, ModuleType):
            return mod_or_name
        return _importlib.import_module(str(mod_or_name))

    def __getattr__(self, name: str) -> _OpNamespace:
        if name.startswith("_"):
            raise AttributeError(name)
        ns = self._ns_cache.get(name)
        if ns is None:
            ns = _OpNamespace(name)
            self._ns_cache[name] = ns
        return ns

    def __iter__(self) -> Iterator[_OpNamespace]:
        for k in list(self._ns_cache.keys()):
            yield self._ns_cache[k]

    def __dir__(self) -> List[str]:
        return sorted(list(self._ns_cache.keys()))


# Singleton ops namespace
ops = _Ops()


# Convenience: allow `import vibetensor.torch.ops as ops` and then `ops.vt.add(...)`.
# (PEP 562 module __getattr__ / __dir__.)
def __getattr__(name: str) -> Any:
    return getattr(ops, name)


def __dir__() -> List[str]:
    # Include both module-level names and delegated ops namespaces.
    return sorted(set(globals().keys()) | set(dir(ops)))
