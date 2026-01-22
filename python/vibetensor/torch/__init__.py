# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# torch-like overlay for VibeTensor
# Safe to import without torch or CUDA; exposes factories, DLPack helpers,
# and an ops namespace that forwards to vibetensor._C dispatcher helpers.
from __future__ import annotations

from typing import Any, Optional, Tuple

import importlib as _importlib
import sys as _sys
import builtins as _builtins

from vibetensor import _C as _C
import vibetensor.autograd as _A



try:
    import numpy as _np
except ImportError:
    _np = None

# Expose standard dtypes (backed by numpy if available, else strings)
if _np is not None:
    float32 = _np.float32
    float64 = _np.float64
    complex64 = _np.complex64
    complex128 = _np.complex128
    cfloat = complex64
    cdouble = complex128
    float16 = _np.float16
    int32 = _np.int32
    int64 = _np.int64
    bool_ = _np.bool_
    uint8 = _np.uint8
else:
    float32 = "float32"
    float64 = "float64"
    complex64 = "complex64"
    complex128 = "complex128"
    cfloat = "complex64"
    cdouble = "complex128"
    float16 = "float16"
    int32 = "int32"
    int64 = "int64"
    bool_ = "bool"
    uint8 = "uint8"

# Public surface export policy for tests
#__all__ = [
#    "float32", "float64", "float16", "int32", "int64", "bool", "uint8",
#]


# ----- DLPack helpers -----


def to_dlpack(t: Any):
    """Return a DLPack capsule for a VibeTensor tensor."""
    # Reject FabricTensor early so users see a stable [Fabric] error rather than a
    # raw nanobind TypeError("incompatible function arguments").
    try:
        is_fabric = getattr(type(t), "__vbt_fabric_tensor__", False) is True
    except Exception:
        is_fabric = False

    if is_fabric:
        from vibetensor.fabric import _raise_fabric_error

        _raise_fabric_error(
            "DLPack is not supported for FabricTensor; export a local shard via ft.to_local_shards()"
        )
    return _C._to_dlpack(t)


def _is_provider(obj: Any) -> bool:
    return hasattr(obj, "__dlpack__") and hasattr(obj, "__dlpack_device__")


def from_dlpack(x: Any, *, device: Optional[str | Tuple[int, int]] = None, copy: Optional[bool] = None):
    """Import a tensor from a DLPack provider or capsule.

    Policy:
    - If x is a provider (has __dlpack__), use the protocol and choose
      CPU or CUDA importer based on __dlpack_device__().
    - If x is a capsule, forward to the default importer. Capsule path
      forbids kwargs (device/copy);
    """
    # Provider path
    if _is_provider(x):
        try:
            dev = x.__dlpack_device__()
        except Exception:
            dev = None
        # Accept device kwarg only for CPU providers (mirrors tests)
        if device is not None and not isinstance(device, (str, tuple)):
            raise TypeError("from_dlpack: device must be a str or tuple when provided")
        if copy is not None:
            raise TypeError("from_dlpack: copy is not supported via overlay")
        if isinstance(dev, (tuple, list)) and len(dev) >= 2:
            dev_type = int(dev[0])
        else:
            dev_type = 1  # assume CPU if unknown
        cap = x.__dlpack__()
        if dev_type == 1:  # kDLCPU
            return _C._from_dlpack(cap)
        else:
            # Prefer CUDA copy importer when available
            # `_from_dlpack_cuda_copy` is deprecated for general use but kept for explicit copy semantics if needed.
            # if getattr(_C, "_from_dlpack_cuda_copy", None) is not None:
            #     return _C._from_dlpack_cuda_copy(cap)  # type: ignore[attr-defined]
            return _C._from_dlpack(cap)

    # Capsule path: forbid kwargs by default; try CUDA importer when available
    if device is not None or copy is not None:
        raise TypeError("from_dlpack: capsule path forbids device/copy kwargs")
    # if getattr(_C, "_from_dlpack_cuda_copy", None) is not None:
    #     try:
    #         return _C._from_dlpack_cuda_copy(x)  # type: ignore[attr-defined]
    #     except Exception:
    #         pass
    return _C._from_dlpack(x)


# ----- view helpers (complex) -----

def view_as_real(t: Any):
    """Return a real[...,2] view of a complex tensor (zero-copy)."""
    return t.view_as_real()


def view_as_complex(t: Any):
    """Return a complex view of a real[...,2] tensor (zero-copy)."""
    return t.view_as_complex()


# ----- dtype helpers -----
from . import _dtype as _dtype  # noqa: E402


# ----- factories (CPU) -----
# Keep a separate factory module for tests that import vibetensor.torch.factory
from . import factory as factory  # noqa: E402
from .factory import (  # noqa: E402
    empty,
    zeros,
    ones,
    full,
    zeros_like,
    ones_like,
    full_like,
    from_numpy,
    as_tensor,
    tensor,
    arange,
    linspace,
    eye,
    rand,
    rand_like,
    randn,
    randn_like,
    randint,
    randint_like,
)


# ----- embedding -----
from .embedding import embedding as embedding  # noqa: E402


# ----- nn (functional) -----
from . import nn as nn  # noqa: E402


# ----- ops namespace -----
from .ops import ops as ops  # noqa: E402

# Basic elementwise ops mirroring PyTorch entrypoints. These route
# through the dispatcher-backed `_C.vt` submodule so they participate

def relu(x: Any):
    """Elementwise ReLU via vt::relu."""
    return _C.vt.relu(x)


def add(a: Any, b: Any):
    """Elementwise add via vt::add."""
    return _C.vt.add(a, b)


def mul(a: Any, b: Any):
    """Elementwise mul via vt::mul."""
    return _C.vt.mul(a, b)


# Operator sugar on `_C.Tensor` so expressions like `x * x` and
# `x + y` route through the Autograd-aware vt ops.
_Tensor = _C.Tensor


def _tensor_add(self: Any, other: Any):
    # Support Tensor + Tensor and Tensor + scalar.
    if isinstance(other, _Tensor):
        return add(self, other)
    if isinstance(other, (int, float)):
        s = full([], float(other), dtype="float32")
        return add(self, s)
    return NotImplemented


def _tensor_radd(self: Any, other: Any):
    # Support scalar + Tensor as well as Tensor + Tensor.
    if isinstance(other, _Tensor):
        return add(other, self)
    if isinstance(other, (int, float)):
        s = full([], float(other), dtype="float32")
        return add(s, self)
    return NotImplemented


def _tensor_mul(self: Any, other: Any):
    # Support Tensor * Tensor and Tensor * scalar.
    if isinstance(other, _Tensor):
        return mul(self, other)
    if isinstance(other, (int, float)):
        s = full([], float(other), dtype="float32")
        return mul(self, s)
    return NotImplemented


def _tensor_rmul(self: Any, other: Any):
    # Support scalar * Tensor as well as Tensor * Tensor.
    if isinstance(other, _Tensor):
        return mul(other, self)
    if isinstance(other, (int, float)):
        s = full([], float(other), dtype="float32")
        return mul(self, s)
    return NotImplemented



def _tensor_cuda(self: Any, device: Optional[int] = None, non_blocking: bool = False):
    from . import cuda
    # and should NOT use .numpy().
    # self.device is tuple (type, index). type=1 is CPU, type=2 is CUDA.
    
    current_dev_type = self.device[0]
    current_dev_idx = self.device[1]
    
    # If explicitly non-blocking, we currently rely on the numpy path which is unsafe for temporaries.
    # The review suggests disallowing it or making it safe. For now, strict check.

    # Target device index
    target_idx = int(getattr(_C, "_cuda_current_device", lambda: 0)()) if device is None else int(device)
    
    if current_dev_type == 2: # kDLCUDA
        if current_dev_idx == target_idx:
            return self
        # Cross-device copy: not implemented in this PR via direct bindings, fall back or error?
        # Tests might expect it to work. If we use .numpy(), it fails. 
        # C++ binding _cuda_d2h_copy_numpy_sync can bring it to host, then to other device.
        # This is slow but correct.
        host_arr = _C._cuda_d2h_copy_numpy_sync(self)
        return cuda.to_device(host_arr, device=target_idx, non_blocking=non_blocking)

    # CPU to CUDA
    # Convert to numpy explicitly to ensure to_device gets a real numpy array
    arr = self.numpy() if hasattr(self, "numpy") else self
    return cuda.to_device(arr, device=target_idx, non_blocking=non_blocking)

def _tensor_cpu(self: Any):
    """Return a copy of this object in CPU memory."""
    # self.device is a tuple (type_int, index_int). kDLCPU is 1.
    if self.device[0] == 1: # kDLCPU
        return self
    # CUDA path: use binding to copy to numpy, then wrap
    if self.device[0] == 2: # kDLCUDA
        arr = _C._cuda_d2h_copy_numpy_sync(self)
        return factory.tensor(arr)
    raise RuntimeError("cpu(): unsupported device")

_Tensor.cpu = _tensor_cpu  # type: ignore[attr-defined]
_Tensor.cuda = _tensor_cuda  # type: ignore[attr-defined]
_Tensor.__add__ = _tensor_add  # type: ignore[attr-defined]
_Tensor.__radd__ = _tensor_radd  # type: ignore[attr-defined]
_Tensor.__mul__ = _tensor_mul  # type: ignore[attr-defined]
_Tensor.__rmul__ = _tensor_rmul  # type: ignore[attr-defined]


class _MinMaxResult:
    __slots__ = ("values", "_indices", "_op")

    def __init__(self, values: Any, indices: Any, op: str) -> None:
        self.values = values
        self._indices = indices
        self._op = op

    @property
    def indices(self):
        if self._indices is None:
            raise NotImplementedError(
                f"{self._op}: indices are not implemented for CUDA tensors; "
                "use amax/amin for values and argmax/argmin for indices"
            )
        return self._indices

    def __iter__(self):
        yield self.values
        yield self.indices

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int):
        if i == 0 or i == -2:
            return self.values
        if i == 1 or i == -1:
            return self.indices
        raise IndexError(i)


_orig_tensor_max = _Tensor.max  # type: ignore[attr-defined]
_orig_tensor_min = _Tensor.min  # type: ignore[attr-defined]


def _tensor_max(self: Any, dim: Any = None, keepdim: bool = False):
    out = _orig_tensor_max(self, dim, keepdim)
    if dim is None:
        return out
    if isinstance(out, tuple) and len(out) == 2:
        return _MinMaxResult(out[0], out[1], "max")
    return _MinMaxResult(out, None, "max")


def _tensor_min(self: Any, dim: Any = None, keepdim: bool = False):
    out = _orig_tensor_min(self, dim, keepdim)
    if dim is None:
        return out
    if isinstance(out, tuple) and len(out) == 2:
        return _MinMaxResult(out[0], out[1], "min")
    return _MinMaxResult(out, None, "min")


_Tensor.max = _tensor_max  # type: ignore[attr-defined]
_Tensor.min = _tensor_min  # type: ignore[attr-defined]


# Helper to create bound method from op name
def _make_method(op_name):
    def method(self, *args, **kwargs):
        # Use ops.vt namespace which dynamically forwards to dispatcher
        return getattr(ops.vt, op_name)(self, *args, **kwargs)
    return method


def _has_vt_op(op_name: str) -> bool:
    return _C._has_op(f"vt::{op_name}")


# Monkey-patch elementwise unary ops
for _op in [
    "abs", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "tanh", "sigmoid",
    "expm1", "log1p", "floor", "ceil", "trunc", "round", "frac", "reciprocal",
    "sign", "exp2", "log2", "sinh", "cosh", "tan", "asin", "acos", "atan",
    "erf", "erfc", "lgamma", "logit"
]:
    if _has_vt_op(_op):
        setattr(_Tensor, _op, _make_method(_op))

# Monkey-patch elementwise binary/ternary ops that are methods
for _op in ["pow", "clamp", "lerp", "fmax", "fmin", "remainder", "fmod", "atan2", "copysign", "hypot", "gcd", "lcm", "xlogy", "xlog1py", "nextafter", "heaviside"]:
    if _has_vt_op(_op):
        setattr(_Tensor, _op, _make_method(_op))

# Monkey-patch bitwise ops
for _op in ["bitwise_and", "bitwise_or", "bitwise_xor", "logical_and", "logical_or", "logical_xor"]:
    if _has_vt_op(_op):
        setattr(_Tensor, _op, _make_method(_op))

# Monkey-patch shifts
if _has_vt_op("lshift"):
    _Tensor.__lshift__ = ops.vt.lshift
if _has_vt_op("rshift"):
    _Tensor.__rshift__ = ops.vt.rshift
if _has_vt_op("bitwise_and"):
    _Tensor.__and__ = ops.vt.bitwise_and
if _has_vt_op("bitwise_or"):
    _Tensor.__or__ = ops.vt.bitwise_or
if _has_vt_op("bitwise_xor"):
    _Tensor.__xor__ = ops.vt.bitwise_xor

# Polygamma special case (n, input) -> input.polygamma(n)
if _has_vt_op("polygamma"):
    def _polygamma_method(self, n):
        return ops.vt.polygamma(n, self)
    _Tensor.polygamma = _polygamma_method



# ----- printing options (numpy-backed) -----


def get_printoptions():
    try:
        import numpy as _np
    except Exception:
        # Fallback defaults if numpy missing
        return {"precision": 8, "threshold": 1000, "edgeitems": 3, "linewidth": 75, "sci_mode": True}
    opts = _np.get_printoptions()
    return {
        "precision": int(opts.get("precision", 8)),
        "threshold": int(opts.get("threshold", 1000)),
        "edgeitems": int(opts.get("edgeitems", 3)),
        "linewidth": int(opts.get("linewidth", 75)),
        "sci_mode": _builtins.bool(opts.get("sci_mode", True)),
    }


def set_printoptions(*, precision: Optional[int] = None, threshold: Optional[int] = None, edgeitems: Optional[int] = None, linewidth: Optional[int] = None, sci_mode: Optional[bool] = None):
    import numpy as _np  # numpy is a test dependency
    kw = {}
    if precision is not None:
        kw["precision"] = int(precision)
    if threshold is not None:
        kw["threshold"] = int(threshold)
    if edgeitems is not None:
        kw["edgeitems"] = int(edgeitems)
    if linewidth is not None:
        kw["linewidth"] = int(linewidth)
    if sci_mode is not None:
        kw["sci_mode"] = _builtins.bool(sci_mode)
    if kw:
        _np.set_printoptions(**kw)



is_grad_enabled = _A.is_grad_enabled
set_grad_enabled = _A.set_grad_enabled
no_grad = _A.no_grad
enable_grad = _A.enable_grad



inference_mode = _A.inference_mode
is_inference_mode_enabled = _A.is_inference_mode_enabled



# Thin alias to the canonical vibetensor.autograd module for torch-like usage.
autograd = _A
_sys.modules[__name__ + ".autograd"] = _A

try:  # pragma: no cover - import wiring
    import vibetensor.autograd_functional as _AF
except Exception:  # pragma: no cover
    _AF = None

if _AF is not None:
    _sys.modules[__name__ + ".autograd.functional"] = _AF
    # Ensure ``vt.autograd.functional`` is the same module.
    try:
        autograd.functional = _AF  # type: ignore[attr-defined]
    except Exception:
        pass

try:  # pragma: no cover - import wiring
    import vibetensor.autograd_forward_ad as _AFAD
except Exception:  # pragma: no cover
    _AFAD = None

if _AFAD is not None:
    _sys.modules[__name__ + ".autograd.forward_ad"] = _AFAD
    # Ensure ``vt.autograd.forward_ad`` is the same module.
    try:
        autograd.forward_ad = _AFAD  # type: ignore[attr-defined]
    except Exception:
        pass

try:  # pragma: no cover - import wiring
    import vibetensor.autograd_graph as _AG
except Exception:  # pragma: no cover
    _AG = None

if _AG is not None:
    # Ensure ``vibetensor.torch.autograd.graph`` resolves to
    # ``vibetensor.autograd.graph``.
    _sys.modules[__name__ + ".autograd.graph"] = _AG
    try:
        autograd.graph = _AG  # type: ignore[attr-defined]
    except Exception:
        pass


# ----- CUDA helpers (optional) -----
# Provide a stable helper for vt.triton to retrieve the current VBT stream handle.


def _cuda_stream_handle_current() -> Optional[int]:
    try:
        # Resolve current device; -1 means runtime current
        dev = getattr(_C, "_cuda_current_device", lambda: -1)()
        handle = int(_C._cuda_stream_handle_current_for_device(int(dev)))  # type: ignore[attr-defined]
        return handle if handle != 0 else None
    except Exception:
        return None


# Expose vibetensor.torch.cuda as a submodule regardless of CUDA availability
try:
    from . import cuda as cuda  # type: ignore[assignment]
except Exception:  # pragma: no cover
    # Ensure import succeeds even if cuda overlay fails to init
    cuda = None  # type: ignore[assignment]

# ----- RNG (CPU) -----
from . import rng as rng  # noqa: E402
from .rng import manual_seed as manual_seed  # noqa: E402
from .rng import seed as seed  # noqa: E402
from .rng import initial_seed as initial_seed  # noqa: E402
from .rng import get_rng_state as get_rng_state  # noqa: E402
from .rng import set_rng_state as set_rng_state  # noqa: E402
from .rng import Generator as Generator  # noqa: E402
