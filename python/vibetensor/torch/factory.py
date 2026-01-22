# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import os as _os

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]

from vibetensor import _C as _C
from . import _dtype as _dt

# Canonical cross-stream RNG misuse error message from C++.
# Mirrors vbt::rng::graph_capture::kErrCudaRngUseOnNonCaptureStream.
# When the extension exposes `_ERR_CUDA_RNG_USE_ON_NON_CAPTURE_STREAM`, we
# take it as the single source of truth; otherwise we fall back to the
# literal string pinned in C++ tests.
_ERR_CUDA_RNG_USE_ON_NON_CAPTURE_STREAM: str = getattr(
    _C,
    "_ERR_CUDA_RNG_USE_ON_NON_CAPTURE_STREAM",
    "rng: CUDA RNG operations on this generator are only allowed "
    "on the captured stream while CUDA Graph capture is active",
)

_HAS_CUDA: bool = bool(getattr(_C, "_has_cuda", False))
_cuda_rng_capture_active = getattr(_C, "_cuda_rng_is_capture_active_for_device", None)

try:  # Import graphs overlay lazily but once.
    from vibetensor.torch.cuda import graphs as _graphs_mod  # type: ignore[import]
except Exception:  # pragma: no cover - graphs absent on some builds
    _graphs_mod = None

_ERR_COMPLEX_DISABLED: str = "complex dtypes are disabled; set VBT_ENABLE_COMPLEX=1"


def _complex_enabled() -> bool:
    # Pinned semantics: enabled iff env is exactly the single character "1".
    return _os.getenv("VBT_ENABLE_COMPLEX", "") == "1"


# ----- dtype normalization -----

def _dtype_token(dtype: Any | None, *, for_full: bool = False) -> str:
    tok = _dt.normalize_dtype_token(dtype, for_full=for_full)
    if tok in ("complex64", "complex128") and not _complex_enabled():
        raise TypeError(_ERR_COMPLEX_DISABLED)
    return tok


# ----- shape/dtype helpers -----

def _normalize_sizes(shape: Any) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (int(shape),)
    if isinstance(shape, (tuple, list)):
        return tuple(int(s) for s in shape)
    raise TypeError("shape must be int or sequence of ints")


def _is_complex_scalar(x: Any) -> bool:
    if isinstance(x, complex):
        return True

    if _np is not None and isinstance(x, _np.generic):
        try:
            return _np.dtype(x.dtype).kind == "c"
        except Exception:
            return False

    return False


def _contains_complex(x: Any) -> bool:
    if _is_complex_scalar(x):
        return True

    if isinstance(x, (list, tuple)):
        return any(_contains_complex(e) for e in x)

    return False


def _validate_generator(gen: Any | None, expected_device: str) -> None:
    if gen is None:
        return
    # Avoid import cycle by string compare
    dev = getattr(gen, "device", None)
    if not isinstance(dev, str) or not dev.startswith(expected_device):
        raise ValueError(f"generator device mismatch: expected {expected_device}, got {dev}")


def _parse_device(device: Any | None) -> Tuple[str, Optional[int]]:
    # Returns (dev_type, index) where dev_type in {"cpu","cuda"}; index is None for cpu
    if device is None:
        return ("cpu", None)
    if isinstance(device, int):
        # Treat as cuda:<index>
        if not getattr(_C, "_has_cuda", False):
            raise ValueError("CUDA is not available")
        di = int(device)
        if di < 0 or di >= int(_C._cuda_device_count()):  # type: ignore[attr-defined]
            raise ValueError("device index out of range")
        return ("cuda", di)
    devs = str(device)
    if devs == "cpu":
        return ("cpu", None)
    if devs == "cuda":
        if not getattr(_C, "_has_cuda", False):
            raise ValueError("CUDA is not available")
        cur = int(getattr(_C, "_cuda_current_device", lambda: 0)())
        return ("cuda", cur)
    if devs.startswith("cuda:"):
        if not getattr(_C, "_has_cuda", False):
            raise ValueError("CUDA is not available")
        try:
            idx = int(devs.split(":", 1)[1])
        except Exception:
            raise ValueError("invalid cuda device string")
        if idx < 0 or idx >= int(_C._cuda_device_count()):  # type: ignore[attr-defined]
            raise ValueError("device index out of range")
        return ("cuda", idx)
    raise ValueError("device must be 'cpu', 'cuda', 'cuda:k', or integer index")


def _guard_rng_cuda_graph_cross_stream(dev_type: str, dev_idx: Optional[int]) -> None:
    """Best-effort guard for CUDA RNG ops under graph capture.

    Preconditions:
      - ``dev_type`` is "cpu" or "cuda" from ``_parse_device``.
      - ``dev_idx`` is None for CPU, or a valid CUDA device index for "cuda".

    When CUDA RNG capture is active for the given CUDA device and the
    *current* stream is not the capture stream, this guard raises a
    RuntimeError whose message matches the canonical C++ guard string
    ``vbt::rng::graph_capture::kErrCudaRngUseOnNonCaptureStream``.

    In all other configurations (CPU-only builds, CUDA without graphs,
    no RNG capture active, or failures to query capture status), this
    guard is a cheap no-op and defers to the C++ allocator and RNG
    kernels for behavior.
    """

    # 1. CPU paths never participate in CUDA RNG capture.
    if dev_type != "cuda" or dev_idx is None:
        return

    # 2. If CUDA is not available, _parse_device would already have
    # rejected CUDA devices, but keep a defensive check.
    if not _HAS_CUDA:
        return

    # 3. If the graphs overlay is missing, we cannot query stream-
    # capture status. This configuration will have skipped all
    # RNG-under-graphs tests via hasattr(vcuda, "graphs").
    if _graphs_mod is None:
        return

    # 4. Query RNG capture status for this device. If the binding is
    # missing or raises, treat this as "no capture" from the Python
    # guard's point of view and let C++ behavior stand.
    if _cuda_rng_capture_active is None:
        return

    try:
        active = bool(_cuda_rng_capture_active(int(dev_idx)))
    except Exception:
        return

    if not active:
        # No RNG capture for this device: arbitrary streams are allowed.
        return

    # 5. RNG capture is active for this device. Check whether the
    # current stream is the capture stream.
    try:
        is_capturing = bool(_graphs_mod.is_current_stream_capturing())
    except Exception:
        # If we cannot query stream capture status, revert to C++
        # behavior rather than second-guessing allocator logic.
        return

    if is_capturing:
        # On the capture stream: let the C++ bridge handle Philox and
        # capture semantics.
        return

    # 6. Cross-stream RNG misuse while capture is active.
    # Optionally mark the active graph context so graph.__exit__ can
    # suppress the expected cudaStreamEndCapture failure after this.
    try:
        get_ctx = getattr(_graphs_mod, "_get_active_graph_context", None)
        ctx = get_ctx() if callable(get_ctx) else None
        if ctx is not None:
            setattr(ctx, "_skip_capture_end", True)
    except Exception:
        # Best-effort only; do not mask the primary error.
        pass

    raise RuntimeError(_ERR_CUDA_RNG_USE_ON_NON_CAPTURE_STREAM)


# ----- factories -----

def empty(shape: Any, *, dtype: Any | None = None, device: str | int = "cpu"):
    # Keep CPU-only semantics for general factories
    if str(device) != "cpu":
        raise ValueError("only cpu device is supported for factories")
    sizes = list(_normalize_sizes(shape))
    tok = _dtype_token(dtype, for_full=False)
    return _C._cpu_empty(sizes, tok)


def zeros(shape: Any, *, dtype: Any | None = None, device: str | int = "cpu"):
    if str(device) != "cpu":
        raise ValueError("only cpu device is supported for factories")
    sizes = list(_normalize_sizes(shape))
    tok = _dtype_token(dtype, for_full=False)
    return _C._cpu_zeros(sizes, tok)


def ones(shape: Any, *, dtype: Any | None = None, device: str | int = "cpu"):
    if str(device) != "cpu":
        raise ValueError("only cpu device is supported for factories")
    return full(shape, 1, dtype=dtype, device=device)


def full(shape: Any, fill_value: Any, *, dtype: Any | None = None, device: str | int = "cpu"):
    if str(device) != "cpu":
        raise ValueError("only cpu device is supported for factories")
    sizes = list(_normalize_sizes(shape))
    tok = _dtype_token(dtype, for_full=True)
    if _is_complex_scalar(fill_value) and tok not in ("complex64", "complex128"):
        raise TypeError("cannot cast complex to real")
    return _C._cpu_full(sizes, tok, fill_value)


def zeros_like(t: Any, *, dtype: Any | None = None):
    # normalize sizes from input; ensure contiguity on output
    sizes = list(int(s) for s in getattr(t, "sizes"))
    tok = _dtype_token(dtype if dtype is not None else getattr(t, "dtype", None), for_full=False)
    # default behavior: create on CPU
    return _C._cpu_zeros(sizes, tok)


def ones_like(t: Any, *, dtype: Any | None = None):
    sizes = list(int(s) for s in getattr(t, "sizes"))
    tok = _dtype_token(dtype if dtype is not None else getattr(t, "dtype", None), for_full=True)
    return _C._cpu_full(sizes, tok, 1)


def full_like(t: Any, fill_value: Any, *, dtype: Any | None = None):
    sizes = list(int(s) for s in getattr(t, "sizes"))
    tok = _dtype_token(dtype if dtype is not None else getattr(t, "dtype", None), for_full=True)
    if _is_complex_scalar(fill_value) and tok not in ("complex64", "complex128"):
        raise TypeError("cannot cast complex to real")
    # Reject non-finite for integer types
    if tok in ("int32", "int64"):
        # Only guard the float() conversion in try/except; do not swallow the finite checks
        try:
            fv = float(fill_value)
        except Exception:
            fv = None  # unable to coerce; let backend handle errors
        else:
            import math as _math
            if _math.isinf(fv) or _math.isnan(fv):
                raise ValueError("fill_value must be finite for integer dtype")
    return _C._cpu_full(sizes, tok, fill_value)


# ----- Random factories (CPU + CUDA) -----

def rand(shape: Any, *, dtype: Any | None = None, device: str | int | None = None, generator: Any | None = None):
    dev_type, dev_idx = _parse_device(device)
    _validate_generator(generator, "cpu" if dev_type == "cpu" else f"cuda:{dev_idx}")
    _guard_rng_cuda_graph_cross_stream(dev_type, dev_idx)
    sizes = list(_normalize_sizes(shape))
    tok = _dtype_token(dtype if dtype is not None else "float32", for_full=False)
    t = _C._cpu_empty(sizes, tok) if dev_type == "cpu" else _C._cuda_empty(sizes, tok, dev_idx)  # type: ignore[attr-defined]
    _C._uniform_(t, 0.0, 1.0)
    return t


def rand_like(t: Any, *, dtype: Any | None = None, device: str | int | None = None, generator: Any | None = None):
    dev_type, dev_idx = _parse_device(device)
    _validate_generator(generator, "cpu" if dev_type == "cpu" else f"cuda:{dev_idx}")
    _guard_rng_cuda_graph_cross_stream(dev_type, dev_idx)
    sizes = list(int(s) for s in getattr(t, "sizes"))
    tok = _dtype_token(dtype if dtype is not None else (getattr(t, "dtype", None) or "float32"), for_full=False)
    out = _C._cpu_empty(sizes, tok) if dev_type == "cpu" else _C._cuda_empty(sizes, tok, dev_idx)  # type: ignore[attr-defined]
    _C._uniform_(out, 0.0, 1.0)
    return out


def randn(shape: Any, *, dtype: Any | None = None, device: str | int | None = None, generator: Any | None = None):
    dev_type, dev_idx = _parse_device(device)
    _validate_generator(generator, "cpu" if dev_type == "cpu" else f"cuda:{dev_idx}")
    _guard_rng_cuda_graph_cross_stream(dev_type, dev_idx)
    sizes = list(_normalize_sizes(shape))
    tok = _dtype_token(dtype if dtype is not None else "float32", for_full=False)
    out = _C._cpu_empty(sizes, tok) if dev_type == "cpu" else _C._cuda_empty(sizes, tok, dev_idx)  # type: ignore[attr-defined]
    _C._normal_(out, 0.0, 1.0)
    return out


def randn_like(t: Any, *, dtype: Any | None = None, device: str | int | None = None, generator: Any | None = None):
    dev_type, dev_idx = _parse_device(device)
    _validate_generator(generator, "cpu" if dev_type == "cpu" else f"cuda:{dev_idx}")
    _guard_rng_cuda_graph_cross_stream(dev_type, dev_idx)
    sizes = list(int(s) for s in getattr(t, "sizes"))
    tok = _dtype_token(dtype if dtype is not None else (getattr(t, "dtype", None) or "float32"), for_full=False)
    out = _C._cpu_empty(sizes, tok) if dev_type == "cpu" else _C._cuda_empty(sizes, tok, dev_idx)  # type: ignore[attr-defined]
    _C._normal_(out, 0.0, 1.0)
    return out


def randint(low: int, high: int, shape: Any, *, dtype: Any | None = None, device: str | int | None = None, generator: Any | None = None):
    dev_type, dev_idx = _parse_device(device)
    _validate_generator(generator, "cpu" if dev_type == "cpu" else f"cuda:{dev_idx}")
    _guard_rng_cuda_graph_cross_stream(dev_type, dev_idx)
    sizes = list(_normalize_sizes(shape))
    tok = _dtype_token(dtype if dtype is not None else "int64", for_full=False)
    out = _C._cpu_empty(sizes, tok) if dev_type == "cpu" else _C._cuda_empty(sizes, tok, dev_idx)  # type: ignore[attr-defined]
    _C._randint_(out, int(low), int(high))
    return out


def randint_like(t: Any, low: int, high: int, *, dtype: Any | None = None, device: str | int | None = None, generator: Any | None = None):
    dev_type, dev_idx = _parse_device(device)
    _validate_generator(generator, "cpu" if dev_type == "cpu" else f"cuda:{dev_idx}")
    _guard_rng_cuda_graph_cross_stream(dev_type, dev_idx)
    sizes = list(int(s) for s in getattr(t, "sizes"))
    tok = _dtype_token(dtype if dtype is not None else (getattr(t, "dtype", None) or "int64"), for_full=False)
    out = _C._cpu_empty(sizes, tok) if dev_type == "cpu" else _C._cuda_empty(sizes, tok, dev_idx)  # type: ignore[attr-defined]
    _C._randint_(out, int(low), int(high))
    return out


# ----- NumPy bridges (CPU) -----

def from_numpy(array: Any):
    if _np is None:
        raise RuntimeError("NumPy is required for from_numpy")
    arr = _np.asarray(array)
    # Byte-order check: reject non-native byte order
    dt = _np.dtype(arr.dtype)
    if dt.kind == "c" and not _complex_enabled():
        raise TypeError(_ERR_COMPLEX_DISABLED)
    byteorder = dt.byteorder
    if byteorder not in ("=", "|"):
        import sys as _sys
        if (byteorder == "<" and _sys.byteorder != "little") or (byteorder == ">" and _sys.byteorder != "big"):
            raise ValueError("given numpy array has byte order different from the native byte order")
    # Stride checks: precedence for non-multiple of itemsize over negative stride
    item_b = int(dt.itemsize)
    strides = tuple(int(s) for s in arr.strides)
    if any((abs(s) % item_b) != 0 for s in strides):
        raise ValueError("given numpy array strides not a multiple of the element byte size")
    if any(s < 0 for s in strides):
        raise ValueError("At least one stride in the given numpy array is negative")
    # If array is not writeable (e.g., broadcast views), fall back to copying
    if not arr.flags.writeable:
        tok = dt.name
        return _C._cpu_from_numpy_copy(_np.ascontiguousarray(arr), tok)
    # Prefer zero-copy import via DLPack when possible
    import vibetensor.torch as vt  # lazy import to avoid cycles
    try:
        return vt.from_dlpack(arr)
    except Exception:
        # Fallback to copy
        tok = dt.name
        return _C._cpu_from_numpy_copy(_np.ascontiguousarray(arr), tok)


def as_tensor(data: Any, dtype: Any | None = None):
    # Scalar support: int, float, bool (create 0-d tensor)
    if isinstance(data, (int, float, bool)):
        tok = _dtype_token(dtype if dtype is not None else ("float32" if isinstance(data, float) else ("bool" if isinstance(data, bool) else "int64")), for_full=True)
        return _C._cpu_full([], tok, data)

    if isinstance(data, complex):
        tok = _dtype_token(dtype if dtype is not None else "complex64", for_full=True)
        if tok not in ("complex64", "complex128"):
            raise TypeError("cannot cast complex to real")
        return _C._cpu_full([], tok, data)

    # NumPy ndarray path: try zero-copy view if dtype matches
    if _np is not None and isinstance(data, _np.ndarray):
        src_tok = _np.dtype(data.dtype).name
        src_is_complex = _np.dtype(data.dtype).kind == "c"

        tok = _dtype_token(src_tok if dtype is None else dtype, for_full=False)

        if src_is_complex and tok not in ("complex64", "complex128"):
            raise TypeError("cannot cast complex to real")

        if dtype is None or tok == src_tok:
            # Prefer DLPack provider path for zero-copy when possible
            import vibetensor.torch as vt  # lazy import to avoid cycles
            try:
                return vt.from_dlpack(data)
            except Exception:
                # DLPack import can fail even for supported dtypes (e.g. some NumPy
                # zero-size arrays) or for dtypes that our DLPack importer does not
                # yet support. Fall back to a copy in those cases.
                return _C._cpu_from_numpy_copy(_np.ascontiguousarray(data), src_tok)

        # Otherwise fallback to copy with requested dtype (cast via NumPy)
        arr = _np.ascontiguousarray(data, dtype=_np.dtype(tok))
        return _C._cpu_from_numpy_copy(arr, tok)
    # Reject VBT tensor input
    if hasattr(data, "device") and hasattr(data, "dtype") and hasattr(data, "sizes"):
        raise TypeError("tensor(): expected Python sequence or NumPy ndarray, not vibetensor tensor")
    # Python sequence: infer dtype
    if isinstance(data, (list, tuple)):
        if _np is None:
            def _shape_and_flatten(x: Any):
                if isinstance(x, (list, tuple)):
                    if len(x) == 0:
                        return (0,), []
                    sub_shape = None
                    flat: list[Any] = []
                    for e in x:
                        sh, fl = _shape_and_flatten(e) if isinstance(e, (list, tuple)) else ((), [e])
                        if sub_shape is None:
                            sub_shape = sh
                        elif sh != sub_shape:
                            raise ValueError("ragged nested sequences are not supported")
                        flat.extend(fl)
                    return (len(x),) + (sub_shape or ()), flat
                return (), [x]

            shape, flat = _shape_and_flatten(data)

            contains_complex = any(isinstance(v, complex) for v in flat)

            if dtype is None:
                if len(flat) == 0:
                    inferred = "float32"
                elif contains_complex:
                    inferred = "complex64"
                elif all(isinstance(v, bool) for v in flat):
                    inferred = "bool"
                else:
                    inferred = "int64"
                    for v in flat:
                        if isinstance(v, bool):
                            continue
                        if isinstance(v, float):
                            inferred = "float32"
                            break
            else:
                inferred = None

            tok = _dtype_token(dtype if dtype is not None else inferred, for_full=False)
            if contains_complex and tok not in ("complex64", "complex128"):
                raise TypeError("cannot cast complex to real")
            return _C._cpu_from_sequence_copy(flat, list(shape), tok)  # type: ignore[attr-defined]

        # NumPy present
        contains_complex = _contains_complex(data)

        if dtype is None:
            if len(data) == 0:
                inferred = "float32"
            elif contains_complex:
                inferred = "complex64"
            elif all(isinstance(v, bool) for v in data):
                inferred = "bool"
            else:
                inferred = "int64"
                for v in data:
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, float):
                        inferred = "float32"
                        break
        else:
            inferred = None

        tok = _dtype_token(dtype if dtype is not None else inferred, for_full=False)
        if contains_complex and tok not in ("complex64", "complex128"):
            raise TypeError("cannot cast complex to real")
        arr = _np.array(data, dtype=_np.dtype(tok))
        return _C._cpu_from_numpy_copy(_np.ascontiguousarray(arr), tok)

    raise TypeError("tensor(): unsupported input type")


def tensor(data: Any, *, dtype: Any | None = None, device: str | None = None):
    if device not in (None, "cpu"):
        raise ValueError("only cpu device is supported for tensor()")
    return as_tensor(data, dtype=dtype)


# ----- RNG helpers -----

def arange(start: Any, end: Any | None = None, step: Any | None = None, *, dtype: Any | None = None):
    if _np is None:
        raise RuntimeError("NumPy is required for arange")
    if end is None:
        start_val, end_val = 0, start
    else:
        start_val, end_val = start, end
    step_val = 1 if step is None else step
    if isinstance(step_val, (int, float)) and step_val == 0:
        raise ValueError("step must be non-zero")
    # dtype inference: any float -> float32 else int64 unless dtype provided
    tok = _dtype_token(dtype if dtype is not None else ("float32" if any(isinstance(v, float) for v in (start_val, end_val, step_val)) else "int64"), for_full=False)
    arr = _np.arange(start_val, end_val, step_val, dtype=_np.dtype(tok))
    return _C._cpu_from_numpy_copy(arr, tok)


def linspace(start: Any, end: Any, *, steps: int = 100, dtype: Any | None = None):
    if _np is None:
        raise RuntimeError("NumPy is required for linspace")
    if not isinstance(steps, int):
        raise TypeError("steps must be an int")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    tok = _dtype_token(dtype if dtype is not None else "float32", for_full=False)
    if steps == 0:
        arr = _np.empty((0,), dtype=_np.dtype(tok))
    elif steps == 1:
        arr = _np.array([start], dtype=_np.dtype(tok))
    else:
        arr = _np.linspace(start, end, num=steps, dtype=_np.dtype(tok))
    return _C._cpu_from_numpy_copy(arr, tok)


def eye(n: int, m: Optional[int] = None, *, dtype: Any | None = None):
    if not isinstance(n, int) or (m is not None and not isinstance(m, int)):
        raise TypeError("n and m must be integers")
    if _np is None:
        raise RuntimeError("NumPy is required for eye")
    tok = _dtype_token(dtype if dtype is not None else "float32", for_full=False)
    arr = _np.eye(N=n, M=(n if m is None else m), dtype=_np.dtype(tok))
    return _C._cpu_from_numpy_copy(arr, tok)
