from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

from vibetensor import _C as _C

_KDLCPU = 1
_KDLCUDA = 2
_KDLCUDAMANAGED = 13


def _shape_tuple(x: Any) -> Tuple[int, ...]:
    shp = getattr(x, "shape", None)
    if isinstance(shp, (tuple, list)) and shp:
        return tuple(int(s) for s in shp)
    sizes = getattr(x, "sizes", None)
    if isinstance(sizes, (tuple, list)):
        return tuple(int(s) for s in sizes)
    raise TypeError("expected a tensor-like with .shape or .sizes")


def _vt_device_tuple(x: Any) -> Tuple[int, int]:
    dev = getattr(x, "device", None)
    if not isinstance(dev, (tuple, list)) or len(dev) < 2:
        raise TypeError("expected a VibeTensor-like object with .device=(type,index)")
    return (int(dev[0]), int(dev[1]))


def _is_vt_tensor(x: Any) -> bool:
    dev = getattr(x, "device", None)
    return isinstance(dev, (tuple, list)) and len(dev) >= 2 and isinstance(dev[0], int)


def _torch_dtype_from_token(like: Any, tok: str) -> Any:
    t = str(tok)
    if t in ("float32", "fp32"):
        fn = getattr(like, "float", None)
        if callable(fn):
            return fn().dtype
    if t in ("float16", "fp16", "half"):
        fn = getattr(like, "half", None)
        if callable(fn):
            return fn().dtype
    if t in ("bfloat16", "bf16"):
        fn = getattr(like, "bfloat16", None)
        if callable(fn):
            return fn().dtype
    return getattr(like, "dtype", None)


def _normalize_dtype_for_like(like: Any, dtype: Any | None) -> Any:
    if dtype is None:
        return getattr(like, "dtype", None)
    if isinstance(dtype, str):
        return _torch_dtype_from_token(like, dtype)
    return dtype


def empty_like(x: Any, *, shape: Optional[Sequence[int]] = None, dtype: Any | None = None) -> Any:
    if _is_vt_tensor(x):
        sizes = list(int(s) for s in (shape if shape is not None else _shape_tuple(x)))
        if dtype is None:
            tok = str(getattr(x, "dtype"))
        elif isinstance(dtype, str):
            tok = str(dtype)
        else:
            tok = str(dtype)
            if tok.startswith("torch."):
                tok = tok.split(".", 1)[1]
        dev_type, dev_index = _vt_device_tuple(x)
        if dev_type == _KDLCPU:
            return _C._cpu_empty(sizes, tok)
        if dev_type in (_KDLCUDA, _KDLCUDAMANAGED):
            return _C._cuda_empty(sizes, tok, int(dev_index))  # type: ignore[attr-defined]
        raise ValueError(f"unsupported device type: {dev_type}")

    new_empty = getattr(x, "new_empty", None)
    if callable(new_empty):
        shp = tuple(int(s) for s in (shape if shape is not None else _shape_tuple(x)))
        dt = _normalize_dtype_for_like(x, dtype)
        dev = getattr(x, "device", None)
        return new_empty(shp, dtype=dt, device=dev)

    raise TypeError("empty_like: unsupported tensor type (expected VibeTensor or PyTorch-like tensor)")


def empty(shape: Sequence[int] | int, *, like: Any, dtype: Any | None = None) -> Any:
    shp = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    return empty_like(like, shape=shp, dtype=dtype)


def zeros_like(x: Any, *, shape: Optional[Sequence[int]] = None, dtype: Any | None = None) -> Any:
    if _is_vt_tensor(x):
        sizes = list(int(s) for s in (shape if shape is not None else _shape_tuple(x)))
        if dtype is None:
            tok = str(getattr(x, "dtype"))
        elif isinstance(dtype, str):
            tok = str(dtype)
        else:
            tok = str(dtype)
            if tok.startswith("torch."):
                tok = tok.split(".", 1)[1]
        dev_type, dev_index = _vt_device_tuple(x)
        if dev_type == _KDLCPU:
            return _C._cpu_zeros(sizes, tok)
        if dev_type in (_KDLCUDA, _KDLCUDAMANAGED):
            return _C._cuda_zeros(sizes, tok, int(dev_index))  # type: ignore[attr-defined]
        raise ValueError(f"unsupported device type: {dev_type}")

    new_zeros = getattr(x, "new_zeros", None)
    if callable(new_zeros):
        shp = tuple(int(s) for s in (shape if shape is not None else _shape_tuple(x)))
        dt = _normalize_dtype_for_like(x, dtype)
        dev = getattr(x, "device", None)
        return new_zeros(shp, dtype=dt, device=dev)

    # Fallback: allocate then rely on downstream kernels to overwrite.
    return empty_like(x, shape=shape, dtype=dtype)


def zeros(shape: Sequence[int] | int, *, like: Any, dtype: Any | None = None) -> Any:
    shp = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
    return zeros_like(like, shape=shp, dtype=dtype)


def as_int_tuple(x: Any) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (int(x),)
    if isinstance(x, (tuple, list)):
        return tuple(int(s) for s in x)
    raise TypeError("expected int or sequence of ints")

