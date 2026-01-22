# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vibetensor import _C as _C

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore[assignment]


def _is_namedtuple_instance(x: Any) -> bool:
    # Heuristic used by PyTorch: a tuple subclass with `_fields`.
    return isinstance(x, tuple) and hasattr(x, "_fields")


def default_convert(data: Any) -> Any:
    """Recursively convert NumPy arrays and Python scalars to VibeTensor tensors.

    This is used when auto-batching is disabled (``batch_size=None`` and
    ``batch_sampler=None``).
    """

    if _np is not None and isinstance(data, _np.ndarray):
        import vibetensor.torch as vt

        return vt.from_numpy(data)

    if isinstance(data, (int, float, bool)):
        import vibetensor.torch as vt

        return vt.as_tensor(data)

    if isinstance(data, _C.Tensor):
        return data

    if isinstance(data, Mapping):
        return {k: default_convert(v) for k, v in data.items()}

    if _is_namedtuple_instance(data):
        return type(data)(*(default_convert(v) for v in data))

    if isinstance(data, tuple):
        return tuple(default_convert(v) for v in data)

    if isinstance(data, list):
        return [default_convert(v) for v in data]

    return data


def default_collate(batch: list[Any]) -> Any:
    """Collate a batch of samples into a single batch.

    Supported leaves (v1):
      - CPU VibeTensor tensors
      - C-contiguous NumPy ndarrays
      - Python scalars (int/float/bool)
      - strings/bytes (returned as list)

    Nested containers (dict/list/tuple/namedtuple) are collated elementwise.
    """

    if not isinstance(batch, list):
        raise TypeError("default_collate: batch must be a list")
    if len(batch) == 0:
        raise ValueError("default_collate: batch must be non-empty")

    elem = batch[0]

    # ----- Tensor leaves -----
    if isinstance(elem, _C.Tensor):
        # Reject CUDA tensors (no stack op; Tensor.numpy() is CPU-only).
        dev = getattr(elem, "device", (1, 0))
        try:
            dev_type = int(dev[0])
        except Exception:
            dev_type = 1
        if dev_type != 1:
            raise TypeError(
                "default_collate: CUDA tensor leaves are not supported; provide a custom collate_fn"
            )

        if _np is None:
            raise RuntimeError("default_collate: NumPy is required for tensor collation")

        arrs = []
        for t in batch:
            if not isinstance(t, _C.Tensor):
                raise TypeError("default_collate: mixed tensor/non-tensor batch")
            d = getattr(t, "device", (1, 0))
            if int(d[0]) != 1:
                raise TypeError(
                    "default_collate: CUDA tensor leaves are not supported; provide a custom collate_fn"
                )
            arrs.append(t.numpy())

        import vibetensor.torch as vt

        stacked = _np.stack(arrs, axis=0)
        return vt.from_numpy(stacked)

    # ----- NumPy leaves -----
    if _np is not None and isinstance(elem, _np.ndarray):
        for a in batch:
            if not isinstance(a, _np.ndarray):
                raise TypeError("default_collate: mixed ndarray/non-ndarray batch")
            if not a.flags.c_contiguous:
                raise TypeError(
                    "default_collate: numpy array leaves must be C-contiguous; provide a custom collate_fn"
                )

        import vibetensor.torch as vt

        stacked = _np.stack(batch, axis=0)
        return vt.from_numpy(stacked)

    # ----- Scalar leaves -----
    if isinstance(elem, (int, float, bool)):
        for x in batch:
            if not isinstance(x, (int, float, bool)):
                raise TypeError("default_collate: mixed scalar/non-scalar batch")
        import vibetensor.torch as vt

        return vt.as_tensor(batch)

    # ----- Strings / bytes -----
    if isinstance(elem, (str, bytes)):
        return list(batch)

    # ----- Nested containers -----
    if isinstance(elem, Mapping):
        # Use keys from the first element.
        return {k: default_collate([d[k] for d in batch]) for k in elem}

    if _is_namedtuple_instance(elem):
        # Collate positional fields.
        return type(elem)(*(default_collate([d[i] for d in batch]) for i in range(len(elem))))

    if isinstance(elem, tuple):
        # PyTorch requires sequences in the batch to have the same length.
        it = iter(batch)
        elem_size = len(elem)
        for s in it:
            if not isinstance(s, tuple) or len(s) != elem_size:
                raise TypeError("default_collate: inconsistent tuple sizes in batch")
        return tuple(default_collate([d[i] for d in batch]) for i in range(elem_size))

    if isinstance(elem, list):
        elem_size = len(elem)
        for s in batch:
            if not isinstance(s, list) or len(s) != elem_size:
                raise TypeError("default_collate: inconsistent list sizes in batch")
        return [default_collate([d[i] for d in batch]) for i in range(elem_size)]

    raise TypeError(
        f"default_collate: unsupported type {type(elem).__name__} in batch; provide a custom collate_fn"
    )
