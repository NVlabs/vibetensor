# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vibetensor import _C as _C


def _is_namedtuple_instance(x: Any) -> bool:
    # Heuristic used by PyTorch: a tuple subclass with `_fields`.
    return isinstance(x, tuple) and hasattr(x, "_fields")


def _is_cpu_tensor(t: _C.Tensor) -> bool:
    dev = getattr(t, "device", None)
    try:
        return int(dev[0]) == 1
    except Exception:
        # Conservative: treat unknown device formats as non-CPU.
        return False


def pin_memory_batch(data: Any) -> Any:
    """Recursively replace CPU tensor leaves with pinned CPU tensors."""

    if isinstance(data, _C.Tensor):
        if not _is_cpu_tensor(data):
            raise TypeError("DataLoader: pin_memory expects CPU tensors")
        return _C._cpu_pin_memory(data)

    if isinstance(data, Mapping):
        return {k: pin_memory_batch(v) for k, v in data.items()}

    if _is_namedtuple_instance(data):
        return type(data)(*(pin_memory_batch(v) for v in data))

    if isinstance(data, tuple):
        return tuple(pin_memory_batch(v) for v in data)

    if isinstance(data, list):
        return [pin_memory_batch(v) for v in data]

    return data
