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


def _is_cpu_tensor(t: _C.Tensor) -> bool:
    dev = getattr(t, "device", None)
    try:
        return int(dev[0]) == 1
    except Exception:
        # Conservative: treat unknown device formats as non-CPU.
        return False


def prefetch_to_device_batch(
    data: Any,
    *,
    device_index: int,
    non_blocking: bool,
    hold_cuda_refs: list[object],
) -> Any:
    """Recursively convert batch leaves to CUDA tensors.

    Leaves:
      - NumPy arrays (C-contiguous): moved via `vibetensor.torch.cuda.to_device`.
      - VibeTensor CPU tensors: converted via `.numpy()` then moved to CUDA.

    When `non_blocking=True`, each created CUDA tensor is appended to
    `hold_cuda_refs` so the caller can synchronize on failures before
    partially-created tensors become unreachable.
    """

    if _np is not None and isinstance(data, _np.ndarray):
        from vibetensor.torch import cuda as _vcuda

        out = _vcuda.to_device(data, device=int(device_index), non_blocking=bool(non_blocking))
        if non_blocking:
            hold_cuda_refs.append(out)
        return out

    if isinstance(data, _C.Tensor):
        if not _is_cpu_tensor(data):
            raise TypeError("DataLoader: prefetch_to_device expects CPU tensors")

        from vibetensor.torch import cuda as _vcuda

        out = _vcuda.to_device(data.numpy(), device=int(device_index), non_blocking=bool(non_blocking))
        if non_blocking:
            hold_cuda_refs.append(out)
        return out

    if isinstance(data, Mapping):
        return {
            k: prefetch_to_device_batch(
                v,
                device_index=device_index,
                non_blocking=non_blocking,
                hold_cuda_refs=hold_cuda_refs,
            )
            for k, v in data.items()
        }

    if _is_namedtuple_instance(data):
        return type(data)(
            *(
                prefetch_to_device_batch(
                    v,
                    device_index=device_index,
                    non_blocking=non_blocking,
                    hold_cuda_refs=hold_cuda_refs,
                )
                for v in data
            )
        )

    if isinstance(data, tuple):
        return tuple(
            prefetch_to_device_batch(
                v,
                device_index=device_index,
                non_blocking=non_blocking,
                hold_cuda_refs=hold_cuda_refs,
            )
            for v in data
        )

    if isinstance(data, list):
        return [
            prefetch_to_device_batch(
                v,
                device_index=device_index,
                non_blocking=non_blocking,
                hold_cuda_refs=hold_cuda_refs,
            )
            for v in data
        ]

    return data
