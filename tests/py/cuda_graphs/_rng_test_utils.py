# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Any, Optional

from vibetensor import _C as _C
import vibetensor.torch as vt


def _has_cuda() -> bool:
    """Return True iff CUDA is built and at least one device is present.

    This mirrors the helper used in other CUDA graphs tests but lives in a
    shared module to avoid copy/paste.
    """

    try:
        return bool(getattr(_C, "_has_cuda", False) and int(getattr(_C, "_cuda_device_count", lambda: 0)()) > 0)  # type: ignore[attr-defined]
    except Exception:
        return False


def _require_cuda_or_skip(reason: str = "CUDA required for RNG-under-graphs tests") -> None:
    if not _has_cuda():
        pytest.skip(reason, allow_module_level=False)


def _require_multi_gpu_or_skip() -> None:
    _require_cuda_or_skip("CUDA with >=2 devices required for multi-GPU RNG-under-graphs tests")
    try:
        n = int(getattr(_C, "_cuda_device_count", lambda: 0)())  # type: ignore[attr-defined]
    except Exception:
        n = 0
    if n < 2:
        pytest.skip("at least 2 CUDA devices required for this test", allow_module_level=False)


def seed_all_cuda(seed: int) -> None:
    """Best-effort helper to seed all CUDA devices to the same seed.

    On CPU-only builds this is a no-op; tests should guard with
    ``_require_cuda_or_skip()`` when they rely on CUDA.
    """

    if not _has_cuda():
        return
    # ``vt.cuda`` is always importable when _has_cuda is True; when CUDA is
    # unavailable it may be ``None`` but we already guarded above.
    cuda_mod = getattr(vt, "cuda", None)
    if cuda_mod is None:  # pragma: no cover - defensive
        return
    cuda_mod.manual_seed_all(int(seed))


def _canonical_vt_device_str(x: Any) -> Optional[str]:
    """Return a canonical device string for a VibeTensor tensor.

    This helper hides the internal DLPack-style (type_code, index) tuple
    used by VibeTensor tensors and returns a stable string like
    "cpu", "cpu:0", "cuda", or "cuda:0" for use in tests.

    Returns None if ``x`` does not have a recognizable device attribute.
    """

    dev = getattr(x, "device", None)
    if dev is None:
        return None

    # If the device is already a string, trust it.
    if isinstance(dev, str):
        return dev

    # DLPack-style (type_code, index) tuple or list.
    if isinstance(dev, (tuple, list)) and len(dev) >= 2:
        try:
            type_code = int(dev[0])
            index = int(dev[1])
        except Exception:
            return None

        # 1 = kDLCPU, 2 = kDLCUDA per DLPack.
        if type_code == 1:
            return "cpu" if index == 0 else f"cpu:{index}"
        if type_code == 2:
            return f"cuda:{index}"

    # Unknown form; let the test handle this gracefully.
    return None
