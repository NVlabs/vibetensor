# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

from vibetensor import _C

__all__ = ("Generator", "manual_seed", "seed", "initial_seed", "get_rng_state", "set_rng_state")

# Canonical guard error message from C++; must match
# vbt::rng::graph_capture::kErrCudaRngMutationDuringCapture.
_ERR_CUDA_RNG_MUTATION_DURING_CAPTURE: str = getattr(
    _C,
    "_ERR_CUDA_RNG_MUTATION_DURING_CAPTURE",
    "rng: generator state mutation is forbidden while CUDA Graph capture is active",
)


def _is_cuda_rng_guard_error(exc: BaseException) -> bool:
    """Return True iff ``exc`` is the CUDA RNG capture guard error.

    We classify guard errors by exact string equality against the canonical
    message exported from C++. This keeps Python and C++ in sync and avoids
    hardcoding the string in many tests.
    """

    return isinstance(exc, RuntimeError) and str(exc) == _ERR_CUDA_RNG_MUTATION_DURING_CAPTURE


def _parse_device(device: str | int | None) -> tuple[str, Optional[int]]:
    # Returns (dev_type, idx). dev_type in {"cpu","cuda"}; idx is None for cpu
    if device is None:
        return ("cpu", None)
    if isinstance(device, int):
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
        if not getattr(_C, "_has_cuda", False) or int(getattr(_C, "_cuda_device_count", lambda: 0)()) <= 0:
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


class Generator:
    """Device-tagged Generator handle that routes to default generators.

    device: 'cpu' or 'cuda:k'
    """

    def __init__(self, device: str | int | None = None):
        dev_type, dev_idx = _parse_device(device)
        self._device = "cpu" if dev_type == "cpu" else f"cuda:{dev_idx}"
        self._dev_type = dev_type
        self._dev_idx = dev_idx

    @property
    def device(self) -> str:
        return f"{self._device}"

    def manual_seed(self, seed: int) -> None:
        if self._dev_type == "cpu":
            _C._rng_manual_seed(int(seed))
        else:
            _C._cuda_rng_manual_seed(int(self._dev_idx), int(seed))  # type: ignore[attr-defined]

    def initial_seed(self) -> int:
        if self._dev_type == "cpu":
            return int(_C._rng_initial_seed())
        return int(_C._cuda_rng_initial_seed(int(self._dev_idx)))  # type: ignore[attr-defined]

    def get_state(self) -> bytes:
        if self._dev_type == "cpu":
            return _C._rng_get_state()
        return _C._cuda_rng_get_state(int(self._dev_idx))  # type: ignore[attr-defined]

    def set_state(self, state: bytes) -> None:
        if not isinstance(state, (bytes, bytearray)):
            raise TypeError("state must be a bytes object")
        if self._dev_type == "cpu":
            _C._rng_set_state(bytes(state))
        else:
            _C._cuda_rng_set_state(int(self._dev_idx), bytes(state))  # type: ignore[attr-defined]


def manual_seed(seed: int) -> int:
    """Set CPU and all CUDA default RNG seeds to `seed` and reset offsets to 0.

    Returns the seed. When CUDA is available, reseeds each device via
    ``_C._cuda_rng_manual_seed``. If a device is participating in an active
    CUDA Graph capture, the guarded binding raises and this function
    re-raises that error instead of silently skipping the device.
    """
    s = int(_C._rng_manual_seed(int(seed)))
    if getattr(_C, "_has_cuda", False):
        try:
            n = int(_C._cuda_device_count())  # type: ignore[attr-defined]
        except Exception:
            n = 0
        for k in range(max(0, n)):
            try:
                _C._cuda_rng_manual_seed(k, s)  # type: ignore[attr-defined]
            except Exception as exc:
                if _is_cuda_rng_guard_error(exc):
                    # Do not silently skip guard errors; surface them.
                    raise
                # Best-effort: skip devices that fail for other reasons.
                pass
    return s


def seed() -> int:
    """Reseed the CPU default RNG with a nondeterministic 64-bit seed. Returns the seed."""
    return int(_C._rng_seed())


def initial_seed() -> int:
    """Return the current seed of the CPU default RNG (does not include offset)."""
    return int(_C._rng_initial_seed())


def get_rng_state(device: str | int | None = None) -> bytes:
    """Return RNG state as 16 bytes little-endian [seed:u64][offset:u64] for the given device (default CPU)."""
    dev_type, dev_idx = _parse_device(device)
    if dev_type == "cpu":
        return _C._rng_get_state()
    return _C._cuda_rng_get_state(int(dev_idx))  # type: ignore[attr-defined]


def set_rng_state(state: bytes, device: str | int | None = None) -> None:
    """Set RNG state from 16 bytes [seed:u64][offset:u64] for the given device (default CPU)."""
    if not isinstance(state, (bytes, bytearray)):
        raise TypeError("state must be a bytes object")
    dev_type, dev_idx = _parse_device(device)
    if dev_type == "cpu":
        _C._rng_set_state(bytes(state))
    else:
        _C._cuda_rng_set_state(int(dev_idx), bytes(state))  # type: ignore[attr-defined]
