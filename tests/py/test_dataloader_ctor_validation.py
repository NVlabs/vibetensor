# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


def _has_cuda_device() -> bool:
    from vibetensor import _C as C

    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def test_dataloader_ctor_validation_stable_substrings(monkeypatch: pytest.MonkeyPatch):
    import vibetensor.torch.utils.data as vtd
    from vibetensor import _C as C

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            return int(idx)

    ds = _DS()

    # ---- prefetch_to_device=False cases ----
    cases = [
        (
            dict(prefetch_to_device=False, device=0),
            ValueError,
            "DataLoader: device requires prefetch_to_device=True",
        ),
        (
            dict(prefetch_to_device=False, non_blocking=True),
            ValueError,
            "DataLoader: non_blocking requires prefetch_to_device=True",
        ),
    ]

    for kwargs, exc_t, msg in cases:
        with pytest.raises(exc_t) as exc:
            vtd.DataLoader(ds, **kwargs)
        assert msg in str(exc.value)

    # ---- prefetch_to_device=True cases ----

    # C3: explicit CPU device is rejected even if CUDA is unavailable.
    with monkeypatch.context() as m:
        m.setattr(C, "_cuda_device_count", lambda: 0, raising=False)

        with pytest.raises(ValueError) as exc_cpu:
            vtd.DataLoader(ds, prefetch_to_device=True, device="cpu")
        assert "DataLoader: device must be a CUDA device when prefetch_to_device=True" in str(exc_cpu.value)

        # C4: CUDA unavailable (after CPU-device check).
        with pytest.raises(ValueError) as exc_no_cuda:
            vtd.DataLoader(ds, prefetch_to_device=True)
        assert "CUDA is not available" in str(exc_no_cuda.value)

    # C5/C6: device parse errors require a real CUDA device.
    if not _has_cuda_device():
        pytest.skip("CUDA device required for device-parse validation")

    parse_cases = [
        (
            dict(prefetch_to_device=True, device="cuda:bad"),
            ValueError,
            "invalid cuda device string",
        ),
        (
            dict(prefetch_to_device=True, device=9999),
            ValueError,
            "device index out of range",
        ),
    ]

    for kwargs, exc_t, msg in parse_cases:
        with pytest.raises(exc_t) as exc:
            vtd.DataLoader(ds, **kwargs)
        assert msg in str(exc.value)
