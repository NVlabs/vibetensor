# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C
import vibetensor.torch.cuda as cuda


def _has_cuda_device() -> bool:
    has_cuda = getattr(C, "_has_cuda", False)
    get_count = getattr(C, "_cuda_device_count", None)
    if not has_cuda or get_count is None:
        return False
    try:
        return int(get_count()) > 0
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda_device(), reason="CUDA device required for fraction env/API test")
def test_fraction_roundtrip_with_cuda_device() -> None:
    cuda.set_per_process_memory_fraction(0.5)
    got = cuda.get_per_process_memory_fraction()
    assert isinstance(got, float)
    assert pytest.approx(0.5) == got


def test_fraction_cpu_only_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force a zero-device view for this test.
    monkeypatch.setattr(C, "_cuda_device_count", lambda: 0, raising=False)

    # Setter should be a validated no-op and getter should return 1.0.
    cuda.set_per_process_memory_fraction(0.5)
    assert cuda.get_per_process_memory_fraction() == 1.0


def test_fraction_missing_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate builds where the CUDA fraction bindings are absent.
    monkeypatch.setattr(C, "_cuda_setMemoryFraction", None, raising=False)
    monkeypatch.setattr(C, "_cuda_getMemoryFraction", None, raising=False)

    cuda.set_per_process_memory_fraction(0.5)
    assert cuda.get_per_process_memory_fraction() == 1.0
