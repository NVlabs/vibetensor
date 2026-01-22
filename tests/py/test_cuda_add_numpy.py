# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


def _cuda_available() -> bool:
    return getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0


def _require_torch_with_cuda():
    try:
        import torch  # noqa: F401
        return bool(getattr(__import__("torch").cuda, "is_available")())
    except Exception:
        return False


@pytest.mark.filterwarnings("ignore:.*")
def test_cuda_add_matches_numpy_across_devices():
    if not _cuda_available():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    if not _require_torch_with_cuda():
        pytest.skip("PyTorch+CUDA not available; required for DLPack bridge in this test", allow_module_level=False)

    import torch

    rng = np.random.default_rng(0)
    # Use a moderately sized, non-power-of-two shape
    shape = (257, 513)
    a_npy = rng.standard_normal(size=shape, dtype=np.float32)
    b_npy = rng.standard_normal(size=shape, dtype=np.float32)
    c_npy = a_npy + b_npy

    ndev = min(int(C._cuda_device_count()), int(torch.cuda.device_count()))
    assert ndev >= 1

    for dev in range(ndev):
        # Set VibeTensor current device via a CUDA stream context
        s = vt.cuda.Stream(device=dev)  # type: ignore[attr-defined]
        with s:
            # Baseline with PyTorch on this device
            a_t = torch.from_numpy(a_npy).to(device=f"cuda:{dev}")
            b_t = torch.from_numpy(b_npy).to(device=f"cuda:{dev}")
            c_t = a_t + b_t
            np.testing.assert_allclose(c_npy, c_t.cpu().numpy(), rtol=1e-5, atol=1e-6)

            # Import tensors into VibeTensor via DLPack (CUDA copy path)
            a_vt = vt.from_dlpack(a_t)
            b_vt = vt.from_dlpack(b_t)

            # Run VibeTensor CUDA add (dispatcher-backed)
            c_vt = C.vt.add(a_vt, b_vt)

            # Ensure device work completed before exposing a raw capsule
            torch.cuda.synchronize(dev)

            # Export result via DLPack and validate numerics against NumPy
            cap = vt.to_dlpack(c_vt)
            c_from_vt_t = torch.utils.dlpack.from_dlpack(cap)
            np.testing.assert_allclose(c_npy, c_from_vt_t.cpu().numpy(), rtol=1e-5, atol=1e-6)
