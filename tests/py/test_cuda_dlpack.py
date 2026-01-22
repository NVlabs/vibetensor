# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


def test_cuda_import_from_dlpack_and_one_shot():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    cap = C._make_cuda_dlpack_1d(16)
    t = vt.from_dlpack(cap)
    # Verify basic properties
    assert t.device == (2, 0) or t.device == (2, getattr(t, "device", (2, 0))[1])  # kDLCUDA
    assert t.sizes == (16,)
    # One-shot: reusing the capsule should fail in importer
    with pytest.raises(Exception):
        _ = vt.from_dlpack(cap)


def test_cuda_import_mixed_device_error_or_ok():
    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0:
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)
    cap = C._make_cuda_dlpack_1d(4)
    # If we have multiple devices, switch device and expect mixed-device error
    if C._cuda_device_count() >= 2:
        # Switch current device to 1 via creating a stream on device 1 and setting current
        s1 = vt.cuda.Stream(device=1)  # type: ignore[attr-defined]
        with s1:
            with pytest.raises(RuntimeError):
                _ = vt.from_dlpack(cap)
    else:
        # Otherwise it should import fine on current device
        _ = vt.from_dlpack(cap)
