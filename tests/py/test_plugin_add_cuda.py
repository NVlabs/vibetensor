# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import numpy as np
import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


def _find_ref_plugin() -> str | None:
    so_name = "libvbt_reference_add.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    # Allow env override
    env = os.environ.get("VBT_REF_PLUGIN_PATH")
    if env:
        return env
    for p in candidates:
        if p.exists():
            return str(p)
    return None


@pytest.mark.skipif(not getattr(C, "_has_cuda", False) or C._cuda_device_count() == 0, reason="CUDA not available for VibeTensor")
def test_plugin_cuda_add_e2e():
    path = _find_ref_plugin()
    if path is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")

    # Load plugin
    try:
        vt.ops.load_library(path)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise

    # Build inputs on CUDA via NumPy copy helper
    N = 1024
    a = np.full((N,), 1.5, dtype=np.float32)
    b = np.full((N,), 2.5, dtype=np.float32)

    ad = vt.cuda.to_device(a, device=0)  # type: ignore[attr-defined]
    bd = vt.cuda.to_device(b, device=0)  # type: ignore[attr-defined]

    # Call vt::add through dispatcher, which should route to plugin CUDA kernel
    cd = C.vt.add(ad, bd)

    out = vt.cuda.from_device(cd)  # type: ignore[attr-defined]
    np.testing.assert_allclose(out, a + b, rtol=1e-5, atol=1e-6)
