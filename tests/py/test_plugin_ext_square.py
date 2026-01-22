# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import numpy as np
import pytest

import vibetensor.torch as vt


def _find_so(name: str) -> str | None:
    so_name = f"lib{name}.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    env = os.environ.get(f"VBT_{name.upper()}_PATH")
    if env:
        return env
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def test_ext_square_cpu_and_cuda_missing():
    path = _find_so("vbt_ext_square")
    if path is None:
        pytest.skip("ext_square plugin not found; ensure build produced libvbt_ext_square.so")
    try:
        vt.ops.load_library(path)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise

    a = np.arange(8, dtype=np.float32)
    ta = vt.from_dlpack(a)
    out = vt.ops.ext.square(ta)
    # Pass provider directly for NumPy from_dlpack protocol
    out_np = np.from_dlpack(out)
    np.testing.assert_allclose(out_np, a * a, rtol=1e-6, atol=1e-7)

    # CUDA path should error due to missing CUDA kernel
    from vibetensor import _C as C
    if getattr(C, "_has_cuda", False) and C._cuda_device_count() > 0:
        tad = vt.cuda.to_device(a, device=0)  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError) as ei:
            _ = vt.ops.ext.square(tad)
        assert "no CUDA kernel registered: ext::square" in str(ei.value)
