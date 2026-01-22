# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import numpy as np
import pytest

from vibetensor import _C as C
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


def test_rollback_restores_vt_add_after_failed_plugin_init():
    src = _find_so("fail_rollback")
    if src is None:
        pytest.skip("fail_rollback plugin not found; ensure build produced libfail_rollback.so")
    # Call vt.add on CPU to capture expected behavior
    a = np.full((4,), 1.0, dtype=np.float32)
    b = np.full((4,), 2.0, dtype=np.float32)
    expect = a + b
    # Attempt to load failing plugin; should raise ValueError
    with pytest.raises(ValueError):
        vt.ops.load_library(src)
    # vt.add should still work after failure
    ta = vt.from_dlpack(a)
    tb = vt.from_dlpack(b)
    out = C.vt.add(ta, tb)
    # Use provider protocol for NumPy
    np.testing.assert_allclose(np.from_dlpack(out), expect, rtol=1e-5, atol=1e-6)
