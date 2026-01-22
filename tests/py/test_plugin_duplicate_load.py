# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest

import vibetensor.torch as vt


def _find_so(name: str) -> str | None:
    so_name = f"lib{name}.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parents[2] / "build-py" / so_name,
    ]
    env = os.environ.get(f"VBT_{name.upper()}_PATH")
    if env:
        return env
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def test_duplicate_load_rejected():
    src = _find_so("vbt_reference_add")
    if src is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")
    # If already loaded, the first call should raise; otherwise, load once then expect duplicate to raise
    already = False
    try:
        from vibetensor import _C as C
        already = bool(C._is_library_loaded(src))
    except Exception:
        already = src in vt.ops.loaded_libraries
    if already:
        with pytest.raises(ValueError):
            vt.ops.load_library(src)
    else:
        vt.ops.load_library(src)
        with pytest.raises(ValueError) as ei:
            vt.ops.load_library(src)
        assert "plugin already loaded: " in str(ei.value)
