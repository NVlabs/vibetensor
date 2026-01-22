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


def test_status_mapping_unsupported_ext_square_float64():
    src = _find_so("vbt_ext_square")
    if src is None:
        pytest.skip("ext_square plugin not found; ensure build produced libvbt_ext_square.so")
    try:
        vt.ops.load_library(src)
    except ValueError as e:
        if "plugin already loaded:" not in str(e):
            raise
    a = np.arange(8, dtype=np.int64)
    ta = vt.from_dlpack(a)
    with pytest.raises(RuntimeError) as ei:
        _ = vt.ops.ext.square(ta)
    assert str(ei.value) == "unsupported: ext::square"


def test_loader_bad_path_value_error_prefix():
    bad = "/definitely/not/exist/libnope.so"
    with pytest.raises(ValueError) as ei:
        vt.ops.load_library(bad)
    assert str(ei.value).startswith("loader: ")
