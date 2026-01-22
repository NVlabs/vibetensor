# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pathlib

import pytest


def test_ops_load_library_smoke():
    import vibetensor.torch as vt
    # Find the reference plugin .so built by CMake
    # Expect it to be named 'libvbt_reference_add.so' on Linux.
    so_name = "libvbt_reference_add.so"
    # Search common paths: current working directory and parent dirs of tests
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    path = None
    for p in candidates:
        if p.exists():
            path = str(p)
            break
    # As a fallback, allow environment to pass path
    path = os.environ.get("VBT_REF_PLUGIN_PATH", path)
    if path is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")
    try:
        vt.ops.load_library(path)
    except ValueError as e:
        # Accept idempotent call when already loaded in this session
        if "plugin already loaded:" not in str(e):
            raise
