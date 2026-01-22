# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only Python startup shim.

Problem:
- This repo runs under a Conda Python binary that has DT_RPATH=$ORIGIN/../lib.
- Importing NumPy / PyTorch can therefore load Conda's libstdc++.so.6 first.
- VibeTensor's native extension may require a newer libstdc++ (GLIBCXX_3.4.30).
  If the older Conda libstdc++ is already loaded, importing vibetensor._C fails.

Solution:
- Preload the toolchain/system libstdc++ (and libgcc_s) with RTLD_GLOBAL very
  early in the process.

Why this file lives under tests/:
- We only need this behavior for test subprocesses spawned by the suite.
- `tests/py/conftest.py` injects this directory into PYTHONPATH so subprocesses
  will import this module automatically via Python's `sitecustomize` hook.

This code is best-effort and intentionally silent on failure.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys


def _looks_like_conda_python() -> bool:
    exe = sys.executable.lower()
    return (
        "conda" in exe
        or "miniconda" in exe
        or bool(os.environ.get("CONDA_PREFIX"))
    )


def _gcc_print_file(name: str) -> str | None:
    try:
        out = subprocess.check_output(["g++", f"-print-file-name={name}"], text=True).strip()
    except Exception:
        return None

    if not out or out == name:
        return None
    return out


def _preload(path: str) -> None:
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


if _looks_like_conda_python():
    p = _gcc_print_file("libstdc++.so.6")
    if p and os.path.exists(p):
        _preload(p)
    else:
        _preload("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")

    p = _gcc_print_file("libgcc_s.so.1")
    if p and os.path.exists(p):
        _preload(p)
    else:
        _preload("/lib/x86_64-linux-gnu/libgcc_s.so.1")
