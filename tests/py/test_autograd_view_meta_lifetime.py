# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import gc
import subprocess

import pytest


from vibetensor import _C


# These tests are subprocess-based so that a segfault in the child process is
# surfaced as returncode != 0 (instead of killing the pytest runner).
if not hasattr(_C, "autograd"):
    pytest.skip("autograd disabled in this build", allow_module_level=True)


def run_py(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2)


def test_view_meta_deleted_base_no_segfault_non_grad():
    code = (
        "import gc\n"
        "from vibetensor import torch as vt\n"
        "from vibetensor import _C\n"
        "x = vt.tensor([1.0, 2.0, 3.0], dtype='float32')\n"
        "v = x[0:1]\n"
        "del x\n"
        "gc.collect()\n"
        "# Try to churn allocator state so a stale base pointer is more likely to crash\n"
        "for _ in range(512):\n"
        "    _t = vt.tensor([0.0], dtype='float32')\n"
        "del _t\n"
        "gc.collect()\n"
        "try:\n"
        "    _C.autograd._graph_get_gradient_edge(v)\n"
        "    print('UNEXPECTED_SUCCESS')\n"
        "except Exception as e:\n"
        "    print(str(e))\n"
    )

    res = run_py(code)
    assert res.returncode == 0
    out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    assert "UNEXPECTED_SUCCESS" not in out
    assert "does not require grad" in out


def test_view_meta_deleted_base_no_segfault_grad():
    code = (
        "import gc\n"
        "from vibetensor import torch as vt\n"
        "from vibetensor import _C\n"
        "x = vt.tensor([1.0, 2.0, 3.0], dtype='float32')\n"
        "x.requires_grad = True\n"
        "v = x[0:1]\n"
        "del x\n"
        "gc.collect()\n"
        "# Allocator churn\n"
        "for _ in range(512):\n"
        "    _t = vt.tensor([0.0], dtype='float32')\n"
        "del _t\n"
        "gc.collect()\n"
        "edge = _C.autograd._graph_get_gradient_edge(v)\n"
        "assert isinstance(edge, tuple) and len(edge) == 2\n"
        "print('OK')\n"
    )

    res = run_py(code)
    assert res.returncode == 0
    out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    assert out.endswith("OK")
