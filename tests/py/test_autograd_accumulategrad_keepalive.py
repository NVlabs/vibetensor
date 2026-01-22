# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess

import pytest

from vibetensor import _C


# Subprocess-based so that a segfault in the child process is surfaced as
# returncode != 0 (instead of killing the pytest runner).
if not hasattr(_C, "autograd"):
    pytest.skip("autograd disabled in this build", allow_module_level=True)


def run_py(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env2,
    )


def test_accumulategrad_deleted_leaf_no_segfault():
    # If AccumulateGrad holds a raw AutogradMeta* and the leaf dies before
    # backward runs, we can hit a use-after-free. This is a subprocess test so
    # a crash becomes returncode != 0.
    code = (
        "import gc\n"
        "from vibetensor import torch as vt\n"
        "import vibetensor.autograd as autograd\n"
        "from vibetensor import _C\n"
        "\n"
        "class F(autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, x):\n"
        "        return _C.vt.mul(x, vt.ones_like(x))\n"
        "    @staticmethod\n"
        "    def backward(ctx, grad_output):\n"
        "        return (grad_output,)\n"
        "\n"
        "x = vt.arange(1024, dtype='float32').detach()\n"
        "x.requires_grad = True\n"
        "y = F.apply(x)\n"
        "g = vt.ones_like(y)\n"
        "del x\n"
        "gc.collect()\n"
        "# Try to churn allocator state so a stale meta pointer is more likely to crash\n"
        "for _ in range(2048):\n"
        "    _t = vt.tensor([0.0], dtype='float32')\n"
        "del _t\n"
        "gc.collect()\n"
        "y.backward(g)\n"
        "print('OK')\n"
    )

    res = run_py(code, env={"VBT_AUTOGRAD_ENABLE_FUNCTION": "1"})
    assert res.returncode == 0
    out = ((res.stdout or "") + "\n" + (res.stderr or "")).strip()
    assert out.endswith("OK")
