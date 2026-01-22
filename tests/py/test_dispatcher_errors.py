# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess


def run_py(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2)


def test_unknown_op_error_mapping_env_gated():
    code = (
        "from vibetensor import _C as _C\n"
        "try:\n"
        "    _C._call_op('__missing__::noop')\n"
        "except Exception as e:\n"
        "    print(str(e))\n"
    )
    # Default (no compat): original message
    res = run_py(code)
    out = (res.stdout or "").strip()
    assert out.startswith("unknown op: __missing__::noop")

    # With compat: mapped message
    res2 = run_py(code, env={"VBT_OPS_COMPAT": "1"})
    out2 = (res2.stdout or "").strip()
    assert out2 == "Didn't find operator '__missing__::noop'"


def test_no_kernel_error_mapping_cpu():
    code = (
        "from vibetensor import _C as _C\n"
        "c_def = getattr(_C, 'def')\n"
        "c_def('x::zero() -> Tensor')\n"
        "try:\n"
        "    _C._call_op('x::zero')\n"
        "except Exception as e:\n"
        "    print(str(e))\n"
    )
    res = run_py(code, env={"VBT_OPS_COMPAT": "1"})
    out = (res.stdout or "").strip()
    assert out == "No kernel found for dispatch key CPU for operator 'x::zero'"
