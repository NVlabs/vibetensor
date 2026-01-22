# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from typing import Dict, Optional


def _run_py(code: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env2,
            timeout=120,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"subprocess timed out: stdout={e.stdout!r} stderr={e.stderr!r}"
        ) from e


def test_cuda_cub_reduce_all_sum_env_smoke():
    code = (
        "import sys\n"
        "import vibetensor.torch as vt\n"
        "from vibetensor import _C as C\n"
        "\n"
        "def _has_cuda() -> bool:\n"
        "    try:\n"
        "        return bool(getattr(C, '_has_cuda', False)) and int(C._cuda_device_count()) > 0\n"
        "    except Exception:\n"
        "        return False\n"
        "\n"
        "if not _has_cuda():\n"
        "    raise SystemExit(0)\n"
        "\n"
        "ok = True\n"
        "\n"
        "n = 1000\n"
        "expected_sum = (n - 1) * n // 2\n"
        "\n"
        "x = vt.arange(n, dtype='float32').cuda()\n"
        "s = x.sum()\n"
        "m = x.mean()\n"
        "ok = ok and (s.device[0] == 2) and abs(float(s.item()) - float(expected_sum)) < 1e-3\n"
        "ok = ok and (m.device[0] == 2) and abs(float(m.item()) - float(expected_sum) / float(n)) < 1e-3\n"
        "\n"
        "x2 = vt.arange(n, dtype='int64').cuda()\n"
        "s2 = x2.sum()\n"
        "ok = ok and (s2.device[0] == 2) and int(s2.item()) == int(expected_sum)\n"
        "\n"
        "sys.exit(0 if ok else 1)\n"
    )

    env = {
        # Enable the internal reduce-all sum fast path.
        "VBT_INTERNAL_CUDA_CUB_REDUCE_ALL_SUM": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
