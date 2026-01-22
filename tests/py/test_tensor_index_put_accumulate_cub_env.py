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
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2)


def test_cuda_index_put_accumulate_cub_env_smoke():
    code = (
        "import numpy as np\n"
        "import vibetensor.torch as vt\n"
        "from vibetensor import _C as C\n"
        "\n"
        "def _to_numpy_cpu(t):\n"
        "    cap = vt.to_dlpack(t)\n"
        "    try:\n"
        "        arr = np.from_dlpack(cap)  # type: ignore[arg-type]\n"
        "    except AttributeError:\n"
        "        class _CapsuleWrapper:\n"
        "            def __init__(self, inner):\n"
        "                self._inner = inner\n"
        "            def __dlpack__(self):\n"
        "                return self._inner\n"
        "        arr = np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]\n"
        "    return arr.reshape(tuple(int(s) for s in t.sizes))\n"
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
        "# --- Float32 accumulate ---\n"
        "x_cpu = vt.zeros((5,), dtype='float32')\n"
        "idx_cpu = vt.tensor([0, 1, 0, 4], dtype='int64')\n"
        "v_cpu = vt.tensor([1.0, 2.0, 3.0, 4.0], dtype='float32')\n"
        "x_cpu.index_put_((idx_cpu,), v_cpu, accumulate=True)\n"
        "x_cpu_np = _to_numpy_cpu(x_cpu)\n"
        "\n"
        "x0_np = np.zeros_like(x_cpu_np, dtype=np.float32)\n"
        "idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)\n"
        "v_np = _to_numpy_cpu(v_cpu)\n"
        "x_cuda = vt.cuda.to_device(x0_np)\n"
        "idx_cuda = vt.cuda.to_device(idx_np)\n"
        "v_cuda = vt.cuda.to_device(v_np)\n"
        "x_cuda.index_put_((idx_cuda,), v_cuda, accumulate=True)\n"
        "x_cuda_np = vt.cuda.from_device(x_cuda)\n"
        "ok = ok and np.allclose(x_cuda_np, x_cpu_np, rtol=1e-5, atol=1e-5)\n"
        "\n"
        "# --- Int64 accumulate ---\n"
        "x_cpu2 = vt.zeros((5,), dtype='int64')\n"
        "idx_cpu2 = vt.tensor([0, 1, 0, 4], dtype='int64')\n"
        "v_cpu2 = vt.tensor([1, 2, 3, 4], dtype='int64')\n"
        "x_cpu2.index_put_((idx_cpu2,), v_cpu2, accumulate=True)\n"
        "x_cpu2_np = _to_numpy_cpu(x_cpu2)\n"
        "\n"
        "x0_np2 = np.zeros_like(x_cpu2_np, dtype=np.int64)\n"
        "idx_np2 = _to_numpy_cpu(idx_cpu2).astype(np.int64)\n"
        "v_np2 = _to_numpy_cpu(v_cpu2).astype(np.int64)\n"
        "x_cuda2 = vt.cuda.to_device(x0_np2)\n"
        "idx_cuda2 = vt.cuda.to_device(idx_np2)\n"
        "v_cuda2 = vt.cuda.to_device(v_np2)\n"
        "x_cuda2.index_put_((idx_cuda2,), v_cuda2, accumulate=True)\n"
        "x_cuda2_np = vt.cuda.from_device(x_cuda2)\n"
        "ok = ok and np.array_equal(x_cuda2_np, x_cpu2_np)\n"
        "\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        # Ensure advanced indexing is enabled and opt into the CUB accumulate prototype.
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_CUDA_CUB_INDEX_PUT_ACCUMULATE": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
