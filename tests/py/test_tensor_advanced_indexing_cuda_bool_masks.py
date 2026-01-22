# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from typing import Dict, Optional

import numpy as np


def _run_py(code: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2)


def test_cuda_bool_mask_enabled_matches_cpu_when_flag_set():
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
        "# CPU baseline: float32 data with 1D bool mask on last dim.\n"
        "x_cpu = vt.arange(6, dtype='float32').reshape((2, 3))\n"
        "mask_cpu = vt.tensor([True, False, True], dtype='bool')\n"
        "y_cpu = x_cpu[:, mask_cpu]\n"
        "\n"
        "x_np = _to_numpy_cpu(x_cpu)\n"
        "mask_np = _to_numpy_cpu(mask_cpu).astype(bool)\n"
        "\n"
        "x_cuda = vt.cuda.to_device(x_np)\n"
        "mask_cuda = vt.cuda.to_device(mask_np)\n"
        "\n"
        "# With the bool-mask flag enabled, CUDA indexing should match CPU.\n"
        "y_cuda = x_cuda[:, mask_cuda]\n"
        "y_cuda_np = vt.cuda.from_device(y_cuda)\n"
        "y_cpu_np = _to_numpy_cpu(y_cpu)\n"
        "\n"
        "ok = y_cuda_np.shape == y_cpu_np.shape and np.allclose(y_cuda_np, y_cpu_np)\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        # Ensure advanced indexing is enabled and bool masks are opted in.
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_cuda_bool_mask_autograd_scatter_adds_into_base_when_v2_enabled():
    code = (
        "import numpy as np\n"
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
        "prev_v2 = C._autograd_indexing_v2_enabled()\n"
        "C._set_autograd_indexing_v2_enabled_for_tests(True)\n"
        "\n"
        "ag = C.autograd\n"
        "prev_grad = bool(ag.is_grad_enabled())\n"
        "prev_cuda = bool(ag.is_cuda_autograd_enabled())\n"
        "try:\n"
        "    ag.set_grad_enabled(True)\n"
        "    ag.set_cuda_autograd_enabled(True)\n"
        "\n"
        "    x_np = np.arange(6, dtype=np.float32).reshape((2, 3))\n"
        "    mask_np = np.array([True, False, True], dtype=bool)\n"
        "\n"
        "    x = vt.cuda.to_device(x_np)\n"
        "    x.set_requires_grad(True)\n"
        "    mask = vt.cuda.to_device(mask_np)\n"
        "\n"
        "    y = x[:, mask]\n"
        "    ones = vt.cuda.to_device(np.ones((2, 2), dtype=np.float32))\n"
        "    y.backward(ones)\n"
        "\n"
        "    g = vt.cuda.from_device(x.grad())\n"
        "    expected = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)\n"
        "    ok = g.shape == expected.shape and np.allclose(g, expected)\n"
        "    import sys as _sys\n"
        "    _sys.exit(0 if ok else 1)\n"
        "finally:\n"
        "    ag.set_cuda_autograd_enabled(prev_cuda)\n"
        "    ag.set_grad_enabled(prev_grad)\n"
        "    C._set_autograd_indexing_v2_enabled_for_tests(prev_v2)\n"
    )

    env = {
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK_CUB": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_cuda_bool_mask_cub_backend_matches_cpu_when_flag_set():
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
        "# CPU baseline: float32 data with 1D bool mask on last dim.\n"
        "x_cpu = vt.arange(6, dtype='float32').reshape((2, 3))\n"
        "mask_cpu = vt.tensor([True, False, True], dtype='bool')\n"
        "y_cpu = x_cpu[:, mask_cpu]\n"
        "\n"
        "x_np = _to_numpy_cpu(x_cpu)\n"
        "mask_np = _to_numpy_cpu(mask_cpu).astype(bool)\n"
        "\n"
        "x_cuda = vt.cuda.to_device(x_np)\n"
        "mask_cuda = vt.cuda.to_device(mask_np)\n"
        "\n"
        "# With the bool-mask + CUB flags enabled, CUDA indexing should match CPU.\n"
        "y_cuda = x_cuda[:, mask_cuda]\n"
        "y_cuda_np = vt.cuda.from_device(y_cuda)\n"
        "y_cpu_np = _to_numpy_cpu(y_cpu)\n"
        "\n"
        "ok = y_cuda_np.shape == y_cpu_np.shape and np.allclose(y_cuda_np, y_cpu_np)\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK_CUB": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_cuda_bool_mask_noncontig_masks_match_cpu_when_flag_set():
    code = (
        "import numpy as np\n"
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
        "x_np = np.arange(6, dtype=np.float32).reshape((2, 3))\n"
        "x_cuda = vt.cuda.to_device(x_np)\n"
        "\n"
        "ok = True\n"
        "\n"
        "# Case 1: stride-2 view (non-contiguous)\n"
        "mask_base_np = np.array([True, False, False, True, True, False], dtype=bool)\n"
        "mask_base_cuda = vt.cuda.to_device(mask_base_np)\n"
        "mask_cuda = mask_base_cuda.as_strided((3,), (2,), 0)\n"
        "expected_mask = np.array([True, False, True], dtype=bool)\n"
        "y_cuda = x_cuda[:, mask_cuda]\n"
        "y_cuda_np = vt.cuda.from_device(y_cuda)\n"
        "expected = x_np[:, expected_mask]\n"
        "ok = ok and (y_cuda_np.shape == expected.shape) and np.allclose(y_cuda_np, expected)\n"
        "\n"
        "# Case 2: negative stride view (reverse)\n"
        "mask_base_np2 = np.array([True, False, False], dtype=bool)\n"
        "mask_base_cuda2 = vt.cuda.to_device(mask_base_np2)\n"
        "mask_cuda2 = mask_base_cuda2.as_strided((3,), (-1,), 2)\n"
        "expected_mask2 = mask_base_np2[::-1]\n"
        "y_cuda2 = x_cuda[:, mask_cuda2]\n"
        "y_cuda2_np = vt.cuda.from_device(y_cuda2)\n"
        "expected2 = x_np[:, expected_mask2]\n"
        "ok = ok and (y_cuda2_np.shape == expected2.shape) and np.allclose(y_cuda2_np, expected2)\n"
        "\n"
        "# Case 3: stride-0 view (broadcast-like)\n"
        "mask_base_np3 = np.array([True, False, False], dtype=bool)\n"
        "mask_base_cuda3 = vt.cuda.to_device(mask_base_np3)\n"
        "mask_cuda3 = mask_base_cuda3.as_strided((3,), (0,), 0)\n"
        "expected_mask3 = np.array([True, True, True], dtype=bool)\n"
        "y_cuda3 = x_cuda[:, mask_cuda3]\n"
        "y_cuda3_np = vt.cuda.from_device(y_cuda3)\n"
        "expected3 = x_np[:, expected_mask3]\n"
        "ok = ok and (y_cuda3_np.shape == expected3.shape) and np.allclose(y_cuda3_np, expected3)\n"
        "\n"
        "# Case 4: positive stride with non-zero offset\n"
        "mask_base_np4 = np.array([False, True, False, False, False, True], dtype=bool)\n"
        "mask_base_cuda4 = vt.cuda.to_device(mask_base_np4)\n"
        "mask_cuda4 = mask_base_cuda4.as_strided((3,), (2,), 1)\n"
        "expected_mask4 = np.array([True, False, True], dtype=bool)\n"
        "y_cuda4 = x_cuda[:, mask_cuda4]\n"
        "y_cuda4_np = vt.cuda.from_device(y_cuda4)\n"
        "expected4 = x_np[:, expected_mask4]\n"
        "ok = ok and (y_cuda4_np.shape == expected4.shape) and np.allclose(y_cuda4_np, expected4)\n"
        "\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_cuda_bool_mask_cub_backend_noncontig_masks_match_cpu_when_flag_set():
    code = (
        "import numpy as np\n"
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
        "x_np = np.arange(6, dtype=np.float32).reshape((2, 3))\n"
        "x_cuda = vt.cuda.to_device(x_np)\n"
        "\n"
        "ok = True\n"
        "\n"
        "# Case 1: stride-2 view (non-contiguous)\n"
        "mask_base_np = np.array([True, False, False, True, True, False], dtype=bool)\n"
        "mask_base_cuda = vt.cuda.to_device(mask_base_np)\n"
        "mask_cuda = mask_base_cuda.as_strided((3,), (2,), 0)\n"
        "expected_mask = np.array([True, False, True], dtype=bool)\n"
        "y_cuda = x_cuda[:, mask_cuda]\n"
        "y_cuda_np = vt.cuda.from_device(y_cuda)\n"
        "expected = x_np[:, expected_mask]\n"
        "ok = ok and (y_cuda_np.shape == expected.shape) and np.allclose(y_cuda_np, expected)\n"
        "\n"
        "# Case 2: negative stride view (reverse)\n"
        "mask_base_np2 = np.array([True, False, False], dtype=bool)\n"
        "mask_base_cuda2 = vt.cuda.to_device(mask_base_np2)\n"
        "mask_cuda2 = mask_base_cuda2.as_strided((3,), (-1,), 2)\n"
        "expected_mask2 = mask_base_np2[::-1]\n"
        "y_cuda2 = x_cuda[:, mask_cuda2]\n"
        "y_cuda2_np = vt.cuda.from_device(y_cuda2)\n"
        "expected2 = x_np[:, expected_mask2]\n"
        "ok = ok and (y_cuda2_np.shape == expected2.shape) and np.allclose(y_cuda2_np, expected2)\n"
        "\n"
        "# Case 3: stride-0 view (broadcast-like)\n"
        "mask_base_np3 = np.array([True, False, False], dtype=bool)\n"
        "mask_base_cuda3 = vt.cuda.to_device(mask_base_np3)\n"
        "mask_cuda3 = mask_base_cuda3.as_strided((3,), (0,), 0)\n"
        "expected_mask3 = np.array([True, True, True], dtype=bool)\n"
        "y_cuda3 = x_cuda[:, mask_cuda3]\n"
        "y_cuda3_np = vt.cuda.from_device(y_cuda3)\n"
        "expected3 = x_np[:, expected_mask3]\n"
        "ok = ok and (y_cuda3_np.shape == expected3.shape) and np.allclose(y_cuda3_np, expected3)\n"
        "\n"
        "# Case 4: positive stride with non-zero offset\n"
        "mask_base_np4 = np.array([False, True, False, False, False, True], dtype=bool)\n"
        "mask_base_cuda4 = vt.cuda.to_device(mask_base_np4)\n"
        "mask_cuda4 = mask_base_cuda4.as_strided((3,), (2,), 1)\n"
        "expected_mask4 = np.array([True, False, True], dtype=bool)\n"
        "y_cuda4 = x_cuda[:, mask_cuda4]\n"
        "y_cuda4_np = vt.cuda.from_device(y_cuda4)\n"
        "expected4 = x_np[:, expected_mask4]\n"
        "ok = ok and (y_cuda4_np.shape == expected4.shape) and np.allclose(y_cuda4_np, expected4)\n"
        "\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK_CUB": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"



def test_cuda_extended_float16_dtype_matches_cpu_when_flag_set():
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
        "# CPU baseline: float16 data with int32 indices.\n"
        "x_np = np.arange(6, dtype=np.float16).reshape((2, 3))\n"
        "idx_np = np.array([0, 2], dtype=np.int32)\n"
        "\n"
        "x_cpu = vt.from_numpy(x_np)\n"
        "idx_cpu = vt.from_numpy(idx_np)\n"
        "y_cpu = x_cpu[:, idx_cpu]\n"
        "y_cpu_np = _to_numpy_cpu(y_cpu)\n"
        "\n"
        "# CUDA path under the extended-dtype flag.\n"
        "x_cuda = vt.cuda.to_device(x_np)\n"
        "idx_cuda = vt.cuda.to_device(idx_np)\n"
        "y_cuda = x_cuda[:, idx_cuda]\n"
        "y_cuda_np = vt.cuda.from_device(y_cuda)\n"
        "\n"
        "ok = y_cuda_np.shape == y_cpu_np.shape and np.allclose(y_cuda_np, y_cpu_np.astype(y_cuda_np.dtype))\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_EXTENDED_DTYPE": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_cuda_bool_mask_policy_error_when_flag_disabled() -> None:
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
        "x_cpu = vt.arange(6, dtype='float32').reshape((2, 3))\n"
        "mask_cpu = vt.tensor([True, False, True], dtype='bool')\n"
        "x_np = _to_numpy_cpu(x_cpu)\n"
        "mask_np = _to_numpy_cpu(mask_cpu).astype(bool)\n"
        "x_cuda = vt.cuda.to_device(x_np)\n"
        "mask_cuda = vt.cuda.to_device(mask_np)\n"
        "\n"
        "ok = False\n"
        "try:\n"
        "    _ = x_cuda[:, mask_cuda]\n"
        "except Exception as e:\n"
        "    msg = str(e)\n"
        "    ok = 'CUDA advanced indexing does not support boolean mask indices' in msg\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        # AE enabled but Bool-mask flag left unset/false.
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_cuda_bool_mask_unsupported_for_float16_even_with_extended_dtype_flag() -> None:
    code = (
        "import numpy as np\n"
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
        "# Float16 data with Bool mask; Bool masks on CUDA\n"
        "# must remain restricted to float32/int64 data even when\n"
        "# the extended-dtype flag is enabled.\n"
        "x_np = np.arange(6, dtype=np.float16).reshape((2, 3))\n"
        "mask_np = np.array([True, False, True], dtype=bool)\n"
        "x_cuda = vt.cuda.to_device(x_np)\n"
        "mask_cuda = vt.cuda.to_device(mask_np)\n"
        "\n"
        "ok = False\n"
        "try:\n"
        "    _ = x_cuda[:, mask_cuda]\n"
        "except Exception as e:\n"
        "    msg = str(e)\n"
        "    ok = 'CUDA advanced indexing does not support boolean mask indices' in msg\n"
        "import sys as _sys\n"
        "_sys.exit(0 if ok else 1)\n"
    )

    env = {
        # AE, Bool-mask flag, and extended-dtype flag all enabled.
        "VBT_ENABLE_ADVANCED_INDEXING": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK": "1",
        "VBT_INTERNAL_ADV_INDEX_CUDA_EXTENDED_DTYPE": "1",
    }

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
