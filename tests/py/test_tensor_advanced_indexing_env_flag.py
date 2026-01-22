# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
from typing import Dict, Optional

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


def _to_numpy_cpu(t):
    """Convert a vibetensor tensor (CPU) to a NumPy array via DLPack."""
    cap = vt.to_dlpack(t)
    # NumPy's from_dlpack historically accepted raw DLPack capsules
    # directly, but newer versions expect an object with a __dlpack__
    # method. Support both by wrapping the capsule when needed.
    try:
        arr = np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover - tiny adapter
                return self._inner

        arr = np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]
    return arr.reshape(tuple(int(s) for s in t.sizes))


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def _run_py(code: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2)




def test_env_disables_advanced_indexing_for_vt_and_tensor_index():
    code = (
        "import vibetensor.torch as vt\n"
        "from vibetensor import _C as C\n"
        "x = vt.arange(4, dtype='float32')\n"
        "idx = vt.tensor([0, 2], dtype='int64')\n"
        "ok = True\n"
        "try:\n"
        "    _ = x[idx]\n"
        "    ok = False\n"
        "except RuntimeError as e:\n"
        "    if 'advanced indexing disabled' not in str(e):\n"
        "        ok = False\n"
        "try:\n"
        "    _ = x.index(idx)\n"
        "    ok = False\n"
        "except RuntimeError as e:\n"
        "    if 'advanced indexing disabled' not in str(e):\n"
        "        ok = False\n"
        "# Basic indexing should remain enabled.\n"
        "_ = x[1]\n"
        "_ = x.index(1)\n"
        "import sys\n"
        "sys.exit(0 if ok else 1)\n"
    )

    res = _run_py(code, env={"VBT_ENABLE_ADVANCED_INDEXING": "0"})
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"


@pytest.mark.parametrize(
    "env_val",
    [
        None,  # unset -> default enabled
        "",  # empty -> enabled
        "1",
        "true",
        "True",
        " yes ",
        "garbage",
    ],
)
def test_env_truthy_values_keep_advanced_indexing_enabled(env_val):
    code = (
        "import vibetensor.torch as vt\n"
        "from vibetensor import _C as C\n"
        "x = vt.arange(4, dtype='float32')\n"
        "idx = vt.tensor([0, 2], dtype='int64')\n"
        "ok = True\n"
        "try:\n"
        "    _ = x[idx]\n"
        "    _ = x.index(idx)\n"
        "except Exception as e:\n"
        "    print('UNEXPECTED', type(e).__name__, str(e))\n"
        "    ok = False\n"
        "import sys\n"
        "sys.exit(0 if ok else 1)\n"
    )

    env: Dict[str, str] = {}
    if env_val is not None:
        env["VBT_ENABLE_ADVANCED_INDEXING"] = env_val

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"env_val={env_val!r} stdout={res.stdout!r} stderr={res.stderr!r}"


# --- _C.Tensor.index parity tests ---------------------------------------------


def _assert_index_value_parity(x, *indices):
    # Helper that compares x[indices] vs x.index(*indices) via NumPy.
    if len(indices) == 1:
        idx_obj = indices[0]
    else:
        idx_obj = indices
    y_get = x[idx_obj]
    y_index = x.index(*indices)

    y_get_np = _to_numpy_cpu(y_get)
    y_index_np = _to_numpy_cpu(y_index)

    assert y_get_np.shape == y_index_np.shape
    np.testing.assert_allclose(y_get_np, y_index_np)


def _assert_index_error_parity(x, indices, match: str):
    if len(indices) == 1:
        idx_obj = indices[0]
    else:
        idx_obj = indices

    with pytest.raises(Exception, match=match) as exc1:
        _ = x[idx_obj]
    exc_type = type(exc1.value)

    with pytest.raises(exc_type, match=match) as exc2:
        _ = x.index(*indices)

    # Error surface should be identical.
    assert str(exc1.value) == str(exc2.value)


def test_tensor_index_basic_parity_cpu():
    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)

        x = vt.arange(6, dtype="float32").reshape((2, 3))

        # Integer, slices, ellipsis, and None patterns.
        cases = [
            (1,),
            (slice(None), 1),
            (...,),
            (None, slice(None), 1),
        ]

        for indices in cases:
            _assert_index_value_parity(x, *indices)
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_tensor_index_advanced_parity_cpu():
    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)

        x = vt.arange(6, dtype="float32")
        idx = vt.tensor([0, 3, 5], dtype="int64")

        _assert_index_value_parity(x, idx)
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_tensor_index_advanced_parity_cuda_matches_getitem():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor", allow_module_level=False)

    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)

        # CPU baseline used to construct CUDA tensors.
        x_cpu = vt.arange(6, dtype="float32")
        idx_cpu = vt.tensor([0, 3, 5], dtype="int64")

        x_np = _to_numpy_cpu(x_cpu)
        idx_np = _to_numpy_cpu(idx_cpu).astype(np.int64)

        x_cuda = vt.cuda.to_device(x_np)
        idx_cuda = vt.cuda.to_device(idx_np)

        if x_cuda.device[0] != 2:  # kDLCUDA
            pytest.skip("CUDA device not available for vt.cuda helpers")

        # Compare CUDA results via vt.cuda.from_device instead of CPU DLPack.
        y_get = x_cuda[idx_cuda]
        y_index = x_cuda.index(idx_cuda)

        y_get_np = vt.cuda.from_device(y_get)
        y_index_np = vt.cuda.from_device(y_index)

        assert y_get_np.shape == y_index_np.shape
        np.testing.assert_allclose(y_get_np, y_index_np)
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_tensor_index_error_parity_for_unsupported_patterns():
    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(True)

        # Sequence-of-scalars (1-D)
        x1 = vt.arange(4, dtype="float32")
        _assert_index_error_parity(
            x1,
            ([0, 1],),
            match="advanced indexing pattern is not supported",
        )

        # Multiple tensor indices (2-D)
        x2 = vt.arange(9, dtype="float32").reshape((3, 3))
        idx0 = vt.tensor([0, 2], dtype="int64")
        idx1 = vt.tensor([0, 1], dtype="int64")
        _assert_index_error_parity(
            x2,
            (idx0, idx1),
            match="multiple tensor/bool indices",
        )

        # Suffix basic indices after advanced index.
        x3 = vt.arange(6, dtype="float32").reshape((2, 3))
        idx = vt.tensor([0, 2], dtype="int64")
        _assert_index_error_parity(
            x3,
            (idx, 0),
            match="suffix basic indices",
        )
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)
