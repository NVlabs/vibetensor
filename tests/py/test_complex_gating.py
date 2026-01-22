# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C


_ERR_COMPLEX_DISABLED = "complex dtypes are disabled; set VBT_ENABLE_COMPLEX=1"


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def _run_py(code: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    # Make the subprocess deterministic regardless of the parent env.
    env2.pop("VBT_ENABLE_COMPLEX", None)
    if env:
        env2.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env2)


def _to_numpy_cpu(t):
    cap = vt.to_dlpack(t)
    try:
        arr = np.from_dlpack(cap)  # type: ignore[arg-type]
    except AttributeError:
        # Older NumPy expects a provider with __dlpack__.
        class _CapsuleWrapper:
            def __init__(self, inner):
                self._inner = inner

            def __dlpack__(self):  # pragma: no cover
                return self._inner

        arr = np.from_dlpack(_CapsuleWrapper(cap))  # type: ignore[arg-type]
    return arr.reshape(tuple(int(s) for s in t.sizes))


@pytest.mark.parametrize(
    "env_val, enabled",
    [
        (None, False),
        ("", False),
        ("0", False),
        ("true", False),
        ("True", False),
        (" 1", False),
        ("1 ", False),
        ("01", False),
        ("1", True),
    ],
)
def test_complex_env_parsing_exact_one(env_val: str | None, enabled: bool) -> None:
    code = (
        "import vibetensor.torch as vt\n"
        f"ERR = {_ERR_COMPLEX_DISABLED!r}\n"
        f"ENABLED = {1 if enabled else 0}\n"
        "ok = True\n"
        "try:\n"
        "    t = vt.tensor(1+2j)\n"
        "    if not ENABLED:\n"
        "        ok = False\n"
        "    else:\n"
        "        if getattr(t, 'dtype', None) not in ('complex64', 'complex128'):\n"
        "            ok = False\n"
        "except Exception as e:\n"
        "    if ENABLED:\n"
        "        ok = False\n"
        "    else:\n"
        "        if type(e).__name__ != 'TypeError' or str(e) != ERR:\n"
        "            ok = False\n"
        "import sys\n"
        "sys.exit(0 if ok else 1)\n"
    )

    env: Dict[str, str] = {}
    if env_val is not None:
        env["VBT_ENABLE_COMPLEX"] = env_val

    res = _run_py(code, env=env)
    assert res.returncode == 0, f"env_val={env_val!r} stdout={res.stdout!r} stderr={res.stderr!r}"


def test_complex_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VBT_ENABLE_COMPLEX", raising=False)

    with pytest.raises(TypeError) as excinfo:
        _ = vt.tensor(1 + 2j)

    assert str(excinfo.value) == _ERR_COMPLEX_DISABLED


def test_complex_dtype_aliases_are_recognized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VBT_ENABLE_COMPLEX", raising=False)

    for tok in ("cfloat", "cdouble"):
        with pytest.raises(TypeError) as excinfo:
            _ = vt.zeros((1,), dtype=tok)

        assert str(excinfo.value) == _ERR_COMPLEX_DISABLED


def test_complex_sequence_inference_is_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VBT_ENABLE_COMPLEX", raising=False)

    with pytest.raises(TypeError) as excinfo:
        _ = vt.tensor([1 + 2j])

    assert str(excinfo.value) == _ERR_COMPLEX_DISABLED


def test_complex_sequence_inference_enabled_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    t = vt.tensor([1 + 2j])
    assert t.dtype == "complex64"
    np.testing.assert_allclose(_to_numpy_cpu(t), np.array([1 + 2j], dtype=np.complex64))


def test_cannot_cast_complex_to_real(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VBT_ENABLE_COMPLEX", raising=False)

    with pytest.raises(TypeError, match=r"cannot cast complex to real"):
        _ = vt.tensor(1 + 2j, dtype="float32")

    with pytest.raises(TypeError, match=r"cannot cast complex to real"):
        _ = vt.full((1,), 1 + 2j, dtype="float32")

    x = np.array([1 + 2j], dtype=np.complex64)
    with pytest.raises(TypeError, match=r"cannot cast complex to real"):
        _ = vt.as_tensor(x, dtype="float32")


def test_complex_construction_enabled_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    t = vt.tensor(1 + 2j)
    assert t.dtype == "complex64"

    z = vt.zeros((2,), dtype="cfloat")
    assert z.dtype == "complex64"

    f = vt.full((3,), 1 + 2j, dtype="complex64")
    f_np = _to_numpy_cpu(f)
    np.testing.assert_allclose(f_np, np.array([1 + 2j, 1 + 2j, 1 + 2j], dtype=np.complex64))


def test_from_numpy_complex_is_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VBT_ENABLE_COMPLEX", raising=False)

    x = np.array([1 + 2j], dtype=np.complex64)
    with pytest.raises(TypeError) as excinfo:
        _ = vt.from_numpy(x)
    assert str(excinfo.value) == _ERR_COMPLEX_DISABLED


def test_from_dlpack_complex_disabled_does_not_consume_capsule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VBT_ENABLE_COMPLEX", raising=False)

    x = np.array([1 + 2j], dtype=np.complex64)
    cap = x.__dlpack__()

    # Gate should fail *without* consuming; repeated calls see the same error.
    for _ in range(2):
        with pytest.raises(TypeError) as excinfo:
            _ = vt.from_dlpack(cap)
        assert str(excinfo.value) == _ERR_COMPLEX_DISABLED


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available for VibeTensor")
def test_complex_cuda_to_device_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VBT_ENABLE_COMPLEX", "1")

    x = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    t = vt.cuda.to_device(x)
    assert t.device[0] == 2
    assert t.dtype == "complex64"

    y = vt.cuda.from_device(t)
    np.testing.assert_allclose(y, x)
