# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import os
import pathlib
import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


def _find_so(name: str) -> str | None:
    so_name = f"lib{name}.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / "build-py" / so_name,
    ]
    env = os.environ.get(f"VBT_{name.upper()}_PATH")
    if env:
        return env
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _rtld_noload_mode() -> int | None:
    # Mirror the dlopen flags used in C++ tests (RTLD_NOLOAD requires a base mode).
    if not hasattr(os, "RTLD_NOLOAD"):
        return None
    mode = os.RTLD_NOLOAD
    mode |= getattr(os, "RTLD_LAZY", 0)
    mode |= getattr(os, "RTLD_LOCAL", 0)
    return mode


@pytest.fixture(scope="session")
def _p3_fail_init_load_attempt() -> tuple[str, str]:
    path = _find_so("p3_fail_init")
    if path is None:
        pytest.skip("p3_fail_init plugin not found; ensure build produced libp3_fail_init.so")

    # Ensure the plugin is not already loaded in this process.
    mode = _rtld_noload_mode()
    if mode is not None:
        try:
            h = ctypes.CDLL(path, mode=mode)
        except OSError:
            h = None
        if h is not None:
            del h
            gc.collect()
            pytest.skip("p3_fail_init plugin already loaded")

    with pytest.raises(ValueError) as ei:
        vt.ops.load_library(path)

    return path, str(ei.value)


def test_plugin_fail_init_load_library_raises_with_plugin_message(
    _p3_fail_init_load_attempt: tuple[str, str],
) -> None:
    _path, msg = _p3_fail_init_load_attempt
    assert "plugin init failed" in msg


@pytest.mark.xfail(
    strict=True,
    reason="dispatcher/p3: init failure should be atomic (op must not be registered)",
)
def test_plugin_fail_init_failure_does_not_register_op(
    _p3_fail_init_load_attempt: tuple[str, str],
) -> None:
    _path, _msg = _p3_fail_init_load_attempt
    assert C._has_op("vt::p3_fail") is False


@pytest.mark.xfail(
    strict=True,
    reason="dispatcher/p3: init failure should dlclose the plugin (RTLD_NOLOAD must fail)",
)
def test_plugin_fail_init_failure_dlcloses_library(
    _p3_fail_init_load_attempt: tuple[str, str],
) -> None:
    path, _msg = _p3_fail_init_load_attempt
    mode = _rtld_noload_mode()
    if mode is None:
        pytest.skip("RTLD_NOLOAD not supported on this platform")

    # Current loader behavior parks failed-init handles; future behavior should dlclose.
    if C._is_library_loaded(path):
        try:
            h = ctypes.CDLL(path, mode=mode)
        except OSError as e:
            pytest.skip(f"RTLD_NOLOAD check failed unexpectedly: {e}")
        else:
            del h
            gc.collect()
            pytest.fail("expected RTLD_NOLOAD to fail after init failure")
    else:
        with pytest.raises(OSError):
            ctypes.CDLL(path, mode=mode)
