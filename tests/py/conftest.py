# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import sysconfig

import pytest


def _ensure_test_bootstrap_on_pythonpath(root: str) -> None:
    """Ensure subprocesses can import the test-only `sitecustomize`.

    Some tests spawn fresh Python processes that import NumPy/PyTorch before
    VibeTensor. Under Conda, that can load an older libstdc++ first and break
    subsequent imports of vibetensor._C.

    We ship a test-only `sitecustomize.py` under `tests/py/_vbt_bootstrap/` and
    ensure that directory is present on PYTHONPATH so subprocesses inherit it.
    """

    bootstrap = os.path.join(root, "tests", "py", "_vbt_bootstrap")
    if not os.path.isdir(bootstrap):
        return

    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if bootstrap in parts:
        return
    parts.insert(0, bootstrap)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def _prime_vibetensor_extension_from_build() -> None:
    """Ensure tests import vibetensor._C from the local build tree.

    Prefer the extension built by CMake under ``build-py/python/vibetensor/_C*.so``.

    Older build layouts placed the extension at ``build-py/_C*.so``; we fall back
    to that path for compatibility.

    This helper is intended only for the test environment and is a no-op when the
    build directory or extension is missing.
    """

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _ensure_test_bootstrap_on_pythonpath(root)
    build_root = os.path.join(root, "build-py")
    if not os.path.isdir(build_root):
        return

    new_layout = sorted(glob.glob(os.path.join(build_root, "python", "vibetensor", "_C.*.so")))
    old_layout = sorted(glob.glob(os.path.join(build_root, "_C.*.so")))
    candidates = new_layout if new_layout else old_layout
    if not candidates:
        return

    candidates = sorted(set(candidates))

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if ext_suffix:
        abi_matches = [p for p in candidates if p.endswith(ext_suffix)]
        if abi_matches:
            if len(abi_matches) != 1:
                raise RuntimeError(
                    "Multiple vibetensor._C extensions match EXT_SUFFIX:\n" + "\n".join(abi_matches)
                )
            path = abi_matches[0]
        else:
            path = candidates[0]
    else:
        path = candidates[0]
    spec = importlib.util.spec_from_file_location("vibetensor._C", path)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    # Register before execution to avoid recursive imports loading a different
    # copy of the extension.
    sys.modules["vibetensor._C"] = module
    spec.loader.exec_module(module)


_prime_vibetensor_extension_from_build()


@pytest.fixture(autouse=True)
def _reset_autograd_modes() -> None:
    """Keep global grad/inference mode state isolated between tests.

    Several tests intentionally toggle grad-mode / inference-mode. This fixture
    forces a known default state (grad enabled, inference disabled) before and
    after each test to prevent order-dependent failures.
    """

    try:
        import vibetensor._C as _C
    except Exception:
        yield
        return

    ag = getattr(_C, "autograd", None)
    if ag is None:
        yield
        return

    set_grad = getattr(ag, "set_grad_enabled", None)
    set_inf = getattr(ag, "_set_inference_mode_enabled", None)
    set_mt = getattr(ag, "set_multithreading_enabled", None)
    set_view = getattr(ag, "set_view_replay_enabled", None)

    if callable(set_inf):
        set_inf(False)
    if callable(set_grad):
        set_grad(True)
    if callable(set_mt):
        set_mt(False)
    if callable(set_view):
        set_view(False)

    yield

    if callable(set_inf):
        set_inf(False)
    if callable(set_grad):
        set_grad(True)
    if callable(set_mt):
        set_mt(False)
    if callable(set_view):
        set_view(False)
