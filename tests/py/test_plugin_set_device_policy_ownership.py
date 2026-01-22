# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

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


def _run_py(code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env2 = dict(os.environ)
    if env:
        env2.update(env)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env2
    )


@pytest.mark.parametrize(
    "mode, expected_substr",
    [
        ("def_null", "def_string is null"),
        ("def_empty", "def_string is empty"),
        ("policy_not_owned", "op not owned"),
    ],
)
def test_plugin_load_library_rejects_bad_registrations(mode: str, expected_substr: str):
    path = _find_so("set_device_policy_core_reject")
    if path is None:
        pytest.skip(
            "set_device_policy_core_reject plugin not found; "
            "ensure build produced libset_device_policy_core_reject.so"
        )

    code = (
        "import os, sys\n"
        "import vibetensor.torch as vt\n"
        "path = os.environ['VBT_TEST_PLUGIN_PATH']\n"
        "expect = os.environ['VBT_EXPECT_SUBSTR']\n"
        "try:\n"
        "    vt.ops.load_library(path)\n"
        "except ValueError as e:\n"
        "    msg = str(e)\n"
        "    if expect in msg:\n"
        "        sys.exit(0)\n"
        "    print(msg)\n"
        "    sys.exit(1)\n"
        "print('unexpected success')\n"
        "sys.exit(1)\n"
    )

    cp = _run_py(
        code,
        env={
            "VBT_TEST_PLUGIN_PATH": path,
            "VBT_SET_DEVICE_POLICY_CORE_REJECT_MODE": mode,
            "VBT_EXPECT_SUBSTR": expected_substr,
        },
    )
    assert cp.returncode == 0, f"stdout={cp.stdout}\nstderr={cp.stderr}"


def test_plugin_host_apis_are_init_only():
    # Ensure we're in the default "ok" mode.
    os.environ.pop("VBT_SET_DEVICE_POLICY_CORE_REJECT_MODE", None)

    path = _find_so("set_device_policy_core_reject")
    if path is None:
        pytest.skip(
            "set_device_policy_core_reject plugin not found; "
            "ensure build produced libset_device_policy_core_reject.so"
        )

    try:
        vt.ops.load_library(path)
    except ValueError as e:
        # Accept idempotent call when already loaded in this session.
        if "plugin already loaded:" not in str(e):
            raise

    a = np.arange(4, dtype=np.float32)
    ta = vt.from_dlpack(a)

    with pytest.raises(ValueError) as ei_def:
        _ = vt.ops.p2_reject.post_def(ta)
    assert "def: init-only" in str(ei_def.value)

    with pytest.raises(ValueError) as ei_pol:
        _ = vt.ops.p2_reject.post_policy(ta)
    assert "set_device_policy: init-only" in str(ei_pol.value)
