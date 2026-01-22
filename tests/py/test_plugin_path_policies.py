# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import stat
import shutil
import pathlib
import tempfile
import pytest

import vibetensor.torch as vt


def _find_so(name: str) -> str | None:
    so_name = f"lib{name}.so"
    candidates = [
        pathlib.Path.cwd() / so_name,
        pathlib.Path(__file__).resolve().parent.parent.parent / so_name,
        pathlib.Path(__file__).resolve().parents[2] / "build-py" / so_name,
    ]
    env = os.environ.get(f"VBT_{name.upper()}_PATH")
    if env:
        return env
    for p in candidates:
        if p.exists():
            return str(p)
    return None


@pytest.mark.skipif(not (os.name == "posix"), reason="POSIX-only path policy tests")
def test_world_writable_parent_disallowed_then_allowed():
    src = _find_so("vbt_reference_add")
    if src is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")
    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        # Make parent world-writable without sticky bit
        os.chmod(td, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0o777
        dst = tdp / "libvbt_reference_add.so"
        shutil.copy2(src, dst)
        # Without env gate, should be rejected
        with pytest.raises(ValueError) as ei:
            vt.ops.load_library(str(dst))
        assert "world-writable parent disallowed" in str(ei.value)
        # With env gate, should succeed
        os.environ["VBT_ALLOW_WORLD_WRITABLE"] = "1"
        try:
            vt.ops.load_library(str(dst))
        finally:
            os.environ.pop("VBT_ALLOW_WORLD_WRITABLE", None)


@pytest.mark.skipif(not (os.name == "posix"), reason="POSIX-only path policy tests")
def test_symlink_path_rejected():
    src = _find_so("vbt_reference_add")
    if src is None:
        pytest.skip("reference plugin not found; ensure build produced libvbt_reference_add.so")
    with tempfile.TemporaryDirectory() as td:
        link = pathlib.Path(td) / "libref_add_link.so"
        os.symlink(src, link)
        with pytest.raises(ValueError) as ei:
            vt.ops.load_library(str(link))
        assert "symlink in path rejected" in str(ei.value)
