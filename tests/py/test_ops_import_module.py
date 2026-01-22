# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


def test_ops_import_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create a temporary package with a registrar module
    pkg = tmp_path / "myext"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "reg.py").write_text(
        "from vibetensor.library import Library\n"
        "lib = Library('ext', 'DEF')\n"
        "lib.define('ext::noop(Tensor) -> Tensor')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from vibetensor.torch import ops

    # First import returns the module
    mod1 = ops.import_module("myext.reg")
    assert getattr(mod1, "__name__", None) == "myext.reg"

    # Second import should be idempotent and return the same module object
    mod2 = ops.import_module("myext.reg")
    assert mod1 is mod2

    # Importing an existing module object should return it unchanged
    mod3 = ops.import_module(mod1)
    assert mod3 is mod1
