# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from vibetensor.torch import ops
from vibetensor import _C as _C


def test_loaded_libraries_surface_and_error_mapping(monkeypatch: pytest.MonkeyPatch):
    # Surface exists and returns a list
    libs = ops.loaded_libraries
    assert isinstance(libs, list)

    # _is_library_loaded should be False for missing paths
    assert hasattr(_C, "_is_library_loaded")
    assert _C._is_library_loaded("/this/path/does/not/exist.so") is False

    # Under compat, loader errors are mapped to OSError
    monkeypatch.setenv("VBT_OPS_COMPAT", "1")
    with pytest.raises(OSError):
        ops.load_library("/definitely/missing.so")
