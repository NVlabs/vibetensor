# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest

from vibetensor import _C as C
import vibetensor.torch as vt


@pytest.mark.cuda
def test_triton_stream_handle_and_no_torch_import():
    # Ensure vibetensor.triton can be imported without torch being imported
    if "torch" in sys.modules:
        del sys.modules["torch"]
    import importlib
    import vibetensor.triton as vt_triton  # noqa: F401
    assert "torch" not in sys.modules

    if not getattr(C, "_has_cuda", False) or C._cuda_device_count() <= 0:  # type: ignore[attr-defined]
        pytest.skip("CUDA not available for VibeTensor")

    # Prepare a non-default VBT stream and make it current; verify we can fetch a valid handle
    s = vt.cuda.Stream(priority=0)  # type: ignore[attr-defined]
    with s:
        handle = vt._cuda_stream_handle_current()
        assert isinstance(handle, int) and handle != 0
