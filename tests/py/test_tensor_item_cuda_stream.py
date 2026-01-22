# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import vibetensor._C as C
import vibetensor.torch as vbt


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and vbt.cuda.is_available() and C._cuda_device_count() > 0


if not _cuda_only():
    pytest.skip("CUDA not available", allow_module_level=True)


@pytest.mark.cuda
def test_tensor_item_nondefault_cuda_stream_is_correct():
    # Regression test: Tensor.item() must observe values produced on a
    # non-default CUDA stream.
    s = vbt.cuda.Stream()

    t = vbt.ones((1024,), dtype=vbt.float32).cuda()
    with s:
        out = t.sum()
        val = out.item()

    assert float(val) == pytest.approx(1024.0)
