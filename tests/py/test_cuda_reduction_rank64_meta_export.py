# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor._C as C
import vibetensor.torch as vbt


def _cuda_only() -> bool:
    return getattr(C, "_has_cuda", False) and vbt.cuda.is_available() and C._cuda_device_count() > 0


if not _cuda_only():
    pytest.skip("CUDA not available", allow_module_level=True)


def _to_numpy_cpu(t) -> np.ndarray:
    cap = vbt.to_dlpack(t)
    return np.from_dlpack(cap)


@pytest.mark.cuda
def test_cuda_reduction_rank_64_meta_export_does_not_throw():
    # Reduction TensorIterators do not coalesce dims, so a high-rank tensor should
    # propagate its full rank into DeviceStrideMeta export.
    shape = (2,) + (1,) * 63  # rank 64, numel 2
    t = vbt.ones(shape, dtype=vbt.float32).cuda()

    out = t.sum(dim=0)

    out_cpu = out.cpu()
    arr = _to_numpy_cpu(out_cpu)
    assert arr.shape == (1,) * 63
    assert float(arr.item()) == pytest.approx(2.0)
