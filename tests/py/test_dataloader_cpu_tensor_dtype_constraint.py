# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest


def _has_cuda() -> bool:
    from vibetensor import _C as C

    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_dataloader_cpu_tensor_dtype_constraint():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    import vibetensor.torch as vt
    import vibetensor.torch.utils.data as vtd

    class _DS:
        def __len__(self):
            return 1

        def __getitem__(self, idx: int):
            _ = int(idx)
            return vt.from_numpy(np.asarray([1, 2], dtype=np.float16))

    dl = vtd.DataLoader(_DS(), batch_size=None, prefetch_to_device=True, device=0)

    with pytest.raises(TypeError) as exc:
        _ = next(iter(dl))

    assert "Tensor.numpy(): unsupported dtype" in str(exc.value)
