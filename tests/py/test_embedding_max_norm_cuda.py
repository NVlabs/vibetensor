# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor.torch as vt
from vibetensor import _C as C

pytestmark = pytest.mark.cuda


def _has_cuda() -> bool:
    try:
        return bool(getattr(C, "_has_cuda", False)) and int(C._cuda_device_count()) > 0
    except Exception:
        return False


def test_functional_embedding_max_norm_cuda_changes_only_referenced_rows():
    if not _has_cuda():
        pytest.skip("CUDA not available for VibeTensor")

    w_cpu = np.array(
        [
            [3.0, 4.0, 0.0],  # ||.||2 = 5  -> scaled
            [0.1, 0.2, 0.3],  # ||.||2 < 1  -> unchanged
            [0.0, 5.0, 12.0],  # ||.||2 = 13 -> scaled
            [1.0, 0.0, 0.0],  # ||.||2 = 1  -> unchanged (== max_norm)
        ],
        dtype=np.float32,
    )
    weight = vt.cuda.to_device(w_cpu)

    idx_cpu = np.array([0, 2, 3, 0], dtype=np.int64)
    idx = vt.cuda.to_device(idx_cpu)

    w_before = vt.cuda.from_device(weight)

    out = vt.nn.functional.embedding(idx, weight, max_norm=1.0, norm_type=2.0)

    w_after = vt.cuda.from_device(weight)
    out_np = vt.cuda.from_device(out)

    w_expected = w_before.copy()
    for row in np.unique(idx_cpu):
        r = w_before[int(row)]
        norm = float(np.linalg.norm(r, ord=2))
        if norm > 1.0:
            w_expected[int(row)] = r * (1.0 / (norm + 1e-7))

    np.testing.assert_allclose(w_after, w_expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_np, w_expected[idx_cpu], rtol=1e-6, atol=1e-6)
