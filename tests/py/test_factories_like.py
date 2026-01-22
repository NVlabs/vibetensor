# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import vibetensor.torch as vt


def test_like_variants_contiguity_and_dtype():
    base = vt.zeros((2, 3), dtype="int64")
    noncontig = base.view((2, 3)).transpose(0, 1)
    z = vt.zeros_like(noncontig)
    assert z.is_contiguous()
    assert z.dtype == base.dtype
    f = vt.full_like(base, 3.0, dtype="int64")
    assert f.dtype == "int64"
    with pytest.raises(ValueError):
        _ = vt.full_like(base, float('inf'), dtype="int64")
