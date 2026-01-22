# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_tensor_creation_and_properties():
    from vibetensor import _C as C
    # Use default vt.unit to obtain a Tensor
    t = C.vt.unit()
    assert t.sizes == ()
    assert t.is_contiguous()
    assert t.is_non_overlapping_and_dense()
    assert t.version() == 0

    # clone returns a contiguous tensor with reset version
    c = t.clone()
    assert c.sizes == t.sizes
    assert c.is_contiguous()
    assert c.version() == 0


def test_inplace_ops_bump_version():
    from vibetensor import _C as C
    t = C.vt.unit()
    v0 = t.version()
    # add_ with itself is allowed (no partial overlap) on scalar
    t.add_(t)
    assert t.version() == v0 + 1
    t.relu_()
    assert t.version() == v0 + 2
