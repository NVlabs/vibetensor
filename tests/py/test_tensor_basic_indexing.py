# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_basic_getitem_shapes_cpu():
    import vibetensor.torch as vt

    # 2x3 tensor with values 0..5 (shape-only checks here)
    x = vt.arange(6, dtype="int64").reshape((2, 3))
    assert tuple(x.sizes) == (2, 3)

    row1 = x[1]
    assert tuple(row1.sizes) == (3,)

    col1 = x[:, 1]
    assert tuple(col1.sizes) == (2,)

    rev = x[1:, ::-1]
    assert tuple(rev.sizes) == (1, 3)


def test_zero_dim_indexing_matrix():
    from vibetensor import _C as C

    x = C.vt.unit()  # 0-d scalar tensor
    assert tuple(x.sizes) == ()

    # () and ... are allowed and return 0-d views
    y = x[()]
    z = x[...]
    assert tuple(y.sizes) == ()
    assert tuple(z.sizes) == ()

    # None inserts new axes
    y1 = x[None]
    assert tuple(y1.sizes) == (1,)
    y2 = x[None, ...]
    assert tuple(y2.sizes) == (1,)

    # Dim-consuming basic indices are rejected
    with pytest.raises(IndexError, match="invalid index of a 0-dim tensor"):
        _ = x[0]
    with pytest.raises(IndexError, match="invalid index of a 0-dim tensor"):
        _ = x[1:]


def test_advanced_indices_rejected_when_flag_disabled():
    import vibetensor.torch as vt
    from vibetensor import _C as C

    x = vt.arange(4, dtype="float32")

    prev = C._advanced_indexing_enabled()
    try:
        C._set_advanced_indexing_enabled_for_tests(False)

        # Scalar boolean
        with pytest.raises(RuntimeError, match="advanced indexing disabled"):
            _ = x[True]
        with pytest.raises(RuntimeError, match="advanced indexing disabled"):
            _ = x.index(True)

        # Tensor index
        idx = vt.arange(2, dtype="int64")
        with pytest.raises(RuntimeError, match="advanced indexing disabled"):
            _ = x[idx]
        with pytest.raises(RuntimeError, match="advanced indexing disabled"):
            _ = x.index(idx)

        # Sequence-of-scalars
        with pytest.raises(RuntimeError, match="advanced indexing disabled"):
            _ = x[[0, 1]]
        with pytest.raises(RuntimeError, match="advanced indexing disabled"):
            _ = x.index([0, 1])
    finally:
        C._set_advanced_indexing_enabled_for_tests(prev)


def test_setitem_cpu_scalar_and_cuda_guard():
    import vibetensor.torch as vt
    from vibetensor import _C as C

    # CPU: scalar assignment should succeed and bump version
    x = vt.zeros((2, 2), dtype="float32")
    v0 = x.version()
    x[0, 0] = 1.0
    assert x.version() == v0 + 1

    if getattr(C, "_has_cuda", False):
        y = C._make_cuda_tensor([2, 2], "float32", 0.0)
        with pytest.raises(
            TypeError,
            match="Tensor.__setitem__ on CUDA requires a Tensor value",
        ):
            y[0, 0] = 1.0


def test_too_many_indices_error_substring():
    import vibetensor.torch as vt

    x = vt.arange(4, dtype="int64")

    # 1-D tensor indexed with two integers -> too many indices.
    with pytest.raises(
        IndexError,
        match="too many indices for tensor of dimension",
    ):
        _ = x[0, 0]



def test_multiple_ellipsis_error_substring():
    import vibetensor.torch as vt

    x = vt.arange(4, dtype="int64")

    # Multiple ellipses in a single index tuple are rejected.
    with pytest.raises(
        IndexError,
        match="index: at most one ellipsis is allowed",
    ):
        _ = x[..., ...]
