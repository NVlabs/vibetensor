# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import vibetensor.torch as vt


def test_as_strided_is_marked_as_view_for_requires_grad_setter():
    x = vt.zeros((2, 3), dtype="float32")
    y = x.as_strided((2, 3), (3, 1), 0)
    with pytest.raises(RuntimeError, match="non-view leaf tensors"):
        y.set_requires_grad(True)


def test_reshape_aliasing_is_marked_as_view_for_requires_grad_setter():
    x = vt.zeros((2, 3), dtype="float32")
    y = x.reshape((6,))
    with pytest.raises(RuntimeError, match="non-view leaf tensors"):
        y.set_requires_grad(True)


def test_reshape_copy_is_not_marked_as_view():
    x = vt.zeros((2, 3), dtype="float32")
    t = x.transpose(0, 1)
    y = t.reshape((6,))
    y.set_requires_grad(True)
    assert y.requires_grad is True
    assert y.is_contiguous()
