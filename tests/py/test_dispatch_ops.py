# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_vt_dispatch_wrappers_work():
    from vibetensor import _C as C
    a = C.vt.unit()
    b = C.vt.unit()

    out0 = C.vt.unit()
    assert hasattr(out0, "sizes")

    out1 = C.vt.relu(a)
    assert out1.sizes == a.sizes

    out2 = C.vt.add(a, b)
    assert out2.sizes == a.sizes

    out3 = C.vt.mul(a, b)
    assert out3.sizes == a.sizes
