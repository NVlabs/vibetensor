# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor import _C as _C
from vibetensor.library import Library


def test_call_op_kwargs_precheck_and_kwargs_flow():
    # Prepare a base-only op and verify pre-check error
    c_def = getattr(_C, "def")
    c_def("y::g(Tensor) -> Tensor")
    with pytest.raises(TypeError, match="kwargs not supported for base kernels"):
        _C._call_op_kwargs("y::g")

    # Prepare a Python override for a unary op and ensure kwargs arrive
    c_def("x::fwd(Tensor) -> Tensor")
    lib = Library("x", "DEF")

    seen = {}

    def cpu_impl(x, **kwargs):
        seen.update(kwargs)
        return x

    lib.impl("fwd", cpu_impl, dispatch_key="CPU")

    t = _C.vt.unit()
    out = _C._call_op_kwargs("x::fwd", t, alpha=7, name="hi")
    assert out is not None  # TensorImpl
    assert seen == {"alpha": 7, "name": "hi"}

    # Result-count invariant: bad return should raise
    def bad_impl(x, **kwargs):
        return 123  # not a Tensor

    lib.impl("fwd", bad_impl, dispatch_key="CPU", allow_override=True)
    with pytest.raises(Exception):
        _C._call_op_kwargs("x::fwd", t, z=1)
