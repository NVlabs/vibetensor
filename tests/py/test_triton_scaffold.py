# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Import guard: must not import torch or triton
import vibetensor.triton as vt_triton  # noqa: F401

from vibetensor import _C as C
import vibetensor.torch as vt


def test_triton_register_cpu_redispatch_add():
    print("[debug] entering test_triton_register_cpu_redispatch_add")
    # Ensure CPU tensors route to base kernel (add)
    a = vt.ones((4,), dtype="float32")
    b = vt.ones((4,), dtype="float32")
    print("[debug] created inputs")

    # Define a dummy kernel that would be clearly different
    def k(a_t, b_t):  # pragma: no cover
        raise AssertionError("should not be called for CPU tensors")

    # Register override for vt::add
    vt_triton.register("vt::add", k)
    print("[debug] registered override")

    # Call via dispatcher wrapper; ensure call succeeds (override should not run)
    out = C.vt.add(a, b)
    print("[debug] called C.vt.add")
    # Validate tensor metadata (numeric checked elsewhere in base tests)
    assert tuple(out.sizes) == (4,)
    assert out.dtype == "float32"
    print("[debug] metadata assertions passed")


def test_triton_register_validates_types():
    def k(*args):  # pragma: no cover
        return args
    with pytest.raises(TypeError):
        vt_triton.register(123, k)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vt_triton.register("vt::add", 123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vt_triton.register("vt::add", k, signature=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vt_triton.register("vt::add", k, grid_fn=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vt_triton.register("vt::add", k, meta=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vt_triton.register("vt::add", k, num_warps=1.2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vt_triton.register("vt::add", k, shared_mem="1")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vt_triton.register("vt::add", k, allow_broadcast=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        vt_triton.register("vt::add", k, num_stages=0)
