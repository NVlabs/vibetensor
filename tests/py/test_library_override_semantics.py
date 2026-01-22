# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from vibetensor.library import Library, _REG


def test_allow_override_semantics_impl_and_fallback():
    fq = "x::foo"
    _REG.pop(fq, None)
    lib = Library("x", "DEF")
    lib.define("x::foo() -> Tensor")

    def f1():
        return 1

    def f2():
        return 2

    # Initial CPU impl
    lib.impl("foo", f1, dispatch_key="CPU")
    assert _REG[fq].impls["CPU"] is f1

    # Idempotent with same function
    lib.impl("foo", f1, dispatch_key="CPU")
    assert _REG[fq].impls["CPU"] is f1

    # Duplicate with different function without allow_override should fail
    with pytest.raises(ValueError, match="duplicate CPU impl: x::foo"):
        lib.impl("foo", f2, dispatch_key="CPU")

    # With allow_override, replacement should succeed
    lib.impl("foo", f2, dispatch_key="CPU", allow_override=True)
    assert _REG[fq].impls["CPU"] is f2

    # Fallback semantics: wildcard
    def fb1():
        return 10

    def fb2():
        return 20

    lib.fallback("foo", fb1)
    assert _REG[fq].wildcard_fallback is fb1

    # Idempotent same object
    lib.fallback("foo", fb1)
    assert _REG[fq].wildcard_fallback is fb1

    # Duplicate without allow_override fails
    with pytest.raises(ValueError, match="duplicate fallback impl: x::foo"):
        lib.fallback("foo", fb2)

    # Replace with allow_override
    lib.fallback("foo", fb2, allow_override=True)
    assert _REG[fq].wildcard_fallback is fb2

    # Keyed fallback
    def fbc1():
        return 30

    def fbc2():
        return 40

    lib.fallback("foo", fbc1, dispatch_key="CPU")
    assert _REG[fq].fallbacks["CPU"] is fbc1

    # Duplicate fails without allow_override
    with pytest.raises(ValueError, match="duplicate CPU fallback: x::foo"):
        lib.fallback("foo", fbc2, dispatch_key="CPU")

    # Replace with allow_override
    lib.fallback("foo", fbc2, dispatch_key="CPU", allow_override=True)
    assert _REG[fq].fallbacks["CPU"] is fbc2
