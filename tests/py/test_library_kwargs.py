# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple

import pytest

from vibetensor.library import _boxed_mux, _REG, Library
from vibetensor import _C as _C


class DummyTensor:
    def __init__(self, device: Tuple[int, int]) -> None:
        self.device = device


def test_boxed_mux_kwargs_pass_and_kwargs_only_redispatch(monkeypatch: pytest.MonkeyPatch):
    fq = "ext::kwop"
    # Ensure clean registry for this fqname
    _REG.pop(fq, None)

    # Install a CPU impl that returns kwargs for verification
    lib = Library("ext", "DEF")
    lib.define("ext::kwop(Tensor) -> Tensor")

    seen: dict[str, Any] = {}

    def cpu_impl(x: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return (x, kwargs)

    lib.impl("kwop", cpu_impl, dispatch_key="CPU")

    # Build mux and call with kwargs -> should reach cpu_impl
    mux = _boxed_mux(fq)
    x = DummyTensor((1, 0))  # CPU
    out = mux(x, alpha=3, beta="b")
    assert isinstance(out, tuple) and out[0] is x
    assert seen == {"alpha": 3, "beta": "b"}

    # kwargs-only call must redispatch to base (0-arity path)
    sentinel = object()

    def redispatch_stub(*args: Any) -> Any:  # no args expected
        return sentinel

    monkeypatch.setattr(_C, "_redispatch_boxed_current", redispatch_stub)

    out2 = mux(**{"foo": 1})
    assert out2 is sentinel

    # Errors (other than NotImplementedError) from target propagate
    def bad_impl(x: Any, **kwargs: Any) -> Any:
        raise ValueError("boom")

    lib.impl("kwop", bad_impl, dispatch_key="CPU", allow_override=True)
    with pytest.raises(ValueError, match="boom"):
        mux(x, gamma=True)
