# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

import vibetensor  # noqa: F401 - ensure import-time patches run
import vibetensor.autograd as A
import vibetensor.torch as vt
import vibetensor.torch.autograd as At


def test_autograd_symbols_present() -> None:
    for name in ("backward", "grad", "GradcheckError", "gradcheck", "gradgradcheck"):
        assert hasattr(A, name)


def test_gradcheckerror_is_runtimeerror_subclass() -> None:
    assert issubclass(A.GradcheckError, RuntimeError)


def test_gradcheckerror_constructor_is_simple_runtimeerror() -> None:
    err = A.GradcheckError("msg")
    assert isinstance(err, A.GradcheckError)
    assert isinstance(err, RuntimeError)
    assert "msg" in str(err)


def test_backward_signature_shape() -> None:
    sig = inspect.signature(A.backward)
    assert list(sig.parameters.keys()) == [
        "tensors",
        "grad_tensors",
        "retain_graph",
        "create_graph",
        "inputs",
    ]
    assert sig.parameters["inputs"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["create_graph"].default is False


def test_grad_signature_shape() -> None:
    sig = inspect.signature(A.grad)
    assert list(sig.parameters.keys()) == [
        "outputs",
        "inputs",
        "grad_outputs",
        "retain_graph",
        "create_graph",
        "allow_unused",
        "is_grads_batched",
        "materialize_grads",
    ]
    assert sig.parameters["create_graph"].default is False


def test_autograd_gradcheck_and_gradgradcheck_signatures() -> None:
    sig_gc = inspect.signature(A.gradcheck)
    assert list(sig_gc.parameters.keys()) == [
        "fn",
        "inputs",
        "eps",
        "atol",
        "rtol",
        "raise_exception",
        "fast_mode",
    ]
    assert sig_gc.parameters["eps"].default == 1e-3
    assert sig_gc.parameters["atol"].default == 1e-4
    assert sig_gc.parameters["rtol"].default == 1e-2
    assert sig_gc.parameters["raise_exception"].default is True
    assert sig_gc.parameters["fast_mode"].default is None

    sig_ggc = inspect.signature(A.gradgradcheck)
    assert list(sig_ggc.parameters.keys()) == [
        "fn",
        "inputs",
        "eps",
        "atol",
        "rtol",
        "raise_exception",
        "fast_mode",
    ]
    assert sig_ggc.parameters["eps"].default == 1e-3
    assert sig_ggc.parameters["atol"].default == 1e-4
    assert sig_ggc.parameters["rtol"].default == 1e-2
    assert sig_ggc.parameters["raise_exception"].default is True
    assert sig_ggc.parameters["fast_mode"].default is None


def test_torch_autograd_alias_identity() -> None:
    assert vt.autograd is A
    assert At is A


def test_torch_autograd_forwards_new_symbols() -> None:
    for name in ("backward", "grad", "GradcheckError", "gradcheck", "gradgradcheck"):
        assert getattr(vt.autograd, name) is getattr(A, name)


def test_torch_all_remains_minimal() -> None:
    assert getattr(vt, "__all__", None) in ((), [], None)
