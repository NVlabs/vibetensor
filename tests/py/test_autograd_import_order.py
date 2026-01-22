# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import vibetensor._C as C


def _fresh_import(name: str):
    return importlib.import_module(name)


def test_no_grad_aliasing_across_surfaces():
    # Import base package and overlays in a stable order
    import vibetensor  # noqa: F401
    A = _fresh_import("vibetensor.autograd")
    vt = _fresh_import("vibetensor.torch")

    ag = C.autograd

    assert ag.no_grad is A.no_grad  # type: ignore[attr-defined]
    assert ag.enable_grad is A.enable_grad  # type: ignore[attr-defined]
    assert vt.no_grad is A.no_grad
    assert vt.enable_grad is A.enable_grad


def test_inference_mode_aliasing_across_surfaces():
    import vibetensor  # noqa: F401
    A = _fresh_import("vibetensor.autograd")
    vt = _fresh_import("vibetensor.torch")

    ag = C.autograd

    # Inference helpers should exist on the C submodule
    assert hasattr(ag, "is_inference_mode_enabled")
    assert hasattr(ag, "_set_inference_mode_enabled")

    # And all three surfaces should agree on the context manager identity
    assert ag.inference_mode is A.inference_mode  # type: ignore[attr-defined]
    assert vt.inference_mode is A.inference_mode
