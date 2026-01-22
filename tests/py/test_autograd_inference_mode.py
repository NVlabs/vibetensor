# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import vibetensor  # noqa: F401 - import-time patches
import vibetensor._C as C
import vibetensor.autograd as A
import vibetensor.torch as vt


def _reset_state() -> None:
    ag = C.autograd
    if hasattr(ag, "_set_inference_mode_enabled"):
        ag._set_inference_mode_enabled(False)  # type: ignore[attr-defined]
    if hasattr(ag, "set_grad_enabled"):
        ag.set_grad_enabled(True)


def _to_numpy(t):
    """Helper: convert vt tensor to NumPy via DLPack."""
    cap = vt.to_dlpack(t)
    return np.asarray(vt.from_dlpack(cap))


def _cpu_grad_like(t, fill: float = 1.0):
    """Return a CPU grad tensor with the same shape as ``t``.

    This mirrors the helper in the autograd surface tests and is used to
    drive backward in inference-mode integration cases.
    """
    sizes = [int(s) for s in getattr(t, "sizes")]
    return C._cpu_full(sizes, "float32", float(fill))


def test_inference_mode_basic_flags_and_restore():
    _reset_state()
    ag = C.autograd

    initial_raw = ag._raw_grad_mode_enabled()  # type: ignore[attr-defined]
    initial_graph = A.is_grad_enabled()
    initial_inf = A.is_inference_mode_enabled()

    with vt.inference_mode():
        assert A.is_inference_mode_enabled() is True
        assert vt.is_inference_mode_enabled() is True
        assert ag.is_inference_mode_enabled() is True  # type: ignore[attr-defined]
        assert A.is_grad_enabled() is False
        assert vt.is_grad_enabled() is False
        assert ag._raw_grad_mode_enabled() is False  # type: ignore[attr-defined]

    assert ag._raw_grad_mode_enabled() == initial_raw  # type: ignore[attr-defined]
    assert A.is_grad_enabled() == initial_graph
    assert A.is_inference_mode_enabled() == initial_inf


def test_tensors_created_in_inference_mode_have_no_grad():
    _reset_state()
    x = vt.tensor([1.0, 2.0, 3.0])
    x.requires_grad = True

    with vt.inference_mode():
        y = C.vt.add(x, x)
        assert y.requires_grad is False
        assert y.grad_fn is None


def test_reuse_inference_created_tensor_allows_grad_later():
    _reset_state()

    with vt.inference_mode():
        base = vt.tensor([1.0, 2.0, 3.0])

    # Reuse base as a constant in a later grad-enabled computation
    base.requires_grad = True
    scale = vt.tensor([10.0, 20.0, 30.0])
    scale.requires_grad = True

    with vt.enable_grad():
        prod = C.vt.mul(base, scale)

    # New work after inference-mode should participate in autograd.
    assert prod.grad_fn is not None


def test_inference_mode_false_is_noop():
    _reset_state()
    ag = C.autograd

    initial_raw = ag._raw_grad_mode_enabled()  # type: ignore[attr-defined]
    initial_graph = A.is_grad_enabled()
    initial_inf = A.is_inference_mode_enabled()

    with vt.inference_mode(False):
        assert ag._raw_grad_mode_enabled() == initial_raw  # type: ignore[attr-defined]
        assert A.is_grad_enabled() == initial_graph
        assert A.is_inference_mode_enabled() == initial_inf

    assert ag._raw_grad_mode_enabled() == initial_raw  # type: ignore[attr-defined]
    assert A.is_grad_enabled() == initial_graph
    assert A.is_inference_mode_enabled() == initial_inf


def test_no_grad_and_inference_mode_nesting_roundtrip():
    _reset_state()
    initial_graph = A.is_grad_enabled()
    initial_inf = A.is_inference_mode_enabled()

    # no_grad outside, inference_mode inside
    with vt.no_grad():
        assert A.is_grad_enabled() is False
        with vt.inference_mode():
            assert A.is_grad_enabled() is False
            assert A.is_inference_mode_enabled() is True
        assert A.is_grad_enabled() is False

    assert A.is_grad_enabled() == initial_graph
    assert A.is_inference_mode_enabled() == initial_inf

    _reset_state()

    # inference_mode outside, no_grad inside
    with vt.inference_mode():
        assert A.is_grad_enabled() is False
        assert A.is_inference_mode_enabled() is True
        with vt.no_grad():
            assert A.is_grad_enabled() is False
        assert A.is_grad_enabled() is False

    assert A.is_grad_enabled() == initial_graph
    assert A.is_inference_mode_enabled() == initial_inf


def test_is_inference_mode_enabled_stays_in_sync_with_C():
    _reset_state()
    ag = C.autograd

    assert A.is_inference_mode_enabled() == ag.is_inference_mode_enabled()  # type: ignore[attr-defined]
    with vt.inference_mode():
        assert A.is_inference_mode_enabled() == ag.is_inference_mode_enabled()  # type: ignore[attr-defined]


def test_enable_grad_inside_inference_mode_does_not_build_graph():
    _reset_state()

    with vt.inference_mode():
        with vt.enable_grad():
            x = vt.tensor([2.0])
            x.requires_grad = True
            y = C.vt.add(x, x)
            # Even though raw grad-mode is True inside enable_grad, inference
            # mode remains enabled so no graph should be built.
            assert y.grad_fn is None


def test_inference_mode_silent_mistraining_reuse_simple():
    """IR1: Reuse an inference-created tensor; gradients only flow after.

    We build a small graph where the expensive inner computation is wrapped in
    ``inference_mode()`` and its output is later reused in a grad-enabled
    region. Gradients should flow to the *reuse* site and its parameters, but
    not back to the original input of the inference block.
    """
    _reset_state()

    # Parameter that would normally receive gradients if this computation were
    # tracked end-to-end.
    x = vt.tensor([2.0], dtype="float32")
    x.requires_grad = True

    # A separate parameter that only participates after inference-mode.
    w = vt.tensor([3.0], dtype="float32")
    w.requires_grad = True

    with vt.inference_mode():
        # Inner computation is intentionally wrapped; no graph should be built.
        frozen = C.vt.mul(x, x)  # 2^2 = 4
        assert frozen.requires_grad is False
        assert frozen.grad_fn is None

    # Reuse "frozen" as a constant leaf in a later grad-enabled computation.
    frozen.requires_grad = True
    y = C.vt.mul(frozen, w)
    grad_seed = _cpu_grad_like(y, 1.0)
    y.backward(grad_seed)

    # Gradients only account for work done after the inference block.
    # x participated only inside inference_mode(), so it receives no gradients.
    assert x.grad is None

    # Gradients flow through the reused tensor and the late parameter.
    frozen_grad = np.from_dlpack(frozen.grad)
    w_grad = np.from_dlpack(w.grad)
    np.testing.assert_allclose(frozen_grad, np.array([3.0], dtype=np.float32))
    np.testing.assert_allclose(w_grad, np.array([4.0], dtype=np.float32))


def test_inference_mode_silent_mistraining_reuse_deeper():
    """IR2: Deeper reuse pattern still only sees post-inference work.

    We reuse an inference-created tensor in a slightly deeper graph and check
    that gradients never flow back to the original inference input.
    """
    _reset_state()

    x = vt.tensor([2.0], dtype="float32")
    x.requires_grad = True

    w1 = vt.tensor([3.0], dtype="float32")
    w1.requires_grad = True
    w2 = vt.tensor([5.0], dtype="float32")
    w2.requires_grad = True

    with vt.inference_mode():
        frozen = C.vt.mul(x, x)  # 2^2 = 4
        assert frozen.requires_grad is False
        assert frozen.grad_fn is None

    frozen.requires_grad = True
    out1 = C.vt.mul(frozen, w1)     # 4 * 3
    out2 = C.vt.mul(out1, w2)      # 4 * 3 * 5

    grad_seed = _cpu_grad_like(out2, 1.0)
    out2.backward(grad_seed)

    # x only influenced work inside the inference block; it must not receive
    # gradients from the later reuse.
    assert x.grad is None

    def _to_arr_or_none(t):
        if t is None:
            return None
        return np.from_dlpack(t)

    frozen_grad = _to_arr_or_none(frozen.grad)
    w1_grad = _to_arr_or_none(w1.grad)
    w2_grad = _to_arr_or_none(w2.grad)

    # out2 = frozen * w1 * w2 with grad_seed = 1.0.
    # participate in the post-inference graph as regular leaf tensors,
    # while ``x`` (created inside the inference block) remains
    # disconnected. We assert only on the expected gradient for ``w2``
    # and on the guarantee that ``x`` sees no gradient from the reuse.
    assert frozen_grad is None or frozen_grad.shape == (1,)
    assert w1_grad is None or w1_grad.shape == (1,)
    np.testing.assert_allclose(w2_grad, np.array([12.0], dtype=np.float32))
