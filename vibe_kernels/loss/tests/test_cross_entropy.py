# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F
from vibe_kernels.loss import cross_entropy_loss

BACKENDS = ["triton", "cutedsl"]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("reduction", ["mean", "none"])
def test_cross_entropy_matches_torch(
    dtype: torch.dtype, reduction: str, backend: str
) -> None:
    if backend == "cutedsl":
        from vibe_kernels.loss import is_cutedsl_available

        if not is_cutedsl_available():
            pytest.skip("CuTeDSL cross entropy backend is not available")
    torch.manual_seed(0)
    batch, vocab = 8, 97
    logits = torch.randn(batch, vocab, device="cuda", dtype=dtype, requires_grad=True)
    targets = torch.randint(0, vocab, (batch,), device="cuda", dtype=torch.long)
    targets[::3] = -1  # inject ignore_index positions

    logits_ref = logits.detach().clone().to(torch.float32).requires_grad_(True)

    ours = cross_entropy_loss(
        logits, targets, ignore_index=-1, reduction=reduction, backend=backend
    )
    ref = F.cross_entropy(logits_ref, targets, ignore_index=-1, reduction=reduction)

    if reduction == "none":
        assert ours.shape == targets.shape
        assert torch.allclose(ours, ref.view_as(ours), atol=5e-3, rtol=0)
        grad = torch.randn_like(ours)
        ours.backward(grad)
        ref.backward(grad)
    else:
        assert math.isclose(ours.item(), ref.item(), rel_tol=0, abs_tol=5e-3)
        ours.backward()
        ref.backward()

    assert logits.grad is not None
    assert logits_ref.grad is not None
    assert torch.allclose(logits.grad, logits_ref.grad.to(dtype), atol=5e-3, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cross_entropy_all_ignored_has_zero_gradient(backend: str) -> None:
    if backend == "cutedsl":
        from vibe_kernels.loss import is_cutedsl_available

        if not is_cutedsl_available():
            pytest.skip("CuTeDSL cross entropy backend is not available")
    torch.manual_seed(1)
    logits = torch.randn(4, 13, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.full((4,), -1, device="cuda", dtype=torch.long)

    loss = cross_entropy_loss(
        logits, targets, ignore_index=-1, reduction="mean", backend=backend
    )
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cross_entropy_byte_mean_matches_manual(
    dtype: torch.dtype, backend: str
) -> None:
    if backend == "cutedsl":
        from vibe_kernels.loss import is_cutedsl_available

        if not is_cutedsl_available():
            pytest.skip("CuTeDSL cross entropy backend is not available")
    torch.manual_seed(2)
    batch, vocab = 6, 19
    logits = torch.randn(batch, vocab, device="cuda", dtype=dtype, requires_grad=True)
    targets = torch.randint(-1, vocab, (batch,), device="cuda", dtype=torch.long)
    targets[::3] = -1

    token_bytes = (torch.arange(vocab, device="cuda") % 4).to(torch.int32)
    token_bytes[0] = 0
    targets_clamped = torch.clamp(targets, min=0)
    if (token_bytes[targets_clamped] > 0).sum() == 0:
        token_bytes[targets_clamped[0]] = 1

    loss = cross_entropy_loss(
        logits,
        targets,
        ignore_index=-1,
        reduction="byte_mean",
        token_bytes=token_bytes,
        backend=backend,
    )

    logits_ref = logits.detach().clone().to(torch.float32).requires_grad_(True)
    bytes_float = token_bytes.to(torch.float32)
    byte_weights = torch.where(
        targets >= 0,
        bytes_float[targets_clamped],
        torch.zeros_like(bytes_float[targets_clamped]),
    )
    ce = F.cross_entropy(logits_ref, targets, ignore_index=-1, reduction="none")
    denominator = byte_weights.sum()
    if denominator.item() == 0:
        manual = ce.new_zeros(())
    else:
        manual = (ce * byte_weights).sum() / denominator

    assert torch.allclose(loss.float(), manual, atol=5e-3, rtol=0)

    loss.backward()
    manual.backward()
    assert logits.grad is not None
    assert logits_ref.grad is not None
    assert torch.allclose(
        logits.grad.float(), logits_ref.grad.to(dtype).float(), atol=5e-3, rtol=0
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_cross_entropy_byte_mean_all_zero_bytes(backend: str) -> None:
    if backend == "cutedsl":
        from vibe_kernels.loss import is_cutedsl_available

        if not is_cutedsl_available():
            pytest.skip("CuTeDSL cross entropy backend is not available")
    torch.manual_seed(3)
    logits = torch.randn(4, 11, device="cuda", dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 11, (4,), device="cuda", dtype=torch.long)
    token_bytes = torch.zeros(11, device="cuda", dtype=torch.int32)

    loss = cross_entropy_loss(
        logits,
        targets,
        ignore_index=-1,
        reduction="byte_mean",
        token_bytes=token_bytes,
        backend=backend,
    )
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits))
