# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from vibe_kernels.sampling import sample_logits
from vibe_kernels.sampling.kernel import _select_topk  # type: ignore[attr-defined]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize(
    "batch,vocab,top_k",
    [
        (2, 17, 5),
        (4, 320, 32),
        (1, 32000, 40),
        (3, 655, 128),
    ],
)
def test_triton_topk_matches_torch(batch: int, vocab: int, top_k: int) -> None:
    torch.manual_seed(0)
    logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)
    triton_vals, triton_idx = _select_topk(logits, top_k)
    ref_vals, ref_idx = torch.topk(logits, top_k, dim=-1)

    assert torch.allclose(triton_vals, ref_vals.to(torch.float32), atol=1e-5, rtol=0)
    assert torch.equal(triton_idx, ref_idx.to(torch.int32))


def test_sample_logits_topk_matches_reference() -> None:
    torch.manual_seed(0)
    batch, vocab = 3, 17
    logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)
    top_k = 5
    generator = torch.Generator(device="cuda").manual_seed(123)
    samples = sample_logits(logits, top_k=top_k, temperature=1.0, generator=generator)

    values, indices = torch.topk(logits, top_k, dim=-1)
    probs = F.softmax(values, dim=-1)
    for row in range(batch):
        idx = samples[row].item()
        assert idx in indices[row].tolist()

    # Compare distribution statistics across many samples
    repeats = 1024
    tiled_logits = logits[0:1].expand(repeats, -1).contiguous()
    generator = torch.Generator(device="cuda").manual_seed(456)
    draws = sample_logits(
        tiled_logits, top_k=top_k, temperature=1.0, generator=generator
    )
    counts = torch.bincount(draws, minlength=vocab).float()
    expected = torch.zeros(vocab, device="cuda")
    expected[indices[0]] = probs[0]
    freq = counts / counts.sum()
    assert torch.allclose(freq, expected, atol=0.03, rtol=0)


def test_sample_logits_without_topk_matches_softmax_distribution() -> None:
    torch.manual_seed(1)
    vocab = 9
    logits = torch.arange(vocab, device="cuda", dtype=torch.float32).unsqueeze(0)
    repeats = 2048
    tiled = logits.expand(repeats, -1).contiguous()
    generator = torch.Generator(device="cuda").manual_seed(999)
    draws = sample_logits(tiled, temperature=1.0, generator=generator)
    counts = torch.bincount(draws, minlength=vocab).float()
    freq = counts / counts.sum()
    expected = F.softmax(logits.squeeze(0), dim=-1)
    assert torch.allclose(freq.cpu(), expected.cpu(), atol=0.02, rtol=0)


def test_sample_logits_multisample_distribution_matches_torch() -> None:
    torch.manual_seed(2)
    vocab = 11
    repeats = 4096
    num_samples = 3
    logits = torch.randn(1, vocab, device="cuda", dtype=torch.float32)
    temperature = 0.85
    tiled = logits.expand(repeats, -1).contiguous()

    generator = torch.Generator(device="cuda").manual_seed(123)
    ours = sample_logits(
        tiled,
        temperature=temperature,
        generator=generator,
        num_samples=num_samples,
    )
    counts = torch.bincount(ours.view(-1).to(torch.long), minlength=vocab).float()

    generator_ref = torch.Generator(device="cuda").manual_seed(123)
    probs = F.softmax((logits / temperature), dim=-1).expand(repeats, -1)
    baseline = torch.multinomial(
        probs,
        num_samples=num_samples,
        replacement=True,
        generator=generator_ref,
    )
    baseline_counts = torch.bincount(
        baseline.view(-1).to(torch.long), minlength=vocab
    ).float()

    freq = counts / counts.sum()
    ref_freq = baseline_counts / baseline_counts.sum()
    assert torch.allclose(freq, ref_freq, atol=0.02, rtol=0)


def test_sample_logits_multisample_topk_distribution_matches_torch() -> None:
    torch.manual_seed(3)
    batch, vocab, top_k, num_samples = 2, 64, 8, 4
    logits = torch.randn(batch, vocab, device="cuda", dtype=torch.float32)
    temperature = 0.9

    generator = torch.Generator(device="cuda").manual_seed(321)
    samples = sample_logits(
        logits,
        top_k=top_k,
        temperature=temperature,
        generator=generator,
        num_samples=num_samples,
    )
    assert samples.shape == (batch, num_samples)

    values, indices = torch.topk(logits, top_k, dim=-1)
    membership = (samples.unsqueeze(-1) == indices.unsqueeze(1)).any(dim=-1)
    assert torch.all(membership)

    repeats = 2048
    row_logits = logits[:1].expand(repeats, -1).contiguous()
    generator2 = torch.Generator(device="cuda").manual_seed(654)
    ours = sample_logits(
        row_logits,
        top_k=top_k,
        temperature=temperature,
        generator=generator2,
        num_samples=num_samples,
    )
    ours_counts = torch.bincount(ours.view(-1), minlength=vocab).float()

    generator_ref = torch.Generator(device="cuda").manual_seed(654)
    top_vals, top_idx = torch.topk(row_logits, top_k, dim=-1)
    probs = F.softmax(top_vals / temperature, dim=-1)
    baseline = torch.multinomial(
        probs,
        num_samples=num_samples,
        replacement=True,
        generator=generator_ref,
    )
    mapped = top_idx.gather(-1, baseline)
    baseline_counts = torch.bincount(mapped.view(-1), minlength=vocab).float()

    ours_freq = ours_counts / ours_counts.sum()
    ref_freq = baseline_counts / baseline_counts.sum()
    compare_idx = top_idx[0]
    assert torch.allclose(
        ours_freq[compare_idx], ref_freq[compare_idx], atol=0.03, rtol=0
    )


def test_temperature_zero_returns_argmax() -> None:
    torch.manual_seed(2)
    logits = torch.randn(6, 13, device="cuda")
    samples = sample_logits(logits, temperature=0.0)
    assert torch.all(samples == torch.argmax(logits, dim=-1))
