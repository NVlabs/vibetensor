# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from vibe_kernels.attention import KVCache, reshape_for_gqa, restore_from_gqa

pytestmark = pytest.mark.skipif(  # type: ignore[assignment]
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_reshape_for_gqa_roundtrip():
    torch.manual_seed(0)
    batch, hq, hk, seqlen, dim = 2, 6, 3, 5, 8
    q = torch.randn(batch, hq, seqlen, dim, device="cuda")
    k = torch.randn(batch, hk, seqlen, dim, device="cuda")
    v = torch.randn_like(k)

    q_gqa, k_gqa, v_gqa, group = reshape_for_gqa(q, k, v)
    assert group == hq // hk
    assert q_gqa.shape == (batch * group, hk, seqlen, dim)

    q_back = restore_from_gqa(q_gqa, group, hk)
    assert torch.allclose(q_back, q)


def test_kv_cache_insert_clone():
    cache = KVCache(num_layers=2, num_heads=4, head_dim=16, max_tokens=32)
    k = torch.randn(1, 4, 10, 16, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    keys, values = cache.insert(0, k, v)
    assert keys.shape[-2] == 10 and cache.pos == 10

    clone = cache.clone_for_batch(batch_size=3)
    assert clone.cache.shape[2] == 3
    assert torch.allclose(clone.cache[:, :, 0], cache.cache)
    clone.reset()
    assert clone.pos == 0
