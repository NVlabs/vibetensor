# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, TYPE_CHECKING

from vibe_kernels.common.tensor_types import TensorLike

if TYPE_CHECKING:
    import torch


def reshape_for_gqa(
    q: TensorLike,
    k: TensorLike,
    v: TensorLike,
) -> Tuple[TensorLike, TensorLike, TensorLike, int]:
    batch, heads_q, seqlen, dim = q.shape
    _, heads_k, _, _ = k.shape
    assert heads_q % heads_k == 0, "Query heads must be a multiple of key/value heads"
    group = heads_q // heads_k
    if group == 1:
        return q.contiguous(), k.contiguous(), v.contiguous(), group
    q_reshaped = (
        q.view(batch, heads_k, group, seqlen, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * group, heads_k, seqlen, dim)
        .contiguous()
    )
    k_reshaped = (
        k.unsqueeze(1)
        .repeat(1, group, 1, 1, 1)
        .reshape(batch * group, heads_k, seqlen, dim)
        .contiguous()
    )
    v_reshaped = (
        v.unsqueeze(1)
        .repeat(1, group, 1, 1, 1)
        .reshape(batch * group, heads_k, seqlen, dim)
        .contiguous()
    )
    return q_reshaped, k_reshaped, v_reshaped, group


def restore_from_gqa(
    tensor: TensorLike,
    group_size: int,
    kv_heads: int,
) -> TensorLike:
    if group_size == 1:
        return tensor
    batch_group, _, seqlen, dim = tensor.shape
    batch = batch_group // group_size
    return (
        tensor.view(batch, group_size, kv_heads, seqlen, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, kv_heads * group_size, seqlen, dim)
        .contiguous()
    )


@dataclass
class KVCache:
    num_layers: int
    num_heads: int
    head_dim: int
    max_tokens: int
    device: Any = None

    def __post_init__(self) -> None:
        import torch as _torch
        device = self.device if self.device is not None else _torch.device("cuda")
        shape = (self.num_layers, 2, self.num_heads, self.max_tokens, self.head_dim)
        self.cache = _torch.zeros(shape, device=device, dtype=_torch.float16)
        self.pos = 0

    def reset(self) -> None:
        self.pos = 0

    def insert(
        self, layer_idx: int, k: TensorLike, v: TensorLike
    ) -> Tuple[TensorLike, TensorLike]:
        assert k.shape == v.shape
        _, heads, tokens, dim = k.shape
        assert heads == self.num_heads and dim == self.head_dim
        t0, t1 = self.pos, self.pos + tokens
        if t1 > self.max_tokens:
            raise RuntimeError("KV cache capacity exceeded")
        self.cache[layer_idx, 0, :, t0:t1] = k.squeeze(0)
        self.cache[layer_idx, 1, :, t0:t1] = v.squeeze(0)
        self.pos = t1
        return (
            self.cache[layer_idx, 0, :, : self.pos],
            self.cache[layer_idx, 1, :, : self.pos],
        )

    def clone_for_batch(self, batch_size: int) -> "KVCache":
        cloned = KVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_tokens=self.max_tokens,
            device=self.cache.device,
        )
        cloned.cache = (
            self.cache.unsqueeze(2).repeat(1, 1, batch_size, 1, 1, 1).contiguous()
        )
        cloned.pos = self.pos
        return cloned
