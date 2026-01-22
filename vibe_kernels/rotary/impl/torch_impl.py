# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import torch


def apply_rotary_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seqlen, head_dim = q.shape
    head_dim_half = head_dim // 2

    if positions is None:
        positions = (
            torch.arange(seqlen, device=q.device)
            .view(1, 1, seqlen)
            .expand(batch, heads, seqlen)
        )

    positions = positions.long()

    # Gather cos/sin: (B, H, S, D/2)
    cos_expanded = cos[positions]
    sin_expanded = sin[positions]

    q1 = q[..., :head_dim_half]
    q2 = q[..., head_dim_half:]
    k1 = k[..., :head_dim_half]
    k2 = k[..., head_dim_half:]

    # q1*cos + q2*sin
    q_out1 = q1 * cos_expanded + q2 * sin_expanded
    # -q1*sin + q2*cos
    q_out2 = -q1 * sin_expanded + q2 * cos_expanded

    # Same for k
    k_out1 = k1 * cos_expanded + k2 * sin_expanded
    k_out2 = -k1 * sin_expanded + k2 * cos_expanded

    q_out = torch.cat([q_out1, q_out2], dim=-1)
    k_out = torch.cat([k_out1, k_out2], dim=-1)

    return q_out, k_out
