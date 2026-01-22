# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F

Reduction = Literal["none", "mean", "byte_mean"]


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -1,
    reduction: Reduction = "mean",
    token_bytes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions")

    vocab_size = logits.shape[-1]
    logits_2d = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # F.cross_entropy handles ignore_index
    losses = F.cross_entropy(
        logits_2d, targets_flat, ignore_index=ignore_index, reduction="none"
    )

    # Reshape back to target shape
    losses = losses.view(targets.shape)

    if reduction == "none":
        return losses

    mask = targets != ignore_index
    valid_count = mask.sum()

    if reduction == "mean":
        if valid_count == 0:
            return losses.sum() * 0
        return losses.sum() / valid_count

    if reduction == "byte_mean":
        if token_bytes is None:
            raise ValueError("reduction='byte_mean' requires token_bytes")

        # Ensure token_bytes is on correct device
        token_bytes = token_bytes.to(device=logits.device)

        # Safe gathering
        targets_clamped = targets.clone()
        targets_clamped[~mask] = 0

        # Gather bytes per token
        # token_bytes is (vocab_size,)
        bytes_per_token = F.embedding(
            targets_clamped, token_bytes.unsqueeze(1)
        ).squeeze(-1)

        # Apply mask
        bytes_per_token = bytes_per_token * mask

        total_bytes = bytes_per_token.sum()
        if total_bytes == 0:
            return losses.sum() * 0

        weighted_loss = (losses * bytes_per_token).sum()
        return weighted_loss / total_bytes

    raise ValueError(f"Unsupported reduction: {reduction}")
