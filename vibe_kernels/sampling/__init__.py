# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampling utilities implemented with Triton."""

from .kernel import sample_logits  # type: ignore[import]

__all__ = ["sample_logits"]
