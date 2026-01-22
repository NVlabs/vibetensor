# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash Attention CUTE (CUDA Template Engine) implementation."""

__version__ = "0.1.0"

import cutlass.cute as cute
from vibe_attention.cute_dsl_utils import cute_compile_patched

from .interface import flash_attn_func, flash_attn_varlen_func

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
