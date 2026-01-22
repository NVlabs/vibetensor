# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike


@cute.kernel
def _rope_kernel(
    x_tensor: cute.Tensor,
    cos_tensor: cute.Tensor,
    sin_tensor: cute.Tensor,
    positions_tensor: cute.Tensor,
    out_tensor: cute.Tensor,
    shape: cute.Shape,
):
    """CuTeDSL ROPE kernel (single tensor).

    Expects flattened tensors and a shape tuple ``(batch, seq_len, num_heads, head_dim)``.
    Rotates the last dimension of ``x_tensor`` into ``out_tensor`` using caller-provided
    ``cos``/``sin`` tables and integer ``positions``.
    """
    # Thread and block indices
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Unpack logical shape
    batch, seq_len, num_heads, head_dim = shape

    # Constants
    threads_per_block = 256

    # Global thread id across all CTAs
    global_tid = bidx * threads_per_block + tidx

    # Each head_dim is split into two halves to match Triton rotary semantics
    pairs_per_head = head_dim // 2
    total_pairs = batch * seq_len * num_heads * pairs_per_head

    if global_tid < total_pairs:
        # Decompose global thread ID into (batch, seq, head, pair)
        pair_idx = global_tid % pairs_per_head
        tmp = global_tid // pairs_per_head
        head_idx = tmp % num_heads
        tmp = tmp // num_heads
        seq_idx = tmp % seq_len
        batch_idx = tmp // seq_len

        # Compute base index into x/outs for this (batch, seq, head)
        x_base = (
            batch_idx * seq_len * num_heads * head_dim
            + seq_idx * num_heads * head_dim
            + head_idx * head_dim
        )

        # First half and second half of head_dim form the rotation pairs
        x0_idx = x_base + pairs_per_head + pair_idx
        x1_idx = x_base + pair_idx

        # Positions are laid out as (batch, seq_len, num_heads)
        row_idx = batch_idx * seq_len * num_heads + seq_idx * num_heads + head_idx
        pos = positions_tensor[row_idx]
        freq_idx = pos * pairs_per_head + pair_idx

        # Load inputs
        x0 = x_tensor[x0_idx]
        x1 = x_tensor[x1_idx]
        cos_val = cos_tensor[freq_idx]
        sin_val = sin_tensor[freq_idx]

        # Apply rotation with Triton-compatible sign convention
        # y1 = x1 * cos + x2 * sin
        # y2 = -x1 * sin + x2 * cos
        out0 = x0 * cos_val + x1 * sin_val
        out1 = -x0 * sin_val + x1 * cos_val

        # Store outputs
        out_tensor[x0_idx] = out0
        out_tensor[x1_idx] = out1


@cute.kernel
def _rope_qk_kernel_flat_pow2(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    cos_tensor: cute.Tensor,
    sin_tensor: cute.Tensor,
    positions_tensor: cute.Tensor,
    out_q_tensor: cute.Tensor,
    out_k_tensor: cute.Tensor,
    shape: cute.Shape,
    shift,
    mask,
):
    """CuTeDSL ROPE kernel optimized for power-of-2 head_dim."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    batch, seq_len, num_heads, head_dim = shape
    pairs_per_head = head_dim // 2
    total_pairs = batch * seq_len * num_heads * pairs_per_head

    threads_per_block = 256
    global_tid = bidx * threads_per_block + tidx

    if global_tid < total_pairs:
        pair_idx = global_tid & mask
        token_idx = global_tid >> shift

        base_idx = token_idx * head_dim
        pos_idx = token_idx

        x0_idx = base_idx + pair_idx
        x1_idx = base_idx + pairs_per_head + pair_idx

        pos = positions_tensor[pos_idx]
        freq_idx = pos * pairs_per_head + pair_idx

        cos_val = cos_tensor[freq_idx]
        sin_val = sin_tensor[freq_idx]

        q0 = q_tensor[x0_idx]
        q1 = q_tensor[x1_idx]
        q_out0 = q0 * cos_val + q1 * sin_val
        q_out1 = -q0 * sin_val + q1 * cos_val
        out_q_tensor[x0_idx] = q_out0
        out_q_tensor[x1_idx] = q_out1

        k0 = k_tensor[x0_idx]
        k1 = k_tensor[x1_idx]
        k_out0 = k0 * cos_val + k1 * sin_val
        k_out1 = -k0 * sin_val + k1 * cos_val
        out_k_tensor[x0_idx] = k_out0
        out_k_tensor[x1_idx] = k_out1


@cute.kernel
def _rope_qk_kernel_flat_generic(
    q_tensor: cute.Tensor,
    k_tensor: cute.Tensor,
    cos_tensor: cute.Tensor,
    sin_tensor: cute.Tensor,
    positions_tensor: cute.Tensor,
    out_q_tensor: cute.Tensor,
    out_k_tensor: cute.Tensor,
    shape: cute.Shape,
):
    """CuTeDSL ROPE kernel for generic head_dim."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    batch, seq_len, num_heads, head_dim = shape
    pairs_per_head = head_dim // 2
    total_pairs = batch * seq_len * num_heads * pairs_per_head

    threads_per_block = 256
    global_tid = bidx * threads_per_block + tidx

    if global_tid < total_pairs:
        pair_idx = global_tid % pairs_per_head
        token_idx = global_tid // pairs_per_head

        base_idx = token_idx * head_dim
        pos_idx = token_idx

        x0_idx = base_idx + pair_idx
        x1_idx = base_idx + pairs_per_head + pair_idx

        pos = positions_tensor[pos_idx]
        freq_idx = pos * pairs_per_head + pair_idx

        cos_val = cos_tensor[freq_idx]
        sin_val = sin_tensor[freq_idx]

        q0 = q_tensor[x0_idx]
        q1 = q_tensor[x1_idx]
        q_out0 = q0 * cos_val + q1 * sin_val
        q_out1 = -q0 * sin_val + q1 * cos_val
        out_q_tensor[x0_idx] = q_out0
        out_q_tensor[x1_idx] = q_out1

        k0 = k_tensor[x0_idx]
        k1 = k_tensor[x1_idx]
        k_out0 = k0 * cos_val + k1 * sin_val
        k_out1 = -k0 * sin_val + k1 * cos_val
        out_k_tensor[x0_idx] = k_out0
        out_k_tensor[x1_idx] = k_out1


@cute.jit
def _rope_function(
    x_tensor,
    cos_tensor,
    sin_tensor,
    positions_tensor,
    out_tensor,
    shape,
):
    """JIT wrapper for `_rope_kernel`."""
    batch, seq_len, num_heads, head_dim = shape
    pairs_per_head = head_dim // 2
    total_pairs = batch * seq_len * num_heads * pairs_per_head
    threads_per_block = 256
    blocks = (total_pairs + threads_per_block - 1) // threads_per_block

    _rope_kernel(
        x_tensor,
        cos_tensor,
        sin_tensor,
        positions_tensor,
        out_tensor,
        shape,
    ).launch(
        grid=[blocks, 1, 1],
        block=[threads_per_block, 1, 1],
    )


@cute.jit
def _rope_qk_function_flat_pow2(
    q_tensor,
    k_tensor,
    cos_tensor,
    sin_tensor,
    positions_tensor,
    out_q_tensor,
    out_k_tensor,
    shape,
    shift,
    mask,
):
    """JIT wrapper for `_rope_qk_kernel_flat_pow2`."""
    batch, seq_len, num_heads, head_dim = shape
    pairs_per_head = head_dim // 2
    total_pairs = batch * seq_len * num_heads * pairs_per_head
    threads_per_block = 256
    blocks = (total_pairs + threads_per_block - 1) // threads_per_block

    _rope_qk_kernel_flat_pow2(
        q_tensor,
        k_tensor,
        cos_tensor,
        sin_tensor,
        positions_tensor,
        out_q_tensor,
        out_k_tensor,
        shape,
        shift,
        mask,
    ).launch(
        grid=[blocks, 1, 1],
        block=[threads_per_block, 1, 1],
    )


@cute.jit
def _rope_qk_function_flat_generic(
    q_tensor,
    k_tensor,
    cos_tensor,
    sin_tensor,
    positions_tensor,
    out_q_tensor,
    out_k_tensor,
    shape,
):
    """JIT wrapper for `_rope_qk_kernel_flat_generic`."""
    batch, seq_len, num_heads, head_dim = shape
    pairs_per_head = head_dim // 2
    total_pairs = batch * seq_len * num_heads * pairs_per_head
    threads_per_block = 256
    blocks = (total_pairs + threads_per_block - 1) // threads_per_block

    _rope_qk_kernel_flat_generic(
        q_tensor,
        k_tensor,
        cos_tensor,
        sin_tensor,
        positions_tensor,
        out_q_tensor,
        out_k_tensor,
        shape,
    ).launch(
        grid=[blocks, 1, 1],
        block=[threads_per_block, 1, 1],
    )


def _validate_common_inputs(
    x: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    positions: TensorLike,
) -> Tuple[int, int, int, int, TensorLike, TensorLike, TensorLike]:
    """Shared validation logic for ROPE kernels.

    Returns (batch, seq_len, num_heads, head_dim, cos_2d, sin_2d, positions_int32).
    """
    if x.device.type != "cuda":  # pragma: no cover
        raise RuntimeError("cutedsl_rope requires CUDA tensors")
    if x.ndim != 4:
        raise ValueError("expected x of shape (batch, seq_len, num_heads, head_dim)")

    batch, seq_len, num_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for ROPE")

    head_dim_half = head_dim // 2

    cos_2d = cos.reshape(-1, cos.shape[-1]).contiguous()
    sin_2d = sin.reshape(-1, sin.shape[-1]).contiguous()
    if cos_2d.shape != sin_2d.shape:
        raise ValueError("cos and sin must have the same shape")
    if cos_2d.shape[1] != head_dim_half:
        raise ValueError("cos/sin last dimension must equal head_dim // 2")

    if positions.shape != (batch, seq_len, num_heads):
        raise ValueError(
            "positions must have shape (batch, seq_len, num_heads) to match x"
        )

    if positions.dtype not in (torch.int32, torch.int64):
        positions_int = positions.to(torch.int32)
    else:
        positions_int = positions.to(torch.int32)

    return batch, seq_len, num_heads, head_dim, cos_2d, sin_2d, positions_int


def _validate_bhsd_inputs(
    x: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    positions: TensorLike,
) -> Tuple[int, int, int, int, TensorLike, TensorLike, TensorLike]:
    """Shared validation logic for ROPE kernels (BHSD layout).

    Returns (batch, heads, seq_len, head_dim, cos_2d, sin_2d, positions_int32).
    """
    if x.device.type != "cuda":  # pragma: no cover
        raise RuntimeError("cutedsl_rope requires CUDA tensors")
    if x.ndim != 4:
        raise ValueError("expected x of shape (batch, heads, seq_len, head_dim)")

    batch, heads, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for ROPE")

    head_dim_half = head_dim // 2

    cos_2d = cos.reshape(-1, cos.shape[-1]).contiguous()
    sin_2d = sin.reshape(-1, sin.shape[-1]).contiguous()
    if cos_2d.shape != sin_2d.shape:
        raise ValueError("cos and sin must have the same shape")
    if cos_2d.shape[1] != head_dim_half:
        raise ValueError("cos/sin last dimension must equal head_dim // 2")

    if positions.shape != (batch, heads, seq_len):
        raise ValueError("positions must have shape (batch, heads, seq_len) to match x")

    if positions.dtype not in (torch.int32, torch.int64):
        positions_int = positions.to(torch.int32)
    else:
        positions_int = positions.to(torch.int32)

    return batch, heads, seq_len, head_dim, cos_2d, sin_2d, positions_int


def cutedsl_rope(
    x: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    positions: TensorLike,
):
    """Apply ROPE to ``x`` using CuTeDSL (no Quack dependency).

    Args:
        x: Tensor of shape ``(batch, seq_len, num_heads, head_dim)``, CUDA, fp16/bf16/fp32.
        cos: Precomputed cosine table. The last dimension must be ``head_dim // 2``;
            a common shape is ``(max_position, head_dim // 2)``.
        sin: Precomputed sine table with the same shape/dtype as ``cos``.
        positions: Integer tensor of shape ``(batch, seq_len, num_heads)`` giving the
            position index for each token into the first dimension of ``cos``/``sin``.

    Returns:
        Tensor of same shape/dtype as ``x`` with rotary embedding applied.
    """
    batch, seq_len, num_heads, head_dim, cos_2d, sin_2d, positions_int = (
        _validate_common_inputs(x, cos, sin, positions)
    )

    dtype = x.dtype

    # Flatten views for CuTeDSL kernel
    x_flat = x.contiguous().view(-1)
    cos_flat = cos_2d.contiguous().view(-1)
    sin_flat = sin_2d.contiguous().view(-1)
    positions_flat = positions_int.contiguous().view(-1)

    out_flat = alloc.empty_like(x_flat)

    # Ensure CUDA context is initialized
    cutlass.cuda.initialize_cuda_context()

    x_tensor = from_dlpack(x_flat).mark_layout_dynamic()
    cos_tensor = from_dlpack(cos_flat).mark_layout_dynamic()
    sin_tensor = from_dlpack(sin_flat).mark_layout_dynamic()
    pos_tensor = from_dlpack(positions_flat).mark_layout_dynamic()
    out_tensor = from_dlpack(out_flat).mark_layout_dynamic()

    shape = (batch, seq_len, num_heads, head_dim)
    compile_key = (dtype, *shape)

    if not hasattr(cutedsl_rope, "compile_cache"):
        cutedsl_rope.compile_cache = {}  # type: ignore[attr-defined]

    cache = cutedsl_rope.compile_cache  # type: ignore[attr-defined]
    if compile_key not in cache:
        cache[compile_key] = cute.compile(
            _rope_function,
            x_tensor,
            cos_tensor,
            sin_tensor,
            pos_tensor,
            out_tensor,
            shape,
        )

    cache[compile_key](
        x_tensor,
        cos_tensor,
        sin_tensor,
        pos_tensor,
        out_tensor,
        shape,
    )

    torch.cuda.synchronize()
    return out_flat.view(batch, seq_len, num_heads, head_dim)


def apply_rotary_cutedsl(
    q: TensorLike,
    k: TensorLike,
    cos: TensorLike,
    sin: TensorLike,
    positions: Optional[TensorLike],
) -> Tuple[TensorLike, TensorLike]:
    """Apply ROPE to query and key tensors in a single CuTeDSL kernel.

    Args:
        q, k: Tensors of shape ``(batch, heads, seq_len, head_dim)``.
        cos, sin, positions: As in :func:`cutedsl_rope`.

    Returns:
        Tuple ``(q_rot, k_rot)`` with the same shape/dtype as ``q``/``k``.
    """
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape for fused CuTeDSL ROPE")
    if q.dtype != k.dtype:
        raise ValueError("q and k must have the same dtype for fused CuTeDSL ROPE")

    batch, heads, seq_len, head_dim, cos_2d, sin_2d, positions_int = (
        _validate_bhsd_inputs(q, cos, sin, positions)
    )

    dtype = q.dtype

    # Tensors are expected to be contiguous BHSD.
    # If they came from PyTorch (B, H, S, D), they are usually contiguous.
    q_flat = q.contiguous().view(-1)
    k_flat = k.contiguous().view(-1)
    cos_flat = cos_2d.contiguous().view(-1)
    sin_flat = sin_2d.contiguous().view(-1)
    positions_flat = positions_int.contiguous().view(-1)

    out_q_flat = alloc.empty_like(q_flat)
    out_k_flat = alloc.empty_like(k_flat)

    cutlass.cuda.initialize_cuda_context()

    q_tensor = from_dlpack(q_flat).mark_layout_dynamic()
    k_tensor = from_dlpack(k_flat).mark_layout_dynamic()
    cos_tensor = from_dlpack(cos_flat).mark_layout_dynamic()
    sin_tensor = from_dlpack(sin_flat).mark_layout_dynamic()
    pos_tensor = from_dlpack(positions_flat).mark_layout_dynamic()
    out_q_tensor = from_dlpack(out_q_flat).mark_layout_dynamic()
    out_k_tensor = from_dlpack(out_k_flat).mark_layout_dynamic()

    shape = (batch, seq_len, heads, head_dim)
    compile_key = (dtype, *shape)

    # Calculate shift/mask if pairs_per_head is power of 2
    pairs_per_head = head_dim // 2
    if pairs_per_head > 0 and (pairs_per_head & (pairs_per_head - 1) == 0):
        shift = pairs_per_head.bit_length() - 1
        mask = pairs_per_head - 1
    else:
        shift = -1
        mask = -1

    if not hasattr(apply_rotary_cutedsl, "compile_cache"):
        apply_rotary_cutedsl.compile_cache = {}  # type: ignore[attr-defined]

    cache = apply_rotary_cutedsl.compile_cache  # type: ignore[attr-defined]

    if shift != -1:
        # Power of 2 path
        compile_key = (dtype, *shape, "pow2")
        if compile_key not in cache:
            cache[compile_key] = cute.compile(
                _rope_qk_function_flat_pow2,
                q_tensor,
                k_tensor,
                cos_tensor,
                sin_tensor,
                pos_tensor,
                out_q_tensor,
                out_k_tensor,
                shape,
                shift,
                mask,
            )
        cache[compile_key](
            q_tensor,
            k_tensor,
            cos_tensor,
            sin_tensor,
            pos_tensor,
            out_q_tensor,
            out_k_tensor,
            shape,
            shift,
            mask,
        )
    else:
        # Generic path
        compile_key = (dtype, *shape, "gen")
        if compile_key not in cache:
            cache[compile_key] = cute.compile(
                _rope_qk_function_flat_generic,
                q_tensor,
                k_tensor,
                cos_tensor,
                sin_tensor,
                pos_tensor,
                out_q_tensor,
                out_k_tensor,
                shape,
            )
        cache[compile_key](
            q_tensor,
            k_tensor,
            cos_tensor,
            sin_tensor,
            pos_tensor,
            out_q_tensor,
            out_k_tensor,
            shape,
        )

    torch.cuda.synchronize()
    q_out = out_q_flat.view(batch, heads, seq_len, head_dim)
    k_out = out_k_flat.view(batch, heads, seq_len, head_dim)
    return q_out, k_out


__all__ = ["cutedsl_rope", "apply_rotary_cutedsl"]
