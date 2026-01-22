# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from typing import Literal, Optional, Tuple

import torch  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

from . import cute_topk as _cute_topk


_MAX_TOP_K = 256
_STREAM_CHUNK_SIZE = 1024
_MAX_STREAM_CHUNKS = 128
_MAX_STAGE2_CANDIDATES = 8192
_MAX_SAMPLES = 16

# Environment overrides used for tuning:
# - AIKF_MAX_STREAM_CHUNKS: caps how many streaming chunks per row are used in stage 1
# - AIKF_STAGE1_WARPS / AIKF_STAGE1_STAGES: launch configuration for the stage-1 kernel
# - AIKF_STAGE2_TILE / AIKF_STAGE2_WARPS / AIKF_STAGE2_STAGES: retained for compatibility,
#   and configure the second-stage Triton reducer


@triton.jit
def _fpval_to_key(x):
    top_mask: tl.constexpr = 1 << 31
    full_mask: tl.constexpr = (1 << 32) - 1
    tm = tl.full(x.shape, top_mask, dtype=x.dtype)
    fm = tl.full(x.shape, full_mask, dtype=x.dtype)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def _key_to_fpval(x):
    top_mask: tl.constexpr = 1 << 31
    full_mask: tl.constexpr = (1 << 32) - 1
    tm = tl.full(x.shape, top_mask, dtype=x.dtype)
    fm = tl.full(x.shape, full_mask, dtype=x.dtype)
    return x ^ tl.where((x & tm) == 0, fm, tm)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit
def _topk_stage1_kernel(
    out_vals_ptr,
    out_idx_ptr,
    logits_ptr,
    stride_out_vals,
    stride_out_idx,
    stride_logits_m,
    stride_logits_n,
    rows,
    cols,
    num_chunks,
    K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if row >= rows or chunk >= num_chunks:
        return

    row_logits = logits_ptr + row * stride_logits_m
    out_vals_row = out_vals_ptr + row * stride_out_vals + chunk * K
    out_idx_row = out_idx_ptr + row * stride_out_idx + chunk * K

    chunk_start = chunk * CHUNK_SIZE
    offsets = tl.arange(0, CHUNK_SIZE)
    absolute_offsets = chunk_start + offsets
    mask = absolute_offsets < cols

    vals = tl.load(
        row_logits + absolute_offsets * stride_logits_n,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)

    for i in tl.static_range(0, K):
        max_val, rel_idx = tl.max(vals, axis=0, return_indices=True)
        valid = max_val != -float("inf")
        global_idx = tl.where(valid, (chunk_start + rel_idx).to(tl.int32), -1)
        tl.store(out_vals_row + i, tl.where(valid, max_val, -float("inf")))
        tl.store(out_idx_row + i, global_idx)
        selected = offsets == rel_idx
        vals = tl.where(selected, -float("inf"), vals)


@triton.jit
def _topk_stage2_kernel(
    out_vals_ptr,
    out_idx_ptr,
    stage_vals_ptr,
    stage_idx_ptr,
    stride_out_vals,
    stride_out_idx,
    stride_stage_vals,
    stride_stage_idx,
    rows,
    total_candidates,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    out_vals_row = out_vals_ptr + row * stride_out_vals
    out_idx_row = out_idx_ptr + row * stride_out_idx
    stage_vals_row = stage_vals_ptr + row * stride_stage_vals
    stage_idx_row = stage_idx_ptr + row * stride_stage_idx

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_candidates

    vals = tl.load(
        stage_vals_row + offsets,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".cg",
    ).to(tl.float32)
    idxs = tl.load(
        stage_idx_row + offsets,
        mask=mask,
        other=-1,
        cache_modifier=".cg",
    ).to(tl.int32)

    neg_inf = -float("inf")
    for i in tl.static_range(0, K):
        max_val, rel_idx = tl.max(vals, axis=0, return_indices=True)
        has = max_val != neg_inf
        selected = offsets == rel_idx
        chosen_idx = tl.max(tl.where(selected, idxs, -1), axis=0).to(tl.int32)
        chosen_idx = tl.where(has, chosen_idx, -1)
        chosen_val = tl.where(has, max_val, neg_inf)
        tl.store(out_vals_row + i, chosen_val)
        tl.store(out_idx_row + i, chosen_idx)
        vals = tl.where(selected, neg_inf, vals)
        idxs = tl.where(selected, -1, idxs)


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _compute_vocab_bits(vocab_size: int) -> int:
    """Compute number of bits needed to represent vocab indices."""
    if vocab_size <= 1:
        return 1
    return (vocab_size - 1).bit_length()


def _select_topk(logits: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Avoid host-side dtype conversion; cast to fp32 inside kernels after load
    logits_x = logits.contiguous()
    rows, cols = logits_x.shape

    if top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be <= {_MAX_TOP_K}, got {top_k}")

    if cols == 0:
        raise ValueError("logits must have non-zero vocabulary dimension")

    if rows == 0:
        device = logits.device
        return (
            torch.empty((0, top_k), device=device, dtype=torch.float32),
            torch.empty((0, top_k), device=device, dtype=torch.int32),
        )

    chunk_size = _STREAM_CHUNK_SIZE
    num_chunks = math.ceil(cols / chunk_size)

    max_stream_chunks = _get_env_int("AIKF_MAX_STREAM_CHUNKS", _MAX_STREAM_CHUNKS)

    # Heuristic A: target candidate pool near ~500 (tunable)
    target_cand = _get_env_int("AIKF_TARGET_STAGE2_CANDIDATES", 500)
    cs_from_cand = chunk_size
    if target_cand > 0:
        desired_chunks = max(1, int(round(target_cand / max(1, top_k))))
        desired_chunks = min(desired_chunks, max_stream_chunks)
        cand_chunk = max(top_k, math.ceil(cols / max(1, desired_chunks)))
        cand_chunk = min(cand_chunk, cols)
        cand_chunk = 1 << (cand_chunk - 1).bit_length()
        cs_from_cand = cand_chunk

    # Heuristic B: ensure enough parallelism for small batches (target CTAs)
    target_ctas = _get_env_int("AIKF_STAGE1_TARGET_CTAS", 128)
    desired_chunks_parallel = max(1, math.ceil(target_ctas / max(1, rows)))
    desired_chunks_parallel = min(desired_chunks_parallel, max_stream_chunks)
    par_chunk = max(top_k, math.ceil(cols / desired_chunks_parallel))
    par_chunk = min(par_chunk, cols)
    par_chunk = 1 << (par_chunk - 1).bit_length()

    # Default: prioritize candidate cap heuristic; optionally favor parallelism via env
    prefer_parallel = _get_env_int("AIKF_STAGE1_PARALLEL_HINT", 0) != 0
    chunk_size = min(cs_from_cand, par_chunk) if prefer_parallel else cs_from_cand
    num_chunks = math.ceil(cols / chunk_size)

    # Enforce stream chunk cap
    if num_chunks > max_stream_chunks:
        chunk_size = max(top_k, math.ceil(cols / max_stream_chunks))
        chunk_size = min(chunk_size, cols)
        chunk_size = 1 << (chunk_size - 1).bit_length()
        num_chunks = math.ceil(cols / chunk_size)

    # Cap total stage-2 candidates
    max_candidates = num_chunks * top_k
    if max_candidates > _MAX_STAGE2_CANDIDATES:
        max_chunks = max(1, _MAX_STAGE2_CANDIDATES // top_k)
        chunk_size = max(top_k, math.ceil(cols / max_chunks))
        chunk_size = 1 << (chunk_size - 1).bit_length()
        num_chunks = math.ceil(cols / chunk_size)
        max_candidates = num_chunks * top_k

    # If still exceeding, grow chunk size until under cap
    while num_chunks * top_k > _MAX_STAGE2_CANDIDATES:
        next_chunk_size = 1 << chunk_size.bit_length()
        if next_chunk_size == chunk_size:
            next_chunk_size *= 2
        chunk_size = min(next_chunk_size, cols)
        num_chunks = math.ceil(cols / chunk_size)

    max_candidates = num_chunks * top_k
    block_size = _next_power_of_two(max_candidates)
    device = logits.device

    # Optional alternate implementations via AIKF_TOPK_IMPL: torch | singlepass | streaming | stream(default)
    topk_impl = os.environ.get("AIKF_TOPK_IMPL", "stream")

    if topk_impl == "torch":
        vals, idx = torch.topk(logits_x, top_k, dim=-1)
        return vals.to(torch.float32).contiguous(), idx.to(torch.int32).contiguous()

    if topk_impl == "streaming":
        out_vals = torch.empty((rows, top_k), device=device, dtype=torch.float32)
        out_idx = torch.empty((rows, top_k), device=device, dtype=torch.int32)

        # Configure block size
        block_n = _get_env_int("AIKF_STREAMING_BLOCK_N", 128)
        block_n = min(block_n, 256)
        block_n = max(block_n, 64)

        # Pad K to next power of 2 for Triton requirements
        k_padded = _next_power_of_two(top_k)

        grid = (rows,)
        num_warps = _get_env_int("AIKF_STREAMING_WARPS", 4)

        _streaming_topk_kernel[grid](
            out_vals,
            out_idx,
            logits_x,
            out_vals.stride(0),
            out_idx.stride(0),
            logits_x.stride(0),
            logits_x.stride(1),
            rows,
            cols,
            K=top_k,
            K_PADDED=k_padded,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        return out_vals.contiguous(), out_idx.contiguous()

    if topk_impl == "singlepass":
        out_vals = torch.empty((rows, top_k), device=device, dtype=torch.float32)
        out_idx = torch.empty((rows, top_k), device=device, dtype=torch.int32)

        tile_size = _get_env_int("AIKF_SINGLEPASS_TILE", 256)
        tile_size = 1 << (tile_size - 1).bit_length()
        tile_size = min(tile_size, _next_power_of_two(cols))

        acc_mult = _get_env_int("AIKF_SINGLEPASS_ACC_MULT", 0)
        if acc_mult <= 0:
            if top_k <= 64:
                acc_mult = 4
            elif top_k <= 128:
                acc_mult = 3
            else:
                acc_mult = 2
        acc_size = _next_power_of_two(min(2048, acc_mult * top_k))

        grid_sp = (rows,)
        if tile_size >= 2048:
            sp_warps = 8
        elif tile_size >= 512:
            sp_warps = 8
        elif tile_size >= 128:
            sp_warps = 4
        else:
            sp_warps = 2
        sp_warps = _get_env_int("AIKF_SINGLEPASS_WARPS", sp_warps)
        sp_stages = _get_env_int("AIKF_SINGLEPASS_STAGES", 2)

        _topk_singlepass_tiles_kernel[grid_sp](  # type: ignore[misc]
            out_vals,
            out_idx,
            logits_x,
            out_vals.stride(0),
            out_idx.stride(0),
            logits_x.stride(0),
            logits_x.stride(1),
            rows,
            cols,
            K=top_k,
            TILE_SIZE=tile_size,
            ACC_SIZE=acc_size,
            num_warps=sp_warps,
            num_stages=sp_stages,
        )
        return out_vals.contiguous(), out_idx.contiguous()

    stage_vals = torch.full(
        (rows, block_size), -float("inf"), device=device, dtype=logits_x.dtype
    )
    stage_idx = torch.full((rows, block_size), -1, device=device, dtype=torch.int32)

    grid_stage1 = (rows, num_chunks)
    # Heuristic: for small rows (nanochat), fewer warps can be better at huge chunks
    if chunk_size >= 8192:
        stage1_warps = 16
    elif chunk_size >= 2048:
        stage1_warps = 8
    elif chunk_size >= 256:
        stage1_warps = 4
    else:
        stage1_warps = 2
    stage1_warps = _get_env_int("AIKF_STAGE1_WARPS", stage1_warps)
    stage1_stages = _get_env_int("AIKF_STAGE1_STAGES", 2)
    _topk_stage1_kernel[grid_stage1](  # type: ignore[misc]
        stage_vals,
        stage_idx,
        logits_x,
        stage_vals.stride(0),
        stage_idx.stride(0),
        logits_x.stride(0),
        logits_x.stride(1),
        rows,
        cols,
        num_chunks,
        K=top_k,
        CHUNK_SIZE=chunk_size,
        num_warps=stage1_warps,
        num_stages=stage1_stages,
    )

    out_vals = torch.empty((rows, top_k), device=device, dtype=torch.float32)
    out_idx = torch.empty((rows, top_k), device=device, dtype=torch.int32)

    grid_stage2 = (rows,)
    # Favor more warps once block_size >= 512 (empirically faster on nanochat)
    if block_size >= 2048:
        stage2_warps = 8
    elif block_size >= 512:
        stage2_warps = 8
    elif block_size >= 128:
        stage2_warps = 4
    else:
        stage2_warps = 2
    stage2_warps = _get_env_int("AIKF_STAGE2_WARPS", stage2_warps)
    stage2_stages = _get_env_int("AIKF_STAGE2_STAGES", 2)

    impl = os.environ.get("AIKF_STAGE2_IMPL", "tiles")
    if impl == "tiles":
        tile_size = _get_env_int("AIKF_STAGE2_TILE", 0)
        if tile_size <= 0:
            tile_size = 256 if max_candidates >= 1024 else 128
        tile_size = 1 << (tile_size - 1).bit_length()
        tile_size = min(tile_size, block_size)

        acc_mult = _get_env_int("AIKF_STAGE2_ACC_MULT", 0)
        if acc_mult <= 0:
            if top_k <= 64:
                acc_mult = 4
            elif top_k <= 128:
                acc_mult = 3
            else:
                acc_mult = 2
        acc_size = _next_power_of_two(min(2048, acc_mult * top_k))

        _topk_stage2_tiles_kernel[grid_stage2](  # type: ignore[misc]
            out_vals,
            out_idx,
            stage_vals,
            stage_idx,
            out_vals.stride(0),
            out_idx.stride(0),
            stage_vals.stride(0),
            stage_idx.stride(0),
            rows,
            max_candidates,
            K=top_k,
            TILE_SIZE=tile_size,
            ACC_SIZE=acc_size,
            num_warps=stage2_warps,
            num_stages=stage2_stages,
        )
    else:
        _topk_stage2_kernel[grid_stage2](  # type: ignore[misc]
            out_vals,
            out_idx,
            stage_vals,
            stage_idx,
            out_vals.stride(0),
            out_idx.stride(0),
            stage_vals.stride(0),
            stage_idx.stride(0),
            rows,
            max_candidates,
            K=top_k,
            BLOCK_SIZE=block_size,
            num_warps=stage2_warps,
            num_stages=stage2_stages,
        )

    return out_vals.contiguous(), out_idx.contiguous()


@triton.jit
def _sample_from_logits(
    out_ptr,
    logits_ptr,
    indices_ptr,
    rand_ptr,
    rows,
    cols,
    stride_logits,
    stride_indices,
    HAS_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    row_offset_logits = row * stride_logits
    row_offset_indices = row * stride_indices
    offsets = tl.arange(0, BLOCK_SIZE)

    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(
            logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf")
        )
        vals = vals.to(tl.float32)
        block_max = tl.max(tl.where(mask, vals, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(
            logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf")
        )
        vals = vals.to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    threshold = tl.load(rand_ptr + row) * sum_exp
    cumulative = tl.zeros([1], dtype=tl.float32)
    chosen = tl.full([1], -1, dtype=tl.int32)

    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(
            logits_ptr + row_offset_logits + idx, mask=mask, other=-float("inf")
        )
        vals = vals.to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        cumsum = tl.cumsum(exp_vals, axis=0)
        block_cum = cumulative + cumsum
        candidates = (block_cum >= threshold) & mask
        valid_candidates = candidates & (chosen == -1)
        num_valid = tl.sum(valid_candidates.to(tl.float32), axis=0)
        if num_valid > 0:
            idx_candidates = tl.where(valid_candidates, idx, cols)
            pos = tl.min(idx_candidates, axis=0).to(tl.int32)
            chosen = tl.where(chosen == -1, pos, chosen)
        cumulative += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    chosen = tl.where(chosen == -1, (cols - 1).to(tl.int32), chosen)
    chosen_scalar = tl.sum(chosen, axis=0)

    if HAS_INDICES:
        result = tl.load(indices_ptr + row_offset_indices + chosen_scalar)
    else:
        result = chosen_scalar.to(tl.int32)
    tl.store(out_ptr + row, result)


@triton.jit
def _sample_from_logits_multi(
    out_ptr,
    logits_ptr,
    indices_ptr,
    rand_ptr,
    rows,
    cols,
    num_samples,
    stride_logits,
    stride_indices,
    stride_out,
    stride_rand,
    HAS_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_SAMPLES: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    row_logits = logits_ptr + row * stride_logits
    row_indices = indices_ptr + row * stride_indices
    row_out = out_ptr + row * stride_out
    row_rand = rand_ptr + row * stride_rand

    offsets = tl.arange(0, BLOCK_SIZE)
    sample_offsets = tl.arange(0, MAX_SAMPLES)
    sample_mask = sample_offsets < num_samples

    max_val = tl.full([1], -float("inf"), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(row_logits + idx, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(tl.where(mask, vals, -float("inf")), axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(row_logits + idx, mask=mask, other=-float("inf")).to(tl.float32)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)

    total = tl.maximum(sum_exp, 1e-20)
    rand_vals = tl.load(row_rand + sample_offsets, mask=sample_mask, other=0.0).to(
        tl.float32
    )
    rand_vals = tl.minimum(rand_vals, 0.99999994)

    results = tl.full([MAX_SAMPLES], 0, dtype=tl.int32)

    for sample_idx in tl.static_range(0, MAX_SAMPLES):
        idx_const = tl.full([1], sample_idx, dtype=tl.int32)
        active = idx_const < num_samples
        selector = sample_offsets == sample_idx
        rand_val = tl.sum(tl.where(selector, rand_vals, 0.0), axis=0)
        threshold = tl.where(active, rand_val * total, 0.0)

        cumulative = tl.zeros([1], dtype=tl.float32)
        found = tl.zeros([1], dtype=tl.int32)
        chosen = tl.full([1], cols - 1, dtype=tl.int32)

        for start in range(0, cols, BLOCK_SIZE):
            idx = start + offsets
            mask = idx < cols
            vals = tl.load(row_logits + idx, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            exp_vals = tl.exp(vals - max_val)
            block_sum = tl.sum(tl.where(mask, exp_vals, 0.0), axis=0)
            cum_block = cumulative + tl.cumsum(exp_vals, axis=0)
            cond = (cum_block >= threshold) & mask
            has = tl.max(cond.to(tl.int32), axis=0)
            rel = tl.min(tl.where(cond, offsets, BLOCK_SIZE), axis=0)
            update = active & (found == 0) & (has != 0)
            chosen = tl.where(update, (start + rel).to(tl.int32), chosen)
            found = tl.where(update, tl.full(found.shape, 1, dtype=tl.int32), found)
            cumulative += block_sum

        default_idx = tl.full([1], cols - 1, dtype=tl.int32)
        final_idx = tl.where(found == 1, chosen, default_idx)
        final_idx = tl.where(active, final_idx, tl.full([1], 0, dtype=tl.int32))
        index_value = tl.sum(final_idx, axis=0).to(tl.int32)

        if HAS_INDICES:
            token = tl.load(row_indices + index_value)
        else:
            token = index_value

        token_scalar = tl.sum(
            tl.where(active, token, tl.full([1], 0, dtype=token.dtype)), axis=0
        ).to(tl.int32)
        token_vec = tl.full([MAX_SAMPLES], token_scalar, dtype=tl.int32)
        active_mask = tl.full(
            [MAX_SAMPLES], tl.sum(active.to(tl.int32), axis=0) != 0, dtype=tl.int1
        )
        selector = (sample_offsets == sample_idx) & active_mask
        results = tl.where(selector, token_vec, results)

    tl.store(row_out + sample_offsets, results, mask=sample_mask)


def _sample_logits_torch(
    logits: torch.Tensor,
    *,
    top_k: Optional[int],
    temperature: float,
    generator: Optional[torch.Generator],
    num_samples: int,
) -> torch.Tensor:
    batch, vocab = logits.shape
    temp = max(temperature, 1e-6)
    if top_k is not None and top_k < vocab:
        values, indices = torch.topk(logits, top_k, dim=-1)
        probs = F.softmax(values / temp, dim=-1)
        draws = torch.multinomial(
            probs,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        tokens = indices.gather(-1, draws)
    else:
        probs = F.softmax(logits / temp, dim=-1)
        tokens = torch.multinomial(
            probs,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
    if num_samples == 1:
        return tokens.squeeze(-1)
    return tokens


def _supports_cutedsl_topk(logits: torch.Tensor, top_k: int) -> bool:
    if logits.ndim != 2:
        return False
    _, vocab = logits.shape
    if top_k <= 0 or vocab <= 0:
        return False
    if vocab > 4096 or top_k > 128:
        return False
    if (vocab & (vocab - 1)) != 0:
        return False
    if (top_k & (top_k - 1)) != 0:
        return False
    return True


def _select_topk_cutedsl(
    logits: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k selector using CuTeDSL; falls back to Triton when unsupported."""
    if not _supports_cutedsl_topk(logits, top_k):
        return _select_topk(logits, top_k)

    vals, idx = _cute_topk.topk(logits, top_k)
    return vals, idx


def sample_logits(
    logits: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    num_samples: int = 1,
    replacement: bool = True,
    backend: Literal["triton", "cutedsl", "torch"] = "triton",
) -> torch.Tensor:
    if logits.device.type != "cuda":  # pragma: no cover
        raise RuntimeError("sample_logits requires CUDA tensors")
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab)")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if num_samples > _MAX_SAMPLES:
        raise ValueError(f"num_samples must be <= {_MAX_SAMPLES}")
    if not replacement:
        raise NotImplementedError("replacement=False is not supported yet")
    if backend not in ("triton", "cutedsl", "torch"):
        raise ValueError("backend must be one of 'triton', 'cutedsl', or 'torch'")

    batch, vocab = logits.shape
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")
    if top_k is not None and top_k > vocab:
        top_k = None

    if temperature == 0 or (top_k is not None and top_k == 1):
        argmax = torch.argmax(logits, dim=-1)
        if num_samples == 1:
            return argmax
        return argmax.unsqueeze(-1).expand(-1, num_samples)

    if backend == "torch":
        return _sample_logits_torch(
            logits,
            top_k=top_k,
            temperature=temperature,
            generator=generator,
            num_samples=num_samples,
        )

    scaled_logits = (logits / max(temperature, 1e-6)).contiguous()

    if top_k is not None and top_k < vocab:
        if backend == "cutedsl":
            candidate_logits, candidate_indices = _select_topk_cutedsl(
                scaled_logits, top_k
            )
        else:
            candidate_logits, candidate_indices = _select_topk(scaled_logits, top_k)
        cols = top_k
    else:
        candidate_logits = scaled_logits.to(torch.float32)
        candidate_indices = None
        cols = vocab

    candidate_logits = candidate_logits.contiguous()

    stride_logits = candidate_logits.stride(0)
    stride_indices = candidate_indices.stride(0) if candidate_indices is not None else 0
    grid = (batch,)

    if num_samples == 1:
        rand = torch.rand(
            batch, device=logits.device, dtype=torch.float32, generator=generator
        )
        output = torch.empty(batch, device=logits.device, dtype=torch.int32)
        _sample_from_logits[grid](  # type: ignore[misc]
            output,
            candidate_logits,
            candidate_indices if candidate_indices is not None else candidate_logits,
            rand,
            batch,
            cols,
            stride_logits,
            stride_indices,
            HAS_INDICES=1 if candidate_indices is not None else 0,
            BLOCK_SIZE=128,
        )
        return output.to(torch.long)

    rand = torch.rand(
        (batch, num_samples),
        device=logits.device,
        dtype=torch.float32,
        generator=generator,
    )
    output = torch.empty((batch, num_samples), device=logits.device, dtype=torch.int32)

    _sample_from_logits_multi[grid](  # type: ignore[misc]
        output,
        candidate_logits,
        candidate_indices if candidate_indices is not None else candidate_logits,
        rand,
        batch,
        cols,
        num_samples,
        stride_logits,
        stride_indices,
        output.stride(0),
        rand.stride(0),
        HAS_INDICES=1 if candidate_indices is not None else 0,
        BLOCK_SIZE=128,
        MAX_SAMPLES=_MAX_SAMPLES,
    )

    return output.to(torch.long)


@triton.jit
def _topk_stage2_tiles_kernel(
    out_vals_ptr,
    out_idx_ptr,
    stage_vals_ptr,
    stage_idx_ptr,
    stride_out_vals,
    stride_out_idx,
    stride_stage_vals,
    stride_stage_idx,
    rows,
    total_candidates,
    K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    ACC_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    out_vals_row = out_vals_ptr + row * stride_out_vals
    out_idx_row = out_idx_ptr + row * stride_out_idx
    stage_vals_row = stage_vals_ptr + row * stride_stage_vals
    stage_idx_row = stage_idx_ptr + row * stride_stage_idx

    ts_offsets = tl.arange(0, TILE_SIZE)

    acc_vals = tl.full([ACC_SIZE], -float("inf"), dtype=tl.float32)
    acc_idxs = tl.full([ACC_SIZE], -1, dtype=tl.int32)
    acc_offsets = tl.arange(0, ACC_SIZE)

    for start in range(0, total_candidates, TILE_SIZE):
        idx = start + ts_offsets
        mask = idx < total_candidates
        vals = tl.load(
            stage_vals_row + idx, mask=mask, other=-float("inf"), cache_modifier=".cg"
        ).to(tl.float32)
        idxs = tl.load(
            stage_idx_row + idx, mask=mask, other=-1, cache_modifier=".cg"
        ).to(tl.int32)

        tmp_vals = vals
        tmp_idxs = idxs
        for i in tl.static_range(0, K):
            max_val, rel = tl.max(tmp_vals, axis=0, return_indices=True)
            has = max_val != -float("inf")
            sel = ts_offsets == rel
            cand_idx = tl.max(tl.where(sel, tmp_idxs, -1), axis=0).to(tl.int32)

            min_val, min_pos = tl.min(acc_vals, axis=0, return_indices=True)
            replace = has & (max_val > min_val)
            pos_mask = acc_offsets == min_pos
            acc_vals = tl.where(replace & pos_mask, max_val, acc_vals)
            acc_idxs = tl.where(replace & pos_mask, cand_idx, acc_idxs)

            tmp_vals = tl.where(sel, -float("inf"), tmp_vals)
            tmp_idxs = tl.where(sel, -1, tmp_idxs)

    neg_inf = -float("inf")
    for i in tl.static_range(0, K):
        max_val, _ = tl.max(acc_vals, axis=0, return_indices=True)
        has = max_val != neg_inf
        # tie-break: among acc entries equal to max_val, pick smallest index
        eq = acc_vals == max_val
        large = (1 << 31) - 1
        idx_masked = tl.where(eq, acc_idxs, large)
        min_idx, pos = tl.min(idx_masked, axis=0, return_indices=True)
        sel = acc_offsets == pos
        chosen_idx = tl.where(has, min_idx.to(tl.int32), -1)
        tl.store(out_vals_row + i, tl.where(has, max_val, neg_inf))
        tl.store(out_idx_row + i, chosen_idx)
        acc_vals = tl.where(sel, neg_inf, acc_vals)
        acc_idxs = tl.where(sel, -1, acc_idxs)


@triton.jit
def _topk_singlepass_tiles_kernel(
    out_vals_ptr,
    out_idx_ptr,
    logits_ptr,
    stride_out_vals,
    stride_out_idx,
    stride_logits_m,
    stride_logits_n,
    rows,
    cols,
    K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    ACC_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    out_vals_row = out_vals_ptr + row * stride_out_vals
    out_idx_row = out_idx_ptr + row * stride_out_idx
    logits_row = logits_ptr + row * stride_logits_m

    ts_offsets = tl.arange(0, TILE_SIZE)

    acc_vals = tl.full([ACC_SIZE], -float("inf"), dtype=tl.float32)
    acc_idxs = tl.full([ACC_SIZE], -1, dtype=tl.int32)
    acc_offsets = tl.arange(0, ACC_SIZE)

    for start in range(0, cols, TILE_SIZE):
        idx = start + ts_offsets
        mask = idx < cols
        vals = tl.load(
            logits_row + idx * stride_logits_n, mask=mask, other=-float("inf")
        ).to(tl.float32)

        tmp_vals = vals
        tmp_idxs = idx.to(tl.int32)
        for _ in tl.static_range(0, K):
            max_val, rel = tl.max(tmp_vals, axis=0, return_indices=True)
            has = max_val != -float("inf")
            sel = ts_offsets == rel
            cand_idx = tl.max(tl.where(sel, tmp_idxs, -1), axis=0).to(tl.int32)

            min_val, min_pos = tl.min(acc_vals, axis=0, return_indices=True)
            replace = has & (max_val > min_val)
            pos_mask = acc_offsets == min_pos
            acc_vals = tl.where(replace & pos_mask, max_val, acc_vals)
            acc_idxs = tl.where(replace & pos_mask, cand_idx, acc_idxs)

            tmp_vals = tl.where(sel, -float("inf"), tmp_vals)
            tmp_idxs = tl.where(sel, -1, tmp_idxs)

    neg_inf = -float("inf")
    for i in tl.static_range(0, K):
        max_val, _ = tl.max(acc_vals, axis=0, return_indices=True)
        has = max_val != neg_inf
        eq = acc_vals == max_val
        large = (1 << 31) - 1
        idx_masked = tl.where(eq, acc_idxs, large)
        min_idx, pos = tl.min(idx_masked, axis=0, return_indices=True)
        sel = acc_offsets == pos
        chosen_idx = tl.where(has, min_idx.to(tl.int32), -1)
        tl.store(out_vals_row + i, tl.where(has, max_val, neg_inf))
        tl.store(out_idx_row + i, chosen_idx)
        acc_vals = tl.where(sel, neg_inf, acc_vals)
        acc_idxs = tl.where(sel, -1, acc_idxs)


@triton.jit
def _pack_key(val, idx, vocab_bits: tl.constexpr):
    val_as_u32 = val.to(tl.uint32, bitcast=True)
    sorted_key = _fpval_to_key(val_as_u32)
    reversed_idx = ((1 << vocab_bits) - 1) - idx
    packed = (sorted_key.to(tl.uint64) << vocab_bits) | reversed_idx.to(tl.uint64)
    return packed


@triton.jit
def _unpack_key(packed, vocab_bits: tl.constexpr):
    mask = (1 << vocab_bits) - 1
    reversed_idx = (packed & mask).to(tl.int32)
    sorted_key = (packed >> vocab_bits).to(tl.uint32)
    val_as_u32 = _key_to_fpval(sorted_key)
    val = val_as_u32.to(tl.float32, bitcast=True)
    idx = ((1 << vocab_bits) - 1) - reversed_idx
    return val, idx


@triton.jit
def _streaming_topk_kernel(
    out_vals_ptr,
    out_idx_ptr,
    logits_ptr,
    stride_out_vals,
    stride_out_idx,
    stride_logits_m,
    stride_logits_n,
    rows,
    cols,
    K: tl.constexpr,
    K_PADDED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return

    out_vals_row = out_vals_ptr + row * stride_out_vals
    out_idx_row = out_idx_ptr + row * stride_out_idx
    logits_row = logits_ptr + row * stride_logits_m

    n_tiles = tl.cdiv(cols, BLOCK_N)
    tile_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, K_PADDED)

    # Initialize accumulator
    acc_vals = tl.full([K_PADDED], -float("inf"), dtype=tl.float32)
    acc_idxs = tl.full([K_PADDED], -1, dtype=tl.int32)

    # Iterate through tiles
    for tile_idx in range(n_tiles):
        tile_start = tile_idx * BLOCK_N
        idx = tile_start + tile_offsets
        mask = idx < cols

        # Stream load with cache hint
        vals = tl.load(
            logits_row + idx * stride_logits_n,
            mask=mask,
            other=-float("inf"),
            cache_modifier=".cg",
        ).to(tl.float32)

        # Extract top-K_PADDED from this tile
        tmp_vals = vals
        tmp_idxs = idx.to(tl.int32)

        for i in tl.static_range(0, K_PADDED):
            max_val, rel = tl.max(tmp_vals, axis=0, return_indices=True)
            has = max_val != -float("inf")
            sel = tile_offsets == rel
            cand_idx = tl.max(tl.where(sel, tmp_idxs, -1), axis=0).to(tl.int32)

            # Merge into accumulator: replace minimum if this candidate is better
            min_val, min_pos = tl.min(acc_vals, axis=0, return_indices=True)
            replace = has & (max_val > min_val)
            pos_mask = k_offsets == min_pos
            acc_vals = tl.where(replace & pos_mask, max_val, acc_vals)
            acc_idxs = tl.where(replace & pos_mask, cand_idx, acc_idxs)

            tmp_vals = tl.where(sel, -float("inf"), tmp_vals)
            tmp_idxs = tl.where(sel, -1, tmp_idxs)

    # Final extraction of top-K from accumulator with tie-breaking
    neg_inf = -float("inf")
    k_mask = k_offsets < K
    for i in tl.static_range(0, K_PADDED):
        max_val, _ = tl.max(acc_vals, axis=0, return_indices=True)
        has = max_val != neg_inf
        # Tie-break: among acc entries equal to max_val, pick smallest index
        eq = acc_vals == max_val
        large = (1 << 31) - 1
        idx_masked = tl.where(eq, acc_idxs, large)
        min_idx, pos = tl.min(idx_masked, axis=0, return_indices=True)
        sel = k_offsets == pos
        chosen_idx = tl.where(has, min_idx.to(tl.int32), -1)

        should_store = (i < K) & has
        i_mask = k_offsets == i
        tl.store(
            out_vals_row + k_offsets,
            tl.where(has, max_val, neg_inf),
            mask=i_mask & k_mask,
        )
        tl.store(out_idx_row + k_offsets, chosen_idx, mask=i_mask & k_mask)

        acc_vals = tl.where(sel, neg_inf, acc_vals)
        acc_idxs = tl.where(sel, -1, acc_idxs)


__all__ = ["sample_logits"]
