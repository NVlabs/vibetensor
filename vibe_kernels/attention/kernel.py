# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from typing import Optional

import torch  # type: ignore[import]
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

try:  # pragma: no cover
    from triton.tools.tensor_descriptor import TensorDescriptor  # type: ignore[import]
except ImportError:  # pragma: no cover
    TensorDescriptor = None  # type: ignore[assignment]


def _is_cuda() -> bool:
    return torch.cuda.is_available()


def _device_capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability() if _is_cuda() else (0, 0)


def _supports_host_descriptor() -> bool:
    return False


def _is_hopper() -> bool:
    major, _ = _device_capability()
    return _is_cuda() and major == 9


def _is_blackwell() -> bool:
    major, _ = _device_capability()
    return _is_cuda() and major == 10


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _pick_num_warps(block_size: int) -> int:
    if block_size >= 32768:
        return 32
    if block_size >= 8192:
        return 16
    if block_size >= 2048:
        return 8
    if block_size >= 1024:
        return 4
    if block_size >= 256:
        return 2
    return 1


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
    USE_DESC: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    col_offsets = tl.arange(0, HEAD_DIM)
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if USE_DESC:
            k_tile = desc_k.load([offsetk_y, 0])
        else:
            k_rows = offsetk_y + tl.arange(0, BLOCK_N)
            k_ptrs = desc_k + k_rows[:, None] * HEAD_DIM + col_offsets[None, :]
            k_tile = tl.load(k_ptrs)
        k = tl.trans(k_tile)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            if USE_DESC:
                v_tile = desc_v.load([offsetv_y, 0])
            else:
                v_rows = offsetv_y + tl.arange(0, BLOCK_N)
                v_ptrs = desc_v + v_rows[:, None] * HEAD_DIM + col_offsets[None, :]
                v_tile = tl.load(v_ptrs)
            v = v_tile
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    if TensorDescriptor is None:
        return
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s,
        num_warps=w,
        pre_hook=_host_descriptor_pre_hook,
    )
    for BM in [32, 64, 128]
    for BN in [16, 32, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]


def _keep_fwd_config(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        _is_cuda()
        and _is_hopper()
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def _prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    HEAD_DIM = kwargs["HEAD_DIM"]
    STAGE = kwargs.get("STAGE", 1)
    is_causal = STAGE == 3
    valid = []
    for conf in configs:
        BLOCK_M = conf.kwargs.get("BLOCK_M", 0)
        BLOCK_N = conf.kwargs.get("BLOCK_N", 0)
        if BLOCK_M > N_CTX or BLOCK_N > N_CTX:
            continue
        if BLOCK_N > HEAD_DIM:
            continue
        if N_CTX % BLOCK_M != 0:
            continue
        if is_causal and BLOCK_N > BLOCK_M:
            continue
        valid.append(conf)
    return valid


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=list(filter(_keep_fwd_config, configs)),
    key=[
        "N_CTX",
        "HEAD_DIM",
        "FP8_OUTPUT",
        "warp_specialize",
        "USE_DESC",
        "INPUT_IS_BF16",
    ],
    prune_configs_by={"early_config_prune": _prune_invalid_configs},
)
@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
    USE_DESC: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    if FP8_OUTPUT:
        dtype = tl.float8e5
    elif INPUT_IS_BF16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    if USE_DESC:
        desc_q = _maybe_make_tensor_desc(
            desc_q,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        if FP8_OUTPUT:
            desc_v = _maybe_make_tensor_desc(
                desc_v,
                shape=[HEAD_DIM, y_dim],
                strides=[N_CTX, 1],
                block_shape=[HEAD_DIM, BLOCK_N],
            )
        else:
            desc_v = _maybe_make_tensor_desc(
                desc_v,
                shape=[y_dim, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=[BLOCK_N, HEAD_DIM],
            )
        desc_k = _maybe_make_tensor_desc(
            desc_k,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        desc_o = _maybe_make_tensor_desc(
            desc_o,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    col_offsets = tl.arange(0, HEAD_DIM)
    if USE_DESC:
        q = desc_q.load([qo_offset_y, 0])
    else:
        row_offsets = qo_offset_y + tl.arange(0, BLOCK_M)
        q_ptrs = desc_q + row_offsets[:, None] * HEAD_DIM + col_offsets[None, :]
        q = tl.load(q_ptrs)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
            USE_DESC,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
            USE_DESC,
        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    if USE_DESC:
        desc_o.store([qo_offset_y, 0], acc.to(dtype))
    else:
        row_offsets_o = qo_offset_y + tl.arange(0, BLOCK_M)
        o_ptrs = desc_o + row_offsets_o[:, None] * HEAD_DIM + col_offsets[None, :]
        tl.store(o_ptrs, acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(
    O,
    DO,
    Delta,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    k,
    v,
    sm_scale,
    DO,
    M,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n,
    start_m,
    num_steps,
    MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for _ in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        dv += tl.dot(pT.to(DTYPE), do)
        Di = tl.load(D + offs_m)
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dk += tl.dot(dsT.to(DTYPE), tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,
    do,
    m,
    D,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m,
    start_n,
    num_steps,
    MASK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for _ in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        dq += tl.dot(ds.to(DTYPE), tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


_bwd_configs = [
    triton.Config(
        {"BLOCK_M1": 32, "BLOCK_N1": 128, "BLOCK_M2": 128, "BLOCK_N2": 32},
        num_stages=s, num_warps=w,
    )
    for s in [3, 4, 5]
    for w in [4, 8]
] + [
    triton.Config(
        {"BLOCK_M1": 64, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 64},
        num_stages=s, num_warps=w,
    )
    for s in [3, 4, 5]
    for w in [4, 8]
] + [
    triton.Config(
        {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32},
        num_stages=s, num_warps=w,
    )
    for s in [3, 4]
    for w in [4]
]


def _prune_bwd_configs(configs, named_args, **kwargs):
    N_CTX = named_args.get("N_CTX") or kwargs.get("N_CTX")
    if N_CTX is None:
        return configs
    valid = []
    for conf in configs:
        BLOCK_N1 = conf.kwargs.get("BLOCK_N1", 0)
        BLOCK_M1 = conf.kwargs.get("BLOCK_M1", 0)
        BLOCK_M2 = conf.kwargs.get("BLOCK_M2", 0)
        BLOCK_N2 = conf.kwargs.get("BLOCK_N2", 0)
        if BLOCK_N1 > N_CTX or BLOCK_M2 > N_CTX:
            continue
        if N_CTX % BLOCK_N1 != 0:
            continue
        if N_CTX % BLOCK_M2 != 0:
            continue
        valid.append(conf)
    return valid


@triton.autotune(
    configs=_bwd_configs,
    key=["N_CTX", "HEAD_DIM"],
    prune_configs_by={"early_config_prune": _prune_bwd_configs},
)
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    if INPUT_IS_BF16:
        DTYPE: tl.constexpr = tl.bfloat16
    else:
        DTYPE: tl.constexpr = tl.float16
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    offs_k = tl.arange(0, HEAD_DIM)
    start_n = pid * BLOCK_N1
    start_m = start_n
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        start_m,
        num_steps,
        MASK=True,
        DTYPE=DTYPE,
    )
    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        start_m,
        num_steps,
        MASK=False,
        DTYPE=DTYPE,
    )
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    m = tl.load(M + offs_m)
    m = m[:, None]
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,
        MASK=True,
        DTYPE=DTYPE,
    )
    end_n -= num_steps * MASK_BLOCK_N2
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,
        do,
        m,
        D,
        stride_tok,
        stride_d,
        H,
        N_CTX,
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,
        MASK=False,
        DTYPE=DTYPE,
    )
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize):
        HEAD_DIM = q.shape[-1]
        assert q.shape == k.shape == v.shape
        assert HEAD_DIM in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        env_disable_ws = os.environ.get("AIKF_DISABLE_WARP_SPECIALIZE")
        if env_disable_ws is not None and env_disable_ws != "0":
            warp_specialize = False

        use_desc = False
        if (
            TensorDescriptor is not None
            and _supports_host_descriptor()
            and not (_is_hopper() and warp_specialize)
        ):
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dummy_block = [1, 1]
            desc_q = TensorDescriptor(
                q,
                shape=[y_dim, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=dummy_block,
            )
            desc_v = TensorDescriptor(
                v,
                shape=[y_dim, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=dummy_block,
            )
            desc_k = TensorDescriptor(
                k,
                shape=[y_dim, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=dummy_block,
            )
            desc_o = TensorDescriptor(
                o,
                shape=[y_dim, HEAD_DIM],
                strides=[HEAD_DIM, 1],
                block_shape=dummy_block,
            )
            use_desc = True
        else:
            desc_q, desc_v, desc_k, desc_o = q, v, k, o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device=q.device)

        triton.set_allocator(alloc_fn)

        def grid(meta):
            return (
                triton.cdiv(q.shape[2], meta["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        if _is_blackwell() and warp_specialize:
            if HEAD_DIM == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](  # type: ignore[misc]
            sm_scale,
            M,
            q.shape[0],
            q.shape[1],
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=warp_specialize,
            IS_HOPPER=_is_hopper(),
            USE_DESC=use_desc,
            INPUT_IS_BF16=q.dtype == torch.bfloat16,
            **extra_kern_args,
        )
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.head_dim = HEAD_DIM
        ctx.causal = causal
        ctx.warp_specialize = warp_specialize
        return o

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, M = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]

        for candidate in (128, 64, 32, 16):
            if N_CTX % candidate == 0:
                PRE_BLOCK = candidate
                break
        else:
            PRE_BLOCK = 16

        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634
        arg_k = k * (ctx.sm_scale * RCP_LN2)
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](  # type: ignore[misc]
            o,
            grad_out,
            delta,
            BATCH,
            N_HEAD,
            N_CTX,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.head_dim,
        )

        def bwd_grid(meta):
            return (N_CTX // meta["BLOCK_N1"], 1, BATCH * N_HEAD)

        _attn_bwd[bwd_grid](  # type: ignore[misc]
            q,
            arg_k,
            v,
            ctx.sm_scale,
            grad_out,
            dq,
            dk,
            dv,
            M,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            N_HEAD,
            N_CTX,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.head_dim,
            INPUT_IS_BF16=q.dtype == torch.bfloat16,
        )
        return dq, dk, dv, None, None, None


_attention = _FlashAttention.apply


def _reshape_for_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    B, Hq, T, D = q.shape
    _, Hk, _, _ = k.shape
    assert Hq % Hk == 0, "Query heads must be a multiple of key/value heads"
    group_size = Hq // Hk
    if group_size == 1:
        return q.contiguous(), k.contiguous(), v.contiguous(), group_size
    q_reshaped = (
        q.view(B, Hk, group_size, T, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B * group_size, Hk, T, D)
        .contiguous()
    )
    k_reshaped = (
        k.unsqueeze(1)
        .repeat(1, group_size, 1, 1, 1)
        .reshape(B * group_size, Hk, T, D)
        .contiguous()
    )
    v_reshaped = (
        v.unsqueeze(1)
        .repeat(1, group_size, 1, 1, 1)
        .reshape(B * group_size, Hk, T, D)
        .contiguous()
    )
    return q_reshaped, k_reshaped, v_reshaped, group_size


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    warp_specialize: bool = False,
) -> torch.Tensor:
    """Fused scaled dot-product attention with optional GQA support.

    Args:
        q, k, v: `(batch, heads, seq_len, head_dim)` tensors in bf16/fp16.
        causal: whether to apply causal masking.
        sm_scale: optional softmax scale (defaults to `1/sqrt(head_dim)`).
        warp_specialize: enable warp specialization (recommended on Hopper+).
    """

    if not _is_cuda():  # pragma: no cover - GPU required
        raise RuntimeError("fused_attention requires CUDA tensors")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    original_dtype = q.dtype
    q_kernel, k_kernel, v_kernel, group_size = _reshape_for_gqa(q, k, v)
    kernel_dtype = q_kernel.dtype
    if kernel_dtype == torch.float32:
        kernel_dtype = torch.float16
        q_kernel = q_kernel.to(kernel_dtype)
        k_kernel = k_kernel.to(kernel_dtype)
        v_kernel = v_kernel.to(kernel_dtype)
    elif kernel_dtype not in {torch.float16, torch.bfloat16, torch.float8_e5m2}:
        raise TypeError(f"Unsupported attention dtype: {kernel_dtype}")

    out = _attention(q_kernel, k_kernel, v_kernel, causal, sm_scale, warp_specialize)
    if out.dtype != original_dtype:
        out = out.to(original_dtype)

    if group_size == 1:
        return out

    B, Hk, T, D = k.shape
    out = (
        out.view(B, group_size, Hk, T, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, Hk * group_size, T, D)
        .contiguous()
    )
    return out


__all__ = ["fused_attention"]
