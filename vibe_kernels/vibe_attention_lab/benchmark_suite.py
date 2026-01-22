# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import sys
import time
import types
from unittest.mock import MagicMock

import torch

# ==========================================
# 0. SETUP STANDALONE PATH
# ==========================================
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))  # .../flex_attention_lab
# Get the package root (parent of script dir)
package_root = os.path.dirname(script_dir)  # .../kernel_factory

# Add package root to sys.path
if package_root not in sys.path:
    sys.path.insert(0, package_root)

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import math as mlir_math
    from vibe_attention import utils
    from vibe_attention.block_sparsity import BlockSparseTensorsTorch

    # NEW IMPORTS from standalone package
    from vibe_attention.interface import _flash_attn_fwd
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)


# Helper: Block Sparsity Generator (Adapted for standalone)
def generate_block_sparsity(mask_fn, B, H, SEQ_Q, SEQ_K, tile_size=128, device="cuda"):
    num_q_blocks = (SEQ_Q + tile_size - 1) // tile_size
    num_k_blocks = (SEQ_K + tile_size - 1) // tile_size

    mask_block_idx = torch.zeros(
        (B, H, num_q_blocks, num_k_blocks), dtype=torch.int32, device=device
    )
    mask_block_cnt = torch.zeros((B, H, num_q_blocks), dtype=torch.int32, device=device)
    full_block_idx = torch.zeros(
        (B, H, num_q_blocks, num_k_blocks), dtype=torch.int32, device=device
    )
    full_block_cnt = torch.zeros((B, H, num_q_blocks), dtype=torch.int32, device=device)

    for b in range(B):
        for h in range(H):
            for q_blk in range(num_q_blocks):
                q_start = q_blk * tile_size
                q_end = min(q_start + tile_size, SEQ_Q)
                mask_cnt = 0
                full_cnt = 0
                for k_blk in range(num_k_blocks):
                    k_start = k_blk * tile_size
                    k_end = min(k_start + tile_size, SEQ_K)
                    q_range = torch.arange(q_start, q_end, device=device)
                    k_range = torch.arange(k_start, k_end, device=device)
                    Q_GRID, K_GRID = torch.meshgrid(q_range, k_range, indexing="ij")
                    is_masked = mask_fn(b, h, Q_GRID, K_GRID)
                    total_pixels = (q_end - q_start) * (k_end - k_start)
                    masked_pixels = is_masked.sum().item()
                    if masked_pixels == total_pixels:
                        continue
                    elif masked_pixels == 0:
                        full_block_idx[b, h, q_blk, full_cnt] = k_blk
                        full_cnt += 1
                    else:
                        mask_block_idx[b, h, q_blk, mask_cnt] = k_blk
                        mask_cnt += 1
                mask_block_cnt[b, h, q_blk] = mask_cnt
                full_block_cnt[b, h, q_blk] = full_cnt

    target_dim = num_k_blocks
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx[..., :target_dim].contiguous(),
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx[..., :target_dim].contiguous(),
    )


# ==========================================
# 1. REFERENCE IMPLEMENTATIONS (PyTorch)
# ==========================================
def attention_ref_torch(q, k, v, score_mod_fn=None, mask_mod_fn=None, aux_tensors=None):
    q = q.float()
    k = k.float()
    v = v.float()
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if score_mod_fn is not None:
        q_idx = torch.arange(Sq, device=q.device).view(1, 1, Sq, 1).expand(B, H, Sq, Sk)
        k_idx = torch.arange(Sk, device=q.device).view(1, 1, 1, Sk).expand(B, H, Sq, Sk)
        b_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1).expand(B, H, Sq, Sk)
        h_idx = torch.arange(H, device=q.device).view(1, H, 1, 1).expand(B, H, Sq, Sk)
        scores = score_mod_fn(scores, b_idx, h_idx, q_idx, k_idx, aux_tensors)
    if mask_mod_fn is not None:
        q_idx = torch.arange(Sq, device=q.device).view(1, 1, Sq, 1).expand(B, H, Sq, Sk)
        k_idx = torch.arange(Sk, device=q.device).view(1, 1, 1, Sk).expand(B, H, Sq, Sk)
        b_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1).expand(B, H, Sq, Sk)
        h_idx = torch.arange(H, device=q.device).view(1, H, 1, 1).expand(B, H, Sq, Sk)
        mask = mask_mod_fn(b_idx, h_idx, q_idx, k_idx, aux_tensors)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)


# ==========================================
# 2. CUTE DSL DEFINITIONS
# ==========================================


# T5 Relative Bias
@cute.jit
def t5_score_mod_cute(score, b, h, q_idx, kv_idx, aux_tensors):
    bias_table = aux_tensors[0]
    rel_pos = mlir_math.absi(q_idx[0] - kv_idx[0])
    return score + bias_table[b[0], h[0], rel_pos]


def t5_score_mod_torch(score, b, h, q, k, aux):
    bias_table = aux[0]
    rel_pos = (q - k).abs()
    bias_vals = torch.zeros_like(score)
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            bias_vals[i, j] = bias_table[i, j, rel_pos[i, j]]
    return score + bias_vals


# ALiBi
@cute.jit
def alibi_score_mod_cute(score, b, h, q_idx, kv_idx, aux_tensors):
    slopes = aux_tensors[0]
    slope = slopes[h[0]]
    dist = mlir_math.absi(q_idx[0] - kv_idx[0])
    return score - slope * dist


def alibi_score_mod_torch(score, b, h, q, k, aux):
    slopes = aux[0]
    sl = slopes.view(1, -1, 1, 1)
    dist = (q - k).abs()
    return score - sl * dist


# ==========================================
# 3. RUNNER
# ==========================================
def run_test(
    name,
    cute_score_mod,
    cute_mask_mod,
    torch_score_mod,
    torch_mask_mod,
    aux_tensors_cute,
    aux_tensors_torch,
    B=1,
    H=4,
    SEQ=1024,
    mask_fn_for_sparsity=None,
):
    print(f"\n--- Testing {name} [Standalone Package] ---")
    device = "cuda"
    dtype = torch.float16
    q = torch.randn((B, SEQ, H, 64), dtype=dtype, device=device)
    k = torch.randn((B, SEQ, H, 64), dtype=dtype, device=device)
    v = torch.randn((B, SEQ, H, 64), dtype=dtype, device=device)
    if mask_fn_for_sparsity is None:
        mask_fn_for_sparsity = lambda b, h, q, k: torch.zeros_like(q, dtype=torch.bool)
    block_sparse = generate_block_sparsity(mask_fn_for_sparsity, B, H, SEQ, SEQ)

    print("Running CuTe DSL...")
    try:
        _flash_attn_fwd(
            q,
            k,
            v,
            score_mod=cute_score_mod,
            mask_mod=cute_mask_mod,
            block_sparse_tensors=block_sparse,
            aux_tensors=aux_tensors_cute,
            m_block_size=128,
            n_block_size=128,
        )
        torch.cuda.synchronize()
        t0 = time.time()
        out_cute, _ = _flash_attn_fwd(
            q,
            k,
            v,
            score_mod=cute_score_mod,
            mask_mod=cute_mask_mod,
            block_sparse_tensors=block_sparse,
            aux_tensors=aux_tensors_cute,
            m_block_size=128,
            n_block_size=128,
        )
        torch.cuda.synchronize()
        t_cute = (time.time() - t0) * 1000
        print(f"CuTe Time: {t_cute:.3f} ms")
    except Exception as e:
        print(f"CuTe Execution Failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Running PyTorch Reference...")
    try:
        t0 = time.time()
        out_ref = attention_ref_torch(
            q,
            k,
            v,
            score_mod_fn=torch_score_mod,
            mask_mod_fn=torch_mask_mod,
            aux_tensors=aux_tensors_torch,
        )
        torch.cuda.synchronize()
        t_ref = (time.time() - t0) * 1000
        print(f"Ref Time:  {t_ref:.3f} ms")
        print(f"Speedup:   {t_ref / t_cute:.2f}x")
        diff = (out_cute - out_ref).abs().max().item()
        print(f"Max Difference: {diff:.6f}")
        if diff < 0.02:
            print("✅ PASSED")
        else:
            print("❌ FAILED (Numerics)")
    except Exception as e:
        print(f"Reference Failed: {e}")


def main():
    device = "cuda"
    B, H, SEQ = 2, 4, 1024

    # Test 1: T5
    bias_table = torch.randn((B, H, SEQ + 1), dtype=torch.float16, device=device)
    run_test(
        "T5 Relative Bias",
        t5_score_mod_cute,
        None,
        t5_score_mod_torch,
        None,
        [bias_table],
        [bias_table],
        B=B,
        H=H,
        SEQ=SEQ,
    )

    # Test 2: ALiBi
    slopes = torch.tensor(
        [1.0 / 2 ** (h + 1) for h in range(H)], device=device, dtype=torch.float32
    )
    run_test(
        "ALiBi",
        alibi_score_mod_cute,
        None,
        alibi_score_mod_torch,
        None,
        [slopes],
        [slopes],
        B=B,
        H=H,
        SEQ=SEQ,
    )

    # Test 3: Document Masking
    # We need doc_mask definitions here
    # I'll define them inline for simplicity
    doc_ids = torch.randint(0, 4, (B, SEQ), device=device, dtype=torch.int32)
    doc_ids, _ = torch.sort(doc_ids, dim=1)

    @cute.jit
    def doc_mask_mod_cute(b, h, q_idx, kv_idx, aux_tensors):
        doc_ids = aux_tensors[0]
        d_q = doc_ids[b[0], q_idx[0]]
        d_k = doc_ids[b[0], kv_idx[0]]
        d_q_ssa = utils.scalar_to_ssa(d_q, cutlass.Int32)
        d_k_ssa = utils.scalar_to_ssa(d_k, cutlass.Int32)
        return d_q_ssa == d_k_ssa

    def doc_mask_mod_torch(b, h, q, k, aux):
        doc_ids = aux[0]
        d_q = doc_ids.unsqueeze(1).unsqueeze(-1)
        d_k = doc_ids.unsqueeze(1).unsqueeze(-2)
        return d_q != d_k

    def doc_sparsity_fn(b, h, q_grid, k_grid):
        d_q = doc_ids[b, q_grid]
        d_k = doc_ids[b, k_grid]
        return d_q != d_k

    run_test(
        "Document Masking",
        None,
        doc_mask_mod_cute,
        None,
        doc_mask_mod_torch,
        [doc_ids],
        [doc_ids],
        B=B,
        H=H,
        SEQ=SEQ,
        mask_fn_for_sparsity=doc_sparsity_fn,
    )

    # Test 4: PrefixLM
    prefix_len_val = 256
    prefix_tensor = torch.tensor([prefix_len_val], dtype=torch.int32, device=device)

    @cute.jit
    def prefix_mask_mod_cute(b, h, q_idx, kv_idx, aux_tensors):
        prefix_len = aux_tensors[0][0]
        p_ssa = utils.scalar_to_ssa(prefix_len, cutlass.Int32)
        is_prefix = kv_idx < p_ssa
        is_causal = kv_idx <= q_idx
        return is_prefix | is_causal

    def prefix_mask_mod_torch(b, h, q, k, aux):
        prefix_len = aux[0].item()
        is_prefix = k < prefix_len
        is_causal = k <= q
        keep = is_prefix | is_causal
        return ~keep

    def prefix_sparsity_fn(b, h, q_grid, k_grid):
        return (k_grid >= prefix_len_val) & (k_grid > q_grid)

    run_test(
        "PrefixLM",
        None,
        prefix_mask_mod_cute,
        None,
        prefix_mask_mod_torch,
        [prefix_tensor],
        [prefix_tensor],
        B=B,
        H=H,
        SEQ=SEQ,
        mask_fn_for_sparsity=prefix_sparsity_fn,
    )

    # Test 5: Causal (Pure Flex)
    @cute.jit
    def causal_mask_mod_cute(b, h, q_idx, kv_idx, aux_tensors):
        # q_idx, kv_idx are scalar SSA inputs from mask.py (no [0])
        # but if they are passed as args, they should be usable directly.
        # Wait, prefix_mask used them directly.
        # And doc_mask used q_idx[0] for indexing?
        # Let's follow prefix pattern: direct comparison.
        return kv_idx <= q_idx  # Keep

    def causal_mask_mod_torch(b, h, q, k, aux):
        return ~(k <= q)

    def causal_sparsity_fn(b, h, q, k):
        return k > q

    run_test(
        "Causal",
        None,
        causal_mask_mod_cute,
        None,
        causal_mask_mod_torch,
        None,
        None,
        B=B,
        H=H,
        SEQ=SEQ,
        mask_fn_for_sparsity=causal_sparsity_fn,
    )

    # Test 6: Sliding Window Attention (SWA)
    window_size = 256
    window_tensor = torch.tensor([window_size], dtype=torch.int32, device=device)

    @cute.jit
    def swa_mask_mod_cute(b, h, q_idx, kv_idx, aux_tensors):
        w_val = aux_tensors[0][0]
        w = utils.scalar_to_ssa(w_val, cutlass.Int32)

        # Use [0] to unwrap TensorSSA to ArithValue for raw mlir_math ops
        q = q_idx[0]
        k = kv_idx[0]

        dist = mlir_math.absi(q - k)  # Scalar ArithValue
        w_scalar = w[0]

        keep = dist <= w_scalar  # Scalar Boolean

        # Return Vector<Int32> as proxy for Boolean (True != 0)
        # We rely on implicit cast or behavior where Boolean can be stored in Int32 fragment
        return utils.scalar_to_ssa(keep, cutlass.Int32)

    def swa_mask_mod_torch(b, h, q, k, aux):
        w = aux[0].item()
        dist = (q - k).abs()
        return ~(dist <= w)  # Mask out

    def swa_sparsity_fn(b, h, q, k):
        return (q - k).abs() > window_size

    run_test(
        "Sliding Window",
        None,
        swa_mask_mod_cute,
        None,
        swa_mask_mod_torch,
        [window_tensor],
        [window_tensor],
        B=B,
        H=H,
        SEQ=SEQ,
        mask_fn_for_sparsity=swa_sparsity_fn,
    )


if __name__ == "__main__":
    main()
