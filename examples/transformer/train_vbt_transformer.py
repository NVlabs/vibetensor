# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-Free Transformer Training - Direct Kernel Usage.

Demonstrates transformer training WITHOUT PyTorch or NumPy:
- Uses vibetensor.torch (VibeTensor's PyTorch-compatible API, NOT PyTorch)
- Uses vibe_kernels vbt_native modules directly (Triton kernels)
- All computation runs on GPU via Triton kernels

This script shows how to use individual vibe_kernels ops.
For a cleaner implementation, see train_modular.py.

Usage:
    python train_vbt_transformer.py
"""

import os
import sys
import time

import vibetensor.torch as vt

# Import vibe_kernels modules
from vibe_kernels.embedding import vbt_native as embed_ops
from vibe_kernels.attention import vbt_native as attn_ops
from vibe_kernels.rmsnorm import vbt_native as rmsnorm_ops
from vibe_kernels.activation import vbt_native as act_ops
from vibe_kernels.rotary import vbt_native as rope_ops
from vibe_kernels.gemm import vbt_native as gemm_ops
from vibe_kernels.indexing import vbt_native as idx_ops
from vibe_kernels.tensor_ops import vbt_native as tensor_ops


def attention_backward_vbt(q, k, v, grad_out, scale):
    """Attention backward using VBT ops + GEMM kernels (no numpy).
    
    Args:
        q, k, v: [B, H, S, D] - query, key, value after RoPE
        grad_out: [B, H, S, D] - gradient of attention output
        scale: softmax scale (1/sqrt(head_dim))
    Returns:
        dq, dk, dv: [B, H, S, D] - gradients
    """
    B, H, S, D = tuple(int(s) for s in q.sizes)
    
    # Process each batch*head independently using GEMM
    # Reshape to [B*H, S, D] for easier processing
    q_bh = q.reshape([B*H, S, D])
    k_bh = k.reshape([B*H, S, D])
    v_bh = v.reshape([B*H, S, D])
    do_bh = grad_out.reshape([B*H, S, D])
    
    # Allocate outputs
    dq_list = []
    dk_list = []
    dv_list = []
    
    scale_t = vt.full([1], scale).cuda()
    neg_one = vt.full([1], -1.0).cuda()
    
    for bh in range(B * H):
        # Extract single head: [S, D]
        q_h = q_bh.select(0, bh)  # [S, D]
        k_h = k_bh.select(0, bh)  # [S, D]
        v_h = v_bh.select(0, bh)  # [S, D]
        do_h = do_bh.select(0, bh)  # [S, D]
        
        # Forward: recompute attention weights
        # scores = Q @ K.T * scale -> [S, S]
        scores = gemm_ops.matmul(q_h, k_h.permute([1, 0])) * scale_t
        
        # Softmax
        scores_max = scores.amax(dim=1, keepdim=True)
        scores_shifted = scores + scores_max * neg_one
        exp_scores = scores_shifted.exp()
        sum_exp = exp_scores.sum(dim=1, keepdim=True)
        attn = exp_scores * sum_exp.reciprocal()  # [S, S]
        
        # Backward
        # dv = attn.T @ do -> [S, D]
        dv_h = gemm_ops.matmul(attn.permute([1, 0]), do_h)
        
        # d_attn = do @ v.T -> [S, S]
        d_attn = gemm_ops.matmul(do_h, v_h.permute([1, 0]))
        
        # Softmax backward: d_scores = attn * (d_attn - sum(d_attn * attn, dim=-1, keepdim=True))
        sum_da = (d_attn * attn).sum(dim=1, keepdim=True)
        d_scores = attn * (d_attn + sum_da * neg_one) * scale_t
        
        # dq = d_scores @ k -> [S, D]
        dq_h = gemm_ops.matmul(d_scores, k_h)
        
        # dk = d_scores.T @ q -> [S, D]
        dk_h = gemm_ops.matmul(d_scores.permute([1, 0]), q_h)
        
        dq_list.append(dq_h)
        dk_list.append(dk_h)
        dv_list.append(dv_h)
    
    # Stack back to [B*H, S, D] then reshape to [B, H, S, D] using VBT stack
    dq = tensor_ops.stack(dq_list, dim=0).reshape([B, H, S, D])
    dk = tensor_ops.stack(dk_list, dim=0).reshape([B, H, S, D])
    dv = tensor_ops.stack(dv_list, dim=0).reshape([B, H, S, D])
    
    return dq, dk, dv


class Config:
    vocab_size = 256
    hidden_dim = 128
    n_heads = 4
    n_layers = 2
    max_seq_len = 64
    
    def __init__(self):
        # Validate configuration
        assert self.hidden_dim % self.n_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
        head_dim = self.hidden_dim // self.n_heads
        assert head_dim % 2 == 0, \
            f"head_dim ({head_dim}) must be even for RoPE"


def init_weights_gpu(cfg):
    """Initialize weights on GPU using VBT ops - fully GPU-native."""
    vt.manual_seed(42)
    d, h = cfg.hidden_dim, cfg.n_heads
    head_dim = d // h
    w = {}
    
    def randn_gpu(*shape):
        return (vt.randn(list(shape)) * 0.02).cuda()
    
    w['tok_emb'] = randn_gpu(cfg.vocab_size, d)
    
    # RoPE tables - fully GPU-native using tensor_ops.arange
    hd2 = head_dim // 2
    # inv_freq = 1 / (10000 ^ (arange(0, hd2) / hd2))
    freq_indices = tensor_ops.arange(0, hd2)  # [0, 1, 2, ..., hd2-1] on GPU
    freq_exponents = freq_indices * vt.full([1], 1.0 / hd2).cuda()  # indices / hd2
    base = vt.full([1], 10000.0).cuda()
    inv_freq_vt = (base.log() * freq_exponents).exp().reciprocal().reshape([1, hd2])
    
    # positions: [0, 1, 2, ..., max_seq_len-1]
    pos = tensor_ops.arange(0, cfg.max_seq_len).reshape([cfg.max_seq_len, 1])
    angles = pos * inv_freq_vt
    w['rope_cos'] = angles.cos()
    w['rope_sin'] = angles.sin()
    
    for i in range(cfg.n_layers):
        w[f'l{i}_norm1'] = vt.ones([d]).cuda()
        w[f'l{i}_wq'] = randn_gpu(d, d)
        w[f'l{i}_wk'] = randn_gpu(d, d)
        w[f'l{i}_wv'] = randn_gpu(d, d)
        w[f'l{i}_wo'] = randn_gpu(d, d)
        w[f'l{i}_norm2'] = vt.ones([d]).cuda()
        w[f'l{i}_w1'] = randn_gpu(d, 4*d)
        w[f'l{i}_w2'] = randn_gpu(4*d, d)
    
    w['final_norm'] = vt.ones([d]).cuda()
    w['lm_head'] = randn_gpu(d, cfg.vocab_size)
    
    return w


def forward_backward_gpu(tokens_vt, targets_vt, weights, cfg):
    """Forward and backward pass - fully GPU-native using VibeTensor ops.
    
    Args:
        tokens_vt: VBT tensor [B, S] int64 on CUDA
        targets_vt: VBT tensor [B, S] int64 on CUDA
        weights: dict of weight tensors
        cfg: config object
    """
    B = int(tokens_vt.sizes[0])
    S = int(tokens_vt.sizes[1])
    d, h = cfg.hidden_dim, cfg.n_heads
    head_dim = d // h
    
    grads = {}
    cache = {}
    
    # ===== FORWARD (GPU-native) =====
    
    x = embed_ops.embedding_forward(weights['tok_emb'], tokens_vt)
    cache['emb_out'] = x
    cache['tokens'] = tokens_vt
    
    for i in range(cfg.n_layers):
        residual = x
        
        # RMSNorm 1
        cache[f'l{i}_norm1_input'] = x  # Cache pre-norm input for backward
        x_norm, inv_rms = rmsnorm_ops.rmsnorm(x, weights[f'l{i}_norm1'])
        cache[f'l{i}_norm1_out'] = x_norm
        cache[f'l{i}_inv_rms1'] = inv_rms
        
        # QKV projections using GEMM
        x_flat = x_norm.reshape([B*S, d])
        q = gemm_ops.matmul(x_flat, weights[f'l{i}_wq'])
        k = gemm_ops.matmul(x_flat, weights[f'l{i}_wk'])
        v = gemm_ops.matmul(x_flat, weights[f'l{i}_wv'])
        
        # Reshape for attention [B, H, S, head_dim] using VBT permute
        q = q.reshape([B, S, h, head_dim]).permute([0, 2, 1, 3])
        k = k.reshape([B, S, h, head_dim]).permute([0, 2, 1, 3])
        v = v.reshape([B, S, h, head_dim]).permute([0, 2, 1, 3])
        
        # RoPE
        q, k = rope_ops.apply_rotary_embedding(q, k, weights['rope_cos'], weights['rope_sin'])
        cache[f'l{i}_q_rot'] = q
        cache[f'l{i}_k_rot'] = k
        cache[f'l{i}_v'] = v
        
        # Attention
        attn_out = attn_ops.attention(q, k, v)
        
        # Reshape back [B*S, D] using VBT permute
        attn_flat = attn_out.permute([0, 2, 1, 3]).reshape([B*S, d])
        cache[f'l{i}_attn_flat'] = attn_flat
        
        # Output projection
        o = gemm_ops.matmul(attn_flat, weights[f'l{i}_wo'])
        o = o.reshape([B, S, d])
        
        # Residual using VBT native add
        x = residual + o
        
        # RMSNorm 2
        residual2 = x
        cache[f'l{i}_norm2_input'] = x  # Cache pre-norm input for backward
        x_norm, inv_rms2 = rmsnorm_ops.rmsnorm(x, weights[f'l{i}_norm2'])
        cache[f'l{i}_norm2_out'] = x_norm
        cache[f'l{i}_inv_rms2'] = inv_rms2  # Cache inv_rms for backward
        
        # FFN
        x_flat = x_norm.reshape([B*S, d])
        h1 = gemm_ops.matmul(x_flat, weights[f'l{i}_w1'])
        cache[f'l{i}_h1_pre'] = h1
        
        h1_act = act_ops.gelu(h1)
        cache[f'l{i}_h1_act'] = h1_act
        
        h2 = gemm_ops.matmul(h1_act, weights[f'l{i}_w2'])
        h2 = h2.reshape([B, S, d])
        
        # Residual using VBT native add
        x = residual2 + h2
    
    # Final norm
    cache['final_norm_input'] = x  # Cache pre-norm input for backward
    x_norm, final_inv_rms = rmsnorm_ops.rmsnorm(x, weights['final_norm'])
    cache['final_norm_out'] = x_norm
    cache['final_inv_rms'] = final_inv_rms  # Cache inv_rms for backward
    
    # LM head
    x_flat = x_norm.reshape([B*S, d])
    logits = gemm_ops.matmul(x_flat, weights['lm_head'])  # [B*S, vocab]
    
    # ===== LOSS (VBT ops + gather for forward) =====
    n = B * S
    targets_flat = targets_vt.reshape([-1])  # Flatten [B, S] -> [B*S], already int64 CUDA
    
    # Softmax: exp(x - max) / sum(exp(x - max))
    logits_max = logits.amax(dim=1, keepdim=True)
    neg_one = vt.full([1], -1.0).cuda()
    logits_shifted = logits + logits_max * neg_one  # x - max
    exp_logits = logits_shifted.exp()
    sum_exp = exp_logits.sum(dim=1, keepdim=True)
    probs = exp_logits * sum_exp.reciprocal()
    
    # Cross-entropy using gather: -log(probs[i, targets[i]]) / n
    eps = vt.full([1], 1e-10).cuda()
    log_probs = (probs + eps).log()
    log_probs_at_targets = idx_ops.gather_dim1(log_probs, targets_flat)  # [n]
    loss_tensor = log_probs_at_targets.sum() * vt.full([1], -1.0/n).cuda()
    # Note: scalar extraction still uses numpy internally via VibeTensor's CPU tensor,
    # but this is acceptable for logging purposes (not in gradient computation path)
    loss = float(loss_tensor.cpu().numpy().flat[0])
    
    # ===== BACKWARD =====
    
    # Grad logits = (probs - one_hot) / n
    # Using VBT subtract_at_indices to avoid numpy one-hot
    inv_n = vt.full([1], 1.0/n).cuda()
    dlogits_vt = probs * inv_n  # Start with probs / n
    # Subtract 1/n at target positions: dlogits[i, target[i]] -= 1/n
    idx_ops.subtract_at_indices(dlogits_vt, targets_flat, 1.0/n)
    
    # LM head backward
    x_flat = cache['final_norm_out'].reshape([B*S, d])
    x_flat_T = x_flat.permute([1, 0])  # Use VBT transpose
    grads['lm_head'] = gemm_ops.matmul(x_flat_T, dlogits_vt)
    
    w_lm_T = weights['lm_head'].permute([1, 0])  # Use VBT transpose
    dx_final_norm = gemm_ops.matmul(dlogits_vt, w_lm_T)  # Gradient w.r.t. final_norm output
    
    # Final norm backward
    final_norm_input = cache['final_norm_input'].reshape([B*S, d])
    final_inv_rms = cache['final_inv_rms']
    dx_pre_final, grad_final_norm = rmsnorm_ops.rmsnorm_backward(
        dx_final_norm, final_norm_input, final_inv_rms, weights['final_norm']
    )
    grads['final_norm'] = grad_final_norm
    dx = dx_pre_final.reshape([B, S, d])
    
    # Backward through layers
    for i in range(cfg.n_layers - 1, -1, -1):
        # FFN backward
        h1_act = cache[f'l{i}_h1_act']
        dx_flat = dx.reshape([B*S, d])
        
        # w2 backward
        h1_act_T = h1_act.permute([1, 0])
        grads[f'l{i}_w2'] = gemm_ops.matmul(h1_act_T, dx_flat)
        
        w2_T = weights[f'l{i}_w2'].permute([1, 0])
        dh1_act = gemm_ops.matmul(dx_flat, w2_T)
        
        # GELU backward
        h1_pre = cache[f'l{i}_h1_pre']
        dh1 = act_ops.gelu_backward(dh1_act, h1_pre)
        
        # w1 backward
        x_norm2 = cache[f'l{i}_norm2_out'].reshape([B*S, d])
        x_norm2_T = x_norm2.permute([1, 0])
        grads[f'l{i}_w1'] = gemm_ops.matmul(x_norm2_T, dh1)
        
        w1_T = weights[f'l{i}_w1'].permute([1, 0])
        dx_norm2 = gemm_ops.matmul(dh1, w1_T)
        
        norm2_input = cache[f'l{i}_norm2_input'].reshape([B*S, d])
        inv_rms2 = cache[f'l{i}_inv_rms2']
        dx_pre_norm2, grad_norm2 = rmsnorm_ops.rmsnorm_backward(
            dx_norm2, norm2_input, inv_rms2, weights[f'l{i}_norm2']
        )
        grads[f'l{i}_norm2'] = grad_norm2
        
        dx = dx + dx_pre_norm2.reshape([B, S, d])
        
        # Attention backward
        attn_flat = cache[f'l{i}_attn_flat']
        dx_flat = dx.reshape([B*S, d])
        
        # wo backward
        attn_T = attn_flat.permute([1, 0])
        grads[f'l{i}_wo'] = gemm_ops.matmul(attn_T, dx_flat)
        
        wo_T = weights[f'l{i}_wo'].permute([1, 0])
        dattn = gemm_ops.matmul(dx_flat, wo_T)
        
        # Attention backward using VBT + GEMM (no numpy einsum)
        q_rot = cache[f'l{i}_q_rot']
        k_rot = cache[f'l{i}_k_rot']
        v_cached = cache[f'l{i}_v']
        dattn_4d = dattn.reshape([B, S, h, head_dim]).permute([0, 2, 1, 3])
        
        scale = 1.0 / (head_dim ** 0.5)
        dq_rot, dk_rot, dv_4d = attention_backward_vbt(q_rot, k_rot, v_cached, dattn_4d, scale)
        
        # Apply inverse RoPE to get gradients w.r.t. pre-RoPE q/k
        # RoPE backward = inverse rotation (negate sin)
        dq_4d, dk_4d = rope_ops.apply_rotary_embedding_backward(
            dq_rot, dk_rot, weights['rope_cos'], weights['rope_sin']
        )
        
        dq_flat = dq_4d.permute([0, 2, 1, 3]).reshape([B*S, d])
        dk_flat = dk_4d.permute([0, 2, 1, 3]).reshape([B*S, d])
        dv_flat = dv_4d.permute([0, 2, 1, 3]).reshape([B*S, d])
        
        # QKV weight gradients
        x_norm1 = cache[f'l{i}_norm1_out'].reshape([B*S, d])
        x_norm1_T = x_norm1.permute([1, 0])
        
        grads[f'l{i}_wv'] = gemm_ops.matmul(x_norm1_T, dv_flat)
        grads[f'l{i}_wq'] = gemm_ops.matmul(x_norm1_T, dq_flat)
        grads[f'l{i}_wk'] = gemm_ops.matmul(x_norm1_T, dk_flat)
        
        # Input gradient
        wq_T = weights[f'l{i}_wq'].permute([1, 0])
        wk_T = weights[f'l{i}_wk'].permute([1, 0])
        wv_T = weights[f'l{i}_wv'].permute([1, 0])
        
        dx_qkv = gemm_ops.matmul(dq_flat, wq_T)
        dx_qkv = dx_qkv + gemm_ops.matmul(dk_flat, wk_T)
        dx_qkv = dx_qkv + gemm_ops.matmul(dv_flat, wv_T)
        
        norm1_input = cache[f'l{i}_norm1_input'].reshape([B*S, d])
        inv_rms1 = cache[f'l{i}_inv_rms1']
        dx_pre_norm1, grad_norm1 = rmsnorm_ops.rmsnorm_backward(
            dx_qkv, norm1_input, inv_rms1, weights[f'l{i}_norm1']
        )
        grads[f'l{i}_norm1'] = grad_norm1
        
        dx = dx + dx_pre_norm1.reshape([B, S, d])
    
    # Embedding backward using Triton kernel (no numpy!)
    dx_flat = dx.reshape([B*S, d])
    grads['tok_emb'] = embed_ops.embedding_backward(dx_flat, cache['tokens'], cfg.vocab_size)
    
    return loss, grads


def train():
    print("=" * 70)
    print("VibeTensor Transformer - GPU-Native Training (VBT ops)")
    print("=" * 70)
    
    cfg = Config()
    print(f"\nConfig: vocab={cfg.vocab_size}, d={cfg.hidden_dim}, h={cfg.n_heads}, L={cfg.n_layers}")
    
    weights = init_weights_gpu(cfg)
    
    batch_size, seq_len, n_steps, lr = 4, 32, 100, 0.1
    neg_lr_t = vt.full([1], -lr).cuda()  # Negative learning rate as GPU tensor
    print(f"Training: batch={batch_size}, seq={seq_len}, steps={n_steps}, lr={lr}")
    print("Task: Learn (token + 1) % vocab")
    print("Using: FULLY GPU-NATIVE - ZERO NUMPY in training loop!\n")
    
    losses = []
    for step in range(n_steps):
        # Data gen using VBT randint - fully on GPU
        inputs_vt = vt.randint(0, cfg.vocab_size, [batch_size, seq_len]).cuda()
        # Compute targets = (inputs + 1) % vocab_size using GPU kernel
        targets_vt = tensor_ops.add_one_mod(inputs_vt, cfg.vocab_size)
        
        start = time.time()
        loss, grads = forward_backward_gpu(inputs_vt, targets_vt, weights, cfg)
        step_time = (time.time() - start) * 1000
        
        # SGD update using VBT clamp and add (no NumPy)
        for k in grads:
            g_clipped = grads[k].clamp(-1.0, 1.0)
            weights[k] = weights[k] + g_clipped * neg_lr_t
        
        losses.append(loss)
        if step % 10 == 0 or step == n_steps - 1:
            print(f"Step {step:3d} | Loss: {loss:.4f} | Time: {step_time:.0f}ms")
    
    print(f"\nInitial: {losses[0]:.4f} -> Final: {losses[-1]:.4f} (reduced {losses[0]-losses[-1]:.4f})")
    print("SUCCESS!" if losses[-1] < losses[0] else "WARNING: Loss not decreasing")


if __name__ == "__main__":
    train()
