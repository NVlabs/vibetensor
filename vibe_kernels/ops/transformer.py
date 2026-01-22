# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transformer model composed of modular ops."""

import vibetensor.torch as vt
from .base import Op
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .linear import Linear
from .rope import RoPE
from .attention import Attention
from .ffn import FFN
from .loss import CrossEntropyLoss


class TransformerBlock(Op):
    """Single transformer block with attention and FFN."""
    
    def __init__(self, dim: int, n_heads: int, rope: RoPE, *, causal: bool = False, use_triton_attn_bwd: bool = True):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Ops - RoPE is shared across all layers
        self.norm1 = RMSNorm(dim)
        self.wq = Linear(dim, dim)
        self.wk = Linear(dim, dim)
        self.wv = Linear(dim, dim)
        self.wo = Linear(dim, dim)
        self.rope = rope  # Shared RoPE
        self.attn = Attention(n_heads, self.head_dim, causal=bool(causal), use_triton_bwd=bool(use_triton_attn_bwd))
        
        self.norm2 = RMSNorm(dim)
        self.ffn = FFN(dim, 4 * dim)
    
    def fwd(self, x):
        """Forward pass through transformer block.
        
        Args:
            x: [B, S, D] input tensor
        Returns:
            out: [B, S, D] output tensor
        """
        B = int(x.sizes[0])
        S = int(x.sizes[1])
        D = self.dim
        H = self.n_heads
        HD = self.head_dim
        
        # Store for residual
        residual1 = x
        self._cache['residual1'] = residual1
        
        # Attention block (reshape to 2D for consistent norm backward)
        x_flat = x.reshape([B * S, D])
        x_norm_flat = self.norm1.fwd(x_flat)
        q = self.wq.fwd(x_norm_flat)
        k = self.wk.fwd(x_norm_flat)
        v = self.wv.fwd(x_norm_flat)
        
        # Reshape for attention [B, H, S, HD]
        q = q.reshape([B, S, H, HD]).permute([0, 2, 1, 3])
        k = k.reshape([B, S, H, HD]).permute([0, 2, 1, 3])
        v = v.reshape([B, S, H, HD]).permute([0, 2, 1, 3])
        
        # RoPE
        q_rot, k_rot = self.rope.fwd(q, k)
        self._cache['v'] = v
        
        # Attention
        attn_out = self.attn.fwd(q_rot, k_rot, v)
        
        # Reshape back [B*S, D]
        attn_flat = attn_out.permute([0, 2, 1, 3]).reshape([B * S, D])
        self._cache['attn_flat'] = attn_flat
        
        # Output projection
        o = self.wo.fwd(attn_flat)
        o = o.reshape([B, S, D])
        
        # Residual
        x = residual1 + o
        
        # FFN block (reshape to 2D for consistent norm backward)
        residual2 = x
        self._cache['residual2'] = residual2
        
        x_flat = x.reshape([B * S, D])
        x_norm2_flat = self.norm2.fwd(x_flat)
        
        h = self.ffn.fwd(x_norm2_flat)
        h = h.reshape([B, S, D])
        
        # Residual
        x = residual2 + h
        
        return x
    
    def bwd(self, grad_out):
        """Backward pass through transformer block."""
        B = int(grad_out.sizes[0])
        S = int(grad_out.sizes[1])
        D = self.dim
        H = self.n_heads
        HD = self.head_dim
        
        # FFN backward
        dx = grad_out  # Gradient from residual path
        
        # FFN
        dx_flat = dx.reshape([B * S, D])
        dx_norm2 = self.ffn.bwd(dx_flat)
        
        # Norm2 backward
        dx_pre_norm2 = self.norm2.bwd(dx_norm2)
        dx_pre_norm2 = dx_pre_norm2.reshape([B, S, D])
        
        # Residual gradient
        dx = dx + dx_pre_norm2
        
        # Attention backward
        dx_flat = dx.reshape([B * S, D])
        
        # wo backward
        dattn_flat = self.wo.bwd(dx_flat)
        dattn = dattn_flat.reshape([B, S, H, HD]).permute([0, 2, 1, 3])
        
        # Attention backward
        dq_rot, dk_rot, dv = self.attn.bwd(dattn)
        
        # RoPE backward
        dq, dk = self.rope.bwd(dq_rot, dk_rot)
        
        # Reshape gradients
        dq_flat = dq.permute([0, 2, 1, 3]).reshape([B * S, D])
        dk_flat = dk.permute([0, 2, 1, 3]).reshape([B * S, D])
        dv_flat = dv.permute([0, 2, 1, 3]).reshape([B * S, D])
        
        # QKV weight backward
        dx_qkv = self.wq.bwd(dq_flat)
        dx_qkv = dx_qkv + self.wk.bwd(dk_flat)
        dx_qkv = dx_qkv + self.wv.bwd(dv_flat)
        
        # Norm1 backward
        dx_pre_norm1 = self.norm1.bwd(dx_qkv)
        dx_pre_norm1 = dx_pre_norm1.reshape([B, S, D])
        
        # Residual gradient
        dx = dx + dx_pre_norm1
        
        return dx
    
    def update(self, lr):
        """Update all weights in block."""
        self.norm1.update(lr)
        self.wq.update(lr)
        self.wk.update(lr)
        self.wv.update(lr)
        self.wo.update(lr)
        self.norm2.update(lr)
        self.ffn.update(lr)


class Transformer(Op):
    """Full transformer model."""
    
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        *,
        causal: bool = False,
        use_triton_attn_bwd: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        head_dim = dim // n_heads
        
        # Ops - RoPE created AFTER embedding, BEFORE layer weights (matches old script)
        self.embedding = Embedding(vocab_size, dim)
        self.rope = RoPE(head_dim, max_seq_len)  # Shared across all layers
        self.blocks = [
            TransformerBlock(dim, n_heads, self.rope, causal=bool(causal), use_triton_attn_bwd=bool(use_triton_attn_bwd))
            for _ in range(n_layers)
        ]
        self.final_norm = RMSNorm(dim)
        self.lm_head = Linear(dim, vocab_size)
        self.loss_fn = CrossEntropyLoss(vocab_size)
    
    def fwd(self, tokens):
        """Forward pass.
        
        Args:
            tokens: [B, S] int64 token indices
        Returns:
            logits: [B*S, vocab_size] unnormalized logits
        """
        B = int(tokens.sizes[0])
        S = int(tokens.sizes[1])
        
        # Embedding
        x = self.embedding.fwd(tokens)
        self._cache['tokens'] = tokens
        
        # Transformer blocks
        for block in self.blocks:
            x = block.fwd(x)
        
        # Final norm (reshape to 2D for consistent backward)
        x_flat = x.reshape([B * S, self.dim])
        x_norm = self.final_norm.fwd(x_flat)
        
        # LM head
        logits = self.lm_head.fwd(x_norm)
        
        return logits
    
    def compute_loss(self, logits, targets):
        """Compute cross-entropy loss.
        
        Args:
            logits: [B*S, vocab_size]
            targets: [B, S] int64 targets
        Returns:
            loss: scalar float
        """
        targets_flat = targets.reshape([-1])
        return self.loss_fn.fwd(logits, targets_flat)
    
    def bwd(self):
        """Backward pass (assumes loss.fwd was called)."""
        B = int(self._cache['tokens'].sizes[0])
        S = int(self._cache['tokens'].sizes[1])
        
        # Loss backward
        dlogits = self.loss_fn.bwd()
        
        # LM head backward
        dx_flat = self.lm_head.bwd(dlogits)
        
        # Final norm backward
        dx = self.final_norm.bwd(dx_flat)
        dx = dx.reshape([B, S, self.dim])
        
        # Transformer blocks backward (reverse order)
        for block in reversed(self.blocks):
            dx = block.bwd(dx)
        
        # Embedding backward
        dx_flat = dx.reshape([B * S, self.dim])
        self.embedding.bwd(dx_flat)
    
    def update(self, lr):
        """Update all weights."""
        self.embedding.update(lr)
        for block in self.blocks:
            block.update(lr)
        self.final_norm.update(lr)
        self.lm_head.update(lr)
