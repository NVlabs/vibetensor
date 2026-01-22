# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collection of custom Triton kernels for nanochat.

Note: All PyTorch-based modules use lazy imports to allow PyTorch-free usage
when only using VibeTensor native ops (kernel_factory.ops).
"""

__all__ = [
    "GEMMTiling",
    "make_default_gemm_configs",
    "triton_gemm",
    "triton_gemm_backward",
    "is_triton_available",
    "FusedEmbeddingRMSNorm",
    "apply_rotary_embedding",
    "layernorm",
    "CuTeDSLLayerNorm",
    "is_cutedsl_layernorm_available",
    "cutedsl_layernorm",
    "RMSNorm",
    "fused_attention",
    "relu_squared",
    "softcap_tanh_projection",
    "elementwise_add",
    "elementwise_mul",
    "elementwise_where",
    "elementwise_lerp",
    "rowwise_l2_norm",
    "cross_entropy_loss",
    "sample_logits",
    "TritonAdamW",
    "TritonDistAdamW",
    "TritonMuon",
    "TritonDistMuon",
    "compute_global_grad_norm",
    "clip_grad_norm_",
    # Indexing ops
    "gather",
    "gather_with_grad",
    "scatter_add",
    "scatter_add_",
]

# Lazy import caches
_gemm_exports = None
_layernorm_exports = None
_activation_exports = None
_indexing_exports = None
_optim_exports = None


def __getattr__(name):
    global _gemm_exports, _layernorm_exports, _activation_exports
    global _indexing_exports, _optim_exports
    
    # GEMM exports
    if name in ("GEMMTiling", "is_triton_available", "make_default_gemm_configs",
                "triton_gemm", "triton_gemm_backward"):
        if _gemm_exports is None:
            from .gemm import (
                GEMMTiling, is_triton_available, make_default_gemm_configs,
                triton_gemm, triton_gemm_backward
            )
            _gemm_exports = {
                "GEMMTiling": GEMMTiling,
                "is_triton_available": is_triton_available,
                "make_default_gemm_configs": make_default_gemm_configs,
                "triton_gemm": triton_gemm,
                "triton_gemm_backward": triton_gemm_backward,
            }
        return _gemm_exports[name]
    
    # Attention
    if name == "fused_attention":
        from .attention import fused_attention
        return fused_attention
    
    # Embedding
    if name == "FusedEmbeddingRMSNorm":
        from .embedding import FusedEmbeddingRMSNorm
        return FusedEmbeddingRMSNorm
    
    # LayerNorm exports
    if name in ("CuTeDSLLayerNorm", "cutedsl_layernorm", "is_cutedsl_layernorm_available", "layernorm"):
        if _layernorm_exports is None:
            from importlib import import_module
            _ln_mod = import_module(".layernorm", __name__)
            _layernorm_exports = {
                "CuTeDSLLayerNorm": _ln_mod.CuTeDSLLayerNorm,
                "cutedsl_layernorm": _ln_mod.cutedsl_layernorm,
                "is_cutedsl_layernorm_available": _ln_mod.is_cutedsl_layernorm_available,
                "layernorm": _ln_mod.layernorm,
            }
        return _layernorm_exports[name]
    
    # RMSNorm
    if name == "RMSNorm":
        from .rmsnorm import RMSNorm
        return RMSNorm
    
    # Rotary
    if name == "apply_rotary_embedding":
        from .rotary import apply_rotary_embedding
        return apply_rotary_embedding
    
    # Sampling
    if name == "sample_logits":
        from .sampling import sample_logits
        return sample_logits
    
    # Activation ops
    if name in ("relu_squared", "softcap_tanh_projection", "elementwise_add",
                "elementwise_mul", "elementwise_where", "elementwise_lerp", "rowwise_l2_norm"):
        if _activation_exports is None:
            from . import activation as _act
            _activation_exports = {
                "relu_squared": _act.relu_squared,
                "softcap_tanh_projection": _act.softcap_tanh_projection,
                "elementwise_add": _act.elementwise_add,
                "elementwise_mul": _act.elementwise_mul,
                "elementwise_where": _act.elementwise_where,
                "elementwise_lerp": _act.elementwise_lerp,
                "rowwise_l2_norm": _act.rowwise_l2_norm,
            }
        return _activation_exports[name]
    
    # Indexing ops
    if name in ("gather", "gather_with_grad", "scatter_add", "scatter_add_"):
        if _indexing_exports is None:
            from .indexing import gather, gather_with_grad, scatter_add, scatter_add_
            _indexing_exports = {
                "gather": gather,
                "gather_with_grad": gather_with_grad,
                "scatter_add": scatter_add,
                "scatter_add_": scatter_add_,
            }
        return _indexing_exports[name]
    
    # Optim (commented out in original, but keep for compatibility)
    if name in ("TritonAdamW", "TritonDistAdamW", "TritonMuon", "TritonDistMuon",
                "compute_global_grad_norm", "clip_grad_norm_"):
        if _optim_exports is None:
            from importlib import import_module
            _optim_mod = import_module(".optim", __name__)
            _optim_exports = {
                "TritonAdamW": _optim_mod.TritonAdamW,
                "TritonDistAdamW": _optim_mod.TritonDistAdamW,
                "TritonMuon": _optim_mod.TritonMuon,
                "TritonDistMuon": _optim_mod.TritonDistMuon,
                "compute_global_grad_norm": _optim_mod.compute_global_grad_norm,
                "clip_grad_norm_": _optim_mod.clip_grad_norm_,
            }
        return _optim_exports[name]
    
    # Loss
    if name == "cross_entropy_loss":
        from .loss import cross_entropy_loss
        return cross_entropy_loss
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
