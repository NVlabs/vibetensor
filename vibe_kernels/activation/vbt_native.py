# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure VibeTensor implementation of activation functions using Triton kernels.

This module provides GELU, SiLU (Swish), and ReLU activations for use
with VibeTensor tensors, with NO PyTorch dependency at runtime.

Usage:
    import vibetensor.torch as vt
    from vibe_kernels.activation import vbt_native as act_ops
    
    x = vt.cuda.to_device(np.random.randn(4, 256).astype(np.float32))
    y_gelu = act_ops.gelu(x)
    y_silu = act_ops.silu(x)
"""

from __future__ import annotations

import math
import threading
from typing import Tuple, Dict

import triton
import triton.language as tl


def _get_vbt_modules():
    """Lazy import VibeTensor modules."""
    import vibetensor.torch as vt
    from vibetensor import _C
    import vibetensor.triton as vt_triton
    return vt, _C, vt_triton


# -----------------------------------------------------------------------------
# Activation Kernels
# -----------------------------------------------------------------------------

@triton.jit
def _gelu_fwd_kernel(
    out_ptr,
    inp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU forward: x * 0.5 * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # GELU approximation using tanh
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coef = 0.044715
    
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + coef * x3)
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2inner = tl.exp(2.0 * inner)
    tanh_inner = (exp_2inner - 1.0) / (exp_2inner + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _gelu_bwd_kernel(
    grad_inp_ptr,
    grad_out_ptr,
    inp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GELU backward."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    sqrt_2_over_pi = 0.7978845608028654
    coef = 0.044715
    
    x2 = x * x
    x3 = x2 * x
    inner = sqrt_2_over_pi * (x + coef * x3)
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2inner = tl.exp(2.0 * inner)
    tanh_inner = (exp_2inner - 1.0) / (exp_2inner + 1.0)
    
    # Derivative of GELU
    sech2 = 1.0 - tanh_inner * tanh_inner
    d_inner = sqrt_2_over_pi * (1.0 + 3.0 * coef * x2)
    
    grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
    grad_inp = grad_out * grad
    
    tl.store(grad_inp_ptr + offsets, grad_inp, mask=mask)


@triton.jit
def _silu_fwd_kernel(
    out_ptr,
    inp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU (Swish) forward: x * sigmoid(x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    sigmoid_x = tl.sigmoid(x)
    y = x * sigmoid_x
    
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _silu_bwd_kernel(
    grad_inp_ptr,
    grad_out_ptr,
    inp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU backward: grad_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    sigmoid_x = tl.sigmoid(x)
    # d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    grad = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
    grad_inp = grad_out * grad
    
    tl.store(grad_inp_ptr + offsets, grad_inp, mask=mask)


@triton.jit
def _relu_fwd_kernel(
    out_ptr,
    inp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU forward: max(0, x)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(x > 0, x, 0.0)
    
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_bwd_kernel(
    grad_inp_ptr,
    grad_out_ptr,
    inp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU backward: grad_out * (x > 0)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    grad_inp = tl.where(x > 0, grad_out, 0.0)
    
    tl.store(grad_inp_ptr + offsets, grad_inp, mask=mask)


# -----------------------------------------------------------------------------
# Kernel Cache
# -----------------------------------------------------------------------------

_kernel_cache: Dict[str, Tuple[int, int, int, int]] = {}
_cache_lock = threading.Lock()


def _get_activation_kernel(
    name: str,
    kernel_fn,
    block_size: int,
    device_idx: int,
    is_backward: bool = False,
) -> Tuple[int, int, int, int]:
    """Get compiled kernel handle for activation function."""
    vt, _C, vt_triton = _get_vbt_modules()
    from triton.compiler.compiler import ASTSource
    from triton.runtime import driver
    
    suffix = "_bwd" if is_backward else "_fwd"
    key = f"{name}{suffix}_{block_size}_{device_idx}"
    
    with _cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]
    
    if is_backward:
        signature = "*fp32,*fp32,*fp32,i32"
    else:
        signature = "*fp32,*fp32,i32"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    
    meta = {"BLOCK_SIZE": block_size}
    
    arg_names = kernel_fn.arg_names
    params = kernel_fn.params
    sig_map = {}
    idx = 0
    for i, p in enumerate(params):
        nm = arg_names[i]
        is_constexpr = bool(getattr(p, "is_constexpr", False))
        if is_constexpr:
            sig_map[nm] = "constexpr"
        else:
            sig_map[nm] = sig_tokens[idx]
            idx += 1
    
    target = driver.active.get_current_target()
    src = ASTSource(kernel_fn, sig_map, meta, {})
    compiled = triton.compile(src, target=target, options={"num_warps": 4})
    
    asm = getattr(compiled, "asm", {})
    if "ptx" not in asm:
        raise RuntimeError("Triton compile produced no PTX")
    ptx_val = asm["ptx"]
    ptx = ptx_val if isinstance(ptx_val, bytes) else str(ptx_val).encode("utf-8")
    
    entry = getattr(compiled, "name", None)
    if not isinstance(entry, str):
        md = getattr(compiled, "metadata", None)
        if hasattr(md, "name"):
            entry = md.name
    if not isinstance(entry, str) or not entry:
        entry = f"_{name}{suffix}_kernel"
    
    md = getattr(compiled, "metadata", None)
    shared_mem = 0
    if hasattr(md, "shared"):
        shared_mem = int(md.shared)
    
    total_params = vt_triton._count_entry_params(ptx, entry)
    extra_params = max(0, total_params - len(sig_tokens))
    
    mod_h = _C._cuda_module_load_ptx(ptx)
    func_h = _C._cuda_module_get_function(mod_h, entry)
    
    with _cache_lock:
        _kernel_cache[key] = (mod_h, func_h, extra_params, shared_mem)
    
    return mod_h, func_h, extra_params, shared_mem


# -----------------------------------------------------------------------------
# Generic activation launcher
# -----------------------------------------------------------------------------

def _activation_forward(x, name: str, kernel_fn):
    """Generic activation forward pass."""
    vt, _C, _ = _get_vbt_modules()
    
    x_sizes = tuple(int(s) for s in x.sizes)
    x_dtype = str(x.dtype)
    x_device = x.device
    
    if x_device[0] != 2:
        raise RuntimeError(f"{name} requires CUDA tensors")
    if x_dtype != "float32":
        raise TypeError(f"{name} requires float32 (for now)")
    
    device_idx = int(x_device[1])
    n_elements = 1
    for s in x_sizes:
        n_elements *= s
    
    out = _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    
    block_size = 1024
    _, func_h, extra_params, shmem = _get_activation_kernel(
        name, kernel_fn, block_size, device_idx, is_backward=False
    )
    
    grid = ((n_elements + block_size - 1) // block_size, 1, 1)
    block = (128, 1, 1)  # num_warps=4
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [out, x, n_elements]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return out


def _activation_backward(grad_output, x, name: str, kernel_fn):
    """Generic activation backward pass."""
    vt, _C, _ = _get_vbt_modules()
    
    x_sizes = tuple(int(s) for s in x.sizes)
    x_dtype = str(x.dtype)
    x_device = x.device
    
    if x_device[0] != 2:
        raise RuntimeError(f"{name}_backward requires CUDA tensors")
    if x_dtype != "float32":
        raise TypeError(f"{name}_backward requires float32 (for now)")
    
    device_idx = int(x_device[1])
    n_elements = 1
    for s in x_sizes:
        n_elements *= s
    
    grad_input = _C._cuda_empty(list(x_sizes), x_dtype, device_idx)
    
    block_size = 1024
    _, func_h, extra_params, shmem = _get_activation_kernel(
        name, kernel_fn, block_size, device_idx, is_backward=True
    )
    
    grid = ((n_elements + block_size - 1) // block_size, 1, 1)
    block = (128, 1, 1)
    stream = _C._cuda_stream_handle_current_for_device(device_idx)
    
    args = [grad_input, grad_output, x, n_elements]
    args.extend([None] * extra_params)
    
    _C._cuda_launch(func_h, grid, block, shmem, stream, args)
    
    return grad_input


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def gelu(x):
    """GELU activation (tanh approximation).
    
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    """
    return _activation_forward(x, "gelu", _gelu_fwd_kernel)


def gelu_backward(grad_output, x):
    """GELU backward pass."""
    return _activation_backward(grad_output, x, "gelu", _gelu_bwd_kernel)


def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x).
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    """
    return _activation_forward(x, "silu", _silu_fwd_kernel)


def silu_backward(grad_output, x):
    """SiLU backward pass."""
    return _activation_backward(grad_output, x, "silu", _silu_bwd_kernel)


def relu(x):
    """ReLU activation: max(0, x).
    
    Pure VibeTensor implementation - NO PyTorch dependency.
    """
    return _activation_forward(x, "relu", _relu_fwd_kernel)


def relu_backward(grad_output, x):
    """ReLU backward pass."""
    return _activation_backward(grad_output, x, "relu", _relu_bwd_kernel)


# Aliases
swish = silu
swish_backward = silu_backward


__all__ = [
    "gelu",
    "gelu_backward",
    "silu",
    "silu_backward",
    "swish",
    "swish_backward",
    "relu",
    "relu_backward",
]
