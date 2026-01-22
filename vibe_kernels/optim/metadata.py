# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata utilities for packing optimizer tensor lists.

The fused kernels operate on batched lists of parameter / gradient / state
pointers.  This module will own the helper dataclasses and construction
routines.  The current revision only sketches the API so that the surrounding
code can type-check; the actual implementations will be filled in alongside the
Triton kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import torch
from torch.nn import Parameter

from vibe_kernels.common.tensor_types import TensorLike

Tensor = TensorLike


@dataclass
class TensorPointerBatch:
    """Pointers and shape information for a batch of tensors.

    Attributes:
        param_ptrs: Device tensor of raw pointers to parameter shards.
        grad_ptrs: Device tensor of raw pointers to gradient shards.
        state1_ptrs: Optional device tensor of raw pointers to the first
            optimizer state tensor (e.g. exp_avg).
        state2_ptrs: Optional device tensor of raw pointers to a second state
            tensor (e.g. exp_avg_sq).
        numel: Device tensor containing the element counts per tensor in the
            batch.
        stride: Device tensor describing the stride (in elements) between
            successive ILP lanes inside the kernel.
    """

    param_ptrs: Tensor
    grad_ptrs: Tensor
    state1_ptrs: Optional[Tensor]
    state2_ptrs: Optional[Tensor]
    numel: Tensor
    stride: Tensor


@dataclass
class ShardMetadata:
    """Describes the slice of a parameter owned by the current rank."""

    parameter: Parameter
    offset: int
    length: int


def pack_tensor_pointer_batch(
    parameters: Sequence[Parameter],
    gradients: Sequence[Optional[Tensor]],
    state_one: Optional[Sequence[Tensor]] = None,
    state_two: Optional[Sequence[Tensor]] = None,
    *,
    chunk_size: int = 4096,
    device: Optional[torch.device] = None,
) -> Tuple[TensorPointerBatch, int]:
    """Build the pointer metadata required by the fused kernels.

    Args:
        parameters: Ordered sequence of parameters that will be updated.
        gradients: Gradients matching ``parameters`` (``None`` entries are
            skipped).
        state_one: Optional first optimizer state (e.g. exp_avg).
        state_two: Optional second optimizer state (e.g. exp_avg_sq).
        chunk_size: Number of elements each kernel block will process.
        device: Device on which to place the metadata tensors.

    Returns:
        A ``TensorPointerBatch`` describing the packed tensor pointers and the
        total number of valid tensor entries that were emitted.
    """

    if len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must be the same length")
    if state_one is not None and len(state_one) != len(parameters):
        raise ValueError("state_one length must match parameters")
    if state_two is not None and len(state_two) != len(parameters):
        raise ValueError("state_two length must match parameters")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    chosen_device: Optional[torch.device] = device
    if chosen_device is None:
        for param in parameters:
            if isinstance(param, Parameter):
                chosen_device = param.device
                break
        else:
            chosen_device = torch.device("cpu")

    param_ptrs: list[int] = []
    grad_ptrs: list[int] = []
    state1_ptrs: list[int] = []
    state2_ptrs: list[int] = []
    lengths: list[int] = []
    strides: list[int] = []

    for idx, (param, grad) in enumerate(zip(parameters, gradients)):
        if grad is None:
            continue
        if grad.device != param.device:
            raise ValueError("Gradient and parameter must live on the same device")
        if not param.is_contiguous():
            raise ValueError("Non-contiguous parameters are not yet supported")
        if not grad.is_contiguous():
            raise ValueError("Non-contiguous gradients are not yet supported")

        state1 = state_one[idx] if state_one is not None else None
        state2 = state_two[idx] if state_two is not None else None
        if state1 is not None and not state1.is_contiguous():
            raise ValueError("Non-contiguous optimizer state tensors are not supported")
        if state2 is not None and not state2.is_contiguous():
            raise ValueError("Non-contiguous optimizer state tensors are not supported")

        numel = param.numel()
        elem_size = param.element_size()
        grad_elem_size = grad.element_size()
        if grad.numel() != numel:
            raise ValueError("Gradient and parameter must have identical numel")
        if state1 is not None and state1.numel() != numel:
            raise ValueError("state_one tensor must match parameter numel")
        if state2 is not None and state2.numel() != numel:
            raise ValueError("state_two tensor must match parameter numel")

        state1_elem_size = state1.element_size() if state1 is not None else 0
        state2_elem_size = state2.element_size() if state2 is not None else 0

        base_param_ptr = param.data_ptr()
        base_grad_ptr = grad.data_ptr()
        base_state1_ptr = state1.data_ptr() if state1 is not None else 0
        base_state2_ptr = state2.data_ptr() if state2 is not None else 0

        for offset in range(0, numel, chunk_size):
            chunk_len = min(chunk_size, numel - offset)
            byte_offset = offset * elem_size
            param_ptrs.append(base_param_ptr + byte_offset)
            grad_ptrs.append(base_grad_ptr + offset * grad_elem_size)
            if state1 is not None:
                state1_ptrs.append(base_state1_ptr + offset * state1_elem_size)
            if state2 is not None:
                state2_ptrs.append(base_state2_ptr + offset * state2_elem_size)
            lengths.append(chunk_len)
            strides.append(1)

    count = len(param_ptrs)
    param_tensor = torch.tensor(param_ptrs, dtype=torch.int64, device=chosen_device)
    grad_tensor = torch.tensor(grad_ptrs, dtype=torch.int64, device=chosen_device)
    state1_tensor: Optional[Tensor]
    state2_tensor: Optional[Tensor]
    if state1_ptrs:
        state1_tensor = torch.tensor(
            state1_ptrs, dtype=torch.int64, device=chosen_device
        )
    else:
        state1_tensor = None
    if state2_ptrs:
        state2_tensor = torch.tensor(
            state2_ptrs, dtype=torch.int64, device=chosen_device
        )
    else:
        state2_tensor = None
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=chosen_device)
    stride_tensor = torch.tensor(strides, dtype=torch.int32, device=chosen_device)

    batch = TensorPointerBatch(
        param_ptrs=param_tensor,
        grad_ptrs=grad_tensor,
        state1_ptrs=state1_tensor,
        state2_ptrs=state2_tensor,
        numel=lengths_tensor,
        stride=stride_tensor,
    )
    return batch, count
