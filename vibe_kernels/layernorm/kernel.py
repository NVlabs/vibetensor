# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast, Literal, Optional, Tuple, Union

import torch

from vibe_kernels.common import alloc
from vibe_kernels.common.tensor_types import TensorLike
from vibe_kernels.common.utils import validate_backend

from .impl import torch_impl

# Lazy import for CuTeDSL implementation
_cutedsl_impl = None


def _get_cutedsl_impl():
    global _cutedsl_impl
    if _cutedsl_impl is None:
        try:
            from .impl import cutedsl_impl

            _cutedsl_impl = cutedsl_impl
        except Exception:
            _cutedsl_impl = False
    return _cutedsl_impl


def is_cutedsl_available() -> bool:
    return _get_cutedsl_impl() is not False


class _CuTeDSLLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        return_rstd: bool,
        return_mean: bool,
    ):
        impl = _get_cutedsl_impl()
        if impl is False:
            raise RuntimeError("CuTeDSL LayerNorm backend is not available")

        need_stats = any(ctx.needs_input_grad[:3])
        want_rstd = return_rstd or need_stats
        want_mean = return_mean or need_stats

        # Call the JIT-compiled kernel
        result = impl.layernorm(
            x,
            weight,
            bias=bias,
            eps=eps,
            return_rstd=want_rstd,
            return_mean=want_mean,
        )

        if want_rstd and want_mean:
            out, rstd, mean = result
        elif want_rstd:
            out, rstd = result
            mean = None
        elif want_mean:
            out, mean = result
            rstd = None
        else:
            out = result
            rstd = None
            mean = None

        if need_stats:
            if rstd is None or mean is None:
                raise RuntimeError("LayerNorm backward requires rstd and mean tensors")
            saved_tensors: Tuple[torch.Tensor, ...]
            if bias is not None:
                saved_tensors = (
                    x.detach(),
                    weight.detach(),
                    bias.detach(),
                    rstd.detach(),
                    mean.detach(),
                )
            else:
                saved_tensors = (
                    x.detach(),
                    weight.detach(),
                    rstd.detach(),
                    mean.detach(),
                )
            ctx.save_for_backward(*saved_tensors)

        ctx.has_bias = bias is not None
        ctx.has_saved_stats = need_stats
        ctx.needs_dx = ctx.needs_input_grad[0]
        ctx.needs_dw = ctx.needs_input_grad[1]
        ctx.needs_db = ctx.needs_input_grad[2] and bias is not None

        outputs: list[torch.Tensor] = [out]
        if return_rstd:
            if rstd is None:
                raise RuntimeError("Expected rstd tensor but kernel did not return it")
            outputs.append(rstd)
        if return_mean:
            if mean is None:
                raise RuntimeError("Expected mean tensor but kernel did not return it")
            outputs.append(mean)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not getattr(ctx, "has_saved_stats", False):
            return None, None, None, None, None, None
        saved = ctx.saved_tensors
        if ctx.has_bias:
            x, weight, bias, rstd, mean = saved
        else:
            x, weight, rstd, mean = saved
            bias = None
        grad_out = grad_outputs[0]
        if grad_out is None:
            grad_out = alloc.zeros_like(x)

        impl = _get_cutedsl_impl()
        if impl is False:
            raise RuntimeError("CuTeDSL LayerNorm backend is not available")

        needs_db = bool(ctx.needs_db and ctx.has_bias)
        stream = int(torch.cuda.current_stream().cuda_stream)
        dx, dw, db = impl.layernorm_bwd(
            x,
            weight,
            grad_out,
            mean,
            rstd,
            has_bias=needs_db,
            stream=stream,
        )
        if not ctx.needs_dx:
            dx = None
        if not ctx.needs_dw:
            dw = None
        if not needs_db:
            db = None
        return dx, dw, db, None, None, None


def layernorm(
    x: TensorLike,
    weight: TensorLike,
    *,
    bias: TensorLike | None = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
    backend: Literal["auto", "cutedsl", "torch"] = "auto",
) -> Union[
    TensorLike,
    Tuple[TensorLike, TensorLike],
    Tuple[TensorLike, TensorLike, TensorLike],
]:
    """LayerNorm with support for CuTeDSL and Torch backends."""

    backend = validate_backend(backend, ["cutedsl", "torch"])

    if backend == "auto":
        if is_cutedsl_available():
            backend = "cutedsl"
        else:
            backend = "torch"

    if backend == "cutedsl":
        if not is_cutedsl_available():
            raise RuntimeError("CuTeDSL backend is not available")

        result = _CuTeDSLLayerNormFunction.apply(
            x,
            weight,
            bias,
            float(eps),
            bool(return_rstd),
            bool(return_mean),
        )
        return cast(
            Union[
                TensorLike,
                Tuple[TensorLike, TensorLike],
                Tuple[TensorLike, TensorLike, TensorLike],
            ],
            result,
        )

    if backend == "torch":
        # Fallback to PyTorch implementation
        # Note: torch.nn.functional.layer_norm doesn't support returning rstd/mean directly
        # so we use reference implementations for that if needed.

        if not return_rstd and not return_mean:
            return torch_impl.layernorm_forward(x, weight, bias=bias, eps=eps)

        # If stats needed, use manual calculation
        # This logic was present in the original file's `layernorm` function

        weight_cast = weight.to(x.dtype)
        bias_cast = bias.to(x.dtype) if bias is not None else None
        out = torch.nn.functional.layer_norm(
            x, weight_cast.shape, weight_cast, bias_cast, eps
        )

        rstd = torch_impl.rstd_ref(x, eps=eps)

        if return_rstd and return_mean:
            return out, rstd, torch_impl.mean_ref(x)
        if return_rstd:
            return out, rstd
        return out, torch_impl.mean_ref(x)

    raise ValueError(f"Unknown backend: {backend}")


class CuTeDSLLayerNorm(torch.nn.Module):
    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(self.normalized_shape))
            self.bias = torch.nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not is_cutedsl_available():
            raise RuntimeError("CuTeDSL backend is not available")

        # We need to cast result to Tensor because layernorm can return tuples
        res = layernorm(x, self.weight, bias=self.bias, eps=self.eps, backend="cutedsl")
        if isinstance(res, tuple):
            return res[0]
        return res


# Expose reference implementations
layernorm_ref = torch_impl.layernorm_ref
layernorm_rstd_ref = torch_impl.rstd_ref
layernorm_mean_ref = torch_impl.mean_ref


# Expose cutedsl_layernorm directly for compatibility
def cutedsl_layernorm(*args, **kwargs):
    impl = _get_cutedsl_impl()
    if impl is False:
        raise RuntimeError("CuTeDSL backend is not available")
    return impl.layernorm(*args, **kwargs)


__all__ = [
    "layernorm",
    "CuTeDSLLayerNorm",
    "is_cutedsl_available",
    "cutedsl_layernorm",
    "layernorm_ref",
    "layernorm_rstd_ref",
    "layernorm_mean_ref",
]
