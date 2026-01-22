// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdint.h>

#include "vbt/plugin/vbt_plugin.h"

// C-side ABI layout checks for vt_tensor_iter handle & metadata.

_Static_assert(VT_TENSOR_ITER_MAX_RANK == 64,
               "VT_TENSOR_ITER_MAX_RANK must remain 64");

_Static_assert(VT_TENSOR_ITER_MAX_OPERANDS == 64,
               "VT_TENSOR_ITER_MAX_OPERANDS must remain 64");

_Static_assert(VT_TENSOR_ITER_CUDA_MAX_NDIM <= VT_TENSOR_ITER_MAX_RANK,
               "VT_TENSOR_ITER_CUDA_MAX_NDIM must not exceed TI max rank");

// Basic shape of vt_tensor_iter_desc.
_Static_assert(sizeof(((vt_tensor_iter_desc*)0)->sizes) ==
                   sizeof(int64_t) * VT_TENSOR_ITER_MAX_RANK,
               "vt_tensor_iter_desc::sizes must have VT_TENSOR_ITER_MAX_RANK entries");

_Static_assert(sizeof(((vt_tensor_iter_desc*)0)->strides) >=
                   sizeof(int64_t) * VT_TENSOR_ITER_MAX_OPERANDS *
                       VT_TENSOR_ITER_MAX_RANK,
               "vt_tensor_iter_desc::strides must cover operands x rank");

// Alias info bitmasks must fit into 64 bits.
_Static_assert(sizeof(((vt_tensor_iter_alias_info*)0)->output_may_alias_input) ==
                   sizeof(uint64_t),
               "output_may_alias_input must be 64-bit mask");

_Static_assert(sizeof(((vt_tensor_iter_alias_info*)0)->input_may_alias_output) ==
                   sizeof(uint64_t),
               "input_may_alias_output must be 64-bit mask");

// CUDA descriptor arrays must respect VT_TENSOR_ITER_CUDA_MAX_NDIM.
_Static_assert(sizeof(((vt_tensor_iter_cuda_desc*)0)->sizes) ==
                   sizeof(int64_t) * VT_TENSOR_ITER_CUDA_MAX_NDIM,
               "vt_tensor_iter_cuda_desc::sizes must have CUDA_MAX_NDIM entries");

_Static_assert(sizeof(((vt_tensor_iter_cuda_desc*)0)->strides) ==
                   sizeof(int64_t) * VT_TENSOR_ITER_CUDA_MAX_NDIM,
               "vt_tensor_iter_cuda_desc::strides must have CUDA_MAX_NDIM entries");

// No runtime tests; this TU is compiled and linked into a gtest binary so that
// the static assertions run at compile time.
