// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1, "VBT_WITH_CUDA must be 0 or 1");

#if !VBT_WITH_CUDA
#  error "cub_wrapped.cuh is CUDA-only"
#endif

// CUB/Thrust namespace hygiene:
// Wrap all CUB and Thrust symbols into a private namespace to avoid ODR/macro
// collisions across shared libraries.
#if defined(THRUST_CUB_WRAPPED_NAMESPACE) || defined(CUB_WRAPPED_NAMESPACE) || \
    defined(CUB_NS_PREFIX) || defined(CUB_NS_POSTFIX) || defined(CUB_NS_QUALIFIER)
#  error "CUB/Thrust namespace wrapper macros must not be set before including cub_wrapped.cuh"
#endif

#define THRUST_CUB_WRAPPED_NAMESPACE vbt_cuda_cccl

#include <cub/cub.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#undef THRUST_CUB_WRAPPED_NAMESPACE

// Undefine wrapper macros after include to avoid leaking across translation units.
#ifdef CUB_WRAPPED_NAMESPACE
#  undef CUB_WRAPPED_NAMESPACE
#endif
#ifdef CUB_NS_PREFIX
#  undef CUB_NS_PREFIX
#endif
#ifdef CUB_NS_POSTFIX
#  undef CUB_NS_POSTFIX
#endif
#ifdef CUB_NS_QUALIFIER
#  undef CUB_NS_QUALIFIER
#endif

// Thrust also defines namespace wrapper macros. Undefine them to avoid leaking
// into later includes within this TU.
#ifdef THRUST_WRAPPED_NAMESPACE
#  undef THRUST_WRAPPED_NAMESPACE
#endif
#ifdef THRUST_NS_PREFIX
#  undef THRUST_NS_PREFIX
#endif
#ifdef THRUST_NS_POSTFIX
#  undef THRUST_NS_POSTFIX
#endif
#ifdef THRUST_NS_QUALIFIER
#  undef THRUST_NS_QUALIFIER
#endif

namespace vbt { namespace cuda { namespace cub_wrapped {

namespace cub = ::vbt_cuda_cccl::cub;
namespace thrust = ::vbt_cuda_cccl::thrust;

}}} // namespace vbt::cuda::cub_wrapped
