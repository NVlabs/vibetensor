// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

#if !VBT_WITH_CUDA
#  error "cub_detail_compat.cuh is CUDA-only"
#endif

#include "cub_wrapped.cuh"  // CUB + Thrust in a private namespace

// CCCL/CUB compatibility shims.
//
// We intentionally prefer Thrust iterators here, since their APIs are stable
// across CCCL/CUB releases, while some CUB iterators have changed arity.

namespace vbt {
namespace cuda {
namespace cub_detail {

#if defined(CUB_VERSION) && (CUB_VERSION >= 300000)
inline constexpr bool kCubV3Plus = true;
#else
inline constexpr bool kCubV3Plus = false;
#endif

template <class T>
using CountingIterator = vbt::cuda::cub_wrapped::thrust::counting_iterator<T>;

using DiscardIterator =
    decltype(vbt::cuda::cub_wrapped::thrust::make_discard_iterator());

template <class InputIt, class UnaryFn>
using TransformIterator =
    decltype(vbt::cuda::cub_wrapped::thrust::make_transform_iterator(
        std::declval<InputIt>(), std::declval<UnaryFn>()));

}  // namespace cub_detail
}  // namespace cuda
}  // namespace vbt
