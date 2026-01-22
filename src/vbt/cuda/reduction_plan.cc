// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/reduction_plan.h"

#include <cstdint>

#include "vbt/core/checked_math.h"

namespace vbt {
namespace cuda {
namespace reduction {

namespace {

// UB-free magnitude for int64 (never calls abs()).
static inline std::uint64_t uabs_i64(std::int64_t x) noexcept {
  const std::uint64_t ux = static_cast<std::uint64_t>(x);
  return (x >= 0) ? ux : static_cast<std::uint64_t>(0) - ux;
}

static void stable_sort_dims_by_abs_stride(std::int32_t* dims,
                                          std::int32_t n,
                                          const std::int64_t* strides) noexcept {
  for (std::int32_t i = 1; i < n; ++i) {
    const std::int32_t key_dim = dims[i];
    const std::uint64_t key_mag = uabs_i64(strides[key_dim]);

    std::int32_t j = i - 1;
    while (j >= 0) {
      const std::uint64_t prev_mag = uabs_i64(strides[dims[j]]);
      if (prev_mag <= key_mag) {
        break;
      }
      dims[j + 1] = dims[j];
      --j;
    }
    dims[j + 1] = key_dim;
  }
}

}  // namespace

CudaReducePlanBuildResult build_cuda_reduce_plan_noalloc(
    const vbt::core::DeviceStrideMeta& out_meta,
    const vbt::core::DeviceStrideMeta& in_meta,
    std::span<const int64_t> reduce_dims,
    std::int64_t out_numel,
    std::int64_t slice_len) noexcept {
  CudaReducePlanBuildResult r{};
  r.plan.out_numel = out_numel;
  r.plan.slice_len = slice_len;

  // Empty cases are handled by the reduction dispatcher; treat them as
  // ineligible here to avoid accidental staged attempts.
  if (out_numel <= 0 || slice_len <= 0) {
    r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    return r;
  }

  const std::int64_t iter_ndim64 = in_meta.ndim;
  if (iter_ndim64 < 0 || iter_ndim64 > vbt::core::kTensorIterMaxRank) {
    r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    return r;
  }
  if (out_meta.ndim != in_meta.ndim) {
    r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    return r;
  }

  const std::int32_t iter_ndim = static_cast<std::int32_t>(iter_ndim64);
  r.plan.iter_ndim = iter_ndim;

  if (reduce_dims.empty() || reduce_dims.size() > static_cast<std::size_t>(iter_ndim)) {
    r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    return r;
  }

  // Validate dim ordering and build a reduced-dims mask.
  bool is_reduced[vbt::core::kTensorIterMaxRank]{};
  std::int32_t red_dims[vbt::core::kTensorIterMaxRank]{};
  const std::int32_t red_ndim = static_cast<std::int32_t>(reduce_dims.size());
  std::int64_t prev = -1;
  for (std::int32_t i = 0; i < red_ndim; ++i) {
    const std::int64_t d64 = reduce_dims[static_cast<std::size_t>(i)];
    if (d64 < 0 || d64 >= iter_ndim64 || d64 <= prev) {
      r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
      return r;
    }
    prev = d64;
    const std::int32_t d = static_cast<std::int32_t>(d64);
    is_reduced[d] = true;
    red_dims[i] = d;
  }

  std::int32_t kept_dims[vbt::core::kTensorIterMaxRank]{};
  std::int32_t kept_ndim = 0;
  for (std::int32_t d = 0; d < iter_ndim; ++d) {
    if (!is_reduced[d]) {
      kept_dims[kept_ndim++] = d;
    }
  }

  // Basic validation: sizes must be >= 0 and the input/out products must agree
  // when representable.
  std::int64_t prod_sizes = 1;
  for (std::int32_t d = 0; d < iter_ndim; ++d) {
    const std::int64_t s = in_meta.sizes[d];
    if (s < 0 || out_meta.sizes[d] != s) {
      r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
      return r;
    }
    if (!vbt::core::checked_mul_i64(prod_sizes, s, prod_sizes)) {
      r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
      return r;
    }
  }

  std::int64_t prod_out_slice = 0;
  if (!vbt::core::checked_mul_i64(out_numel, slice_len, prod_out_slice)) {
    r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    return r;
  }
  if (prod_out_slice != prod_sizes) {
    r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    return r;
  }

  // Kept-dim negative stride policy (active dims only).
  for (std::int32_t i = 0; i < kept_ndim; ++i) {
    const std::int32_t d = kept_dims[i];
    if (in_meta.sizes[d] <= 1) {
      continue;
    }
    if (in_meta.strides[d] < 0 || out_meta.strides[d] < 0) {
      r.ineligible_reason = CudaReduceIneligibleReason::KeptNegativeStride;
      return r;
    }
  }

  // Reduced-dim stride checks (active reduced dims only).
  std::int32_t active_red_dims[vbt::core::kTensorIterMaxRank]{};
  std::int32_t active_red_ndim = 0;
  for (std::int32_t i = 0; i < red_ndim; ++i) {
    const std::int32_t d = red_dims[i];
    if (in_meta.sizes[d] > 1) {
      active_red_dims[active_red_ndim++] = d;
    }
  }

  bool any_neg_red_stride = false;
  for (std::int32_t i = 0; i < active_red_ndim; ++i) {
    const std::int32_t d = active_red_dims[i];
    const std::int64_t st = in_meta.strides[d];
    if (st == 0) {
      r.ineligible_reason = CudaReduceIneligibleReason::RedStrideZero;
      return r;
    }
    any_neg_red_stride = any_neg_red_stride || (st < 0);
  }

  if (any_neg_red_stride && active_red_ndim > 1) {
    r.ineligible_reason = CudaReduceIneligibleReason::RedMultiDimNegativeStride;
    return r;
  }

  // Reduced linearization: require active reduced dims to be fully coalescible
  // into a single linear stride.
  std::int64_t red_linear_stride = 1;
  if (active_red_ndim == 0) {
    red_linear_stride = 1;
  } else if (active_red_ndim == 1) {
    red_linear_stride = in_meta.strides[active_red_dims[0]];
  } else {
    stable_sort_dims_by_abs_stride(active_red_dims, active_red_ndim, in_meta.strides);

    const std::int32_t d0 = active_red_dims[0];
    red_linear_stride = in_meta.strides[d0];

    for (std::int32_t i = 1; i < active_red_ndim; ++i) {
      const std::int32_t prev_d = active_red_dims[i - 1];
      const std::int32_t cur_d  = active_red_dims[i];

      const std::int64_t prev_stride = in_meta.strides[prev_d];
      const std::int64_t prev_size   = in_meta.sizes[prev_d];
      std::int64_t expected = 0;
      if (!vbt::core::checked_mul_i64(prev_stride, prev_size, expected)) {
        r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
        return r;
      }
      if (in_meta.strides[cur_d] != expected) {
        r.ineligible_reason = CudaReduceIneligibleReason::RedNotLinearizable;
        return r;
      }
    }
  }

  // Plan kept-dim ordering and positive-only coalescing.
  std::int32_t active_kept_dims[vbt::core::kTensorIterMaxRank]{};
  std::int32_t active_kept_ndim = 0;
  for (std::int32_t i = 0; i < kept_ndim; ++i) {
    const std::int32_t d = kept_dims[i];
    if (in_meta.sizes[d] > 1) {
      active_kept_dims[active_kept_ndim++] = d;
    }
  }

  if (active_kept_ndim > 1) {
    stable_sort_dims_by_abs_stride(active_kept_dims, active_kept_ndim, out_meta.strides);
  }

  // Coalesce in-place in active_kept_dims.
  std::int64_t kept_sizes[vbt::core::kTensorIterMaxRank]{};
  std::int64_t kept_in_strides[vbt::core::kTensorIterMaxRank]{};
  std::int64_t kept_out_strides[vbt::core::kTensorIterMaxRank]{};
  std::int32_t kept_out_ndim = 0;

  for (std::int32_t i = 0; i < active_kept_ndim; ++i) {
    const std::int32_t d = active_kept_dims[i];
    std::int64_t sz = in_meta.sizes[d];
    std::int64_t in_st = in_meta.strides[d];
    std::int64_t out_st = out_meta.strides[d];

    if (kept_out_ndim > 0) {
      const std::int32_t prev = kept_out_ndim - 1;
      const std::int64_t prev_sz = kept_sizes[prev];
      const std::int64_t prev_in_st = kept_in_strides[prev];
      const std::int64_t prev_out_st = kept_out_strides[prev];

      // Positive-only contiguity check.
      if (prev_sz > 0 && prev_in_st > 0 && prev_out_st > 0 &&
          in_st > 0 && out_st > 0) {
        std::int64_t exp_in = 0;
        std::int64_t exp_out = 0;
        if (vbt::core::checked_mul_i64(prev_in_st, prev_sz, exp_in) &&
            vbt::core::checked_mul_i64(prev_out_st, prev_sz, exp_out) &&
            exp_in == in_st && exp_out == out_st) {
          std::int64_t merged = 0;
          if (!vbt::core::checked_mul_i64(prev_sz, sz, merged)) {
            r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
            return r;
          }
          kept_sizes[prev] = merged;
          continue;
        }
      }
    }

    kept_sizes[kept_out_ndim] = sz;
    kept_in_strides[kept_out_ndim] = in_st;
    kept_out_strides[kept_out_ndim] = out_st;
    ++kept_out_ndim;
  }

  // Validate that kept_sizes product matches out_numel.
  std::int64_t kept_prod = 1;
  for (std::int32_t i = 0; i < kept_out_ndim; ++i) {
    if (!vbt::core::checked_mul_i64(kept_prod, kept_sizes[i], kept_prod)) {
      r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
      return r;
    }
  }
  if (kept_prod != out_numel) {
    r.ineligible_reason = CudaReduceIneligibleReason::Overflow;
    return r;
  }

  r.plan.kept_ndim = kept_out_ndim;
  r.plan.red_ndim = active_red_ndim;
  r.plan.red_linear_stride = red_linear_stride;

  for (std::int32_t i = 0; i < kept_out_ndim; ++i) {
    r.plan.kept_sizes[i] = kept_sizes[i];
    r.plan.kept_in_strides[i] = kept_in_strides[i];
    r.plan.kept_out_strides[i] = kept_out_strides[i];
  }

  r.ineligible_reason = CudaReduceIneligibleReason::None;
  return r;
}

}  // namespace reduction
}  // namespace cuda
}  // namespace vbt
