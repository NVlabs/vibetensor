// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_iterator/cpu.h"

#include <stdexcept>
#include <vector>
#include <cassert>

#include "vbt/core/checked_math.h"

namespace vbt {
namespace core {

namespace {

bool try_reduction_contiguous_lastdim_fastpath(const TensorIter& iter,
                                               reduce_loop1d_t loop,
                                               void* ctx) {
#if defined(VBT_TI_FORCE_GENERIC_PATHS)
  (void) iter;
  (void) loop;
  (void) ctx;
  return false;
#else
  const int R = iter.ndim();
  if (R <= 0) {
    return false;
  }

  const auto& S = iter.shape();
  const auto& reduce_dims = iter.reduce_dims();
  if (reduce_dims.size() != 1) {
    return false;
  }
  const int rd = static_cast<int>(reduce_dims[0]);
  if (rd != R - 1) {
    return false;
  }
  const std::size_t rd_idx = static_cast<std::size_t>(rd);

  const int nt = iter.ntensors();
  const int noutputs = iter.noutputs();
  const int ninputs  = iter.ninputs();
  if (ninputs != 1 || noutputs < 1 || noutputs > 2) {
    return false;
  }

  for (int k = 0; k < nt; ++k) {
    const IterOperand& op = iter.operand(k);
    if (op.device.type != kDLCPU) {
      return false;
    }
  }

  const IterOperand& in_op = iter.operand(noutputs);
  TensorImpl* in_tensor = in_op.tensor;
  if (!in_tensor) {
    return false;
  }
  if (!in_tensor->is_non_overlapping_and_dense() ||
      !in_tensor->is_contiguous()) {
    return false;
  }

  const std::int64_t total = iter.numel();
  if (total == 0) {
    return true;
  }

  const std::int64_t reduce_size = S[rd_idx];
  if (reduce_size == 0) {
    return true;
  }

  std::vector<int> keep_dims;
  keep_dims.reserve(static_cast<std::size_t>(R - 1));
  for (int d = 0; d < R; ++d) {
    if (d != rd) {
      keep_dims.push_back(d);
    }
  }

  const std::size_t outer_ndim = keep_dims.size();
  std::vector<std::int64_t> outer_shape(outer_ndim, 0);
  for (std::size_t j = 0; j < outer_ndim; ++j) {
    outer_shape[j] = S[static_cast<std::size_t>(keep_dims[j])];
  }

  std::int64_t outer_count = 1;
  for (std::int64_t s : outer_shape) {
    std::int64_t tmp = 0;
    if (!checked_mul_i64(outer_count, s, tmp)) {
      outer_count = 0;
      break;
    }
    outer_count = tmp;
  }
  if (outer_count == 0) {
    return true;
  }

#ifdef VBT_TI_DEBUG
  {
    std::int64_t expected = 0;
    if (!checked_mul_i64(outer_count, reduce_size, expected)) {
      expected = 0;
    }
    const std::int64_t total_dbg = iter.numel();
    assert(expected == total_dbg);
  }
#endif

  std::vector<char*> data(static_cast<std::size_t>(nt));
  std::vector<std::int64_t> strides(static_cast<std::size_t>(nt));
  std::vector<std::int64_t> outer_coords(outer_ndim, 0);

  auto compute_tile_pointers = [&]() {
    for (int k = 0; k < nt; ++k) {
      const IterOperand& op = iter.operand(k);
      char* base = static_cast<char*>(op.data);
      std::int64_t off_bytes = 0;
      for (std::size_t j = 0; j < outer_ndim; ++j) {
        const int dim = keep_dims[j];
        const std::int64_t idx_d = outer_coords[j];
        std::int64_t term = 0;
        if (!checked_mul_i64(idx_d,
                             op.dim_stride_bytes[static_cast<std::size_t>(dim)],
                             term) ||
            !checked_add_i64(off_bytes, term, off_bytes)) {
          throw std::overflow_error(
              "TI: overflow computing reduction tile base offset");
        }
      }
      data[static_cast<std::size_t>(k)] = base + off_bytes;
      if (k < noutputs) {
        // Outputs are treated as scalars for the reduced dim.
        strides[static_cast<std::size_t>(k)] = 0;
      } else {
        // Single input slice along reduced dim.
        strides[static_cast<std::size_t>(k)] =
            op.dim_stride_bytes[rd_idx];
      }
    }
  };

  auto bump_outer = [&]() -> bool {
    for (int j = static_cast<int>(outer_ndim) - 1; j >= 0; --j) {
      std::int64_t next = outer_coords[static_cast<std::size_t>(j)] + 1;
      if (next < outer_shape[static_cast<std::size_t>(j)]) {
        outer_coords[static_cast<std::size_t>(j)] = next;
        return true;
      }
      outer_coords[static_cast<std::size_t>(j)] = 0;
    }
    return false;
  };

  bool first = true;
  while (true) {
    if (first) {
      first = false;
    } else if (!bump_outer()) {
      break;
    }
    compute_tile_pointers();
    loop(data.data(), strides.data(), reduce_size, ctx);
  }

  return true;
#endif  // VBT_TI_FORCE_GENERIC_PATHS
}

}  // namespace

void for_each_reduction_cpu(const TensorIter& iter,
                            reduce_loop1d_t loop,
                            void* ctx) {
  if (!iter.is_reduction() || iter.num_reduce_dims() == 0 ||
      iter.ninputs() != 1 || iter.noutputs() < 1 || iter.noutputs() > 2) {
    throw std::logic_error(
        "TI: for_each_reduction_cpu precondition failed");
  }
  if (loop == nullptr) {
    throw std::invalid_argument(
        "TI: for_each_reduction_cpu callback must not be null");
  }

  const int nt = iter.ntensors();
  for (int k = 0; k < nt; ++k) {
    const IterOperand& op = iter.operand(k);
    if (op.device.type != kDLCPU) {
      throw std::invalid_argument(
          "TI: for_each_reduction_cpu supports CPU tensors only");
    }
  }

  const int R = iter.ndim();
  if (R <= 0) {
    throw std::logic_error(
        "TI: for_each_reduction_cpu requires rank >= 1");
  }

  VBT_TI_STATS_INC(cpu_reduction_invocations);

#ifdef VBT_TI_DEBUG
  // Invariants: iteration shape rank matches ndim(), and each operand has a
  // stride entry per iteration dim.
  assert(iter.shape().size() == static_cast<std::size_t>(R));
  for (int k = 0; k < nt; ++k) {
    const IterOperand& op_dbg = iter.operand(k);
    assert(op_dbg.dim_stride_bytes.size() == static_cast<std::size_t>(R));
  }
#endif

  const std::int64_t total = iter.numel();
  if (total == 0) {
    return;
  }

  const std::vector<std::int64_t>& S = iter.shape();
  const std::vector<std::int64_t>& reduce_dims = iter.reduce_dims();
  
  // If single reduction dim, try fast path
  if (reduce_dims.size() == 1) {
     if (try_reduction_contiguous_lastdim_fastpath(iter, loop, ctx)) {
       return;
     }
  }

  // Generic path: treat the last reduction dimension as the inner loop.
  // All other dimensions (kept or reduced) are iterated in the outer loop.
  // Since reduced dims have stride 0 in output, iterating them accumulates correctly.
  const int inner_dim = static_cast<int>(reduce_dims.back());
  const std::size_t inner_idx = static_cast<std::size_t>(inner_dim);
  const std::int64_t inner_size = S[inner_idx];

  if (inner_size == 0) {
    return;
  }

  // Build list of outer dims (all except inner_dim)
  std::vector<int> outer_dims;
  outer_dims.reserve(static_cast<std::size_t>(R - 1));
  for (int d = 0; d < R; ++d) {
    if (d != inner_dim) {
      outer_dims.push_back(d);
    }
  }

  const std::size_t outer_ndim = outer_dims.size();
  std::vector<std::int64_t> outer_shape(outer_ndim, 0);
  for (std::size_t j = 0; j < outer_ndim; ++j) {
    outer_shape[j] = S[static_cast<std::size_t>(outer_dims[j])];
  }

  // Compute outer_count via checked multiplication; overflow => treat as zero.
  std::int64_t outer_count = 1;
  for (std::int64_t s : outer_shape) {
    std::int64_t tmp = 0;
    if (!checked_mul_i64(outer_count, s, tmp)) {
      outer_count = 0;
      break;
    }
    outer_count = tmp;
  }

  if (outer_count == 0) {
    return;
  }

  const int noutputs = iter.noutputs();

  std::vector<char*> data(static_cast<std::size_t>(nt));
  std::vector<std::int64_t> strides(static_cast<std::size_t>(nt));
  std::vector<std::int64_t> outer_coords(outer_ndim, 0);

  auto compute_tile_pointers = [&]() {
    for (int k = 0; k < nt; ++k) {
      const IterOperand& op = iter.operand(k);
      char* base = static_cast<char*>(op.data);
      std::int64_t off_bytes = 0;
      for (std::size_t j = 0; j < outer_ndim; ++j) {
        const int dim = outer_dims[j];
        const std::int64_t idx_d = outer_coords[j];
        std::int64_t term = 0;
        if (!checked_mul_i64(idx_d,
                             op.dim_stride_bytes[static_cast<std::size_t>(dim)],
                             term) ||
            !checked_add_i64(off_bytes, term, off_bytes)) {
          throw std::overflow_error(
              "TI: overflow computing reduction tile base offset");
        }
      }
      data[static_cast<std::size_t>(k)] = base + off_bytes;
      if (k < noutputs) {
        // Outputs are treated as scalars for the reduced inner dim.
        strides[static_cast<std::size_t>(k)] = 0;
      } else {
        // Single input slice along reduced inner dim.
        strides[static_cast<std::size_t>(k)] =
            op.dim_stride_bytes[inner_idx];
      }
    }
  };

  auto bump_outer = [&]() -> bool {
    for (int j = static_cast<int>(outer_ndim) - 1; j >= 0; --j) {
      std::int64_t next = outer_coords[static_cast<std::size_t>(j)] + 1;
      if (next < outer_shape[static_cast<std::size_t>(j)]) {
        outer_coords[static_cast<std::size_t>(j)] = next;
        return true;
      }
      outer_coords[static_cast<std::size_t>(j)] = 0;
    }
    return false;
  };

  bool first = true;
  while (true) {
    if (first) {
      first = false;
    } else if (!bump_outer()) {
      break;
    }
    compute_tile_pointers();
    loop(data.data(), strides.data(), inner_size, ctx);
  }
}

} // namespace core
} // namespace vbt
