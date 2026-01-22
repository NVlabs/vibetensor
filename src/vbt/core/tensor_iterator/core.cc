// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/tensor_iterator/core.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <optional>
#include <cstdio>
#include <vector>
#include <cstdio>

#include "vbt/core/checked_math.h"
#include "vbt/core/type_promotion.h"
#include "vbt/core/overlap.h"
#include "vbt/core/broadcast.h"
#include "vbt/core/strided_loop.h"  // checked_abs_i64_hdr
#include "vbt/core/write_guard.h"
#include "vbt/cpu/storage.h"  // TI: new_cpu_storage for resizing

namespace vbt {
namespace core {

#ifdef VBT_TI_STATS
namespace {

TensorIterStats g_tensor_iter_stats{};

}  // anonymous namespace

namespace detail {

TensorIterStats& mutable_tensor_iter_stats() noexcept {
  return g_tensor_iter_stats;
}

}  // namespace detail

const TensorIterStats& get_tensor_iter_stats() noexcept {
  return g_tensor_iter_stats;
}

void reset_tensor_iter_stats() noexcept {
  g_tensor_iter_stats.cpu_invocations.store(0, std::memory_order_relaxed);
  g_tensor_iter_stats.cpu_reduction_invocations.store(
      0, std::memory_order_relaxed);
  g_tensor_iter_stats.cuda_meta_exports.store(0, std::memory_order_relaxed);
  g_tensor_iter_stats.cuda_ti_kernel_launches.store(
      0, std::memory_order_relaxed);
  g_tensor_iter_stats.num_32bit_splits.store(
      0, std::memory_order_relaxed);
}
#endif  // VBT_TI_STATS

void maybe_set_cpu_nod_contig_fastpath(TensorIter& iter) {
  if (iter.is_reduction()) {
    return;
  }

  const int R = iter.ndim();
  if (R != 1) {
    return;
  }

  const auto& shape = iter.shape();
  if (shape.empty()) {
    return;
  }

  const std::int64_t size0 = shape[0];
  if (size0 <= 0) {
    return;
  }

  const int nt = iter.ntensors();
  for (int k = 0; k < nt; ++k) {
    const IterOperand& op = iter.operand(k);
    if (op.device.type != kDLCPU) {
      return;
    }
    if (op.tensor && !op.tensor->is_non_overlapping_and_dense()) {
      return;
    }
    if (op.dim_stride_bytes.size() != 1) {
      return;
    }
    const std::int64_t stride_bytes = op.dim_stride_bytes[0];
    if (stride_bytes < 0) {
      return;
    }
    const auto item_b =
        static_cast<std::int64_t>(itemsize(op.dtype));
    if (item_b <= 0 || stride_bytes != item_b) {
      return;
    }
  }

  iter.cpu_nod_contig_fastpath_ = true;
}

#ifdef VBT_TI_ENABLE_TEST_HOOKS
namespace testing {

TensorIter TensorIterTestHelper::make_iterator_for_shape(
    std::span<const std::int64_t> shape) {
  TensorIter iter;
  iter.shape_.assign(shape.begin(), shape.end());
  iter.reduce_dims_.clear();
  iter.operands_.clear();
  iter.num_outputs_ = 0;
  iter.common_dtype_ = ScalarType::Float32;
  iter.common_device_ = Device::cpu();
  iter.is_reduction_ = false;
  iter.op_signature_ = nullptr;
  iter.mem_overlap_checked_ = false;
  iter.has_any_output_input_alias_ = false;
  iter.alias_status_table_.clear();
  return iter;
}

TensorIter TensorIterTestHelper::make_iterator_for_shape_with_dummy_operand(
    std::span<const std::int64_t> shape) {
  TensorIter iter = make_iterator_for_shape(shape);

  IterOperand op;
  op.tensor = nullptr;
  op.role = IterOperandRole::ReadOnly;

  const std::size_t ndim = iter.shape_.size();
  op.dim_stride_bytes.assign(ndim, 0);
  const auto item_b = static_cast<std::int64_t>(itemsize(iter.common_dtype_));
  std::int64_t elem_stride = 1;
  // Row-major contiguous layout in bytes, with zero-sized dims treated as
  // having stride equal to the running product so numel()==0 shapes still
  // behave like regular tensors for indexing calculations.
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(ndim) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    op.dim_stride_bytes[idx] = elem_stride * item_b;
    const std::int64_t sz = iter.shape_[idx];
    if (sz == 0) {
      continue;
    }
    std::int64_t next = 0;
    if (!checked_mul_i64(elem_stride, sz, next)) {
      // Overflow here simply leaves elem_stride large; can_use_32bit_indexing
      // will conservatively return false in that case.
      break;
    }
    elem_stride = next;
  }

  op.data = nullptr;
  op.is_output = false;
  op.is_read_write = false;
  op.will_resize = false;
  op.dtype = iter.common_dtype_;
  op.device = iter.common_device_;

  iter.operands_.clear();
  iter.operands_.push_back(std::move(op));
  return iter;
}

ScalarType TensorIterTestHelper::common_dtype(const TensorIterBase& iter) {
  return iter.common_dtype_;
}

Device TensorIterTestHelper::common_device(const TensorIterBase& iter) {
  return iter.common_device_;
}

bool TensorIterTestHelper::cpu_nod_contig_fastpath(const TensorIterBase& iter) {
  return iter.cpu_nod_contig_fastpath_;
}

} // namespace testing
#endif  // VBT_TI_ENABLE_TEST_HOOKS

// ==== TensorIterBase introspection ========================================================

int TensorIterBase::ndim() const {
  return static_cast<int>(shape_.size());
}

std::int64_t TensorIterBase::numel() const {
  // Mirror TensorImpl::numel semantics on the iteration shape:
  // start from 1, treat overflow as zero.
  if (shape_.empty()) {
    return 1;
  }
  std::int64_t total = 1;
  for (std::int64_t s : shape_) {
    std::int64_t tmp = 0;
    if (!checked_mul_i64(total, s, tmp)) {
      return 0;
    }
    total = tmp;
  }
  return total;
}

int TensorIterBase::ntensors() const {
  return static_cast<int>(operands_.size());
}

int TensorIterBase::noutputs() const {
  return num_outputs_;
}

int TensorIterBase::ninputs() const {
  return ntensors() - noutputs();
}

const std::vector<std::int64_t>& TensorIterBase::shape() const {
  return shape_;
}

const std::vector<std::int64_t>& TensorIterBase::reduce_dims() const {
  return reduce_dims_;
}

const IterOperand& TensorIterBase::operand(int idx) const {
  assert(idx >= 0 && idx < ntensors());
  return operands_[static_cast<std::size_t>(idx)];
}

IterOperand& TensorIterBase::operand(int idx) {
  assert(idx >= 0 && idx < ntensors());
  return operands_[static_cast<std::size_t>(idx)];
}

bool TensorIterBase::is_reduction() const {
  return is_reduction_;
}

int TensorIterBase::num_reduce_dims() const {
  return static_cast<int>(reduce_dims_.size());
}

bool TensorIterBase::is_trivial_1d() const {
  // Treat zero-sized and scalar/all-size-1 iterators as trivial.
  const std::int64_t n = numel();
  return n <= 1 || ndim() <= 1;
}

bool TensorIterBase::can_use_32bit_indexing() const {
  const std::int64_t n = numel();
  // Zero-numel iterators are always safe, including overflow-sentinel shapes.
  if (n == 0) {
    return true;
  }

  constexpr std::int64_t kI32Max =
      static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
  if (n > kI32Max) {
    return false;
  }

  const int R = ndim();
  if (R <= 0) {
    return true;
  }

  const int nt = ntensors();
  const auto& S = shape_;

  for (int k = 0; k < nt; ++k) {
    const IterOperand& op = operands_[static_cast<std::size_t>(k)];
    const std::int64_t item_b =
        static_cast<std::int64_t>(itemsize(op.dtype));
    if (item_b <= 0) {
      return false;
    }

    if (op.dim_stride_bytes.size() != static_cast<std::size_t>(R)) {
#ifdef VBT_TI_DEBUG
      assert(!"TI: dim_stride_bytes rank mismatch in can_use_32bit_indexing");
#endif
      return false;
    }

    std::int64_t max_offset_elems = 0;

    for (int d = 0; d < R; ++d) {
      const std::int64_t size_d = S[static_cast<std::size_t>(d)];
      const std::int64_t max_idx_d = (size_d == 0 ? 0 : size_d - 1);
      if (max_idx_d == 0) {
        continue;
      }

      const std::int64_t stride_bytes =
          op.dim_stride_bytes[static_cast<std::size_t>(d)];
      if (stride_bytes % item_b != 0) {
        return false;
      }

      const std::int64_t elem_stride =
          (stride_bytes == 0) ? 0 : stride_bytes / item_b;

      std::int64_t abs_elem_stride = 0;
      if (!checked_abs_i64_hdr(elem_stride, abs_elem_stride)) {
        return false;
      }

      std::int64_t max_off_d = 0;
      if (!checked_mul_i64(max_idx_d, abs_elem_stride, max_off_d)) {
        return false;
      }

      if (!checked_add_i64(max_offset_elems, max_off_d, max_offset_elems)) {
        return false;
      }

      if (max_offset_elems > kI32Max) {
        return false;
      }
    }
  }

  return true;
}

ScalarType TensorIterBase::computation_dtype() const {
  return opmath_dtype(common_dtype_);
}

const IterOpSignature* TensorIterBase::op_signature() const noexcept {
  return op_signature_;
}

bool TensorIterBase::mem_overlap_checked() const noexcept {
  return mem_overlap_checked_;
}

bool TensorIterBase::has_any_output_input_alias() const noexcept {
  return has_any_output_input_alias_;
}

MemOverlapStatus TensorIterBase::alias_status(int out_index, int in_index) const {
  const int out_count = noutputs();
  const int in_count  = ninputs();
  assert(out_index >= 0 && out_index < out_count);
  assert(in_index >= 0 && in_index < in_count);

#ifdef VBT_TI_DEBUG
  if (!mem_overlap_checked_ || alias_status_table_.empty()) {
    assert(!"TensorIterBase::alias_status used without overlap metadata");
  }
#endif

  if (!mem_overlap_checked_ || alias_status_table_.empty()) {
    return MemOverlapStatus::No;
  }

  const std::size_t idx = static_cast<std::size_t>(out_index) *
                          static_cast<std::size_t>(in_count) +
                          static_cast<std::size_t>(in_index);
  if (idx >= alias_status_table_.size()) {
    return MemOverlapStatus::No;
  }
  return alias_status_table_[idx];
}

void TensorIterBase::export_device_meta(int operand_index,
                                        DeviceStrideMeta* out_meta,
                                        std::int64_t max_ndim) const {
  if (out_meta == nullptr) {
    throw std::invalid_argument(
        "TI: export_device_meta: out_meta must not be null");
  }

  const int nt = ntensors();
  if (operand_index < 0 || operand_index >= nt) {
    throw std::out_of_range(
        "TI: export_device_meta: operand index out of range");
  }

  if (max_ndim < 1 || max_ndim > kTensorIterMaxRank) {
    throw std::invalid_argument(
        "TI: export_device_meta: max_ndim out of range");
  }

  const int R = ndim();
#ifdef VBT_TI_DEBUG
  assert(R >= 0);
  assert(R <= kTensorIterMaxRank);
#endif
  if (R < 0) {
    throw std::logic_error(
        "TI: export_device_meta: negative ndim");
  }
  if (R > max_ndim) {
    throw std::invalid_argument(
        "TI: export_device_meta: iteration rank exceeds max_ndim");
  }

  const IterOperand& op = operands_[static_cast<std::size_t>(operand_index)];
  const auto item_b = static_cast<std::int64_t>(itemsize(op.dtype));
  if (item_b <= 0) {
    throw std::logic_error(
        "TI: export_device_meta: invalid itemsize for operand");
  }

#ifdef VBT_TI_DEBUG
  assert(op.dim_stride_bytes.size() == static_cast<std::size_t>(R));
#endif

  for (int d = 0; d < R; ++d) {
    const auto stride_bytes =
        op.dim_stride_bytes[static_cast<std::size_t>(d)];
    if (stride_bytes % item_b != 0) {
      throw std::logic_error(
          "TI: export_device_meta: stride_bytes not divisible by itemsize");
    }
  }

  DeviceStrideMeta local{};
  local.ndim = R;

  for (int d = 0; d < R; ++d) {
    local.sizes[d] = shape_[static_cast<std::size_t>(d)];

    const auto stride_bytes =
        op.dim_stride_bytes[static_cast<std::size_t>(d)];
    if (stride_bytes == 0) {
      local.strides[d] = 0;
    } else {
      local.strides[d] = stride_bytes / item_b;
    }
  }

  for (int d = R; d < kTensorIterMaxRank; ++d) {
    local.sizes[d]   = 0;
    local.strides[d] = 0;
  }

#ifdef VBT_TI_DEBUG
  const std::int64_t n = numel();
  if (R == 0) {
    // numel() is either 0 or 1 for rank-0 iterators.
    assert(n == 0 || n == 1);
  } else {
    std::int64_t prod = 1;
    bool any_zero = false;
    for (int d = 0; d < R; ++d) {
      if (local.sizes[d] == 0) {
        any_zero = true;
      }
      std::int64_t tmp = 0;
      if (!checked_mul_i64(prod, local.sizes[d], tmp)) {
        prod = 0;
        any_zero = true;
        break;
      }
      prod = tmp;
    }
    if (any_zero) {
      assert(n == 0);
    } else {
      assert(n == prod);
    }
  }
#endif

#ifdef VBT_TI_STATS
  if (common_device_.type == kDLCUDA) {
    VBT_TI_STATS_INC(cuda_meta_exports);
  }
#endif

  *out_meta = local;
}

// ==== TensorIterBase::for_each_cpu =======================================================

void TensorIterBase::for_each_cpu(loop1d_t loop, void* ctx) const {
  if (common_device_.type != kDLCPU) {
    throw std::logic_error(
        "TI: for_each_cpu only valid for CPU iterators");
  }
  if (is_reduction_) {
    throw std::logic_error(
        "TI: for_each_cpu is only valid for non-reduction iterators");
  }
  if (loop == nullptr) {
    throw std::invalid_argument("TI: for_each_cpu callback must not be null");
  }

  VBT_TI_STATS_INC(cpu_invocations);

  const std::int64_t total = numel();
  if (total == 0) {
    // Zero-sized iterator: nothing to do.
    return;
  }

  const int R = ndim();
  const int nt = ntensors();
  assert(nt > 0);
  assert(noutputs() > 0 && noutputs() <= nt);
#ifdef VBT_TI_DEBUG
  // Invariants: iteration shape rank matches ndim(), and each operand has a
  // stride entry per iteration dim.
  assert(shape_.size() == static_cast<std::size_t>(R));
  for (const IterOperand& op : operands_) {
    assert(op.dim_stride_bytes.size() == static_cast<std::size_t>(R));
  }
#endif

#if !defined(VBT_TI_FORCE_GENERIC_PATHS)
  if (cpu_nod_contig_fastpath_) {
#ifdef VBT_TI_DEBUG
    assert(R == 1);
#endif
    std::vector<char*> data(static_cast<std::size_t>(nt));
    std::vector<std::int64_t> strides(static_cast<std::size_t>(nt));
    for (int k = 0; k < nt; ++k) {
      const IterOperand& op = operands_[static_cast<std::size_t>(k)];
      data[static_cast<std::size_t>(k)] = static_cast<char*>(op.data);
      strides[static_cast<std::size_t>(k)] =
          op.dim_stride_bytes[static_cast<std::size_t>(0)];
    }
    loop(data.data(), strides.data(), total, ctx);
    return;
  }
#endif

  // Scalar / all-size-1 path: ndim()==0 but numel()==1.
  if (R == 0) {
    std::vector<char*> data(static_cast<std::size_t>(nt));
    std::vector<std::int64_t> strides(static_cast<std::size_t>(nt), 0);
    for (int k = 0; k < nt; ++k) {
      data[static_cast<std::size_t>(k)] = static_cast<char*>(operands_[static_cast<std::size_t>(k)].data);
    }
    loop(data.data(), strides.data(), /*size=*/1, ctx);
    return;
  }

  // General path: R >= 1
  const std::vector<std::int64_t>& S = shape_;
  const int inner_dim = R - 1;  // fastest-moving dim
  const std::int64_t inner_size = S[static_cast<std::size_t>(inner_dim)];

  // Build list of outer dims (all except inner_dim) and their sizes.
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
#ifdef VBT_TI_DEBUG
  {
    std::int64_t expected = 0;
    if (!checked_mul_i64(outer_count, inner_size, expected)) {
      expected = 0;
    }
    assert(expected == total);
  }
#endif
  if (outer_count == 0) {
    return;
  }

  // Arrays reused across tiles.
  std::vector<char*> data(static_cast<std::size_t>(nt));
  std::vector<std::int64_t> strides(static_cast<std::size_t>(nt));

  if (outer_ndim == 0) {
    // Only inner dimension exists (R == 1): single tile.
    for (int k = 0; k < nt; ++k) {
      const IterOperand& op = operands_[static_cast<std::size_t>(k)];
      data[static_cast<std::size_t>(k)] = static_cast<char*>(op.data);
      strides[static_cast<std::size_t>(k)] = op.dim_stride_bytes[static_cast<std::size_t>(inner_dim)];
    }
    loop(data.data(), strides.data(), inner_size, ctx);
    return;
  }

  std::vector<std::int64_t> outer_coords(outer_ndim, 0);

  auto compute_tile_pointers = [&](void) {
    for (int k = 0; k < nt; ++k) {
      const IterOperand& op = operands_[static_cast<std::size_t>(k)];
      char* base = static_cast<char*>(op.data);
      std::int64_t off_bytes = 0;
      for (std::size_t j = 0; j < outer_ndim; ++j) {
        const int dim = outer_dims[j];
        const std::int64_t idx_d = outer_coords[j];
        std::int64_t term = 0;
        if (!checked_mul_i64(idx_d, op.dim_stride_bytes[static_cast<std::size_t>(dim)], term) ||
            !checked_add_i64(off_bytes, term, off_bytes)) {
          throw std::overflow_error("TI: overflow computing tile base offset");
        }
      }
      data[static_cast<std::size_t>(k)] = base + off_bytes;
      strides[static_cast<std::size_t>(k)] = op.dim_stride_bytes[static_cast<std::size_t>(inner_dim)];
    }
  };

  auto bump_outer = [&]() -> bool {
    // Lexicographic counter over outer_shape (row-major).
    for (int j = static_cast<int>(outer_ndim) - 1; j >= 0; --j) {
      std::int64_t next = outer_coords[static_cast<std::size_t>(j)] + 1;
      if (next < outer_shape[static_cast<std::size_t>(j)]) {
        outer_coords[static_cast<std::size_t>(j)] = next;
        return true;
      }
      outer_coords[static_cast<std::size_t>(j)] = 0;
    }
    return false;  // finished
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

// ==== TensorIterConfig ===================================================================

TensorIterConfig& TensorIterConfig::add_output(OptionalTensorImplRef out,
                                               IterOperandRole role,
                                               bool allow_resize) {
  if (!inputs_.empty()) {
    throw std::invalid_argument("TI: outputs must be added before inputs");
  }
  if (!out.defined()) {
    throw std::invalid_argument("TI: outputs must be defined");
  }
  if (role == IterOperandRole::ReduceOutput && !is_reduction_) {
    throw std::invalid_argument(
        "TI: ReduceOutput outputs are only valid for reduction iterators");
  }
  if (role != IterOperandRole::WriteOnly &&
      role != IterOperandRole::ReadWrite &&
      role != IterOperandRole::ReduceOutput) {
    throw std::invalid_argument(
        "TI: unsupported output role for Tensor Iterator");
  }

  outputs_.push_back(out);
  output_roles_.push_back(role);
  output_allow_resize_.push_back(allow_resize);
  return *this;
}

TensorIterConfig& TensorIterConfig::add_input(const TensorImpl& in) {
  inputs_.push_back(&in);
  input_is_const_.push_back(false);
  return *this;
}

TensorIterConfig& TensorIterConfig::add_const_input(const TensorImpl& in) {
  inputs_.push_back(&in);
  input_is_const_.push_back(true);
  return *this;
}

TensorIterConfig& TensorIterConfig::check_mem_overlap(bool v) {
  check_mem_overlap_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::check_all_same_dtype(bool v) {
  check_all_same_dtype_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::check_all_same_device(bool v) {
  check_all_same_device_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::enable_fabric_2gpu_elementwise(
    int primary_device) {
  if (is_reduction_) {
    throw std::logic_error(
        "[Fabric] TI Fabric mode is only supported for elementwise iterators");
  }
  if (primary_device < 0) {
    throw std::invalid_argument(
        "[Fabric] TI Fabric mode requires non-negative CUDA device index");
  }
  allow_multi_device_fabric_ = true;
  fabric_primary_device_ = primary_device;
  return *this;
}

TensorIterConfig& TensorIterConfig::promote_inputs_to_common_dtype(bool v) {
  promote_inputs_to_common_dtype_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::promote_integer_inputs_to_float(bool v) {
  promote_integer_inputs_to_float_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::cast_common_dtype_to_outputs(bool v) {
  cast_common_dtype_to_outputs_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::resize_outputs(bool v) {
  resize_outputs_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::allow_cpu_scalars(bool v) {
  allow_cpu_scalars_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::is_reduction(bool v) {
  if (v && allow_multi_device_fabric_) {
    throw std::logic_error(
        "[Fabric] TI Fabric mode is only supported for elementwise iterators");
  }
  is_reduction_ = v;
  if (!is_reduction_) {
    reduce_dims_spec_.clear();
    reduce_keepdim_ = false;
    has_reduce_dims_ = false;
  }
  return *this;
}

TensorIterConfig& TensorIterConfig::enforce_linear_iteration(bool v) {
  enforce_linear_iteration_ = v;
  return *this;
}

TensorIterConfig& TensorIterConfig::set_reduce_dims(
    std::span<const std::int64_t> dims,
    bool keepdim) {
  if (!is_reduction_) {
    throw std::logic_error(
        "TI: set_reduce_dims requires is_reduction(true)");
  }
  if (inputs_.empty()) {
    throw std::logic_error(
        "TI: set_reduce_dims requires at least one input");
  }
  if (dims.size() == 0) {
    throw std::invalid_argument(
        "TI: set_reduce_dims requires at least one dim");
  }
  
  // Normalize and sort dims
  const TensorImpl* in0 = inputs_[0];
  const auto& sizes = in0->sizes();
  const std::int64_t R = static_cast<std::int64_t>(sizes.size());
  
  std::vector<std::int64_t> sorted_dims;
  sorted_dims.reserve(dims.size());
  for (std::int64_t d : dims) {
      if (d < 0) d += R;
      if (d < 0 || d >= R) throw std::invalid_argument("TI: reduction dim out of range");
      sorted_dims.push_back(d);
  }
  std::sort(sorted_dims.begin(), sorted_dims.end());
  sorted_dims.erase(std::unique(sorted_dims.begin(), sorted_dims.end()), sorted_dims.end());

  reduce_dims_spec_ = sorted_dims;
  reduce_keepdim_ = keepdim;
  has_reduce_dims_ = true;
  return *this;
}

TensorIterConfig& TensorIterConfig::declare_static_dtype_and_device(ScalarType dtype, Device device) {
  static_dtype_ = dtype;
  static_device_ = device;
  return *this;
}

TensorIterConfig& TensorIterConfig::declare_static_dtype(ScalarType dtype) {
  static_dtype_ = dtype;
  return *this;
}

TensorIterConfig& TensorIterConfig::declare_static_device(Device device) {
  static_device_ = device;
  return *this;
}

TensorIterConfig& TensorIterConfig::declare_static_shape(std::span<const std::int64_t> shape) {
  static_shape_.assign(shape.begin(), shape.end());
  has_static_shape_ = true;
  return *this;
}

TensorIterConfig& TensorIterConfig::set_max_rank(std::int64_t max_rank) {
  max_rank_ = max_rank;
  return *this;
}

TensorIterConfig& TensorIterConfig::set_op_signature(const IterOpSignature* sig) {
  op_signature_ = sig;
  return *this;
}

// and CPU-only single-dim reductions.
TensorIter TensorIterConfig::build() {
  // Reject unsupported flags up front. Promotion flags are implemented in
  if (enforce_linear_iteration_) {
    throw std::logic_error(
        "TI: enforce_linear_iteration is not supported");
  }
  if (outputs_.empty()) {
    throw std::invalid_argument("TI: at least one output is required");
  }
  // Validate max_rank range.
  if (max_rank_ < 1 || max_rank_ > kTensorIterMaxRank) {
    throw std::invalid_argument("TI: max_rank out of range");
  }

  if (allow_multi_device_fabric_ && is_reduction_) {
    throw std::logic_error(
        "[Fabric] TI Fabric mode is only supported for elementwise iterators");
  }

  if (is_reduction_) {
    if (promote_inputs_to_common_dtype_ ||
        promote_integer_inputs_to_float_ ||
        cast_common_dtype_to_outputs_) {
      throw std::logic_error(
          "TI: promotion flags are not supported for reductions");
    }
    if (!has_reduce_dims_) {
      throw std::invalid_argument(
          "TI: reduction dims must be set via set_reduce_dims before build()");
    }

    // ==== Reduction iterator path (single input, single reduction dim) ====
    if (inputs_.size() != 1) {
      throw std::invalid_argument(
          "TI: reductions with multiple inputs are not supported");
    }

    const TensorImpl& in = *inputs_.front();
    const auto& in_sizes = in.sizes();
    const std::size_t R_size = in_sizes.size();
    const std::int64_t R = static_cast<std::int64_t>(R_size);

    if (R == 0) {
      throw std::invalid_argument(
          "TI: cannot build reduction iterator for rank-0 tensors");
    }
    if (R > max_rank_) {
      throw std::invalid_argument("TI: iteration rank exceeds max_rank");
    }

    const Device dev_in = in.device();
    if (dev_in.type != kDLCPU && dev_in.type != kDLCUDA) {
      throw std::invalid_argument(
          "TI: reductions support CPU or CUDA tensors on a single device only");
    }
    const ScalarType dtype_in = in.dtype();

    // Validate outputs and classify roles.
    int value_out_idx = -1;
    int index_out_idx = -1;
    const std::size_t num_outputs = outputs_.size();

    for (std::size_t oi = 0; oi < num_outputs; ++oi) {
      TensorImpl& out_impl = outputs_[oi].get();
      const IterOperandRole role = output_roles_[oi];

      if (out_impl.device().type != kDLCPU && out_impl.device().type != kDLCUDA) {
        throw std::invalid_argument(
            "TI: reductions support CPU or CUDA tensors on a single device only");
      }
      if (out_impl.device() != dev_in) {
        throw std::invalid_argument(
            "TI: reduction outputs must be on the same device as input");
      }

      if (role == IterOperandRole::ReduceOutput) {
        if (value_out_idx != -1) {
          throw std::invalid_argument(
              "TI: reduction requires exactly one ReduceOutput value output");
        }
        if (out_impl.dtype() != dtype_in) {
          throw std::invalid_argument(
              "TI: reduction value outputs must match input dtype");
        }
        value_out_idx = static_cast<int>(oi);
      } else if (role == IterOperandRole::WriteOnly) {
        if (index_out_idx != -1) {
          throw std::invalid_argument(
              "TI: at most one index output is supported for reductions");
        }
        if (out_impl.dtype() != ScalarType::Int64) {
          throw std::invalid_argument(
              "TI: reduction index outputs must have dtype Int64");
        }
        index_out_idx = static_cast<int>(oi);
      } else {
        throw std::invalid_argument(
            "TI: invalid reduction output roles");
      }
    }

    if (value_out_idx == -1) {
      throw std::invalid_argument(
          "TI: reduction requires exactly one ReduceOutput value output");
    }

    if (reduce_dims_spec_.empty()) {
      throw std::logic_error(
          "TI: internal error: reduce_dims_spec_ must not be empty");
    }

    // Compute expected value-output shape.
    std::vector<std::int64_t> expected_value_sizes;
    expected_value_sizes.reserve(R_size);
    
    // Determine if a dim is reduced
    auto is_reduced = [&](std::int64_t d) {
        for (auto rd : reduce_dims_spec_) if (rd == d) return true;
        return false;
    };

    if (reduce_keepdim_) {
      expected_value_sizes.assign(in_sizes.begin(), in_sizes.end());
      for (auto rd : reduce_dims_spec_) {
          expected_value_sizes[static_cast<std::size_t>(rd)] = 1;
      }
    } else {
      for (std::size_t i = 0; i < R_size; ++i) {
        if (is_reduced(static_cast<std::int64_t>(i))) {
          continue;
        }
        expected_value_sizes.push_back(in_sizes[i]);
      }
    }

    TensorImpl& value_out_impl =
        outputs_[static_cast<std::size_t>(value_out_idx)].get();
    if (value_out_impl.sizes() != expected_value_sizes) {
      throw std::invalid_argument(
          "TI: reduction value output shape does not match input shape and dim");
    }

    if (index_out_idx != -1) {
      TensorImpl& index_out_impl =
          outputs_[static_cast<std::size_t>(index_out_idx)].get();
      if (index_out_impl.sizes() != expected_value_sizes) {
        throw std::invalid_argument(
            "TI: reduction index output shape must match value output shape");
      }
    }

    if (!check_mem_overlap_) {
      throw std::logic_error(
          "TI: check_mem_overlap(true) is required for reductions");
    }

    TensorIter iter;
    iter.common_dtype_  = dtype_in;
    iter.common_device_ = dev_in;
    iter.num_outputs_   = static_cast<int>(num_outputs);
    iter.is_reduction_  = true;
    iter.op_signature_  = op_signature_;
    iter.shape_.assign(in_sizes.begin(), in_sizes.end());
    iter.reduce_dims_ = reduce_dims_spec_;
    iter.mem_overlap_checked_ = false;
    iter.has_any_output_input_alias_ = false;
    iter.alias_status_table_.clear();

    // Populate operands: outputs first, then input.
    auto make_reduction_operand = [&](const TensorImpl& impl,
                                      bool is_output,
                                      IterOperandRole role) {
      IterOperand op;
      op.tensor        = const_cast<TensorImpl*>(&impl);
      op.role          = role;
      op.is_output     = is_output;
      op.is_read_write = (role == IterOperandRole::ReadWrite);
      op.will_resize   = false;
      op.dtype         = impl.dtype();
      op.device        = impl.device();
      op.data          = impl.numel() > 0 ? impl.data() : nullptr;
      op.dim_stride_bytes.clear();

      const std::int64_t item_b =
          static_cast<std::int64_t>(impl.itemsize());

      if (R > 0) {
        if (is_output) {
          const auto& out_strides = impl.strides();
          for (std::size_t d = 0; d < R_size; ++d) {
            if (is_reduced(static_cast<std::int64_t>(d))) {
              op.dim_stride_bytes.push_back(0);
              continue;
            }
            std::size_t out_dim;
            if (reduce_keepdim_) {
              out_dim = d;
            } else {
              // Calculate index in output: count how many non-reduced dims are before d
              out_dim = 0;
              for (std::size_t k = 0; k < d; ++k) {
                  if (!is_reduced(static_cast<std::int64_t>(k))) out_dim++;
              }
            }
            std::int64_t tmp = 0;
            if (!checked_mul_i64(out_strides[out_dim], item_b, tmp)) {
              throw std::overflow_error(
                  "TI: overflow computing reduction output stride bytes");
            }
            op.dim_stride_bytes.push_back(tmp);
          }
        } else {
          const auto& in_strides = impl.strides();
          for (std::size_t i = 0; i < R_size; ++i) {
            std::int64_t tmp = 0;
            if (!checked_mul_i64(in_strides[i], item_b, tmp)) {
              throw std::overflow_error(
                  "TI: overflow computing reduction input stride bytes");
            }
            op.dim_stride_bytes.push_back(tmp);
          }
        }
      }

      return op;
    };

    iter.operands_.clear();
    iter.operands_.reserve(num_outputs + 1);
    // Outputs first.
    for (std::size_t oi = 0; oi < num_outputs; ++oi) {
      TensorImpl& out_impl = outputs_[oi].get();
      IterOperandRole role = output_roles_[oi];
      iter.operands_.push_back(
          make_reduction_operand(out_impl, /*is_output=*/true, role));
    }
    // Single input.
    iter.operands_.push_back(
        make_reduction_operand(in, /*is_output=*/false,
                               IterOperandRole::ReadOnly));

    // Overlap checks: internal overlap on writable outputs, then cross-tensor.
    const int out_count = iter.noutputs();
    const int in_count  = iter.ninputs();
    const int input_index = out_count;

    for (int o = 0; o < out_count; ++o) {
      const IterOperand& op =
          iter.operands_[static_cast<std::size_t>(o)];
      if (op.role == IterOperandRole::WriteOnly ||
          op.role == IterOperandRole::ReadWrite ||
          op.role == IterOperandRole::ReduceOutput) {
        check_writable(*op.tensor);
      }
    }

    // Reject output–output overlap.
    for (int o0 = 0; o0 < out_count; ++o0) {
      const TensorImpl& t0 =
          *iter.operands_[static_cast<std::size_t>(o0)].tensor;
      for (int o1 = o0 + 1; o1 < out_count; ++o1) {
        const TensorImpl& t1 =
            *iter.operands_[static_cast<std::size_t>(o1)].tensor;
        if (get_overlap_status(t0, t1) != MemOverlapStatus::No) {
          throw std::invalid_argument(
              "TI: overlapping outputs are not supported; use distinct tensors");
        }
      }
    }

    iter.mem_overlap_checked_ = true;
    iter.has_any_output_input_alias_ = false;
    const std::size_t table_size =
        static_cast<std::size_t>(out_count) *
        static_cast<std::size_t>(in_count);
    iter.alias_status_table_.assign(table_size, MemOverlapStatus::No);

    // Reject any input–output aliasing.
    for (int o = 0; o < out_count; ++o) {
      const TensorImpl& out_t =
          *iter.operands_[static_cast<std::size_t>(o)].tensor;
      const TensorImpl& in_t =
          *iter.operands_[static_cast<std::size_t>(input_index)].tensor;
      if (get_overlap_status(out_t, in_t) != MemOverlapStatus::No) {
        throw std::invalid_argument(
            "TI: reduction outputs must not alias inputs; clone() the input first");
      }
    }

    return iter;
  }

  // ==== Elementwise iterator path (possibly multi-output) ====

  // Enforce dtype and device invariants across all operands. Shape
  // compatibility is validated by the broadcasting helpers below.
  TensorImpl& out0 = outputs_[0].get();
  const ScalarType dtype_out = out0.dtype();
  const Device     dev_out   = out0.device();

  if (dev_out.type != kDLCPU && dev_out.type != kDLCUDA) {
    throw std::invalid_argument(
        "TI: elementwise iterators support only CPU or CUDA tensors");
  }

  if (allow_multi_device_fabric_) {
    if (dev_out.type != kDLCUDA) {
      throw std::invalid_argument(
          "[Fabric] TI Fabric mode supports only CUDA tensors");
    }
    if (fabric_primary_device_ < 0) {
      throw std::logic_error(
          "[Fabric] TI Fabric mode internal error: missing primary device");
    }
    if (dev_out.index != fabric_primary_device_) {
      throw std::invalid_argument(
          "[Fabric] TI Fabric: outputs must be on the primary CUDA device");
    }
  }

  const int noutputs_cfg = static_cast<int>(outputs_.size());
  const int ninputs_cfg  = static_cast<int>(inputs_.size());

  // Flag dependency checks for promotion.
  if (promote_integer_inputs_to_float_ && !promote_inputs_to_common_dtype_) {
    throw std::logic_error(
        "TI: promote_integer_inputs_to_float requires "
        "promote_inputs_to_common_dtype(true)");
  }
  if (cast_common_dtype_to_outputs_ && !promote_inputs_to_common_dtype_) {
    throw std::logic_error(
        "TI: cast_common_dtype_to_outputs requires "
        "promote_inputs_to_common_dtype(true)");
  }
  if ((promote_inputs_to_common_dtype_ ||
       promote_integer_inputs_to_float_ ||
       cast_common_dtype_to_outputs_) &&
      ninputs_cfg == 0) {
    throw std::logic_error(
        "TI: promotion flags are not supported for nullary or zero-input iterators");
  }

  const bool enable_promotion = promote_inputs_to_common_dtype_;

  // Validate outputs (roles, device) and enforce a single output dtype.
  for (std::size_t idx = 0; idx < outputs_.size(); ++idx) {
    TensorImpl& out_impl = outputs_[idx].get();
    if (output_roles_[idx] == IterOperandRole::ReduceOutput) {
      throw std::invalid_argument(
          "TI: ReduceOutput role is only valid for reduction iterators");
    }
    if (!allow_multi_device_fabric_) {
      if (out_impl.device() != dev_out) {
        throw std::invalid_argument(
            "TI: all operands must be on the same CPU device");
      }
    } else {
      if (out_impl.device().type != kDLCUDA) {
        throw std::invalid_argument(
            "[Fabric] TI Fabric mode supports only CUDA tensors");
      }
      if (out_impl.device().index != fabric_primary_device_) {
        throw std::invalid_argument(
            "[Fabric] TI Fabric: outputs must be on the primary CUDA device");
      }
    }
    if (check_all_same_dtype_ && out_impl.dtype() != dtype_out) {
      if (enable_promotion) {
        throw std::invalid_argument(
            "TI: all outputs must share the same dtype under promotion");
      }
      throw std::invalid_argument(
          "TI: all operands must have the same dtype");
    }
  }

  // Accumulate promotion state from inputs while enforcing device invariants.
  ResultTypeState promotion_state{};
  for (const TensorImpl* in_impl : inputs_) {
    bool is_cpu_scalar = (in_impl->device().type == kDLCPU && in_impl->numel() == 1 && in_impl->sizes().empty());

    if (!allow_multi_device_fabric_) {
      if (allow_cpu_scalars_ && is_cpu_scalar) {
        // Allowed CPU scalar mixed with CUDA/other device
      } else if (in_impl->device() != dev_out) {
        throw std::invalid_argument(
            "TI: all operands must be on the same device (unless allow_cpu_scalars is set)");
      }
    } else {
      if (is_cpu_scalar) {
        throw std::invalid_argument(
            "[Fabric] TI Fabric mode does not support CPU scalar operands");
      }
      if (in_impl->device().type != kDLCUDA) {
        throw std::invalid_argument(
            "[Fabric] TI Fabric mode supports only CUDA tensors");
      }
    }

    if (!enable_promotion) {
      if (check_all_same_dtype_ && in_impl->dtype() != dtype_out) {
        throw std::invalid_argument(
            "TI: all operands must have the same dtype");
      }
    } else {
      update_result_type_state(promotion_state, in_impl->dtype());
    }
  }

  ScalarType common_dtype = dtype_out;
  if (enable_promotion) {
    common_dtype = result_type(promotion_state);
    if (promote_integer_inputs_to_float_ &&
        is_integral_or_bool(common_dtype)) {
      common_dtype = ScalarType::Float32;
    }

    // Enforce output vs common dtype semantics.
    if (!cast_common_dtype_to_outputs_) {
      if (dtype_out != common_dtype) {
        throw std::invalid_argument(
            "TI: output dtype must equal common input dtype");
      }
    } else {
      // Reject in-place + cast combinations.
      bool has_inplace_output = false;
      for (IterOperandRole role : output_roles_) {
        if (role == IterOperandRole::ReadWrite) {
          has_inplace_output = true;
          break;
        }
      }
      if (has_inplace_output) {
        throw std::logic_error(
            "TI: cast_common_dtype_to_outputs is not allowed for in-place iterators");
      }
      if (!can_cast(common_dtype, dtype_out)) {
        throw std::invalid_argument(
            "TI: cannot cast common dtype to output dtype");
      }
    }
  }

  // Compute broadcasted logical shape across all outputs and inputs.
  std::vector<std::vector<std::int64_t>> shapes;
  shapes.reserve(outputs_.size() + inputs_.size());
  for (const auto& out_ref : outputs_) {
    shapes.push_back(out_ref.get().sizes());
  }
  for (const TensorImpl* in_impl : inputs_) {
    shapes.push_back(in_impl->sizes());
  }

  std::vector<std::int64_t> broadcast_shape;
  try {
    broadcast_shape = infer_broadcast_shape_nary(
        std::span<const std::vector<std::int64_t>>(shapes));
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument(
        std::string("TI: all operands must have the same shape or be "
                    "broadcastable: ") + e.what());
  }

  // If static shape is declared, validate it against broadcast shape
  if (has_static_shape_) {
     if (static_shape_ != broadcast_shape) {
         throw std::invalid_argument("TI: inferred shape does not match declared static shape");
     }
  }
  // If static dtype is declared, validate it against common dtype
  if (static_dtype_ != ScalarType::Undefined) {
      if (common_dtype != static_dtype_) {
           throw std::invalid_argument("TI: inferred dtype does not match declared static dtype");
      }
  }
  // If static device is declared, validate
  const Device sentinel{kDLCPU, -1};
  if (static_device_ != sentinel) {
      if (dev_out != static_device_) {
          throw std::invalid_argument("TI: inferred device does not match declared static device");
      }
  }

  // Enforce that all outputs already match the broadcasted shape; TI never
  // resizes outputs.
  if (broadcast_shape != out0.sizes()) {
    if (resize_outputs_ && output_allow_resize_[0]) {
      if (dev_out.type != kDLCPU) {
        throw std::logic_error("TI: resizing non-CPU outputs is not yet supported");
      }
      
      // Detect Memory Format (Channels Last)
      bool use_channels_last = false;
      if (broadcast_shape.size() == 4) {
        bool all_inputs_cl = true;
        bool has_inputs = false;
        for (const TensorImpl* in : inputs_) {
          has_inputs = true;
          if (!in->is_channels_last()) {
            all_inputs_cl = false;
            break;
          }
        }
        if (has_inputs && all_inputs_cl) {
          use_channels_last = true;
        }
      }
      
      std::int64_t numel = 1;
      std::vector<std::int64_t> new_strides(broadcast_shape.size());
      
      if (use_channels_last) {
         // NHWC Strides: (HWC, 1, WC, C) for shape (N, C, H, W)
         std::int64_t n = broadcast_shape[0];
         std::int64_t c = broadcast_shape[1];
         std::int64_t h = broadcast_shape[2];
         std::int64_t w = broadcast_shape[3];
         
         std::int64_t stride_c = 1;
         std::int64_t stride_w = c; // Checked below
         std::int64_t stride_h = 0;
         std::int64_t stride_n = 0;
         
         // Use checked math
         if (!checked_mul_i64(w, c, stride_h)) throw std::overflow_error("TI: overflow");
         if (!checked_mul_i64(stride_h, h, stride_n)) throw std::overflow_error("TI: overflow");
         
         new_strides[0] = stride_n;
         new_strides[1] = 1;
         new_strides[2] = stride_h;
         new_strides[3] = stride_w;
         
         if (!checked_mul_i64(stride_n, n, numel)) throw std::overflow_error("TI: overflow");
         
      } else if (!broadcast_shape.empty()) {
        std::int64_t st = 1;
        for (std::size_t i = broadcast_shape.size(); i-- > 0;) {
          new_strides[i] = st;
          if (!checked_mul_i64(st, broadcast_shape[i], st)) {
             throw std::overflow_error("TI: overflow computing new output strides");
          }
        }
        numel = st;
      } else {
        numel = 1; // Scalar
      }
      
      std::int64_t nbytes = 0;
      if (!checked_mul_i64(numel, static_cast<std::int64_t>(out0.itemsize()), nbytes)) {
         throw std::overflow_error("TI: overflow computing new output size in bytes");
      }
      
      auto new_storage = vbt::cpu::new_cpu_storage(static_cast<std::size_t>(nbytes), false);
      out0.set_storage(std::move(new_storage));
      out0.set_storage_offset(0);
      out0.set_sizes_and_strides(broadcast_shape, new_strides);
    } else {
      throw std::invalid_argument(
          "TI: output[0] shape does not match broadcasted shape (output " +
          shape_to_string(std::span<const std::int64_t>(out0.sizes())) +
          ", broadcast " +
          shape_to_string(std::span<const std::int64_t>(broadcast_shape)) +
          ")");
    }
  }
  
  const auto& sizes_out0 = out0.sizes(); // Refresh reference after potential resize

  // Compute the iteration rank after dropping size-1 dimensions. This bound is
  // independent of permutation and prevents stride-byte overflow in
  // pathological high-rank shapes (including zero-numel ones).
  std::int64_t R_iter_bound = 0;
  for (std::int64_t s : sizes_out0) {
    if (s != 1) {
      ++R_iter_bound;
    }
  }
  if (R_iter_bound > max_rank_) {
    throw std::invalid_argument("TI: iteration rank exceeds max_rank");
  }

  for (std::size_t idx = 1; idx < outputs_.size(); ++idx) {
    TensorImpl& out_i = outputs_[idx].get();
    if (out_i.sizes() != broadcast_shape) {
      if (resize_outputs_ && output_allow_resize_[idx]) {
        if (out_i.device().type != kDLCPU) {
           throw std::logic_error("TI: resizing non-CPU outputs is not yet supported");
        }

        // Detect Memory Format (Channels Last)
        bool use_channels_last = false;
        if (broadcast_shape.size() == 4) {
            bool all_inputs_cl = true;
            bool has_inputs = false;
            for (const TensorImpl* in : inputs_) {
            has_inputs = true;
            if (!in->is_channels_last()) {
                all_inputs_cl = false;
                break;
            }
            }
            if (has_inputs && all_inputs_cl) {
            use_channels_last = true;
            }
        }

        std::int64_t numel = 1;
        std::vector<std::int64_t> new_strides(broadcast_shape.size());
        
        if (use_channels_last) {
            std::int64_t n = broadcast_shape[0];
            std::int64_t c = broadcast_shape[1];
            std::int64_t h = broadcast_shape[2];
            std::int64_t w = broadcast_shape[3];
            
            std::int64_t stride_c = 1;
            std::int64_t stride_w = c;
            std::int64_t stride_h = 0;
            std::int64_t stride_n = 0;
            
            if (!checked_mul_i64(w, c, stride_h)) throw std::overflow_error("TI: overflow");
            if (!checked_mul_i64(stride_h, h, stride_n)) throw std::overflow_error("TI: overflow");
            
            new_strides[0] = stride_n;
            new_strides[1] = 1;
            new_strides[2] = stride_h;
            new_strides[3] = stride_w;
            
            if (!checked_mul_i64(stride_n, n, numel)) throw std::overflow_error("TI: overflow");
        } else if (!broadcast_shape.empty()) {
          std::int64_t st = 1;
          for (std::size_t i = broadcast_shape.size(); i-- > 0;) {
            new_strides[i] = st;
            if (!checked_mul_i64(st, broadcast_shape[i], st)) {
               throw std::overflow_error("TI: overflow computing new output strides");
            }
          }
          numel = st;
        } else {
           numel = 1;
        }

        std::int64_t nbytes = 0;
        if (!checked_mul_i64(numel, static_cast<std::int64_t>(out_i.itemsize()), nbytes)) {
           throw std::overflow_error("TI: overflow computing new output size in bytes");
        }
        auto new_storage = vbt::cpu::new_cpu_storage(static_cast<std::size_t>(nbytes), false);
        out_i.set_storage(std::move(new_storage));
        out_i.set_storage_offset(0);
        out_i.set_sizes_and_strides(broadcast_shape, new_strides);
      } else {
        throw std::invalid_argument(
            "TI: all outputs must match the broadcasted shape");
      }
    }
  }

  TensorIter iter;
  iter.common_dtype_  = common_dtype;
  iter.common_device_ = allow_multi_device_fabric_ ? Device::cuda(fabric_primary_device_) : dev_out;
  iter.num_outputs_   = static_cast<int>(outputs_.size());
  iter.is_reduction_  = false;
  iter.op_signature_  = op_signature_;
  iter.reduce_dims_.clear();
  iter.mem_overlap_checked_ = false;
  iter.has_any_output_input_alias_ = false;
  iter.alias_status_table_.clear();

  // Compute permutation and iteration-space shape using the first output.
  const auto& strides_out = out0.strides();
  const std::size_t ndim_orig = sizes_out0.size();
  const std::int64_t item_b =
      static_cast<std::int64_t>(out0.itemsize());

  std::vector<std::int64_t> signed_stride_bytes(ndim_orig);
  std::vector<std::int64_t> step_bytes(ndim_orig);
  for (std::size_t i = 0; i < ndim_orig; ++i) {
    std::int64_t tmp = 0;
    if (!checked_mul_i64(strides_out[i], item_b, tmp)) {
      throw std::overflow_error("TI: overflow computing stride bytes");
    }
    signed_stride_bytes[i] = tmp;
    std::int64_t abs_st = 0;
    if (!checked_abs_i64_hdr(tmp, abs_st)) {
      throw std::overflow_error(
          "TI: overflow computing absolute stride bytes");
    }
    step_bytes[i] = abs_st;
  }

  // Build permutation in increasing effective step bytes, size-1 dims last.
  std::vector<std::int64_t> perm(ndim_orig);
  for (std::size_t i = 0; i < ndim_orig; ++i) {
    perm[i] = static_cast<std::int64_t>(i);
  }

  std::stable_sort(perm.begin(), perm.end(), [&](std::int64_t a, std::int64_t b) {
    const auto ia = static_cast<std::size_t>(a);
    const auto ib = static_cast<std::size_t>(b);
    const std::int64_t size_a = sizes_out0[ia];
    const std::int64_t size_b = sizes_out0[ib];

    auto eff_step = [&](std::int64_t size, std::int64_t step) {
      if (size == 1) {
        // Size-1 dims are dropped from the iteration space and should come
        // last in the permutation.
        return std::numeric_limits<std::int64_t>::max();
      }
      if (size == 0) {
        // Zero-sized dims participate in the iteration shape but force
        // numel()==0. Put them first so that the inner dimension is never
        // a zero-sized dim; this keeps the "no callbacks for numel==0"
        // behavior simple and matches the broadcast tests.
        return static_cast<std::int64_t>(0);
      }
      return step;
    };

    const std::int64_t sa = eff_step(size_a, step_bytes[ia]);
    const std::int64_t sb = eff_step(size_b, step_bytes[ib]);
    return sa < sb;  // Ascending: zero-sized dims first, size-1 dims last
  });

  iter.shape_.clear();
  std::vector<std::int64_t> kept_dims;
  kept_dims.reserve(ndim_orig);

  for (std::int64_t d : perm) {
    const std::size_t i = static_cast<std::size_t>(d);
    if (sizes_out0[i] == 1) {
      continue;  // drop size-1 dims from iteration space
    }
    iter.shape_.push_back(sizes_out0[i]);
    kept_dims.push_back(d);
  }

  const std::int64_t R_iter =
      static_cast<std::int64_t>(iter.shape_.size());
#ifdef VBT_TI_DEBUG
  // Sanity-check that the final iteration rank matches the pre-checked bound.
  assert(R_iter == R_iter_bound);
#endif
  if (R_iter > max_rank_) {
    throw std::invalid_argument("TI: iteration rank exceeds max_rank");
  }

  const std::int64_t R_broadcast =
      static_cast<std::int64_t>(broadcast_shape.size());

  // Populate operands: outputs first, then inputs.
  auto make_operand = [&](const TensorImpl& impl,
                          bool is_output,
                          IterOperandRole role) {
    IterOperand op;
    op.tensor        = const_cast<TensorImpl*>(&impl);
    op.role          = role;
    op.is_output     = is_output;
    op.is_read_write = (role == IterOperandRole::ReadWrite);
    op.will_resize   = false;
    op.dtype         = impl.dtype();
    op.device        = impl.device();
    op.data          = impl.numel() > 0 ? impl.data() : nullptr;

    op.dim_stride_bytes.clear();
    if (R_iter != 0) {
      const auto& orig_sizes   = impl.sizes();
      const auto& orig_strides = impl.strides();
      const std::size_t r      = orig_sizes.size();
      if (r > static_cast<std::size_t>(R_broadcast)) {
        throw std::invalid_argument(
            "TI: operand rank exceeds broadcast rank");
      }
      const std::int64_t pad =
          R_broadcast - static_cast<std::int64_t>(r);

      std::vector<std::int64_t> logical_elem_strides(
          static_cast<std::size_t>(R_broadcast));
      for (std::int64_t d = 0; d < R_broadcast; ++d) {
        const std::int64_t b = broadcast_shape[static_cast<std::size_t>(d)];
        std::int64_t elem_stride = 0;
        if (d < pad) {
          // Padded leading dims behave as size-1 with stride 0.
          elem_stride = 0;
        } else {
          const std::size_t orig_dim =
              static_cast<std::size_t>(d - pad);
          const std::int64_t s = orig_sizes[orig_dim];
          if (s == b) {
            elem_stride = orig_strides[orig_dim];
          } else {
            // Broadcasted dimension: require size 1.
            if (s != 1) {
              throw std::invalid_argument(
                  "TI: operand shape is not broadcastable to output shape");
            }
            elem_stride = 0;
          }
        }
        logical_elem_strides[static_cast<std::size_t>(d)] = elem_stride;
      }

      const std::int64_t item_b2 =
          static_cast<std::int64_t>(impl.itemsize());
      for (std::int64_t d_orig : kept_dims) {
        const std::size_t dim =
            static_cast<std::size_t>(d_orig);
        const std::int64_t elem_stride =
            logical_elem_strides[static_cast<std::size_t>(dim)];
        std::int64_t tmp = 0;
        if (!checked_mul_i64(elem_stride, item_b2, tmp)) {
          throw std::overflow_error(
              "TI: overflow computing operand stride bytes");
        }
        op.dim_stride_bytes.push_back(tmp);
      }
    }
    return op;
  };

  iter.operands_.clear();
  iter.operands_.reserve(outputs_.size() + inputs_.size());
  // Outputs first.
  for (std::size_t out_index = 0; out_index < outputs_.size(); ++out_index) {
    TensorImpl& impl = outputs_[out_index].get();
    IterOperandRole role = output_roles_[out_index];
    iter.operands_.push_back(
        make_operand(impl, /*is_output=*/true, role));
  }
  // Then inputs (read-only in elementwise mode).
  for (const TensorImpl* in_impl : inputs_) {
    iter.operands_.push_back(
        make_operand(*in_impl, /*is_output=*/false,
                     IterOperandRole::ReadOnly));
  }

  if (allow_multi_device_fabric_) {
    // devices, primary must be present, and remote operands must be read-only
    // inputs.
    std::vector<Device> devices;
    devices.reserve(iter.operands_.size());

    for (const IterOperand& op : iter.operands_) {
      if (op.tensor == nullptr) {
        continue;
      }
      devices.push_back(op.device);
    }

    if (devices.empty()) {
      throw std::invalid_argument(
          "[Fabric] TI Fabric mode requires at least one CUDA tensor operand");
    }

    const DLDeviceType type0 = devices[0].type;
    for (Device d : devices) {
      if (d.type != type0) {
        throw std::invalid_argument(
            "[Fabric] TI Fabric mode requires uniform device types");
      }
    }

    if (type0 != kDLCUDA) {
      throw std::invalid_argument(
          "[Fabric] TI Fabric mode supports only CUDA tensors");
    }

    std::sort(devices.begin(), devices.end(),
              [](Device a, Device b) { return a.index < b.index; });
    devices.erase(std::unique(devices.begin(), devices.end(),
                              [](Device a, Device b) {
                                return a.index == b.index;
                              }),
                  devices.end());

    if (devices.size() > 2) {
      throw std::invalid_argument(
          "[Fabric] TI Fabric mode supports at most two CUDA devices");
    }

    bool primary_in_operands = false;
    for (Device d : devices) {
      if (d.index == fabric_primary_device_) {
        primary_in_operands = true;
        break;
      }
    }
    if (!primary_in_operands) {
      throw std::invalid_argument(
          "[Fabric] TI Fabric compute_device must be one of the operand devices");
    }

    std::optional<int> remote_device;
    for (const IterOperand& op : iter.operands_) {
      if (op.tensor == nullptr) {
        continue;
      }

      if (op.device.index == fabric_primary_device_) {
        continue;
      }

      if (!remote_device.has_value()) {
        remote_device = op.device.index;
      } else if (*remote_device != op.device.index) {
        throw std::logic_error(
            "[Fabric] TI Fabric: at most one remote CUDA device is supported");
      }

      if (op.is_output || op.role != IterOperandRole::ReadOnly) {
        throw std::logic_error(
            "[Fabric] TI Fabric: remote operands must be read-only inputs");
      }
    }
  }

  // Overlap checks: always guard internal overlap on writable outputs, reject
  // output–output overlap, and optionally compute cross-tensor alias metadata.
  const int out_count = iter.noutputs();
  const int in_count  = iter.ninputs();

  for (int o = 0; o < out_count; ++o) {
    const IterOperand& op =
        iter.operands_[static_cast<std::size_t>(o)];
    if (op.role == IterOperandRole::WriteOnly ||
        op.role == IterOperandRole::ReadWrite ||
        op.role == IterOperandRole::ReduceOutput) {
      check_writable(*op.tensor);
    }
  }

  // Output–output overlap is always rejected.
  for (int o0 = 0; o0 < out_count; ++o0) {
    const TensorImpl& t0 =
        *iter.operands_[static_cast<std::size_t>(o0)].tensor;
    for (int o1 = o0 + 1; o1 < out_count; ++o1) {
      const TensorImpl& t1 =
          *iter.operands_[static_cast<std::size_t>(o1)].tensor;
      if (get_overlap_status(t0, t1) != MemOverlapStatus::No) {
        throw std::invalid_argument(
            "TI: overlapping outputs are not supported; use distinct tensors");
      }
    }
  }

  iter.mem_overlap_checked_ = false;
  iter.has_any_output_input_alias_ = false;
  iter.alias_status_table_.clear();

  // Coalescing (optimization) should happen before early return for no-overlap check,
  // so that fastpath logic (which relies on rank-1) can benefit from it.
  
  // Coalesce dimensions (TI Optimization)
  if (!iter.is_reduction()) {
    int i = 0;
    while (i < static_cast<int>(iter.shape_.size()) - 1) {
      const std::size_t idx = static_cast<std::size_t>(i);
      const std::size_t next = idx + 1;
      
      bool can_coalesce = true;
      for (const auto& op : iter.operands_) {
        const std::int64_t s0 = op.dim_stride_bytes[idx];
        const std::int64_t s1 = op.dim_stride_bytes[next];
        const std::int64_t dim_size_next = iter.shape_[next];
        
        std::int64_t expected = 0;
        if (!checked_mul_i64(s1, dim_size_next, expected)) {
          can_coalesce = false;
          break;
        }
        
        if (s0 != expected) {
          can_coalesce = false;
          break;
        }
      }
      
      if (can_coalesce) {
        std::int64_t new_size = 0;
        if (!checked_mul_i64(iter.shape_[idx], iter.shape_[next], new_size)) {
           i++;
           continue;
        }
        iter.shape_[idx] = new_size;
        for (auto& op : iter.operands_) {
          op.dim_stride_bytes[idx] = op.dim_stride_bytes[next];
          op.dim_stride_bytes.erase(op.dim_stride_bytes.begin() + next);
        }
        iter.shape_.erase(iter.shape_.begin() + next);
      } else {
        i++;
      }
    }
  }

  maybe_set_cpu_nod_contig_fastpath(iter);

  if (!check_mem_overlap_ || in_count == 0) {
    return iter;
  }

  iter.mem_overlap_checked_ = true;
  const std::size_t table_size_elem =
      static_cast<std::size_t>(out_count) *
      static_cast<std::size_t>(in_count);
  iter.alias_status_table_.assign(table_size_elem, MemOverlapStatus::No);

  // Validate alias metadata from op_signature_ (if any) and mark allowed
  // in-place alias pairs.
  std::vector<bool> allow_inplace_alias(table_size_elem, false);
  if (iter.op_signature_ && iter.op_signature_->aliases) {
    const IterOpSignature* sig = iter.op_signature_;
    for (std::size_t idx = 0; idx < sig->alias_count; ++idx) {
      const IterAliasInfo& info = sig->aliases[idx];
      if (info.is_view) {
        throw std::invalid_argument(
            "TI: view-style alias entries are not supported");
      }
      if (info.output_index < 0 || info.output_index >= out_count ||
          info.input_index < 0 || info.input_index >= in_count) {
        throw std::invalid_argument(
            "TI: alias indices in IterOpSignature are out of range");
      }
      const std::size_t flat =
          static_cast<std::size_t>(info.output_index) *
              static_cast<std::size_t>(in_count) +
          static_cast<std::size_t>(info.input_index);
      if (allow_inplace_alias[flat]) {
        throw std::invalid_argument(
            "TI: duplicate alias entry in IterOpSignature");
      }
      allow_inplace_alias[flat] = info.is_inplace;
    }
  }

  // Compute cross-tensor overlap and populate alias metadata.
  for (int o = 0; o < out_count; ++o) {
    const TensorImpl& out_t =
        *iter.operands_[static_cast<std::size_t>(o)].tensor;
    for (int i = 0; i < in_count; ++i) {
      const TensorImpl& in_t =
          *iter.operands_[static_cast<std::size_t>(out_count + i)].tensor;
      const std::size_t flat =
          static_cast<std::size_t>(o) *
              static_cast<std::size_t>(in_count) +
          static_cast<std::size_t>(i);

      MemOverlapStatus status = get_overlap_status(out_t, in_t);
      if (!allow_inplace_alias[flat]) {
        if (status == MemOverlapStatus::No) {
          iter.alias_status_table_[flat] = MemOverlapStatus::No;
          continue;
        }
        if (status == MemOverlapStatus::Partial) {
          // Delegate to assert_no_partial_overlap for message parity.
          assert_no_partial_overlap(out_t, in_t);
        }
        if (status == MemOverlapStatus::Full) {
          // Full aliasing is not allowed for out-of-place ops.
          assert_no_overlap(out_t, in_t);
        }
        // semantics that require overlap analysis to succeed when
        // check_mem_overlap(true) is requested.
        throw std::invalid_argument(
            "TI: aliasing on non-NO&D tensors is not supported; "
            "please clone() inputs or use an out-of-place op");
      }

      // In-place alias path: only allow exact full aliasing on declared
      // pairs; everything else is rejected with a clear error.
      if (status == MemOverlapStatus::No) {
        iter.alias_status_table_[flat] = MemOverlapStatus::No;
      } else if (status == MemOverlapStatus::Full) {
        iter.alias_status_table_[flat] = MemOverlapStatus::Full;
        iter.has_any_output_input_alias_ = true;
      } else if (status == MemOverlapStatus::Partial) {
        throw std::invalid_argument(
            "TI: partial overlap is not allowed for in-place aliases; "
            "please clone() inputs before calling this op");
      } else {  // TooHard
        throw std::invalid_argument(
            "TI: in-place aliasing on non-NO&D tensors is not supported; "
            "please clone() inputs or use an out-of-place op");
      }
    }
  }

  return iter;
}

// ==== TensorIter factories ===============================================================

namespace {

static void ensure_cpu_and_dtype_match(TensorImpl& out,
                                       std::initializer_list<const TensorImpl*> ins) {
  const ScalarType dtype_out = out.dtype();
  const Device     dev_out   = out.device();

  if (dev_out.type != kDLCPU) {
    throw std::invalid_argument("TI: only CPU tensors are supported");
  }

  for (const TensorImpl* in : ins) {
    if (in->device() != dev_out) {
      throw std::invalid_argument(
          "TI: all operands must be on the same CPU device");
    }
    if (in->dtype() != dtype_out) {
      throw std::invalid_argument(
          "TI: all operands must have the same dtype");
    }
    if (in->sizes() != out.sizes()) {
      throw std::invalid_argument(
          "TI: all operands must have the same shape");
    }
  }
}

}  // namespace

TensorIter TensorIter::unary_op(TensorImpl& out, const TensorImpl& a) {
  ensure_cpu_and_dtype_match(out, {&a});
  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.check_mem_overlap(false);
  return cfg.build();
}

TensorIter TensorIter::binary_op(TensorImpl& out,
                                 const TensorImpl& a,
                                 const TensorImpl& b) {
  ensure_cpu_and_dtype_match(out, {&a, &b});
  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.add_input(b);
  cfg.check_mem_overlap(false);
  return cfg.build();
}

TensorIter TensorIter::nullary_op(TensorImpl& out) {
  if (out.device().type != kDLCPU) {
    throw std::invalid_argument("TI: only CPU tensors are supported");
  }
  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.check_mem_overlap(false);
  return cfg.build();
}

TensorIter TensorIter::reduce_op(TensorImpl& out,
                                 const TensorImpl& a,
                                 std::span<const std::int64_t> dims) {
  if (dims.size() == 0) {
    throw std::invalid_argument(
        "TI: reduce_op expects at least one dim");
  }

  const auto& in_sizes = a.sizes();
  const std::int64_t R = static_cast<std::int64_t>(in_sizes.size());
  if (R <= 0) {
    throw std::invalid_argument(
        "TI: cannot build reduction iterator for rank-0 tensors");
  }

  std::vector<std::int64_t> norm_dims;
  norm_dims.reserve(dims.size());
  for (std::int64_t d : dims) {
    if (d < 0) d += R;
    if (d < 0 || d >= R) {
      throw std::invalid_argument("TI: reduction dim out of range");
    }
    norm_dims.push_back(d);
  }
  std::sort(norm_dims.begin(), norm_dims.end());
  auto last = std::unique(norm_dims.begin(), norm_dims.end());
  norm_dims.erase(last, norm_dims.end());

  auto is_reduced = [&](std::int64_t d) {
    for (auto rd : norm_dims) {
      if (rd == d) return true;
    }
    return false;
  };

  // Infer keepdim from the value output shape.
  std::vector<std::int64_t> expected_keep(in_sizes.begin(), in_sizes.end());
  for (auto d : norm_dims) {
    expected_keep[static_cast<std::size_t>(d)] = 1;
  }

  std::vector<std::int64_t> expected_drop;
  expected_drop.reserve(static_cast<std::size_t>(R > 0 ? R - norm_dims.size() : 0));
  for (std::int64_t d = 0; d < R; ++d) {
    if (is_reduced(d)) continue;
    expected_drop.push_back(in_sizes[static_cast<std::size_t>(d)]);
  }

  const auto& out_sizes = out.sizes();
  const bool matches_keep = (out_sizes == expected_keep);
  const bool matches_drop = (out_sizes == expected_drop);

  bool keepdim = false;
  if (matches_keep && !matches_drop) {
    keepdim = true;
  } else if (matches_drop && !matches_keep) {
    keepdim = false;
  } else if (matches_keep && matches_drop) {
    // Ambiguous case (e.g. reduced dim size 1), default to keepdim=false usually?
    // Or if out is already defined, we check if it has 1s.
    // If out shape matches both, it means the reduced dimensions were size 1.
    // In that case, keepdim=True -> output has 1s. keepdim=False -> output drops them.
    // Since dropping size-1 dims results in same shape as keeping them (if we reduce them), 
    // actually: if reduced dim is size 1. Keep: size 1. Drop: gone.
    // If we reduce dim of size 1, keep -> 1. Drop -> nothing.
    // (2, 1, 3) reduce dim 1. Keep: (2, 1, 3). Drop: (2, 3).
    // So they are different unless we reduce ALL dims and they are all 1?
    // If shape is (1, 1). Reduce all. Keep: (1, 1). Drop: ().
    // So ambiguous only if we reduce nothing? But dims size > 0.
    // So matches_keep and matches_drop are disjoint usually.
    // Exception: reduce dim of size 1... wait.
    // (2, 1, 3). reduce dim 1.
    // expected_keep: (2, 1, 3).
    // expected_drop: (2, 3).
    // Different.
    // So if matches_keep, keepdim=true.
    // If matches_drop, keepdim=false.
    // If both? Impossible if we reduce at least one dim.
    // Wait, if input is (1,), reduce dim 0. Keep: (1,). Drop: ().
    // If input is (), reduce? No rank 0 rejected.
    
    // Just proceed.
    keepdim = false; 
  } else {
    throw std::invalid_argument(
        "TI: output shape does not match reduction dim for reduce_op");
  }

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.is_reduction(true);
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::ReduceOutput);
  cfg.add_input(a);
  cfg.set_reduce_dims(dims, keepdim);
  return cfg.build();
}

TensorIter TensorIter::reduce_op(TensorImpl& out_values,
                                 TensorImpl& out_indices,
                                 const TensorImpl& a,
                                 std::span<const std::int64_t> dims) {
  if (dims.size() == 0) {
    throw std::invalid_argument(
        "TI: reduce_op expects a single dim; dim=None is implemented at the kernel layer");
  }
  if (dims.size() > 1) {
    throw std::invalid_argument(
        "TI: only a single reduction dim is supported");
  }

  const auto& in_sizes = a.sizes();
  const std::int64_t R = static_cast<std::int64_t>(in_sizes.size());
  if (R <= 0) {
    throw std::invalid_argument(
        "TI: cannot build reduction iterator for rank-0 tensors");
  }

  std::int64_t dim = dims[0];
  if (dim < 0) {
    dim += R;
  }
  if (dim < 0 || dim >= R) {
    throw std::invalid_argument("TI: reduction dim out of range");
  }

  std::vector<std::int64_t> expected_keep(in_sizes.begin(), in_sizes.end());
  expected_keep[static_cast<std::size_t>(dim)] = 1;

  std::vector<std::int64_t> expected_drop;
  expected_drop.reserve(static_cast<std::size_t>(R > 0 ? R - 1 : 0));
  for (std::int64_t d = 0; d < R; ++d) {
    if (d == dim) continue;
    expected_drop.push_back(in_sizes[static_cast<std::size_t>(d)]);
  }

  const auto& val_sizes = out_values.sizes();
  const auto& idx_sizes = out_indices.sizes();
  const bool matches_keep = (val_sizes == expected_keep);
  const bool matches_drop = (val_sizes == expected_drop);

  bool keepdim = false;
  if (matches_keep && !matches_drop) {
    keepdim = true;
  } else if (matches_drop && !matches_keep) {
    keepdim = false;
  } else {
    throw std::invalid_argument(
        "TI: value output shape does not match reduction dim for reduce_op");
  }

  if (idx_sizes != val_sizes) {
    throw std::invalid_argument(
        "TI: index output shape must match value output shape for reduce_op");
  }

  TensorIterConfig cfg;
  cfg.check_mem_overlap(true);
  cfg.is_reduction(true);
  cfg.add_output(OptionalTensorImplRef(&out_values, /*defined=*/true),
                 IterOperandRole::ReduceOutput);
  cfg.add_output(OptionalTensorImplRef(&out_indices, /*defined=*/true),
                 IterOperandRole::WriteOnly);
  cfg.add_input(a);
  cfg.set_reduce_dims(std::span<const std::int64_t>(&dim, 1), keepdim);
  return cfg.build();
}

} // namespace core
} // namespace vbt
