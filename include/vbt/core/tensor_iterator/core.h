// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>
#include <limits>
#include <utility>
#include <atomic>

#include <type_traits>
#include <cassert>

#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor.h"
#include "vbt/core/overlap.h"
#include "vbt/core/checked_math.h"
#include "vbt/core/strided_loop.h"

namespace vbt {
namespace core {

// Maximum supported iteration rank for Tensor Iterator. This is a hard limit
// and becomes ABI-relevant once exposed to plugins.
inline constexpr std::int32_t kTensorIterMaxRank = 64;
static_assert(kTensorIterMaxRank > 0 && kTensorIterMaxRank <= 64,
              "TI: kTensorIterMaxRank must remain in [1, 64] for ABI stability");

// Forward declaration for the concrete iterator returned by the builder.
class TensorIter;

// Stats helper for 32-bit indexing splits (no-op when stats disabled).
void ti_stats_inc_num_32bit_splits() noexcept;

// Internal helper for marking CPU NO&D fast path eligibility.
void maybe_set_cpu_nod_contig_fastpath(TensorIter& iter);

#ifdef VBT_TI_ENABLE_TEST_HOOKS
namespace testing {
struct TensorIterTestHelper;
} // namespace testing
#endif

// Canonical 1D loop callback types used by CPU drivers.
using loop1d_t = void (*)(char** data,
                          const std::int64_t* strides,
                          std::int64_t size,
                          void* ctx);

using reduce_loop1d_t = void (*)(char** data,
                                 const std::int64_t* strides,
                                 std::int64_t size,
                                 void* ctx);

// Logical role of an iterator operand.
enum class IterOperandRole : std::uint8_t {
  ReadOnly,      // input only
  WriteOnly,     // pure output
  ReadWrite,     // in-place (read + write)
  ReduceOutput,  // reduction output
};

// Per-operand metadata used by Tensor Iterator.
struct IterOperand {
  // Non-owning pointer to the logical tensor for this operand.
  // numel() > 0.
  TensorImpl* tensor = nullptr;

  IterOperandRole role = IterOperandRole::ReadOnly;

  // Strides in bytes in iteration space, one per iteration dimension.
  std::vector<std::int64_t> dim_stride_bytes{};

  // Base data pointer for this operand in iteration space. May be null even
  // when tensor != nullptr for zero-numel iterators.
  void* data = nullptr;

  // Flags about how this operand participates in the iterator.
  bool is_output = false;      // index < num_outputs_
  bool is_read_write = false;  // role == ReadWrite
  bool will_resize = false;    // outputs TI may resize

  ScalarType dtype = ScalarType::Float32;
  Device     device = Device::cpu();
};

// Autograd-aware alias metadata for a particular operation. These types are
struct IterAliasInfo {
  int  output_index;  // [0, noutputs)
  int  input_index;   // [0, ninputs)
  bool is_inplace;    // true if aliasing with in-place update is allowed
  bool is_view;
};

struct IterOpSignature {
  const char*          op_fqname;   // e.g., "vt::add_"
  const IterAliasInfo* aliases;     // pointer to static array
  std::size_t          alias_count; // number of entries
};

// Optional reference wrapper for TensorImpl used to model outputs that may be
// logically undefined prior to iterator build(), but are always backed by a
// live TensorImpl object.
class OptionalTensorImplRef {
 public:
  OptionalTensorImplRef() = delete;  // no uninitialized sentinel in public API

  // p must be non-null; defined indicates whether the logical tensor is
  // currently defined (e.g., in-place vs. out-parameter to be filled in
  // build()).
  OptionalTensorImplRef(TensorImpl* p, bool defined)
      : ptr_(p), defined_(defined) {
    if (!p) {
      throw std::invalid_argument(
          "TI: OptionalTensorImplRef requires non-null ptr");
    }
  }

  [[nodiscard]] bool defined() const noexcept { return defined_; }

  TensorImpl& get() const {
#ifdef VBT_TI_DEBUG
    assert(ptr_ != nullptr);
    assert(defined_ && "OptionalTensorImplRef::get() on undefined tensor");
#endif
    if (!defined_) {
      throw std::invalid_argument(
          "TI: OptionalTensorImplRef::get() called on undefined tensor");
    }
    return *ptr_;
  }

  // Internal helper for TI implementation; callers must still honor lifetime
  // rules. This has no iteration semantics on its own.
  TensorImpl* unsafe_raw_ptr() const noexcept { return ptr_; }

 private:
  friend class TensorIterConfig;
  friend class TensorIterBase;

  TensorImpl* ptr_;   // non-null, points to live TensorImpl owned elsewhere
  bool        defined_;
};

// CUDA-facing metadata used to communicate iteration-space sizes and strides
// to device kernels. This struct is POD and is populated by
struct DeviceStrideMeta {
  std::int64_t sizes[kTensorIterMaxRank]{};
  std::int64_t strides[kTensorIterMaxRank]{};
  std::int64_t ndim{0};  // actual rank used
};

static_assert(std::is_trivially_copyable_v<DeviceStrideMeta>,
              "DeviceStrideMeta must be trivially copyable for ABI stability");
static_assert(std::is_standard_layout_v<DeviceStrideMeta>,
              "DeviceStrideMeta must have standard layout");
static_assert(offsetof(DeviceStrideMeta, sizes) == 0,
              "DeviceStrideMeta::sizes must be first");
static_assert(offsetof(DeviceStrideMeta, strides) ==
              sizeof(std::int64_t) * kTensorIterMaxRank,
              "DeviceStrideMeta::strides offset mismatch");
static_assert(offsetof(DeviceStrideMeta, ndim) ==
              2 * sizeof(std::int64_t) * kTensorIterMaxRank,
              "DeviceStrideMeta::ndim offset mismatch");
static_assert(sizeof(DeviceStrideMeta) ==
              (2 * kTensorIterMaxRank + 1) * sizeof(std::int64_t),
              "DeviceStrideMeta size must match packed layout");

// Forward declarations of core iterator classes.
class TensorIterConfig;

class TensorIterBase {
 public:
  TensorIterBase(const TensorIterBase&) = default;
  TensorIterBase& operator=(const TensorIterBase&) = default;
  TensorIterBase(TensorIterBase&&) noexcept = default;
  TensorIterBase& operator=(TensorIterBase&&) noexcept = default;
  virtual ~TensorIterBase() = default;

  int           ndim() const;
  std::int64_t  numel() const;
  int           ntensors() const;
  int           noutputs() const;
  int           ninputs() const;

  const std::vector<std::int64_t>& shape() const;
  const std::vector<std::int64_t>& reduce_dims() const;

  const IterOperand& operand(int idx) const;
  IterOperand&       operand(int idx);

  bool is_reduction() const;
  int  num_reduce_dims() const;
  bool is_trivial_1d() const;
  bool can_use_32bit_indexing() const;
  ScalarType computation_dtype() const;

  const IterOpSignature* op_signature() const noexcept;
  bool                   mem_overlap_checked() const noexcept;
  bool                   has_any_output_input_alias() const noexcept;
  MemOverlapStatus       alias_status(int out_index, int in_index) const;

  using loop1d_t = ::vbt::core::loop1d_t;

  // Single-threaded CPU iteration driver.
  void for_each_cpu(loop1d_t loop, void* ctx) const;

  void export_device_meta(int operand_index,
                          DeviceStrideMeta* out_meta,
                          std::int64_t max_ndim = kTensorIterMaxRank) const;

 protected:
  TensorIterBase() = default;  // only derived classes can construct

  std::vector<std::int64_t>  shape_{};
  std::vector<std::int64_t>  reduce_dims_{};
  std::vector<IterOperand>   operands_{};

  int        num_outputs_{0};
  ScalarType common_dtype_{ScalarType::Float32};
  Device     common_device_{Device::cpu()};
  bool       is_reduction_{false};
  bool       cpu_nod_contig_fastpath_{false};

  const IterOpSignature*        op_signature_{nullptr};
  bool                          mem_overlap_checked_{false};
  bool                          has_any_output_input_alias_{false};
  std::vector<MemOverlapStatus> alias_status_table_{};

  friend class TensorIterConfig;
  friend class TensorIter;
#ifdef VBT_TI_ENABLE_TEST_HOOKS
  friend struct ::vbt::core::testing::TensorIterTestHelper;
#endif
  friend void maybe_set_cpu_nod_contig_fastpath(TensorIter&);
};

// with a full API surface; build() constructs a TensorIter for CPU equal-shape
// elementwise iteration.
class TensorIterConfig final {
 public:
  TensorIterConfig() = default;

  TensorIterConfig(const TensorIterConfig&) = delete;
  TensorIterConfig& operator=(const TensorIterConfig&) = delete;

  TensorIterConfig(TensorIterConfig&&) noexcept = default;
  TensorIterConfig& operator=(TensorIterConfig&&) noexcept = default;

  ~TensorIterConfig() = default;

  // Output/input registration.
  TensorIterConfig& add_output(OptionalTensorImplRef out,
                               IterOperandRole role = IterOperandRole::WriteOnly,
                               bool allow_resize = true);

  TensorIterConfig& add_input(const TensorImpl& in);
  TensorIterConfig& add_const_input(const TensorImpl& in);

  // Behavioral flags (aligned with TI README and PyTorch).
  TensorIterConfig& check_mem_overlap(bool v = true);
  TensorIterConfig& check_all_same_dtype(bool v = true);
  TensorIterConfig& check_all_same_device(bool v = true);

  // Enable Fabric-only 2-GPU elementwise mode.
  // This relaxes device invariants in build() to allow a single remote
  // read-only CUDA device, while still exposing a single common_device_.
  TensorIterConfig& enable_fabric_2gpu_elementwise(int primary_device);

  TensorIterConfig& promote_inputs_to_common_dtype(bool v);
  TensorIterConfig& promote_integer_inputs_to_float(bool v);
  TensorIterConfig& cast_common_dtype_to_outputs(bool v);
  TensorIterConfig& resize_outputs(bool v);
  TensorIterConfig& allow_cpu_scalars(bool v);
  TensorIterConfig& is_reduction(bool v);
  TensorIterConfig& enforce_linear_iteration(bool v = true);
  TensorIterConfig& set_reduce_dims(std::span<const std::int64_t> dims,
                                    bool keepdim = false);

  // Static declarations for dtype, device, and shape.
  TensorIterConfig& declare_static_dtype_and_device(ScalarType, Device);
  TensorIterConfig& declare_static_dtype(ScalarType);
  TensorIterConfig& declare_static_device(Device);
  TensorIterConfig& declare_static_shape(std::span<const std::int64_t> shape);

  TensorIterConfig& set_max_rank(std::int64_t max_rank = kTensorIterMaxRank);
  TensorIterConfig& set_op_signature(const IterOpSignature* sig);

  // Finalizer – constructs a TensorIter from the collected configuration.
  TensorIter build();

 private:
  std::vector<OptionalTensorImplRef> outputs_{};
  std::vector<const TensorImpl*>     inputs_{};

  std::vector<IterOperandRole> output_roles_{};        // parallel to outputs_
  std::vector<bool>            output_allow_resize_{}; // parallel to outputs_
  std::vector<bool>            input_is_const_{};      // parallel to inputs_

  const IterOpSignature* op_signature_{nullptr};  // may be null

  std::int64_t max_rank_{kTensorIterMaxRank};

  bool check_mem_overlap_{true};
  bool check_all_same_dtype_{true};
  bool check_all_same_device_{true};
  bool enforce_linear_iteration_{false};
  bool promote_inputs_to_common_dtype_{false};
  bool promote_integer_inputs_to_float_{false};
  bool cast_common_dtype_to_outputs_{false};
  bool resize_outputs_{true};
  bool allow_cpu_scalars_{false};
  bool is_reduction_{false};

  bool allow_multi_device_fabric_{false};
  int  fabric_primary_device_{-1};

  // Static declaration storage
  ScalarType static_dtype_{ScalarType::Undefined};
  Device     static_device_{kDLCPU, -1};
  std::vector<std::int64_t> static_shape_{};
  bool       has_static_shape_{false};

  std::vector<std::int64_t> reduce_dims_spec_{};
  bool                       reduce_keepdim_{false};
  bool                       has_reduce_dims_{false};
};

// Fabric: canonical helper for building an elementwise 2-GPU iterator.
// This is the only supported way to build a Fabric TI currently.
TensorIter make_fabric_elementwise_2gpu_iter(
    TensorImpl& out,
    const TensorImpl& a,
    const TensorImpl& b,
    Device primary_dev);

// Final, user-facing iterator type returned by the builder.
class TensorIter final : public TensorIterBase {
 public:
  TensorIter() = default;

  TensorIter(const TensorIter&) = delete;
  TensorIter& operator=(const TensorIter&) = delete;

  TensorIter(TensorIter&&) noexcept = default;
  TensorIter& operator=(TensorIter&&) noexcept = default;

  ~TensorIter() override = default;

  // Static factories (unary_op/binary_op/nullary_op for
  // equal-shape CPU elementwise iteration).
  static TensorIter unary_op(TensorImpl& out,
                             const TensorImpl& a);

  static TensorIter unary_inplace(TensorImpl& a) = delete;  // planned later

  static TensorIter binary_op(TensorImpl& out,
                              const TensorImpl& a,
                              const TensorImpl& b);

  static TensorIter nullary_op(TensorImpl& out);

  static TensorIter reduce_op(TensorImpl& out,
                              const TensorImpl& a,
                              std::span<const std::int64_t> dims);

  static TensorIter reduce_op(TensorImpl& out_values,
                              TensorImpl& out_indices,
                              const TensorImpl& a,
                              std::span<const std::int64_t> dims);

  // Convenience wrapper that runs `fn` on each 32-bit-safe sub-iterator.
  // See design/tensor_iter/README_v2.md §4.1.4 for semantics.
  template <class F>
  void with_32bit_indexing(F&& fn) const {
    if (is_reduction()) {
      throw std::logic_error(
          "TI: with_32bit_indexing is only supported for elementwise iterators");
    }

    using Fn = std::decay_t<F>;
    Fn fn_copy(std::forward<F>(fn));

    // Work stack of iterators to process. Start with a copy of *this via the
    // base-class assignment operator to avoid enabling TensorIter copies.
    std::vector<TensorIter> stack;
    stack.emplace_back();
    static_cast<TensorIterBase&>(stack.back()) =
        static_cast<const TensorIterBase&>(*this);

    while (!stack.empty()) {
      TensorIter iter = std::move(stack.back());
      stack.pop_back();

      const std::int64_t n = iter.numel();
      if (n == 0) {
        // Zero-numel iterators are trivially safe; callers typically skip work
        // based on numel()==0, but we still allow fn to observe the iterator.
        if (iter.can_use_32bit_indexing()) {
          fn_copy(iter);
        }
        continue;
      }

      if (iter.can_use_32bit_indexing()) {
        fn_copy(iter);
        continue;
      }

      const int R = iter.ndim();
      if (R <= 0) {
        continue;
      }

      const auto& S = iter.shape();
      const int nt  = iter.ntensors();

      int d_split = -1;
      std::int64_t best_contrib = 0;

      // Choose a split dimension based on the largest contribution to the
      // maximum absolute byte offset, similar to PyTorch's get_dim_to_split.
      for (int d = 0; d < R; ++d) {
        const std::int64_t size_d = S[static_cast<std::size_t>(d)];
        if (size_d <= 1) {
          continue;  // splitting size-0/1 dims is pointless
        }

        std::int64_t max_abs_stride_bytes = 0;
        for (int k = 0; k < nt; ++k) {
          const IterOperand& op = iter.operand(k);
          if (op.dim_stride_bytes.size() <=
              static_cast<std::size_t>(d)) {
            continue;
          }
          const std::int64_t stride_b =
              op.dim_stride_bytes[static_cast<std::size_t>(d)];
          std::int64_t abs_stride_b = 0;
          if (!checked_abs_i64_hdr(stride_b, abs_stride_b)) {
            abs_stride_b = std::numeric_limits<std::int64_t>::max();
          }
          if (abs_stride_b > max_abs_stride_bytes) {
            max_abs_stride_bytes = abs_stride_b;
          }
        }

        if (max_abs_stride_bytes == 0) {
          continue;
        }

        std::int64_t contrib = 0;
        if (!checked_mul_i64(size_d, max_abs_stride_bytes, contrib)) {
          contrib = std::numeric_limits<std::int64_t>::max();
        }
        if (contrib > best_contrib) {
          best_contrib = contrib;
          d_split = d;
        }
      }

      if (d_split < 0) {
        // No meaningful split dimension found; treat as "requires 64-bit"
        // and do not invoke fn on this iterator.
        continue;
      }

      const std::int64_t size_split =
          S[static_cast<std::size_t>(d_split)];
      if (size_split <= 1) {
        continue;
      }

      std::int64_t k_split = size_split / 2;
      if (k_split <= 0) {
        k_split = 1;
      } else if (k_split >= size_split) {
        k_split = size_split - 1;
      }

      TensorIter left;
      TensorIter right;
      static_cast<TensorIterBase&>(left) =
          static_cast<const TensorIterBase&>(iter);
      static_cast<TensorIterBase&>(right) =
          static_cast<const TensorIterBase&>(iter);

      left.shape_[static_cast<std::size_t>(d_split)] = k_split;
      right.shape_[static_cast<std::size_t>(d_split)] =
          size_split - k_split;

      const int nt_local = iter.ntensors();
      bool split_ok = true;
      for (int k = 0; k < nt_local; ++k) {
        IterOperand& left_op =
            left.operands_[static_cast<std::size_t>(k)];
        IterOperand& right_op =
            right.operands_[static_cast<std::size_t>(k)];
        const IterOperand& src_op =
            iter.operands_[static_cast<std::size_t>(k)];

        left_op = src_op;
        right_op = src_op;

        if (!src_op.data) {
          continue;
        }

        const std::int64_t stride_bytes =
            src_op.dim_stride_bytes[static_cast<std::size_t>(d_split)];

        std::int64_t off_bytes = 0;
        if (!checked_mul_i64(k_split, stride_bytes, off_bytes)) {
          split_ok = false;
          break;
        }

        char* base = static_cast<char*>(src_op.data);
        right_op.data = static_cast<void*>(base + off_bytes);
      }

      if (!split_ok) {
        continue;
      }

      ti_stats_inc_num_32bit_splits();
      stack.push_back(std::move(right));
      stack.push_back(std::move(left));
    }
  }

#ifdef VBT_TI_ENABLE_TEST_HOOKS
  friend struct ::vbt::core::testing::TensorIterTestHelper;
#endif
};

#ifdef VBT_TI_ENABLE_TEST_HOOKS
namespace testing {

struct TensorIterTestHelper {
  static TensorIter make_iterator_for_shape(
      std::span<const std::int64_t> shape);

  static TensorIter make_iterator_for_shape_with_dummy_operand(
      std::span<const std::int64_t> shape);

  static ScalarType common_dtype(const TensorIterBase& iter);
  static Device     common_device(const TensorIterBase& iter);
  static bool       cpu_nod_contig_fastpath(const TensorIterBase& iter);
};

} // namespace testing
#endif  // VBT_TI_ENABLE_TEST_HOOKS

#ifdef VBT_TI_STATS

struct TensorIterStats {
  std::atomic<std::uint64_t> cpu_invocations{0};
  std::atomic<std::uint64_t> cpu_reduction_invocations{0};
  std::atomic<std::uint64_t> cuda_meta_exports{0};
  std::atomic<std::uint64_t> cuda_ti_kernel_launches{0};
  std::atomic<std::uint64_t> num_32bit_splits{0};
};

// Read-only snapshot accessor for tests and tooling.
const TensorIterStats& get_tensor_iter_stats() noexcept;

// Reset all counters to zero; safe to call concurrently with readers.
void reset_tensor_iter_stats() noexcept;

namespace detail {
// Internal mutable accessor for instrumentation macros.
TensorIterStats& mutable_tensor_iter_stats() noexcept;
} // namespace detail

// Low-level instrumentation macros used within vbt::core and close friends.
#define VBT_TI_STATS_INC(field)                                                   \
  (::vbt::core::detail::mutable_tensor_iter_stats().field                        \
       .fetch_add(1, std::memory_order_relaxed))

#define VBT_TI_STATS_ADD(field, delta)                                            \
  (::vbt::core::detail::mutable_tensor_iter_stats().field                        \
       .fetch_add((delta), std::memory_order_relaxed))

#else  // !VBT_TI_STATS

// When stats are disabled, TensorIterStats is incomplete and the accessors are
// intentionally absent; instrumentation macros compile to no-ops.
struct TensorIterStats;  // Incomplete when stats are disabled.

inline void reset_tensor_iter_stats() noexcept {}

#define VBT_TI_STATS_INC(field)        ((void)0)
#define VBT_TI_STATS_ADD(field, delta) ((void)0)

#endif // VBT_TI_STATS

inline void ti_stats_inc_num_32bit_splits() noexcept {
  VBT_TI_STATS_INC(num_32bit_splits);
}

} // namespace core
} // namespace vbt
