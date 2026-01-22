// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "vbt/core/tensor.h"

namespace vbt {
namespace core {
namespace indexing {

namespace detail {
enum class CudaBoundsMode : std::uint8_t;
struct AdvancedIndexEnvConfig;
}  // namespace detail

// Classification of one normalized index element.
enum class IndexKind : std::uint8_t {
  None,      // Python None / newaxis
  Ellipsis,  // Python Ellipsis / ... (pre-ellipsis expansion only)
  Integer,   // Scalar integer index
  Boolean,   // Scalar bool index
  Slice,     // Python slice(start, stop, step)
  Tensor     // Tensor / array / sequence index (advanced)
};

// Python-style slice(start, stop, step) with optional bounds.
struct Slice {
  std::optional<std::int64_t> start;  // inclusive; nullopt → depends on step & dim size
  std::optional<std::int64_t> stop;   // exclusive; nullopt → depends on step & dim size
  std::optional<std::int64_t> step;   // nullopt → +1; must never be 0 after validation
};

// Single index element (None, Ellipsis, Integer, Boolean, Slice, or Tensor).
struct TensorIndex {
  struct EllipsisTag {};

  IndexKind kind{IndexKind::None};
  std::int64_t integer{0};             // when kind == Integer
  bool boolean{false};                 // when kind == Boolean
  Slice slice{};                       // when kind == Slice
  vbt::core::TensorImpl tensor{};      // when kind == Tensor (int64/bool tensors)

  TensorIndex() = default;
  explicit TensorIndex(std::nullptr_t) : kind(IndexKind::None) {}
  explicit TensorIndex(EllipsisTag)    : kind(IndexKind::Ellipsis) {}
  explicit TensorIndex(std::int64_t idx) : kind(IndexKind::Integer), integer(idx) {}
  explicit TensorIndex(bool b)         : kind(IndexKind::Boolean), boolean(b) {}
  explicit TensorIndex(Slice s)        : kind(IndexKind::Slice), slice(std::move(s)) {}
  explicit TensorIndex(const vbt::core::TensorImpl& t)
      : kind(IndexKind::Tensor), tensor(t) {}
};

// Full multi-d index specification after Python normalization.
struct IndexSpec {
  // Order matches the normalized Python index tuple after ellipsis expansion.
  std::vector<TensorIndex> items;
};

// Helper predicates
inline bool is_advanced_kind(IndexKind k) {
  return k == IndexKind::Boolean || k == IndexKind::Tensor;
}

inline bool has_any_advanced(const IndexSpec& spec) {
  for (const auto& it : spec.items) {
    if (is_advanced_kind(it.kind)) return true;
  }
  return false;
}

// Count how many tensor dimensions an IndexSpec will consume.
// - Integer, Slice, integer Tensor → +1
// - Bool Tensor (mask)             → +mask.dim()
// - Scalar Boolean, None, Ellipsis → +0
std::int64_t count_specified_dims(const IndexSpec& spec, std::int64_t self_dim);

// Expand (at most one) Ellipsis into full-range slices so that the
// total consumed dimensions match self_dim. Throws std::invalid_argument
// if multiple ellipses appear or if too many indices are provided.
IndexSpec expand_ellipsis_and_validate(const IndexSpec& raw, std::int64_t self_dim);

// Normalized slice parameters for a single dimension.
struct NormalizedSlice {
  std::int64_t start;   // inclusive; may be 0..D-1 for positive step or -1..D-1 for negative step
  std::int64_t step;    // non-zero; may be negative
  std::int64_t length;  // number of elements, >= 0
};

// Normalize a Slice against a given dimension size using Python semantics.
NormalizedSlice normalize_slice(const Slice& s, std::int64_t dim_size);

// Core basic indexing helpers (device-agnostic views + CPU writes).

// Compute a basic-index view of self according to spec.
vbt::core::TensorImpl basic_index(const vbt::core::TensorImpl& self,
                                  const IndexSpec& spec);

void basic_index_put(vbt::core::TensorImpl& self,
                     const IndexSpec& spec,
                     const vbt::core::TensorImpl& value);

// View-only broadcast: return a non-owning broadcasted view of src
// with the exact target_sizes shape. Throws std::invalid_argument
// with a message containing "shape mismatch" if broadcasting is
// not possible.
vbt::core::TensorImpl broadcast_to(const vbt::core::TensorImpl& src,
                                   std::span<const std::int64_t> target_sizes);

struct AdvancedIndex {
  // Restrided source view; sizes equal result_shape.
  vbt::core::TensorImpl src;
  // Broadcasted Int64 index tensors (CPU), one per indexed dimension.
  std::vector<vbt::core::TensorImpl> indices;

  // Per-indexed original dim size and stride (in elements; may be negative).
  std::vector<std::int64_t> indexed_sizes;
  std::vector<std::int64_t> indexed_strides_elems;

  // Number of non-indexed dims before and after the advanced block
  // in the permuted base tensor.
  std::int64_t dims_before{0};
  std::int64_t dims_after{0};

  // Broadcasted shape of the advanced indices and full logical
  // result shape (including prefix/suffix dims).
  std::vector<std::int64_t> index_shape;
  std::vector<std::int64_t> result_shape;

  // Device-agnostic iteration-domain hint for 32-bit loop
  // selection on both CPU and CUDA advanced indexing kernels.
  // - Derived solely from result_shape via safe_numel_from_sizes{,_cuda}.
  // - Let R = safe_numel_from_sizes{,_cuda}(result_shape):
  //     * R == 0 covers both true empty domains and overflow sentinels.
  //     * use32bit_indexing is true when R == 0, or (0 < R <= INT32_MAX).
  // - Independent of layout/contiguity; NOT a safety check for pointer math.
  // - Under current DoS caps, any valid non-empty advanced result that passes
  //   caps satisfies 0 < R <= kAdvIndexMaxResultNumel <= INT32_MAX, so the
  //   hint is true for all routed non-empty results.
  bool use32bit_indexing{false};
};

// Build normalized advanced-index metadata for CPU tensors. The
// input spec must contain at least one advanced index (Boolean or
// Tensor). We support integer tensor indices, boolean tensor
// masks, and scalar booleans (under limited patterns); unsupported
// advanced forms throw.
AdvancedIndex make_advanced_index(const vbt::core::TensorImpl& self,
                                  const IndexSpec& spec);

// Unboxed, device-agnostic entry points (basic + advanced).
// CPU path is implemented in indexing_advanced.cc; CUDA path is implemented
// in indexing_advanced_cuda.cu.
vbt::core::TensorImpl index(const vbt::core::TensorImpl& self,
                            const IndexSpec& spec);

// CUDA-only helper for advanced read indexing. This is called by
// index() when self.device().type == kDLCUDA. It throws std::invalid_argument
// for unsupported patterns or devices.
vbt::core::TensorImpl index_cuda(const vbt::core::TensorImpl& self,
                                 const IndexSpec& spec);

void index_put_(vbt::core::TensorImpl& self,
                const IndexSpec& spec,
                const vbt::core::TensorImpl& value,
                bool accumulate);

vbt::core::TensorImpl advanced_index_cpu(const AdvancedIndex& info);

void advanced_index_put_cpu(AdvancedIndex& info,
                            const vbt::core::TensorImpl& value,
                            bool accumulate);

// CUDA advanced scatter kernel. The metadata contract
// matches AdvancedIndex as produced by cuda_impl::make_advanced_index_cuda
// for CUDA tensors.
void advanced_index_put_cuda(AdvancedIndex& info,
                             const vbt::core::TensorImpl& value,
                             bool accumulate);

namespace cuda_impl {

// CUDA-mode selector for advanced indexing metadata builders.
enum class AdvancedIndexCudaMode : std::uint8_t { Read, Write };

// CUDA AdvancedIndex metadata for reads/writes. We require
// that info.src is a CUDA tensor and indices.size() == 1 with an
// Int32/Int64 CUDA tensor on the same device; reads further restrict
// the index dtype to Int64.
struct AdvancedIndexCudaResult {
  AdvancedIndex info;
  bool          can_use_1d_fastpath{false};
};

AdvancedIndexCudaResult make_advanced_index_cuda(
    const vbt::core::TensorImpl& self,
    const IndexSpec&              spec_raw,
    AdvancedIndexCudaMode         mode);

// Backwards-compatible overload used by existing write-only call sites
inline AdvancedIndexCudaResult make_advanced_index_cuda(
    const vbt::core::TensorImpl& self,
    const IndexSpec&              spec_raw) {
  return make_advanced_index_cuda(self, spec_raw, AdvancedIndexCudaMode::Write);
}

// Canonical CUDA advanced read helper. Chooses between the
// generic TensorIterator-backed kernel and an optional 1D gather fast
// path based on AdvancedIndex metadata and can_use_1d_fastpath.
vbt::core::TensorImpl advanced_index_cuda_impl(AdvancedIndex& info,
                                               bool           coarse_can_use_1d_fastpath);

#if VBT_WITH_CUDA && VBT_INTERNAL_TESTS
// Test-only helpers for introspecting CUDA bounds/launch behavior.
detail::CudaBoundsMode get_effective_cuda_bounds_mode_for_tests() noexcept;

// Return grid_dim.x for the canonical 1D launch used by CUDA advanced
// indexing kernels. This is a thin wrapper around the internal
// make_1d_grid helper and is only intended for white-box tests.
unsigned int get_1d_grid_x_for_tests(std::int64_t N,
                                     int          threads,
                                     int          dev_index);

// Override the cached per-device max grid size used by make_1d_grid.
// Passing std::nullopt clears any override and forces the next call to
// consult cudaGetDeviceProperties again.
void set_device_max_grid_x_override_for_tests(
    int                         dev_index,
    std::optional<unsigned int> max_grid_x);
#endif  // VBT_WITH_CUDA && VBT_INTERNAL_TESTS

} // namespace cuda_impl

// Feature flag for advanced indexing. Implemented in core; tests
// may toggle this via a dedicated helper to exercise the disabled
// path.
bool advanced_indexing_enabled();

// Test-only helper to override the feature flag within a process.
// Production code should not call this outside of tests.
void set_advanced_indexing_enabled_for_tests(bool enabled);

// Global 32-bit optimization flag for advanced indexing loops.
// - Controls whether CPU and CUDA advanced kernels may use 32-bit loop
//   counters and whether the CUDA 1D fast path may run; it does not
//   enable/disable advanced indexing as a feature.
// - Implemented as a process-global configuration in
//   src/vbt/core/indexing_advanced.cc. The default value is derived once
//   from the VBT_MINDEX_32BIT_DISABLE environment variable; tests may
//   override it via the setter below.
// - The setter is intended for tests/benchmarks only; production code
//   should treat this flag as effectively read-only.
bool advanced_index_32bit_enabled();
void set_advanced_index_32bit_enabled_for_tests(bool enabled);

namespace detail {

// Test-only helper that mirrors the internal env mapping used by
// advanced_index_32bit_enabled(). Given a raw getenv-style value
// (or nullptr for the unset case), returns the corresponding default
// enablement for the 32-bit optimization flag.
bool compute_mindex32_default_from_env_value_for_tests(const char* raw) noexcept;

// Test-only helper for the advanced indexing feature flag env mapping.
// Given a raw getenv-style value (or nullptr for the unset case),
// returns the default enablement for advanced indexing as derived
// from VBT_ENABLE_ADVANCED_INDEXING.
bool compute_adv_flag_from_env_raw(const char* raw) noexcept;

// Centralized, cached view of all environment-derived configuration
// computed once per process from the real process environment and
// then treated as immutable; tests may override it via dedicated
// helpers when VBT_INTERNAL_TESTS is enabled.

// CUDA bounds-checking mode selector used internally by CUDA advanced
// indexing helpers. LegacyHost corresponds to the existing D2H/CPU
// normalize/H2D path; DeviceNormalized corresponds to the optional
enum class CudaBoundsMode : std::uint8_t {
  LegacyHost = 0,
  DeviceNormalized = 1,
};

struct AdvancedIndexEnvConfig {
  // Feature flags (defaults; test overrides still exist at the
  // advanced_indexing_enabled / advanced_index_32bit_enabled level).
  bool advanced_indexing_default;  // from VBT_ENABLE_ADVANCED_INDEXING
  bool mindex32_default;           // from VBT_MINDEX_32BIT_DISABLE

  // Debug / logging knobs.
  bool debug_adv_index;            // from VBT_DEBUG_ADV_INDEX (present => true)

  CudaBoundsMode cuda_bounds_default;
  bool           cuda_gpu_bounds_disable; // from VBT_MINDEX_CUDA_GPU_BOUNDS_DISABLE
  std::int64_t   cuda_max_blocks_cap;     // from VBT_MINDEX_CUDA_MAX_BLOCKS (>0 => cap, <=0 => none)

  // Internal, test-only CUDA feature gates. These are derived from
  // VBT_INTERNAL_ADV_INDEX_* envs when VBT_INTERNAL_TESTS is enabled and
  // are always false in production builds.
  bool cuda_allow_bool_mask_indices;  // from VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK
  bool cuda_bool_mask_use_cub;       // from VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK_CUB
  bool cuda_allow_extended_dtypes;   // from VBT_INTERNAL_ADV_INDEX_CUDA_EXTENDED_DTYPE
  bool cuda_cub_index_put_accumulate;  // from VBT_INTERNAL_CUDA_CUB_INDEX_PUT_ACCUMULATE
};

// Return the process-wide, cached env configuration. This function is
// thread-safe and cheap to call from hot paths; environment variables
// are read at most once per process.
const AdvancedIndexEnvConfig& get_advanced_index_env_config() noexcept;

#if VBT_INTERNAL_TESTS
// White-box helpers for tests to inspect and override the env config.
// These are **not** thread-safe with concurrent advanced-index
// operations and must only be used from single-threaded test
// setup/teardown.
AdvancedIndexEnvConfig get_advanced_index_env_config_for_tests() noexcept;
void reset_advanced_index_env_config_for_tests(const AdvancedIndexEnvConfig& cfg) noexcept;
void clear_advanced_index_env_config_override_for_tests() noexcept;

// Return whether a test-only env-config override is currently active.
bool env_config_override_is_active_for_tests() noexcept;

struct EnvProbeCounters {
  std::uint64_t num_getenv_calls_enable_adv{0};
  std::uint64_t num_getenv_calls_mindex32_disable{0};
  std::uint64_t num_getenv_calls_debug_adv_index{0};
  std::uint64_t num_getenv_calls_cuda_gpu_bounds_disable{0};
  std::uint64_t num_getenv_calls_cuda_max_blocks{0};

  // Internal counters for test-only CUDA feature envs.
  std::uint64_t num_getenv_calls_cuda_bool_mask_indices{0};
  std::uint64_t num_getenv_calls_cuda_bool_mask_cub{0};
  std::uint64_t num_getenv_calls_cuda_extended_dtypes{0};
  std::uint64_t num_getenv_calls_cuda_cub_index_put_accumulate{0};
};

EnvProbeCounters get_env_probe_counters_for_tests() noexcept;
void reset_env_probe_counters_for_tests() noexcept;

// RAII helper that temporarily overrides the env config for the
// lifetime of the guard and restores the previous configuration
// (and override state) on destruction. Not thread-safe with
// concurrent advanced-index operations; intended for single-threaded
// test setup/teardown only.
struct AdvancedIndexEnvConfigGuard {
  AdvancedIndexEnvConfig prev;
  bool had_override;

  explicit AdvancedIndexEnvConfigGuard(const AdvancedIndexEnvConfig& cfg)
      : prev(get_advanced_index_env_config_for_tests()),
        had_override(env_config_override_is_active_for_tests()) {
    reset_advanced_index_env_config_for_tests(cfg);
  }

  ~AdvancedIndexEnvConfigGuard() {
    reset_advanced_index_env_config_for_tests(prev);
    if (!had_override) {
      clear_advanced_index_env_config_override_for_tests();
    }
  }
};
#endif  // VBT_INTERNAL_TESTS

} // namespace detail

} // namespace indexing
} // namespace core
} // namespace vbt
