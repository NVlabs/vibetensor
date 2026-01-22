// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/indexing.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <mutex>
#include <string>
#include <cctype>

#include "vbt/core/broadcast.h"
#include "vbt/core/checked_math.h"
#include "vbt/core/dtype.h"
#include "vbt/core/overlap.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/tensor_iterator/core.h"
#include "vbt/core/tensor_iterator/cpu.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/view_ops.h"
#include "vbt/core/write_guard.h"
#include "vbt/core/indexing_advanced_stats.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/logging/logging.h"

namespace vbt {
namespace core {
namespace indexing {

using vbt::core::ScalarType;
using vbt::core::TensorImpl;
using vbt::core::checked_add_i64;
using vbt::core::checked_mul_i64;
using vbt::core::OptionalTensorImplRef;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::IterOperandRole;

static bool compute_mindex32_default_from_env_raw(const char* raw) noexcept {
  if (!raw || *raw == '\0') {
    return true;
  }

  auto to_lower = [](unsigned char c) noexcept {
    return static_cast<char>(std::tolower(c));
  };

  auto equals_ci = [&](const char* s, const char* lit) noexcept {
    while (*s != '\0' && *lit != '\0') {
      if (to_lower(static_cast<unsigned char>(*s)) != *lit) {
        return false;
      }
      ++s;
      ++lit;
    }
    return (*s == '\0') && (*lit == '\0');
  };

  if (equals_ci(raw, "0") ||
      equals_ci(raw, "false") ||
      equals_ci(raw, "no") ||
      equals_ci(raw, "off")) {
    return true;
  }

  // Any other non-empty string disables optimizations.
  return false;
}

// - raw == nullptr, empty, or all-whitespace -> false.
// - Explicit false-like values (0/false/no/off) -> false.
// - Any other non-empty value -> true.
static bool compute_opt_in_flag_from_env_raw(const char* raw) noexcept {
  if (raw == nullptr) {
    return false;
  }

  const unsigned char* p = reinterpret_cast<const unsigned char*>(raw);
  while (*p && std::isspace(*p)) {
    ++p;
  }
  if (*p == '\0') {
    return false;  // all whitespace
  }

  // Delegate classification of false-like vs truthy to the existing
  // advanced-indexing helper so semantics stay consistent.
  return detail::compute_adv_flag_from_env_raw(raw);
}

namespace {

constexpr std::int64_t kAdvIndexMaxIndexNumel  = 10'000'000;   // 1e7
constexpr std::int64_t kAdvIndexMaxResultNumel = 100'000'000;  // 1e8
constexpr std::int64_t kAdvIndexMaxNdim        = 25;
constexpr std::int64_t kAdvIndexMaxMaskNumel   = 10'000'000;   // 1e7 (mask.numel cap)

static_assert(kAdvIndexMaxResultNumel <= std::numeric_limits<std::int32_t>::max(),
              "advanced indexing result cap must fit in int32_t for 32-bit loops");

inline std::int64_t safe_numel_from_sizes(const std::vector<std::int64_t>& sizes) {
  if (sizes.empty()) {
    return 1;  // scalar
  }
  std::int64_t n = 1;
  for (std::int64_t s : sizes) {
    if (s == 0) {
      return 0;
    }
    std::int64_t tmp = 0;
    if (!checked_mul_i64(n, s, tmp)) {
      // Treat overflow as zero to mirror TensorImpl::numel semantics.
      return 0;
    }
    n = tmp;
  }
  return n;
}

static TensorImpl make_empty_like_cpu(const TensorImpl& like,
                                      const std::vector<std::int64_t>& sizes) {
  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;

  auto dev = like.device();
  if (dev.type != kDLCPU) {
    throw std::invalid_argument("advanced_index_cpu: expected CPU tensor");
  }

  const ScalarType dtype = like.dtype();
  const std::size_t item_b = itemsize(dtype);
  const std::int64_t n = safe_numel_from_sizes(sizes);
  const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> strides(sizes.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype, dev);
}

// Shared helper for CPU advanced indexing loops. Iterates linearly over
// the logical result domain described by result_sizes, decodes linear
// indices into coordinates, and invokes the provided body functor with
// the 64-bit linear index and coordinate vector.
template <typename IndexT, typename BodyFn>
inline void for_each_advanced_index_linear(
    IndexT                              N,
    const std::vector<std::int64_t>&    result_sizes,
    BodyFn&&                            body) {
  const std::int64_t R = static_cast<std::int64_t>(result_sizes.size());
  std::vector<std::int64_t> coords(static_cast<std::size_t>(R));

#ifndef NDEBUG
  const std::int64_t debug_numel = safe_numel_from_sizes(result_sizes);
  VBT_ASSERT(debug_numel >= 0);
  if constexpr (std::is_same_v<IndexT, std::int32_t>) {
    if (debug_numel == 0) {
      VBT_ASSERT(N == 0);
    } else {
      VBT_ASSERT(debug_numel <= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()));
      VBT_ASSERT(static_cast<std::int64_t>(N) == debug_numel);
    }
  } else {
    VBT_ASSERT(static_cast<std::int64_t>(N) == debug_numel);
  }
#endif

  if (N == 0) {
    return;
  }

  auto decode_coords = [&](std::int64_t linear64) {
    std::int64_t tmp = linear64;
    for (std::int64_t d = R - 1; d >= 0; --d) {
      const std::int64_t sz = result_sizes[static_cast<std::size_t>(d)];
      std::int64_t c = 0;
      if (sz > 0) {
        c = tmp % sz;
        tmp /= sz;
      }
      coords[static_cast<std::size_t>(d)] = c;
    }
  };

  auto run_one = [&](IndexT i) {
    const std::int64_t linear64 = static_cast<std::int64_t>(i);
    decode_coords(linear64);
    body(linear64, coords);
  };

  if constexpr (std::is_same_v<IndexT, std::int32_t>) {
    for (std::int32_t i = 0; i < N; ++i) {
      run_one(i);
    }
  } else {
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
      run_one(static_cast<IndexT>(i));
    }
  }
}

std::atomic<bool> g_adv_override_active{false};
std::atomic<bool> g_advanced_indexing_enabled{true};  // holds env default or test override

std::atomic<bool> g_mindex32_override_active{false};
std::atomic<bool> g_mindex32_effective{true};  // holds env default or test override

std::once_flag g_env_config_once;
detail::AdvancedIndexEnvConfig g_env_config{};
std::atomic<bool> g_env_config_override_active{false};

#if VBT_INTERNAL_TESTS
// White-box counters for how many times we touched the real
// process environment. These are only used in tests.
detail::EnvProbeCounters g_env_probe_counters{};
#endif

static detail::AdvancedIndexEnvConfig compute_env_config_from_process_env() noexcept {
  detail::AdvancedIndexEnvConfig cfg{};

  // Advanced indexing feature flag default.
  const char* raw_adv = std::getenv("VBT_ENABLE_ADVANCED_INDEXING");
#if VBT_INTERNAL_TESTS
  g_env_probe_counters.num_getenv_calls_enable_adv++;
#endif
  cfg.advanced_indexing_default = detail::compute_adv_flag_from_env_raw(raw_adv);

  // 32-bit optimization flag default (double-negative env).
  const char* raw_mindex32 = std::getenv("VBT_MINDEX_32BIT_DISABLE");
#if VBT_INTERNAL_TESTS
  g_env_probe_counters.num_getenv_calls_mindex32_disable++;
#endif
  cfg.mindex32_default = compute_mindex32_default_from_env_raw(raw_mindex32);

  // Debug flag: presence of the env (non-null) enables debug logging.
  const char* raw_debug = std::getenv("VBT_DEBUG_ADV_INDEX");
#if VBT_INTERNAL_TESTS
  g_env_probe_counters.num_getenv_calls_debug_adv_index++;
#endif
  cfg.debug_adv_index = (raw_debug != nullptr);

  // CUDA bounds kill-switch: truthy values disable DeviceNormalized
  // mode even when it is compiled in.
  const char* raw_gpu_bounds = std::getenv("VBT_MINDEX_CUDA_GPU_BOUNDS_DISABLE");
#if VBT_INTERNAL_TESTS
  g_env_probe_counters.num_getenv_calls_cuda_gpu_bounds_disable++;
#endif
  cfg.cuda_gpu_bounds_disable = false;
  if (raw_gpu_bounds && *raw_gpu_bounds != '\0') {
    std::string s(raw_gpu_bounds);
    auto is_space = [](unsigned char c) noexcept {
      return std::isspace(c) != 0;
    };
    auto it_begin = std::find_if_not(s.begin(), s.end(), is_space);
    auto it_end = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
    if (it_begin < it_end) {
      s.assign(it_begin, it_end);
      std::transform(
          s.begin(), s.end(), s.begin(),
          [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      if (!(s == "0" || s == "false" || s == "no" || s == "off")) {
        cfg.cuda_gpu_bounds_disable = true;
      }
    }
  }

  // CUDA launch cap: positive integer => soft upper bound on blocks;
  // non-positive / malformed => no additional cap beyond device limit.
  const char* raw_max_blocks = std::getenv("VBT_MINDEX_CUDA_MAX_BLOCKS");
#if VBT_INTERNAL_TESTS
  g_env_probe_counters.num_getenv_calls_cuda_max_blocks++;
#endif
  cfg.cuda_max_blocks_cap = 0;
  if (raw_max_blocks && *raw_max_blocks != '\0') {
    char* end = nullptr;
    long long val = std::strtoll(raw_max_blocks, &end, 10);
    if (end != raw_max_blocks && *end == '\0' && val > 0) {
      cfg.cuda_max_blocks_cap = static_cast<std::int64_t>(val);
    }
  }

  cfg.cuda_bounds_default = detail::CudaBoundsMode::LegacyHost;

#if VBT_INTERNAL_TESTS
  // Internal-only CUDA feature flags (bool mask indices + extended dtypes).
  //
  // These env vars are test-only and should never affect production builds.
  const char* raw_bool = std::getenv("VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK");
  g_env_probe_counters.num_getenv_calls_cuda_bool_mask_indices++;
  cfg.cuda_allow_bool_mask_indices = compute_opt_in_flag_from_env_raw(raw_bool);

  const char* raw_bool_cub = std::getenv("VBT_INTERNAL_ADV_INDEX_CUDA_BOOL_MASK_CUB");
  g_env_probe_counters.num_getenv_calls_cuda_bool_mask_cub++;
  cfg.cuda_bool_mask_use_cub = compute_opt_in_flag_from_env_raw(raw_bool_cub);

  const char* raw_dtype = std::getenv("VBT_INTERNAL_ADV_INDEX_CUDA_EXTENDED_DTYPE");
  g_env_probe_counters.num_getenv_calls_cuda_extended_dtypes++;
  cfg.cuda_allow_extended_dtypes = compute_opt_in_flag_from_env_raw(raw_dtype);

  const char* raw_idx_put_acc =
      std::getenv("VBT_INTERNAL_CUDA_CUB_INDEX_PUT_ACCUMULATE");
  g_env_probe_counters.num_getenv_calls_cuda_cub_index_put_accumulate++;
  cfg.cuda_cub_index_put_accumulate =
      compute_opt_in_flag_from_env_raw(raw_idx_put_acc);
#else
  // In non-internal builds, these flags are hard-disabled regardless of env.
  cfg.cuda_allow_bool_mask_indices = false;
  cfg.cuda_bool_mask_use_cub = false;
  cfg.cuda_allow_extended_dtypes = false;
  cfg.cuda_cub_index_put_accumulate = false;
#endif

  return cfg;
}

} // anonymous namespace

namespace detail {

bool compute_mindex32_default_from_env_value_for_tests(const char* raw) noexcept {
  return compute_mindex32_default_from_env_raw(raw);
}

bool compute_adv_flag_from_env_raw(const char* raw) noexcept {
  if (raw == nullptr) {
    return true;  // unset -> enabled by default
  }

  std::string s(raw);
  auto is_space = [](unsigned char c) noexcept {
    return std::isspace(c) != 0;
  };

  // Trim leading whitespace
  auto it_begin = std::find_if_not(s.begin(), s.end(), is_space);
  if (it_begin == s.end()) {
    return true;  // all whitespace -> treat as empty -> enabled
  }
  // Trim trailing whitespace
  auto it_end = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
  s.assign(it_begin, it_end);

  if (s.empty()) {
    return true;  // empty after trimming -> enabled
  }

  std::transform(
      s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (s == "0" || s == "false" || s == "no" || s == "off") {
    return false;
  }

  // Any other non-empty value enables advanced indexing.
  return true;
}

const AdvancedIndexEnvConfig& get_advanced_index_env_config() noexcept {
  if (g_env_config_override_active.load(std::memory_order_acquire)) {
    return g_env_config;
  }
  std::call_once(g_env_config_once, [] {
    g_env_config = compute_env_config_from_process_env();
  });
  return g_env_config;
}

#if VBT_INTERNAL_TESTS
AdvancedIndexEnvConfig get_advanced_index_env_config_for_tests() noexcept {
  (void)get_advanced_index_env_config();
  return g_env_config;
}

void reset_advanced_index_env_config_for_tests(const AdvancedIndexEnvConfig& cfg) noexcept {
  g_env_config = cfg;
  g_env_config_override_active.store(true, std::memory_order_release);
}

void clear_advanced_index_env_config_override_for_tests() noexcept {
  g_env_config_override_active.store(false, std::memory_order_release);
}

EnvProbeCounters get_env_probe_counters_for_tests() noexcept {
  return g_env_probe_counters;
}

void reset_env_probe_counters_for_tests() noexcept {
  g_env_probe_counters = EnvProbeCounters{};
}

bool env_config_override_is_active_for_tests() noexcept {
  return g_env_config_override_active.load(std::memory_order_acquire);
}
#endif  // VBT_INTERNAL_TESTS

}  // namespace detail

bool advanced_indexing_enabled() {
  // Fast path: if a test override is active, env is ignored.
  if (g_adv_override_active.load(std::memory_order_relaxed)) {
    return g_advanced_indexing_enabled.load(std::memory_order_relaxed);
  }
  return detail::get_advanced_index_env_config().advanced_indexing_default;
}

void set_advanced_indexing_enabled_for_tests(bool enabled) {
  g_advanced_indexing_enabled.store(enabled, std::memory_order_relaxed);
  g_adv_override_active.store(true, std::memory_order_relaxed);
}

bool advanced_index_32bit_enabled() {
  // Fast path: if a test override is active, env is ignored.
  if (g_mindex32_override_active.load(std::memory_order_relaxed)) {
    return g_mindex32_effective.load(std::memory_order_relaxed);
  }
  return detail::get_advanced_index_env_config().mindex32_default;
}

void set_advanced_index_32bit_enabled_for_tests(bool enabled) {
  g_mindex32_effective.store(enabled, std::memory_order_relaxed);
  g_mindex32_override_active.store(true, std::memory_order_relaxed);
}

// Allocate a contiguous Int64 tensor on the same CPU device as `like`.
static TensorImpl make_int64_tensor_cpu(const TensorImpl& like,
                                        const std::vector<std::int64_t>& sizes) {
  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;

  auto dev = like.device();
  if (dev.type != kDLCPU) {
    throw std::invalid_argument("make_int64_tensor_cpu: expected CPU tensor");
  }

  const ScalarType dtype = ScalarType::Int64;
  const std::size_t item_b = itemsize(dtype);
  const std::int64_t n = safe_numel_from_sizes(sizes);
  const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> strides(sizes.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype, dev);
}

// Build per-dimension Int64 index tensors from a boolean mask (or a
// conceptual full-true / full-false mask when mask_opt == nullptr).
static void build_bool_indices_from_mask(
    const TensorImpl* mask_opt,
    bool scalar_value,
    const TensorImpl& base_like,
    const std::vector<std::int64_t>& mask_sizes,
    std::vector<TensorImpl>& out_indices,
    std::vector<std::int64_t>& out_index_shape,
    bool is_scalar_bool_path) {
  const std::int64_t K = static_cast<std::int64_t>(mask_sizes.size());
  if (K <= 0) {
    throw std::invalid_argument(
        "make_advanced_index: advanced tensor indices must have dim() > 0");
  }

  const std::int64_t mask_numel = safe_numel_from_sizes(mask_sizes);
  if (mask_numel > kAdvIndexMaxMaskNumel) {
    if (is_scalar_bool_path) {
      throw std::runtime_error(
          std::string(errors::kErrAdvIndexTooLarge) + " (scalar bool)");
    } else {
      throw std::runtime_error(
          std::string(errors::kErrAdvIndexTooLarge) + " (mask)");
    }
  }

  std::int64_t n_true = 0;
  const std::uint8_t* data = nullptr;
  TensorImpl mask_local;

  if (mask_opt) {
    mask_local = *mask_opt;
    if (mask_local.device().type != kDLCPU) {
      throw std::invalid_argument(
          "make_advanced_index: boolean tensor indices must be CPU tensors");
    }
    if (mask_local.dtype() != ScalarType::Bool) {
      throw std::invalid_argument(
          "make_advanced_index: expected Bool dtype for boolean tensor indices");
    }
    if (!mask_local.is_contiguous()) {
      mask_local = vbt::core::clone_cpu(mask_local);
    }
    data = static_cast<const std::uint8_t*>(mask_local.data());
    for (std::int64_t i = 0; i < mask_numel; ++i) {
      if (data[i] != 0) {
        ++n_true;
      }
    }
  } else {
    if (scalar_value) {
      n_true = mask_numel;
    } else {
      n_true = 0;
    }
  }

  out_index_shape.clear();
  out_index_shape.push_back(n_true);

  if (static_cast<std::int64_t>(out_index_shape.size()) > kAdvIndexMaxNdim) {
    throw std::runtime_error(errors::kErrAdvTooManyIndexDims);
  }
  if (n_true > kAdvIndexMaxIndexNumel) {
    throw std::runtime_error(errors::kErrAdvIndexTooLarge);
  }

  out_indices.clear();
  out_indices.reserve(static_cast<std::size_t>(K));

  const std::vector<std::int64_t> idx_sizes{n_true};
  for (std::int64_t k = 0; k < K; ++k) {
    out_indices.push_back(make_int64_tensor_cpu(base_like, idx_sizes));
  }

  if (n_true == 0 || mask_numel == 0) {
    return;
  }

  std::vector<std::int64_t*> idx_ptrs(static_cast<std::size_t>(K));
  for (std::int64_t k = 0; k < K; ++k) {
    idx_ptrs[static_cast<std::size_t>(k)] =
        static_cast<std::int64_t*>(out_indices[static_cast<std::size_t>(k)].data());
  }

  std::vector<std::int64_t> coords(static_cast<std::size_t>(K), 0);
  std::int64_t out_index = 0;

  for (std::int64_t linear = 0; linear < mask_numel; ++linear) {
    bool is_true = false;
    if (mask_opt) {
      is_true = (data[linear] != 0);
    } else {
      is_true = scalar_value;
    }
    if (!is_true) {
      continue;
    }

    std::int64_t tmp = linear;
    for (std::int64_t d = K - 1; d >= 0; --d) {
      const std::int64_t dim_size = mask_sizes[static_cast<std::size_t>(d)];
      std::int64_t coord = 0;
      if (dim_size > 0) {
        coord = tmp % dim_size;
        tmp /= dim_size;
      } else {
        coord = 0;
      }
      coords[static_cast<std::size_t>(d)] = coord;
    }

    for (std::int64_t d = 0; d < K; ++d) {
      idx_ptrs[static_cast<std::size_t>(d)][out_index] =
          coords[static_cast<std::size_t>(d)];
    }
    ++out_index;
  }

  if (out_index != n_true) {
    throw std::logic_error(
        "make_advanced_index: inconsistent boolean index construction");
  }
}

// Shared helper that finalizes AdvancedIndex given a base tensor, per-dim
// index sizes/strides, and raw Int64 index tensors.
static AdvancedIndex build_advanced_index(
    const TensorImpl& base,
    std::int64_t dims_before,
    std::vector<std::int64_t> indexed_sizes,
    std::vector<std::int64_t> indexed_strides_elems,
    std::vector<TensorImpl> indices,
    std::vector<std::int64_t> index_shape) {
  const auto& base_sizes = base.sizes();
  const auto& base_strides = base.strides();
  const std::int64_t B = static_cast<std::int64_t>(base_sizes.size());

  const std::size_t K = indices.size();
  if (indexed_sizes.size() != K ||
      indexed_strides_elems.size() != K) {
    throw std::logic_error(
        "build_advanced_index: inconsistent AdvancedIndex metadata");
  }

  if (dims_before < 0 || dims_before > B) {
    throw std::logic_error(
        "build_advanced_index: dims_before out of range");
  }

  const std::int64_t dims_after =
      B - dims_before - static_cast<std::int64_t>(K);
  if (dims_after < 0) {
    throw std::logic_error(
        "build_advanced_index: negative dims_after");
  }

  const std::int64_t index_ndim =
      static_cast<std::int64_t>(index_shape.size());
  if (index_ndim > kAdvIndexMaxNdim) {
    throw std::runtime_error(errors::kErrAdvTooManyIndexDims);
  }

  const std::int64_t index_numel = safe_numel_from_sizes(index_shape);
  if (index_numel > kAdvIndexMaxIndexNumel) {
    throw std::runtime_error(errors::kErrAdvIndexTooLarge);
  }

  for (std::size_t k = 0; k < K; ++k) {
    const std::int64_t size_k = indexed_sizes[k];
    TensorImpl& idx = indices[k];
    const std::int64_t numel_k = idx.numel();

    if (size_k == 0 && numel_k > 0) {
      throw std::out_of_range(
          std::string(errors::kErrIndexOutOfRange) + " 0");
    }

    auto* data = static_cast<std::int64_t*>(idx.data());
    for (std::int64_t i = 0; i < numel_k; ++i) {
      std::int64_t v = data[i];
      if (v < -size_k || v >= size_k) {
        throw std::out_of_range(
            std::string(errors::kErrIndexOutOfRange) + " " +
            std::to_string(size_k));
      }
      if (v < 0) {
        v += size_k;
      }
      data[i] = v;
    }
  }

  std::vector<std::int64_t> result_shape;
  result_shape.reserve(static_cast<std::size_t>(dims_before + index_ndim +
                                                dims_after));
  for (std::int64_t d = 0; d < dims_before; ++d) {
    result_shape.push_back(base_sizes[static_cast<std::size_t>(d)]);
  }
  result_shape.insert(result_shape.end(),
                      index_shape.begin(), index_shape.end());
  for (std::int64_t d = dims_before + static_cast<std::int64_t>(K);
       d < B; ++d) {
    result_shape.push_back(base_sizes[static_cast<std::size_t>(d)]);
  }

  const std::int64_t result_numel = safe_numel_from_sizes(result_shape);
  if (result_numel > kAdvIndexMaxResultNumel) {
    throw std::runtime_error(errors::kErrAdvResultTooLarge);
  }

  std::vector<std::int64_t> src_sizes = result_shape;
  std::vector<std::int64_t> src_strides(src_sizes.size(), 0);

  for (std::int64_t d = 0; d < dims_before; ++d) {
    src_strides[static_cast<std::size_t>(d)] =
        base_strides[static_cast<std::size_t>(d)];
  }

  for (std::int64_t j = 0; j < index_ndim; ++j) {
    src_strides[static_cast<std::size_t>(dims_before + j)] = 0;
  }

  for (std::int64_t d = 0; d < dims_after; ++d) {
    const std::int64_t base_d =
        dims_before + static_cast<std::int64_t>(K) + d;
    const std::int64_t out_d =
        dims_before + index_ndim + d;
    src_strides[static_cast<std::size_t>(out_d)] =
        base_strides[static_cast<std::size_t>(base_d)];
  }



  TensorImpl src(
      base.storage(),
      src_sizes,
      src_strides,
      base.storage_offset(),
      base.dtype(),
      base.device());

  // When index_shape has zero numel, the kernels will early-out and never
  // read from the index tensors. Avoid reshaping zero-sized index tensors
  // through view()/as_strided, which enforces a non-empty storage range
  // and would throw "as_strided: upper bound overflow" for tensors backed
  // by a zero-byte storage. Instead, rebuild zero-sized index tensors with
  // the desired logical shape but trivial (zero) strides.
  if (index_numel == 0) {
    for (auto& idx : indices) {
      std::vector<std::int64_t> idx_sizes;
      idx_sizes.reserve(src_sizes.size());
      for (std::int64_t d = 0; d < dims_before; ++d) {
        idx_sizes.push_back(1);
      }
      idx_sizes.insert(idx_sizes.end(),
                       index_shape.begin(), index_shape.end());
      for (std::int64_t d = 0; d < dims_after; ++d) {
        idx_sizes.push_back(1);
      }
      std::vector<std::int64_t> idx_strides(idx_sizes.size(), 0);
      idx = TensorImpl(idx.storage(),
                       std::move(idx_sizes),
                       std::move(idx_strides),
                       idx.storage_offset(),
                       idx.dtype(),
                       idx.device());
    }
  } else {
    for (auto& idx : indices) {
      std::vector<std::int64_t> idx_sizes;
      idx_sizes.reserve(src_sizes.size());
      for (std::int64_t d = 0; d < dims_before; ++d) {
        idx_sizes.push_back(1);
      }
      idx_sizes.insert(idx_sizes.end(),
                       index_shape.begin(), index_shape.end());
      for (std::int64_t d = 0; d < dims_after; ++d) {
        idx_sizes.push_back(1);
      }
      idx = vbt::core::reshape(idx, idx_sizes);
    }
  }

  AdvancedIndex info;
  info.src = std::move(src);
  info.indices = std::move(indices);
  info.indexed_sizes = std::move(indexed_sizes);
  info.indexed_strides_elems = std::move(indexed_strides_elems);
  info.dims_before = dims_before;
  info.dims_after = dims_after;
  info.index_shape = std::move(index_shape);
  info.result_shape = std::move(result_shape);

  const std::int64_t R = result_numel;
  info.use32bit_indexing = (R == 0) || (R > 0 && R <= std::numeric_limits<std::int32_t>::max());

  const bool has_nonempty_domain = (result_numel > 0);
  detail::record_cpu_32bit_hint(info.use32bit_indexing, has_nonempty_domain);

  return info;
}

// Scalar-bool-only advanced indexing implementation.
static AdvancedIndex make_advanced_index_scalar_bool_impl(
    const TensorImpl& self,
    const IndexSpec& spec) {
  const auto self_dim =
      static_cast<std::int64_t>(self.sizes().size());
  if (self_dim == 0) {
    throw std::invalid_argument(
        "make_advanced_index: advanced indexing is not supported for 0-d tensors");
  }

  if (spec.items.size() != 1 ||
      spec.items[0].kind != IndexKind::Boolean) {
    throw std::invalid_argument(
        "make_advanced_index: scalar bool advanced indexing only supports a single Boolean index");
  }

  const bool flag = spec.items[0].boolean;

  const auto& base_sizes = self.sizes();
  const auto& base_strides = self.strides();
  const std::int64_t B =
      static_cast<std::int64_t>(base_sizes.size());

  std::vector<std::int64_t> mask_sizes(
      base_sizes.begin(), base_sizes.end());

  std::vector<TensorImpl> indices;
  std::vector<std::int64_t> index_shape;

  build_bool_indices_from_mask(
      /*mask_opt=*/nullptr,
      /*scalar_value=*/flag,
      /*base_like=*/self,
      mask_sizes,
      indices,
      index_shape,
      /*is_scalar_bool_path=*/true);

  std::vector<std::int64_t> indexed_sizes;
  std::vector<std::int64_t> indexed_strides_elems;
  indexed_sizes.reserve(static_cast<std::size_t>(B));
  indexed_strides_elems.reserve(static_cast<std::size_t>(B));
  for (std::int64_t d = 0; d < B; ++d) {
    indexed_sizes.push_back(base_sizes[static_cast<std::size_t>(d)]);
    indexed_strides_elems.push_back(
        base_strides[static_cast<std::size_t>(d)]);
  }

  const std::int64_t dims_before = 0;
  return build_advanced_index(
      self,
      dims_before,
      std::move(indexed_sizes),
      std::move(indexed_strides_elems),
      std::move(indices),
      std::move(index_shape));
}

// Tensor (integer or boolean mask) advanced indexing implementation.
static AdvancedIndex make_advanced_index_tensor_impl(
    const TensorImpl& self,
    const IndexSpec& spec) {
  const auto self_dim =
      static_cast<std::int64_t>(self.sizes().size());

  const std::size_t n_items = spec.items.size();
  std::vector<std::size_t> tensor_positions;
  tensor_positions.reserve(n_items);

  for (std::size_t i = 0; i < n_items; ++i) {
    if (spec.items[i].kind == IndexKind::Tensor) {
      tensor_positions.push_back(i);
    }
  }

  if (tensor_positions.empty()) {
    throw std::logic_error(
        "make_advanced_index: no tensor advanced indices found after normalization");
  }
  if (tensor_positions.size() > 1) {
    throw std::invalid_argument(
        "make_advanced_index: multiple tensor indices are not supported");
  }

  const std::size_t adv_pos = tensor_positions[0];
  const TensorIndex& adv_it = spec.items[adv_pos];
  const TensorImpl& adv_tensor = adv_it.tensor;

  if (adv_tensor.sizes().size() == 0u) {
    throw std::invalid_argument(
        "make_advanced_index: advanced tensor indices must have dim() > 0");
  }

  if (adv_tensor.device() != self.device()) {
    throw std::invalid_argument(
        "make_advanced_index: advanced index tensor must be on the same device as self");
  }
  if (adv_tensor.device().type != kDLCPU) {
    throw std::invalid_argument(
        "make_advanced_index: advanced index tensor must be a CPU tensor");
  }

  const bool is_mask = (adv_tensor.dtype() == ScalarType::Bool);

  IndexSpec prefix_spec;
  prefix_spec.items.assign(
      spec.items.begin(), spec.items.begin() + static_cast<std::ptrdiff_t>(adv_pos));
  IndexSpec suffix_spec;
  suffix_spec.items.assign(
      spec.items.begin() + static_cast<std::ptrdiff_t>(adv_pos + 1),
      spec.items.end());

  // Disallow suffix indices after the advanced block, except for trailing
  // full-range slices (:) which are equivalent to omitted dimensions.
  std::int64_t suffix_full_slices = 0;
  for (const auto& it : suffix_spec.items) {
    if (it.kind == IndexKind::Slice) {
      const Slice& s = it.slice;
      if (!s.start.has_value() && !s.stop.has_value() && !s.step.has_value()) {
        ++suffix_full_slices;
        continue;  // full slice(None)
      }
    }
    throw std::invalid_argument(
        "advanced indexing: suffix indices after the advanced block are not supported");
  }

  TensorImpl base = self;
  if (!prefix_spec.items.empty()) {
    base = basic_index(base, prefix_spec);
  }

  const auto& base_sizes = base.sizes();
  const auto& base_strides = base.strides();
  const std::int64_t B =
      static_cast<std::int64_t>(base_sizes.size());

  if (B == 0) {
    throw std::invalid_argument(
        "make_advanced_index: advanced indexing is not supported for 0-d tensors");
  }

  std::vector<std::int64_t> indexed_sizes;
  std::vector<std::int64_t> indexed_strides_elems;
  std::vector<TensorImpl> indices;
  std::vector<std::int64_t> index_shape;

  if (is_mask) {
    const auto& mask_sizes_vec = adv_tensor.sizes();
    const std::int64_t mask_rank =
        static_cast<std::int64_t>(mask_sizes_vec.size());
    if (mask_rank > B) {
      throw std::invalid_argument(
          "make_advanced_index: boolean mask rank exceeds base rank");
    }

    const std::int64_t dims_before = B - mask_rank;

    for (std::int64_t k = 0; k < mask_rank; ++k) {
      const std::size_t base_d =
          static_cast<std::size_t>(dims_before + k);
      const auto sz_base = base_sizes[base_d];
      const auto sz_mask = mask_sizes_vec[static_cast<std::size_t>(k)];
      if (sz_base != sz_mask) {
        throw std::invalid_argument(
            "make_advanced_index: boolean mask shape must match indexed dimensions");
      }
    }

    std::vector<std::int64_t> mask_sizes(
        mask_sizes_vec.begin(), mask_sizes_vec.end());

    build_bool_indices_from_mask(
        &adv_tensor,
        /*scalar_value=*/false,
        /*base_like=*/base,
        mask_sizes,
        indices,
        index_shape,
        /*is_scalar_bool_path=*/false);

    indexed_sizes.reserve(static_cast<std::size_t>(mask_rank));
    indexed_strides_elems.reserve(static_cast<std::size_t>(mask_rank));
    for (std::int64_t k = 0; k < mask_rank; ++k) {
      const std::size_t base_d =
          static_cast<std::size_t>(dims_before + k);
      indexed_sizes.push_back(base_sizes[base_d]);
      indexed_strides_elems.push_back(base_strides[base_d]);
    }

    return build_advanced_index(
        base,
        dims_before,
        std::move(indexed_sizes),
        std::move(indexed_strides_elems),
        std::move(indices),
        std::move(index_shape));
  }

  // Integer tensor index path.
  TensorImpl index = adv_tensor;
  if (index.dtype() == ScalarType::Int64) {
    if (!index.is_contiguous()) {
      index = vbt::core::clone_cpu(index);
    }
  } else if (index.dtype() == ScalarType::Int32) {
    std::vector<std::int64_t> idx_sizes(
        index.sizes().begin(), index.sizes().end());
    TensorImpl tmp = make_int64_tensor_cpu(self, idx_sizes);
    auto* dst = static_cast<std::int64_t*>(tmp.data());
    auto* src = static_cast<const std::int32_t*>(index.data());
    const std::int64_t n = safe_numel_from_sizes(idx_sizes);
    for (std::int64_t i = 0; i < n; ++i) {
      dst[i] = static_cast<std::int64_t>(src[i]);
    }
    index = std::move(tmp);
  } else {
    throw std::invalid_argument(
        "make_advanced_index: only Int32 and Int64 index tensors are supported");
  }

  std::vector<std::int64_t> idx_sizes_vec(
      index.sizes().begin(), index.sizes().end());
  const std::int64_t adv_index_ndim =
      static_cast<std::int64_t>(idx_sizes_vec.size());
  if (adv_index_ndim > kAdvIndexMaxNdim) {
    throw std::runtime_error(errors::kErrAdvTooManyIndexDims);
  }

  const std::int64_t index_numel =
      safe_numel_from_sizes(idx_sizes_vec);
  if (index_numel > kAdvIndexMaxIndexNumel) {
    throw std::runtime_error(errors::kErrAdvIndexTooLarge);
  }

  indices.clear();
  indices.push_back(std::move(index));
  index_shape = std::move(idx_sizes_vec);

  const std::int64_t dims_before = B - 1 - suffix_full_slices;
  // Note: dims_before should always be in-range after ellipsis expansion and
  // suffix validation (suffix_full_slices <= B - 1). If this trips, it's an
  // internal bug in index normalization rather than a user error.
  if (dims_before < 0 || dims_before >= B) {
    throw std::logic_error(
        "make_advanced_index: computed advanced dim out of range");
  }
  const std::size_t adv_dim =
      static_cast<std::size_t>(dims_before);

  indexed_sizes.push_back(base_sizes[adv_dim]);
  indexed_strides_elems.push_back(base_strides[adv_dim]);

  return build_advanced_index(
      base,
      dims_before,
      std::move(indexed_sizes),
      std::move(indexed_strides_elems),
      std::move(indices),
      std::move(index_shape));
}

AdvancedIndex make_advanced_index(const TensorImpl& self,
                                  const IndexSpec& spec_raw) {
  if (self.device().type != kDLCPU) {
    throw std::invalid_argument(
        "make_advanced_index: advanced indexing requires CPU tensors");
  }

  const auto self_dim = static_cast<std::int64_t>(self.sizes().size());

  if (self_dim == 0 && has_any_advanced(spec_raw)) {
    throw std::invalid_argument(
        "make_advanced_index: advanced indexing is not supported for 0-d tensors");
  }

  if (!has_any_advanced(spec_raw)) {
    throw std::logic_error(
        "make_advanced_index called with basic-only IndexSpec");
  }

  // First, analyze the raw spec to detect scalar-bool-only vs tensor paths.
  bool has_scalar_bool_raw = false;
  bool has_tensor_adv_raw  = false;
  for (const auto& it : spec_raw.items) {
    if (it.kind == IndexKind::Boolean) has_scalar_bool_raw = true;
    if (it.kind == IndexKind::Tensor)  has_tensor_adv_raw  = true;
  }

  if (has_scalar_bool_raw && has_tensor_adv_raw) {
    throw std::invalid_argument(
        "make_advanced_index: mixing scalar bool and tensor/bool-mask indices is not supported");
  }

  // Pure scalar-bool-only advanced indexing (no tensor indices).
  if (has_scalar_bool_raw && !has_tensor_adv_raw) {
    return make_advanced_index_scalar_bool_impl(self, spec_raw);
  }

  // From here on, we only handle tensor-based advanced indexing
  // (integer tensors and boolean masks).
  IndexSpec spec = expand_ellipsis_and_validate(spec_raw, self_dim);

  if (!has_any_advanced(spec)) {
    throw std::logic_error(
        "make_advanced_index called with basic-only IndexSpec");
  }

  bool has_scalar_bool = false;
  bool has_tensor_adv  = false;
  for (const auto& it : spec.items) {
    if (it.kind == IndexKind::Boolean) has_scalar_bool = true;
    if (it.kind == IndexKind::Tensor)  has_tensor_adv  = true;
  }

  if (has_scalar_bool && has_tensor_adv) {
    throw std::invalid_argument(
        "make_advanced_index: mixing scalar bool and tensor/bool-mask indices is not supported");
  }

  // Optional normalization of x[idx, 1] on 2D: rewrite to prefix-only form
  // modeling x[:, k][idx].
  if (self_dim == 2 && spec.items.size() == 2) {
    const TensorIndex& i0 = spec.items[0];
    const TensorIndex& i1 = spec.items[1];
    if (i0.kind == IndexKind::Tensor &&
        i0.tensor.dtype() != ScalarType::Bool &&
        i1.kind == IndexKind::Integer) {
      IndexSpec rewritten;
      rewritten.items.emplace_back(Slice{});  // full slice(None)
      rewritten.items.push_back(i1);
      rewritten.items.push_back(i0);
      spec = std::move(rewritten);
    }
  }

  // Tensor (integer / bool mask) advanced indexing path.
  return make_advanced_index_tensor_impl(self, spec);
}

TensorImpl advanced_index_cpu(const AdvancedIndex& info) {
  if (info.src.device().type != kDLCPU) {
    throw std::invalid_argument(
        "advanced_index_cpu: src must be a CPU tensor");
  }

  const std::size_t K = info.indices.size();
  if (K == 0u) {
    throw std::invalid_argument(
        "advanced_index_cpu: at least one index tensor is required");
  }
  if (K != info.indexed_sizes.size() ||
      K != info.indexed_strides_elems.size()) {
    throw std::invalid_argument(
        "advanced_index_cpu: inconsistent AdvancedIndex metadata");
  }

  std::vector<std::int64_t> result_sizes = info.result_shape;
  TensorImpl out = make_empty_like_cpu(info.src, result_sizes);

  const std::int64_t N = safe_numel_from_sizes(result_sizes);
  if (N == 0) {
    return out;
  }

  const auto& src_strides = info.src.strides();
  const auto& src_sizes   = info.src.sizes();
  const std::int64_t R =
      static_cast<std::int64_t>(result_sizes.size());
  if (R != static_cast<std::int64_t>(src_sizes.size())) {
    throw std::invalid_argument(
        "advanced_index_cpu: src shape must match result_shape");
  }

  const std::int64_t dims_before = info.dims_before;
  const std::int64_t index_ndim =
      static_cast<std::int64_t>(info.index_shape.size());

  std::vector<const std::int64_t*> idx_data(K);
  for (std::size_t k = 0; k < K; ++k) {
    idx_data[k] = static_cast<const std::int64_t*>(info.indices[k].data());
  }

  auto* src_bytes = static_cast<std::uint8_t*>(info.src.data());
  auto* out_bytes = static_cast<std::uint8_t*>(out.data());
  const std::int64_t item_b =
      static_cast<std::int64_t>(info.src.itemsize());

  const bool use32 =
      advanced_index_32bit_enabled() && info.use32bit_indexing;

#ifndef NDEBUG
  if (use32) {
    VBT_ASSERT(N > 0);
    VBT_ASSERT(N <= static_cast<std::int64_t>(
                     std::numeric_limits<std::int32_t>::max()));
  }
#endif

  auto body = [&](std::int64_t linear,
                  const std::vector<std::int64_t>& coords) {
    // Base offset within src coming from prefix/suffix dims.
    std::int64_t src_off_elems = 0;
    for (std::int64_t d = 0; d < R; ++d) {
      const std::int64_t stride_d =
          src_strides[static_cast<std::size_t>(d)];
      if (stride_d == 0) {
        continue;
      }
      const std::int64_t c =
          coords[static_cast<std::size_t>(d)];
      std::int64_t term = 0;
      if (!checked_mul_i64(c, stride_d, term) ||
          !checked_add_i64(src_off_elems, term, src_off_elems)) {
        throw std::runtime_error(
            "advanced_index_cpu: offset overflow");
      }
    }

    // Advanced offset from index tensors.
    std::int64_t idx_linear = 0;
    for (std::int64_t j = 0; j < index_ndim; ++j) {
      const std::int64_t sz =
          info.index_shape[static_cast<std::size_t>(j)];
      const std::int64_t c =
          coords[static_cast<std::size_t>(dims_before + j)];
      idx_linear = idx_linear * sz + c;
    }

    std::int64_t adv_off_elems = 0;
    for (std::size_t k = 0; k < K; ++k) {
      const std::int64_t idx_val = idx_data[k][idx_linear];
      std::int64_t term = 0;
      if (!checked_mul_i64(idx_val,
                           info.indexed_strides_elems[k],
                           term) ||
          !checked_add_i64(adv_off_elems, term, adv_off_elems)) {
        throw std::runtime_error(
            "advanced_index_cpu: offset overflow");
      }
    }

    std::int64_t total_off_elems = 0;
    if (!checked_add_i64(src_off_elems, adv_off_elems,
                         total_off_elems)) {
      throw std::runtime_error(
          "advanced_index_cpu: offset overflow");
    }

    const std::int64_t src_off_bytes =
        total_off_elems * item_b;
    const std::int64_t out_off_bytes =
        linear * item_b;

    const std::uint8_t* src_elem =
        src_bytes + src_off_bytes;
    std::uint8_t* out_elem =
        out_bytes + out_off_bytes;

    std::memcpy(out_elem, src_elem,
                static_cast<std::size_t>(item_b));
  };

  if (use32) {
    const std::int32_t N32 =
        static_cast<std::int32_t>(N);
    for_each_advanced_index_linear<std::int32_t>(
        N32, result_sizes, body);
  } else {
    for_each_advanced_index_linear<std::int64_t>(
        N, result_sizes, body);
  }

  return out;
}

namespace {

static void accumulate_in_place(void* dst,
                                const void* src,
                                ScalarType dt) {
  switch (dt) {
    case ScalarType::Float32: {
      auto* pd = static_cast<float*>(dst);
      const auto* ps = static_cast<const float*>(src);
      *pd += *ps;
      break;
    }
    case ScalarType::Int32: {
      auto* pd = static_cast<std::int32_t*>(dst);
      const auto* ps = static_cast<const std::int32_t*>(src);
      *pd += *ps;
      break;
    }
    case ScalarType::Int64: {
      auto* pd = static_cast<std::int64_t*>(dst);
      const auto* ps = static_cast<const std::int64_t*>(src);
      *pd += *ps;
      break;
    }
    case ScalarType::Bool: {
      auto* pd = static_cast<std::uint8_t*>(dst);
      const auto* ps = static_cast<const std::uint8_t*>(src);
      const bool dv = (*pd != 0);
      const bool sv = (*ps != 0);
      *pd = static_cast<std::uint8_t>(dv || sv);
      break;
    }
    default:
      throw std::invalid_argument(
          "advanced_index_put_cpu: accumulate unsupported for dtype");
  }
}

} // anonymous namespace

void advanced_index_put_cpu(AdvancedIndex& info,
                            const TensorImpl& value,
                            bool accumulate) {
  if (info.src.device().type != kDLCPU) {
    throw std::invalid_argument(
        "advanced_index_put_cpu: src must be a CPU tensor");
  }
  if (value.device().type != kDLCPU) {
    throw std::invalid_argument(
        "advanced_index_put_cpu: value must be a CPU tensor");
  }
  if (value.dtype() != info.src.dtype()) {
    throw std::invalid_argument(
        "advanced_index_put_cpu: dtype mismatch between src and value");
  }

  const std::int64_t result_numel =
      safe_numel_from_sizes(info.result_shape);
  if (result_numel == 0) {
    return;
  }

  TensorImpl value_b = broadcast_to(
      value,
      std::span<const std::int64_t>(
          info.result_shape.data(), info.result_shape.size()));

  const auto& dst_strides = info.src.strides();
  const auto& dst_sizes   = info.src.sizes();
  const auto& val_strides = value_b.strides();
  const auto& val_sizes   = value_b.sizes();

  const std::int64_t R =
      static_cast<std::int64_t>(info.result_shape.size());
  if (R != static_cast<std::int64_t>(dst_sizes.size()) ||
      R != static_cast<std::int64_t>(val_sizes.size())) {
    throw std::invalid_argument(
        "advanced_index_put_cpu: result/value shapes must match src shape");
  }

  const std::size_t K = info.indices.size();
  if (K == 0u ||
      K != info.indexed_sizes.size() ||
      K != info.indexed_strides_elems.size()) {
    throw std::invalid_argument(
        "advanced_index_put_cpu: inconsistent AdvancedIndex metadata");
  }

  const std::int64_t dims_before = info.dims_before;
  const std::int64_t index_ndim =
      static_cast<std::int64_t>(info.index_shape.size());

  std::vector<const std::int64_t*> idx_data(K);
  for (std::size_t k = 0; k < K; ++k) {
    idx_data[k] = static_cast<const std::int64_t*>(info.indices[k].data());
  }

  auto* dst_bytes = static_cast<std::uint8_t*>(info.src.data());
  auto* src_bytes = static_cast<std::uint8_t*>(value_b.data());
  const std::int64_t item_b =
      static_cast<std::int64_t>(info.src.itemsize());

  const bool use32 =
      advanced_index_32bit_enabled() && info.use32bit_indexing;

#ifndef NDEBUG
  if (use32) {
    VBT_ASSERT(result_numel > 0);
    VBT_ASSERT(result_numel <= static_cast<std::int64_t>(
                               std::numeric_limits<std::int32_t>::max()));
  }
#endif

  auto body = [&](std::int64_t linear,
                  const std::vector<std::int64_t>& coords) {
    // Base offset within dst coming from prefix/suffix dims.
    std::int64_t dst_off_elems = 0;
    for (std::int64_t d = 0; d < R; ++d) {
      const std::int64_t stride_d =
          dst_strides[static_cast<std::size_t>(d)];
      if (stride_d == 0) {
        continue;
      }
      const std::int64_t c =
          coords[static_cast<std::size_t>(d)];
      std::int64_t term = 0;
      if (!checked_mul_i64(c, stride_d, term) ||
          !checked_add_i64(dst_off_elems, term, dst_off_elems)) {
        throw std::runtime_error(
            "advanced_index_put_cpu: offset overflow");
      }
    }

    // Advanced offset from index tensors.
    std::int64_t idx_linear = 0;
    for (std::int64_t j = 0; j < index_ndim; ++j) {
      const std::int64_t sz =
          info.index_shape[static_cast<std::size_t>(j)];
      const std::int64_t c =
          coords[static_cast<std::size_t>(dims_before + j)];
      idx_linear = idx_linear * sz + c;
    }

    std::int64_t adv_off_elems = 0;
    for (std::size_t k = 0; k < K; ++k) {
      const std::int64_t idx_val = idx_data[k][idx_linear];
      std::int64_t term = 0;
      if (!checked_mul_i64(idx_val,
                           info.indexed_strides_elems[k],
                           term) ||
          !checked_add_i64(adv_off_elems, term, adv_off_elems)) {
        throw std::runtime_error(
            "advanced_index_put_cpu: offset overflow");
      }
    }

    std::int64_t total_dst_off_elems = 0;
    if (!checked_add_i64(dst_off_elems, adv_off_elems,
                         total_dst_off_elems)) {
      throw std::runtime_error(
          "advanced_index_put_cpu: offset overflow");
    }

    // Offset for value_b based on its strides.
    std::int64_t src_off_elems = 0;
    for (std::int64_t d = 0; d < R; ++d) {
      const std::int64_t stride_d =
          val_strides[static_cast<std::size_t>(d)];
      if (stride_d == 0) {
        continue;
      }
      const std::int64_t c =
          coords[static_cast<std::size_t>(d)];
      std::int64_t term = 0;
      if (!checked_mul_i64(c, stride_d, term) ||
          !checked_add_i64(src_off_elems, term, src_off_elems)) {
        throw std::runtime_error(
            "advanced_index_put_cpu: offset overflow");
      }
    }

    const std::int64_t dst_off_bytes =
        total_dst_off_elems * item_b;
    const std::int64_t src_off_bytes =
        src_off_elems * item_b;

    std::uint8_t* dst_elem =
        dst_bytes + dst_off_bytes;
    const std::uint8_t* src_elem =
        src_bytes + src_off_bytes;

    if (!accumulate) {
      std::memcpy(dst_elem, src_elem,
                  static_cast<std::size_t>(item_b));
    } else {
      accumulate_in_place(dst_elem, src_elem, info.src.dtype());
    }
  };

  if (use32) {
    const std::int32_t N32 =
        static_cast<std::int32_t>(result_numel);
    for_each_advanced_index_linear<std::int32_t>(
        N32, info.result_shape, body);
  } else {
    for_each_advanced_index_linear<std::int64_t>(
        result_numel, info.result_shape, body);
  }
}
TensorImpl index(const TensorImpl& self, const IndexSpec& spec) {
  if (!has_any_advanced(spec)) {
    return basic_index(self, spec);
  }

  if (!advanced_indexing_enabled()) {
    throw std::runtime_error(errors::kErrAdvDisabledCore);
  }

  const auto dev_type = self.device().type;
  if (dev_type == kDLCPU) {
    AdvancedIndex info = make_advanced_index(self, spec);
    return advanced_index_cpu(info);
  }

#if VBT_WITH_CUDA
  if (dev_type == kDLCUDA) {
    auto r = cuda_impl::make_advanced_index_cuda(
        self, spec, cuda_impl::AdvancedIndexCudaMode::Read);
    return cuda_impl::advanced_index_cuda_impl(r.info, r.can_use_1d_fastpath);
  }
#endif

  throw std::invalid_argument(
      "index: advanced indexing is only implemented for CPU"
#if VBT_WITH_CUDA
      " and CUDA"
#endif
      " tensors");
}

void index_put_(TensorImpl& self,
                const IndexSpec& spec,
                const TensorImpl& value,
                bool accumulate) {
  if (!has_any_advanced(spec)) {
    basic_index_put(self, spec, value);
    return;
  }

  if (!advanced_indexing_enabled()) {
    throw std::runtime_error(errors::kErrAdvDisabledCore);
  }

  // Shared aliasing rules for advanced writes (CPU & CUDA).
  vbt::core::check_writable(self);
  vbt::core::assert_no_internal_overlap(self);
  if (self.storage().get() == value.storage().get()) {
    vbt::core::assert_no_partial_overlap(self, value);
  }

  const auto dev_type = self.device().type;
  if (dev_type == kDLCPU) {
    AdvancedIndex info = make_advanced_index(self, spec);
    advanced_index_put_cpu(info, value, accumulate);
    if (info.src.numel() > 0) {
      self.bump_version();
    }
    return;
  }

#if VBT_WITH_CUDA
  if (dev_type == kDLCUDA) {
    auto r = cuda_impl::make_advanced_index_cuda(self, spec);
    advanced_index_put_cuda(r.info, value, accumulate);
    if (r.info.src.numel() > 0) {
      self.bump_version();
    }
    return;
  }
#endif

  throw std::invalid_argument(
      "index_put_: advanced indexing is only implemented for CPU"
#if VBT_WITH_CUDA
      " and CUDA"
#endif
      " tensors");
}

} // namespace indexing
} // namespace core
} // namespace vbt
