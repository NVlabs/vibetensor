// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/reduction_env.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vbt {
namespace cuda {
namespace reduction {

namespace {
std::once_flag            g_env_once;
CudaReductionEnvConfig    g_env_cfg_from_env{};
CudaReductionEnvConfig    g_env_cfg_override{};
std::atomic<bool>         g_env_override_active{false};

static bool compute_opt_in_flag_from_env_raw(const char* raw) noexcept {
  if (raw == nullptr) {
    return false;
  }

  const unsigned char* begin = reinterpret_cast<const unsigned char*>(raw);
  while (*begin && std::isspace(*begin)) {
    ++begin;
  }
  if (*begin == '\0') {
    return false;
  }

  const unsigned char* end =
      begin + std::strlen(reinterpret_cast<const char*>(begin));
  while (end > begin && std::isspace(*(end - 1))) {
    --end;
  }
  if (end == begin) {
    return false;
  }

  auto equals_ci = [&](const char* lit) noexcept {
    const std::size_t lit_len = std::strlen(lit);
    const std::size_t n = static_cast<std::size_t>(end - begin);
    if (n != lit_len) {
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const unsigned char c =
          static_cast<unsigned char>(std::tolower(begin[i]));
      if (static_cast<char>(c) != lit[i]) {
        return false;
      }
    }
    return true;
  };

  if (equals_ci("0") || equals_ci("false") || equals_ci("no") ||
      equals_ci("off")) {
    return false;
  }

  return true;
}

static CudaReductionEnvConfig compute_env_config_from_process_env() noexcept {
  CudaReductionEnvConfig cfg{};

  // Staged kernel enablement (default: off; opt-in only).
  cfg.staged_default = compute_opt_in_flag_from_env_raw(
      std::getenv("VBT_CUDA_REDUCTION_STAGED"));

  // CUDA launch cap: positive integer => soft upper bound on blocks;
  // non-positive / malformed => no additional cap beyond device limit.
  const char* raw_max_blocks = std::getenv("VBT_CUDA_REDUCTION_MAX_BLOCKS");
  cfg.cuda_max_blocks_cap = 0;
  if (raw_max_blocks && *raw_max_blocks != '\0') {
    char* end = nullptr;
    long long val = std::strtoll(raw_max_blocks, &end, 10);
    if (end != raw_max_blocks && *end == '\0' && val > 0) {
      cfg.cuda_max_blocks_cap = static_cast<std::int64_t>(val);
    }
  }

  return cfg;
}
} // namespace

const CudaReductionEnvConfig& get_cuda_reduction_env_config() noexcept {
#if VBT_INTERNAL_TESTS
  if (g_env_override_active.load(std::memory_order_acquire)) {
    return g_env_cfg_override;
  }
#endif

  std::call_once(g_env_once,
                 [] { g_env_cfg_from_env = compute_env_config_from_process_env(); });
  return g_env_cfg_from_env;
}

#if VBT_INTERNAL_TESTS
namespace {
// -1: no override.
std::atomic<int>  g_kernel_policy_override{-1};
std::atomic<long long> g_grid_x_cap_override{-1};
std::atomic<long long> g_k2multi_ctas_per_output_override{-1};
std::atomic<int> g_k2multi_fault_mode_override{-1};
std::atomic<int> g_k2multi_stream_mismatch_injection_override{-1};

std::mutex               g_last_stats_mutex;
CudaReductionLastStats   g_last_stats{};
} // namespace

CudaReductionLastStats get_cuda_reduction_last_stats_for_tests() noexcept {
  std::lock_guard<std::mutex> lock(g_last_stats_mutex);
  return g_last_stats;
}

void reset_cuda_reduction_last_stats_for_tests() noexcept {
  std::lock_guard<std::mutex> lock(g_last_stats_mutex);
  g_last_stats = CudaReductionLastStats{};
}

void set_cuda_reduction_last_stats_for_tests(const CudaReductionLastStats& stats) noexcept {
  std::lock_guard<std::mutex> lock(g_last_stats_mutex);
  g_last_stats = stats;
}

CudaReductionKernelPolicy get_cuda_reduction_kernel_policy_for_tests() noexcept {
  const int raw = g_kernel_policy_override.load(std::memory_order_acquire);
  if (raw < 0) {
    return CudaReductionKernelPolicy::Auto;
  }
  return static_cast<CudaReductionKernelPolicy>(raw);
}

void set_cuda_reduction_kernel_policy_for_tests(CudaReductionKernelPolicy policy) noexcept {
  g_kernel_policy_override.store(static_cast<int>(policy), std::memory_order_release);
}

void clear_cuda_reduction_kernel_policy_override_for_tests() noexcept {
  g_kernel_policy_override.store(-1, std::memory_order_release);
}

bool cuda_reduction_kernel_policy_override_is_active_for_tests() noexcept {
  return g_kernel_policy_override.load(std::memory_order_acquire) >= 0;
}

std::optional<unsigned int> get_cuda_reduction_grid_x_cap_for_tests() noexcept {
  const long long raw = g_grid_x_cap_override.load(std::memory_order_acquire);
  if (raw < 0) {
    return std::nullopt;
  }
  return static_cast<unsigned int>(raw);
}

void set_cuda_reduction_grid_x_cap_for_tests(std::optional<unsigned int> cap) {
  if (cap.has_value() && *cap == 0u) {
    throw std::invalid_argument(
        "cuda_reduction: grid_x_cap_for_tests must be > 0");
  }
  if (cap.has_value()) {
    g_grid_x_cap_override.store(static_cast<long long>(*cap), std::memory_order_release);
  } else {
    g_grid_x_cap_override.store(-1, std::memory_order_release);
  }
}

void clear_cuda_reduction_grid_x_cap_override_for_tests() noexcept {
  g_grid_x_cap_override.store(-1, std::memory_order_release);
}

std::optional<unsigned int> get_cuda_reduction_k2multi_ctas_per_output_for_tests() noexcept {
  const long long raw = g_k2multi_ctas_per_output_override.load(std::memory_order_acquire);
  if (raw < 0) {
    return std::nullopt;
  }
  return static_cast<unsigned int>(raw);
}

void set_cuda_reduction_k2multi_ctas_per_output_for_tests(
    std::optional<unsigned int> ctas_per_output) {
  if (ctas_per_output.has_value() && *ctas_per_output == 0u) {
    throw std::invalid_argument(
        "cuda_reduction: k2multi_ctas_per_output_for_tests must be > 0 "
        "(dispatcher clamps to >= 2 for K2Multi)");
  }
  if (ctas_per_output.has_value()) {
    g_k2multi_ctas_per_output_override.store(static_cast<long long>(*ctas_per_output),
                                             std::memory_order_release);
  } else {
    g_k2multi_ctas_per_output_override.store(-1, std::memory_order_release);
  }
}

void clear_cuda_reduction_k2multi_ctas_per_output_override_for_tests() noexcept {
  g_k2multi_ctas_per_output_override.store(-1, std::memory_order_release);
}

CudaK2MultiFaultMode get_cuda_reduction_k2multi_fault_mode_for_tests() noexcept {
  const int raw = g_k2multi_fault_mode_override.load(std::memory_order_acquire);
  if (raw < 0) {
    return CudaK2MultiFaultMode::None;
  }

  if (raw == static_cast<int>(CudaK2MultiFaultMode::None) ||
      raw == static_cast<int>(CudaK2MultiFaultMode::SignalButSkipPartialWrite)) {
    return static_cast<CudaK2MultiFaultMode>(raw);
  }

  // Defensive: unknown values are treated as None.
  return CudaK2MultiFaultMode::None;
}

void set_cuda_reduction_k2multi_fault_mode_for_tests(CudaK2MultiFaultMode mode) noexcept {
  g_k2multi_fault_mode_override.store(static_cast<int>(mode), std::memory_order_release);
}

void clear_cuda_reduction_k2multi_fault_mode_override_for_tests() noexcept {
  g_k2multi_fault_mode_override.store(-1, std::memory_order_release);
}

bool cuda_reduction_k2multi_fault_mode_override_is_active_for_tests() noexcept {
  return g_k2multi_fault_mode_override.load(std::memory_order_acquire) >= 0;
}

bool cuda_reduction_k2multi_stream_mismatch_injection_enabled_for_tests() noexcept {
  const int raw = g_k2multi_stream_mismatch_injection_override.load(std::memory_order_acquire);
  return raw > 0;
}

void set_cuda_reduction_k2multi_stream_mismatch_injection_enabled_for_tests(bool enabled) noexcept {
  g_k2multi_stream_mismatch_injection_override.store(enabled ? 1 : 0, std::memory_order_release);
}

void clear_cuda_reduction_k2multi_stream_mismatch_injection_override_for_tests() noexcept {
  g_k2multi_stream_mismatch_injection_override.store(-1, std::memory_order_release);
}

bool cuda_reduction_k2multi_stream_mismatch_injection_override_is_active_for_tests() noexcept {
  return g_k2multi_stream_mismatch_injection_override.load(std::memory_order_acquire) >= 0;
}

CudaReductionEnvConfig get_cuda_reduction_env_config_for_tests() noexcept {
  return get_cuda_reduction_env_config();
}

void reset_cuda_reduction_env_config_for_tests(const CudaReductionEnvConfig& cfg) noexcept {
  g_env_cfg_override = cfg;
  g_env_override_active.store(true, std::memory_order_release);
}

void clear_cuda_reduction_env_config_override_for_tests() noexcept {
  g_env_override_active.store(false, std::memory_order_release);
}

bool cuda_reduction_env_config_override_is_active_for_tests() noexcept {
  return g_env_override_active.load(std::memory_order_acquire);
}
#endif  // VBT_INTERNAL_TESTS

} // namespace reduction
} // namespace cuda
} // namespace vbt
