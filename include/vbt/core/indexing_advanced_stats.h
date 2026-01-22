// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>

namespace vbt {
namespace core {
namespace indexing {

struct MIndexAdvancedStats {
  std::atomic<std::uint64_t> cpu_32bit_hint_true{0};
  std::atomic<std::uint64_t> cpu_32bit_hint_false{0};
  std::atomic<std::uint64_t> cuda_32bit_hint_true{0};
  std::atomic<std::uint64_t> cuda_32bit_hint_false{0};
  std::atomic<std::uint64_t> cuda_fast1d_forward_hits{0};

  // Bytes copied D2H for the full bool-mask payload in the legacy CUDA
  // bool-mask indexing path (host scan). The CUB backend should keep this at 0.
  std::atomic<std::uint64_t> cuda_bool_mask_d2h_bytes{0};
};

// Read-only snapshot accessor for tests and tooling.
const MIndexAdvancedStats& get_m_index_advanced_stats() noexcept;

// Test-only helper to zero all counters. Intended for serial test suites.
void reset_m_index_advanced_stats_for_tests() noexcept;

namespace detail {

// Internal helpers used only by indexing_advanced.cc / .cu to update stats.
// - has_nonempty_domain gates updates for zero-numel results.
// - All increments are memory_order_relaxed and may wrap.
void record_cpu_32bit_hint(bool use32bit_hint_true,
                           bool has_nonempty_domain) noexcept;

void record_cuda_32bit_hint(bool use32bit_hint_true,
                            bool has_nonempty_domain) noexcept;

// Increment cuda_fast1d_forward_hits by 1.
// Call sites must ensure the fast path is actually taken and N > 0.
void record_cuda_fast1d_hit() noexcept;

// Add nbytes to cuda_bool_mask_d2h_bytes.
void record_cuda_bool_mask_d2h_bytes(std::uint64_t nbytes) noexcept;

} // namespace detail

} // namespace indexing
} // namespace core
} // namespace vbt
