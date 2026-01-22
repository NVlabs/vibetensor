// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/indexing_advanced_stats.h"

namespace vbt {
namespace core {
namespace indexing {

namespace {

MIndexAdvancedStats g_mindex_stats;  // zero-initialized at startup

} // anonymous namespace

const MIndexAdvancedStats& get_m_index_advanced_stats() noexcept {
  return g_mindex_stats;
}

void reset_m_index_advanced_stats_for_tests() noexcept {
  auto& s = g_mindex_stats;
  s.cpu_32bit_hint_true.store(0, std::memory_order_relaxed);
  s.cpu_32bit_hint_false.store(0, std::memory_order_relaxed);
  s.cuda_32bit_hint_true.store(0, std::memory_order_relaxed);
  s.cuda_32bit_hint_false.store(0, std::memory_order_relaxed);
  s.cuda_fast1d_forward_hits.store(0, std::memory_order_relaxed);
  s.cuda_bool_mask_d2h_bytes.store(0, std::memory_order_relaxed);
}

namespace detail {

void record_cpu_32bit_hint(bool use32bit_hint_true,
                           bool has_nonempty_domain) noexcept {
  if (!has_nonempty_domain) {
    return;
  }
  auto& s = g_mindex_stats;
  if (use32bit_hint_true) {
    s.cpu_32bit_hint_true.fetch_add(1, std::memory_order_relaxed);
  } else {
    s.cpu_32bit_hint_false.fetch_add(1, std::memory_order_relaxed);
  }
}

void record_cuda_32bit_hint(bool use32bit_hint_true,
                            bool has_nonempty_domain) noexcept {
  if (!has_nonempty_domain) {
    return;
  }
  auto& s = g_mindex_stats;
  if (use32bit_hint_true) {
    s.cuda_32bit_hint_true.fetch_add(1, std::memory_order_relaxed);
  } else {
    s.cuda_32bit_hint_false.fetch_add(1, std::memory_order_relaxed);
  }
}

void record_cuda_fast1d_hit() noexcept {
  g_mindex_stats.cuda_fast1d_forward_hits.fetch_add(1, std::memory_order_relaxed);
}

void record_cuda_bool_mask_d2h_bytes(std::uint64_t nbytes) noexcept {
  g_mindex_stats.cuda_bool_mask_d2h_bytes.fetch_add(
      nbytes, std::memory_order_relaxed);
}

} // namespace detail

} // namespace indexing
} // namespace core
} // namespace vbt
