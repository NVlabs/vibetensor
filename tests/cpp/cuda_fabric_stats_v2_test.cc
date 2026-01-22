// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <vector>

#include "vbt/cuda/fabric_addmul_decision.h"
#include "vbt/cuda/fabric_state.h"

using vbt::cuda::fabric::FabricAddMulDecision;
using vbt::cuda::fabric::FabricAddMulFallbackReason;
using vbt::cuda::fabric::FabricInflightGuard;
using vbt::cuda::fabric::FabricStats;
using vbt::cuda::fabric::compute_fabric_bytes;
using vbt::cuda::fabric::record_fabric_decision_stats;

namespace {

void reset_stats(FabricStats& stats) {
  stats.fabric_ops_attempted.store(0, std::memory_order_relaxed);
  stats.fabric_ops_hit.store(0, std::memory_order_relaxed);
  stats.fabric_ops_fallback.store(0, std::memory_order_relaxed);

  stats.remote_bytes_read.store(0, std::memory_order_relaxed);
  stats.remote_bytes_written.store(0, std::memory_order_relaxed);

  stats.reasons.no_p2p.store(0, std::memory_order_relaxed);
  stats.reasons.requires_grad.store(0, std::memory_order_relaxed);
  stats.reasons.in_backward.store(0, std::memory_order_relaxed);
  stats.reasons.small_tensor.store(0, std::memory_order_relaxed);

  stats.inflight_ops_current.store(0, std::memory_order_relaxed);
  stats.inflight_ops_peak.store(0, std::memory_order_relaxed);

  for (int i = 0; i < FabricStats::kMaxFabricDevices; ++i) {
    stats.ops_as_primary[i].store(0, std::memory_order_relaxed);
    stats.ops_as_remote[i].store(0, std::memory_order_relaxed);
    stats.remote_bytes_read_by_device[i].store(0, std::memory_order_relaxed);
    stats.remote_bytes_written_by_device[i].store(0, std::memory_order_relaxed);
  }
}

std::uint64_t sum_arr(
    const std::array<std::atomic<std::uint64_t>, FabricStats::kMaxFabricDevices>& arr) {
  std::uint64_t sum = 0;
  for (const auto& x : arr) {
    sum += x.load(std::memory_order_relaxed);
  }
  return sum;
}

}  // namespace

TEST(FabricStatsV2, ComputeFabricBytesSaturatesOnOverflow) {
  FabricAddMulDecision dec;
  dec.numel = static_cast<std::int64_t>(
      (std::numeric_limits<std::uint64_t>::max() / 4u) + 1u);

  const std::uint64_t bytes = compute_fabric_bytes(dec, /*itemsize=*/4);
  EXPECT_EQ(bytes, std::numeric_limits<std::uint64_t>::max());
}

TEST(FabricStatsV2, ComputeFabricBytesReturnsZeroOnNonPositiveNumel) {
  FabricAddMulDecision dec;
  dec.numel = -1;
  EXPECT_EQ(compute_fabric_bytes(dec, /*itemsize=*/4), 0u);

  dec.numel = 0;
  EXPECT_EQ(compute_fabric_bytes(dec, /*itemsize=*/4), 0u);
}

TEST(FabricStatsV2, RecordDecisionStatsEarlyReturnForSameDevice) {
  FabricStats stats;
  reset_stats(stats);

  FabricAddMulDecision dec;
  dec.other_device = -1;
  dec.primary_device = 0;
  dec.numel = 123;

  record_fabric_decision_stats(stats, dec, /*itemsize=*/4);

  EXPECT_EQ(stats.fabric_ops_attempted.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(sum_arr(stats.ops_as_primary), 0u);
  EXPECT_EQ(sum_arr(stats.ops_as_remote), 0u);
}

TEST(FabricStatsV2, RecordDecisionStatsAggregatesAndPreservesInvariants) {
  FabricStats stats;
  reset_stats(stats);

  FabricAddMulDecision hit;
  hit.use_fabric = true;
  hit.use_copy_fallback = false;
  hit.reason = FabricAddMulFallbackReason::kNone;
  hit.primary_device = 0;
  hit.other_device = 1;
  hit.numel = 3;

  FabricAddMulDecision fb;
  fb.use_fabric = false;
  fb.use_copy_fallback = true;
  fb.reason = FabricAddMulFallbackReason::kNotInSameCliqueOrNoP2P;
  fb.primary_device = 0;
  fb.other_device = 1;
  fb.numel = 3;

  FabricAddMulDecision err;
  err.use_fabric = false;
  err.use_copy_fallback = false;
  err.reason = FabricAddMulFallbackReason::kRequiresGrad;
  err.primary_device = 0;
  err.other_device = 1;
  err.numel = 3;

  record_fabric_decision_stats(stats, hit, /*itemsize=*/4);
  record_fabric_decision_stats(stats, fb, /*itemsize=*/4);
  record_fabric_decision_stats(stats, err, /*itemsize=*/4);

  const std::uint64_t attempted = stats.fabric_ops_attempted.load(std::memory_order_relaxed);
  EXPECT_EQ(attempted, 3u);

  EXPECT_EQ(stats.fabric_ops_hit.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(stats.fabric_ops_fallback.load(std::memory_order_relaxed), 1u);

  EXPECT_EQ(stats.reasons.no_p2p.load(std::memory_order_relaxed), 1u);
  EXPECT_EQ(stats.reasons.requires_grad.load(std::memory_order_relaxed), 1u);

  // Only hit + fallback contribute to bytes.
  const std::uint64_t bytes_one = compute_fabric_bytes(hit, /*itemsize=*/4);
  const std::uint64_t bytes_expected = bytes_one * 2u;

  EXPECT_EQ(stats.remote_bytes_read.load(std::memory_order_relaxed), bytes_expected);
  EXPECT_EQ(stats.remote_bytes_written.load(std::memory_order_relaxed), bytes_expected);

  EXPECT_EQ(sum_arr(stats.ops_as_primary), attempted);
  EXPECT_EQ(sum_arr(stats.ops_as_remote), attempted);

  EXPECT_EQ(sum_arr(stats.remote_bytes_read_by_device), bytes_expected);
  EXPECT_EQ(sum_arr(stats.remote_bytes_written_by_device), bytes_expected);

  // 2-device add/mul: read == write both globally and per-device.
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(stats.remote_bytes_read_by_device[i].load(std::memory_order_relaxed),
              stats.remote_bytes_written_by_device[i].load(std::memory_order_relaxed));
  }
}

TEST(FabricStatsV2, InflightGuardTracksCurrentAndPeakAndIsExceptionSafe) {
  FabricStats stats;
  reset_stats(stats);

  {
    FabricInflightGuard g0(&stats, /*cross_device_inputs=*/false);
    EXPECT_EQ(stats.inflight_ops_current.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(stats.inflight_ops_peak.load(std::memory_order_relaxed), 0u);
  }

  {
    FabricInflightGuard g1(&stats, /*cross_device_inputs=*/true);
    EXPECT_EQ(stats.inflight_ops_current.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(stats.inflight_ops_peak.load(std::memory_order_relaxed), 1u);

    {
      FabricInflightGuard g2(&stats, /*cross_device_inputs=*/true);
      EXPECT_EQ(stats.inflight_ops_current.load(std::memory_order_relaxed), 2u);
      EXPECT_EQ(stats.inflight_ops_peak.load(std::memory_order_relaxed), 2u);
    }

    EXPECT_EQ(stats.inflight_ops_current.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(stats.inflight_ops_peak.load(std::memory_order_relaxed), 2u);
  }

  EXPECT_EQ(stats.inflight_ops_current.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(stats.inflight_ops_peak.load(std::memory_order_relaxed), 2u);

  // Exception path: destructor must run and restore current to 0.
  try {
    FabricInflightGuard g3(&stats, /*cross_device_inputs=*/true);
    throw std::runtime_error("boom");
  } catch (const std::exception&) {
  }

  EXPECT_EQ(stats.inflight_ops_current.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(stats.inflight_ops_peak.load(std::memory_order_relaxed), 2u);
}

TEST(FabricStatsV2, InflightGuardTracksPeakUnderConcurrency) {
  FabricStats stats;
  reset_stats(stats);

  constexpr int kThreads = 8;

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::atomic<int> in_guard{0};
  std::atomic<bool> release{false};

  auto worker = [&]() {
    ready.fetch_add(1, std::memory_order_acq_rel);
    // Spin until the main thread signals all workers to proceed.
    while (!go.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    // Hold a single guard instance so that all threads overlap.
    FabricInflightGuard g(&stats, /*cross_device_inputs=*/true);
    in_guard.fetch_add(1, std::memory_order_acq_rel);

    // Spin until the main thread lets all workers exit, ensuring peak reflects
    // all threads holding the guard concurrently.
    while (!release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back(worker);
  }

  // Wait until all workers are ready, then release them.
  while (ready.load(std::memory_order_acquire) < kThreads) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);

  // Ensure all workers have entered the guarded region before releasing them.
  while (in_guard.load(std::memory_order_acquire) < kThreads) {
    std::this_thread::yield();
  }
  release.store(true, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(stats.inflight_ops_current.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(stats.inflight_ops_peak.load(std::memory_order_relaxed),
            static_cast<std::uint64_t>(kThreads));
}
