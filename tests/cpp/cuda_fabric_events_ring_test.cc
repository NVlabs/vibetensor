// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "vbt/cuda/fabric_events.h"
#include "vbt/cuda/fabric_state.h"

using namespace std::chrono_literals;

using vbt::cuda::fabric::FabricEventsMode;
using vbt::cuda::fabric::FabricEvent;
using vbt::cuda::fabric::FabricEventKind;
using vbt::cuda::fabric::FabricEventLevel;
using vbt::cuda::fabric::FabricEventSnapshot;
using vbt::cuda::fabric::fabric_events_enabled;
using vbt::cuda::fabric::fabric_events_snapshot;
using vbt::cuda::fabric::fabric_events_wait_for_seq;
using vbt::cuda::fabric::get_fabric_events_mode;
using vbt::cuda::fabric::kFabricEventMsgTest;
using vbt::cuda::fabric::record_fabric_event;
using vbt::cuda::fabric::set_fabric_events_mode;

#if defined(VBT_INTERNAL_TESTS)
using vbt::cuda::fabric::reset_fabric_events_for_tests;
using vbt::cuda::fabric::reset_fabric_stats_for_tests;
using vbt::cuda::fabric::debug_fail_fabric_event_record_on_n_for_testing;
using vbt::cuda::fabric::debug_reset_fabric_event_record_injection_for_testing;
#endif

TEST(FabricEventsRing, ModeGateGetSet) {
  set_fabric_events_mode(FabricEventsMode::kOff);
  EXPECT_EQ(get_fabric_events_mode(), FabricEventsMode::kOff);
  EXPECT_FALSE(fabric_events_enabled());

  set_fabric_events_mode(FabricEventsMode::kBasic);
  EXPECT_EQ(get_fabric_events_mode(), FabricEventsMode::kBasic);
  EXPECT_TRUE(fabric_events_enabled());

  set_fabric_events_mode(FabricEventsMode::kOff);
  EXPECT_EQ(get_fabric_events_mode(), FabricEventsMode::kOff);
  EXPECT_FALSE(fabric_events_enabled());
}

#if defined(VBT_INTERNAL_TESTS)

TEST(FabricEventsRing, SnapshotEmpty) {
  set_fabric_events_mode(FabricEventsMode::kBasic);
  reset_fabric_events_for_tests();

  FabricEventSnapshot snap = fabric_events_snapshot(/*min_seq=*/0, /*max_events=*/16);
  EXPECT_EQ(snap.base_seq, 0u);
  EXPECT_EQ(snap.next_seq, 0u);
  EXPECT_GT(snap.capacity, 0u);
  EXPECT_TRUE(snap.events.empty());
}

TEST(FabricEventsRing, RecordAndSnapshot) {
  set_fabric_events_mode(FabricEventsMode::kBasic);
  reset_fabric_events_for_tests();

  FabricEvent ev;
  ev.kind = FabricEventKind::kOpEnqueue;
  ev.level = FabricEventLevel::kInfo;
  ev.message = kFabricEventMsgTest;
  record_fabric_event(std::move(ev));

  FabricEventSnapshot snap = fabric_events_snapshot(/*min_seq=*/0, /*max_events=*/16);
  ASSERT_EQ(snap.events.size(), 1u);
  EXPECT_EQ(snap.base_seq, 0u);
  EXPECT_EQ(snap.next_seq, 1u);

  const auto& e0 = snap.events[0];
  EXPECT_EQ(e0.seq, 0u);
  EXPECT_EQ(e0.kind, FabricEventKind::kOpEnqueue);
  EXPECT_EQ(e0.level, FabricEventLevel::kInfo);
  ASSERT_NE(e0.message, nullptr);
  EXPECT_STREQ(e0.message, kFabricEventMsgTest);
  EXPECT_GT(e0.t_ns, 0u);
}

TEST(FabricEventsRing, OverflowDropsOldest) {
  set_fabric_events_mode(FabricEventsMode::kBasic);
  reset_fabric_events_for_tests();

  const std::size_t cap = fabric_events_snapshot(0, 0).capacity;
  ASSERT_GT(cap, 0u);

  const std::size_t n = cap + 10;
  for (std::size_t i = 0; i < n; ++i) {
    FabricEvent ev;
    ev.kind = FabricEventKind::kOpEnqueue;
    ev.level = FabricEventLevel::kInfo;
    ev.op_id = static_cast<std::uint64_t>(i + 1);
    ev.message = kFabricEventMsgTest;
    record_fabric_event(std::move(ev));
  }

  FabricEventSnapshot snap = fabric_events_snapshot(0, n);
  EXPECT_EQ(snap.next_seq, static_cast<std::uint64_t>(n));
  EXPECT_EQ(snap.base_seq, static_cast<std::uint64_t>(n - cap));
  EXPECT_EQ(snap.dropped_total, static_cast<std::uint64_t>(n - cap));

  ASSERT_EQ(snap.events.size(), cap);
  EXPECT_EQ(snap.events.front().seq, snap.base_seq);
  EXPECT_EQ(snap.events.back().seq, snap.next_seq - 1);
}

TEST(FabricEventsRing, WaitSemantics) {
  set_fabric_events_mode(FabricEventsMode::kOff);
  EXPECT_FALSE(fabric_events_wait_for_seq(1, 1ms));

  set_fabric_events_mode(FabricEventsMode::kBasic);
  reset_fabric_events_for_tests();

  // Timeout when no events arrive.
  EXPECT_FALSE(fabric_events_wait_for_seq(1, 5ms));

  reset_fabric_events_for_tests();

  std::thread t([]() {
    std::this_thread::sleep_for(10ms);
    FabricEvent ev;
    ev.kind = FabricEventKind::kOpEnqueue;
    ev.level = FabricEventLevel::kInfo;
    record_fabric_event(std::move(ev));
  });

  EXPECT_TRUE(fabric_events_wait_for_seq(1, 200ms));
  t.join();
}

TEST(FabricEventsRing, WaitReturnsFalseWhenModeTurnsOff) {
  set_fabric_events_mode(FabricEventsMode::kBasic);
  reset_fabric_events_for_tests();

  std::thread t([]() {
    std::this_thread::sleep_for(10ms);
    set_fabric_events_mode(FabricEventsMode::kOff);
  });

  // Need a target beyond the single mode-change event emitted during disable.
  EXPECT_FALSE(fabric_events_wait_for_seq(2, 200ms));
  t.join();
}

TEST(FabricEventsRing, MirrorsStatsWhenStateInitialized) {
  set_fabric_events_mode(FabricEventsMode::kBasic);
  vbt::cuda::fabric::fabric_state();
  reset_fabric_stats_for_tests();
  reset_fabric_events_for_tests();

  FabricEvent ev;
  ev.kind = FabricEventKind::kOpEnqueue;
  ev.level = FabricEventLevel::kInfo;
  record_fabric_event(std::move(ev));

  auto stats = vbt::cuda::fabric::fabric_stats_snapshot();
  EXPECT_EQ(stats.event_dropped_total, 0u);
  EXPECT_EQ(stats.event_queue_len_peak, 1u);
}

TEST(FabricEventsRing, InjectedFailureBumpsFailuresCounterAndDoesNotThrow) {
  set_fabric_events_mode(FabricEventsMode::kBasic);
  vbt::cuda::fabric::fabric_state();
  reset_fabric_stats_for_tests();
  reset_fabric_events_for_tests();

  debug_reset_fabric_event_record_injection_for_testing();
  debug_fail_fabric_event_record_on_n_for_testing(1);

  FabricEvent ev;
  ev.kind = FabricEventKind::kOpEnqueue;
  ev.level = FabricEventLevel::kInfo;
  ev.message = kFabricEventMsgTest;
  record_fabric_event(std::move(ev));

  auto stats1 = vbt::cuda::fabric::fabric_stats_snapshot();
  EXPECT_EQ(stats1.event_failures_total, 1u);

  FabricEventSnapshot snap1 = fabric_events_snapshot(/*min_seq=*/0, /*max_events=*/16);
  EXPECT_EQ(snap1.next_seq, 0u);
  EXPECT_TRUE(snap1.events.empty());

  debug_reset_fabric_event_record_injection_for_testing();

  FabricEvent ok;
  ok.kind = FabricEventKind::kOpEnqueue;
  ok.level = FabricEventLevel::kInfo;
  ok.message = kFabricEventMsgTest;
  record_fabric_event(std::move(ok));

  FabricEventSnapshot snap2 = fabric_events_snapshot(/*min_seq=*/0, /*max_events=*/16);
  ASSERT_EQ(snap2.events.size(), 1u);
  EXPECT_EQ(snap2.events[0].seq, 0u);
  EXPECT_STREQ(snap2.events[0].message, kFabricEventMsgTest);

  auto stats2 = vbt::cuda::fabric::fabric_stats_snapshot();
  EXPECT_EQ(stats2.event_failures_total, 1u);
}

TEST(FabricEventsRing, ResetClearsMirroredPeaksAndDropsButNotFailures) {
  set_fabric_events_mode(FabricEventsMode::kBasic);
  vbt::cuda::fabric::fabric_state();
  reset_fabric_stats_for_tests();
  reset_fabric_events_for_tests();

  // Force a record failure to set event_failures_total.
  debug_reset_fabric_event_record_injection_for_testing();
  debug_fail_fabric_event_record_on_n_for_testing(1);

  FabricEvent bad;
  bad.kind = FabricEventKind::kOpEnqueue;
  bad.level = FabricEventLevel::kInfo;
  record_fabric_event(std::move(bad));

  debug_reset_fabric_event_record_injection_for_testing();

  const std::size_t cap = fabric_events_snapshot(0, 0).capacity;
  ASSERT_GT(cap, 0u);

  const std::size_t n = cap + 5;
  for (std::size_t i = 0; i < n; ++i) {
    FabricEvent ev;
    ev.kind = FabricEventKind::kOpEnqueue;
    ev.level = FabricEventLevel::kInfo;
    record_fabric_event(std::move(ev));
  }

  auto before = vbt::cuda::fabric::fabric_stats_snapshot();
  EXPECT_EQ(before.event_failures_total, 1u);
  EXPECT_GT(before.event_queue_len_peak, 0u);
  EXPECT_GT(before.event_dropped_total, 0u);

  reset_fabric_events_for_tests();

  auto after = vbt::cuda::fabric::fabric_stats_snapshot();
  EXPECT_EQ(after.event_failures_total, 1u);
  EXPECT_EQ(after.event_queue_len_peak, 0u);
  EXPECT_EQ(after.event_dropped_total, 0u);
}

#endif  // VBT_INTERNAL_TESTS
