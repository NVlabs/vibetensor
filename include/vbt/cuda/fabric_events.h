// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace vbt { namespace cuda { namespace fabric {

enum class FabricEventsMode : std::uint8_t {
  kOff   = 0,
  kBasic = 1,
};

enum class FabricEventKind : std::uint8_t {
  // Op lifecycle
  kOpEnqueue = 0,
  kOpComplete,
  kOpFallback,
  kOpError,

  // Config changes
  kModeChanged,
  kEventLifetimeToggled,
  kEventsModeChanged,
};

enum class FabricEventLevel : std::uint8_t { kDebug = 0, kInfo, kWarn, kError };

using FabricSeqNo = std::uint64_t;
using FabricOpId  = std::uint64_t;

// Append-only allowlist of stable message tokens.
// Producers must pass one of these pointers (or nullptr). Never pass a string literal.
extern const char kFabricEventMsgTest[];
extern const char kFabricEventMsgEventsModeChanged[];
extern const char kFabricEventMsgModeChanged[];
extern const char kFabricEventMsgEventLifetimeEnabled[];
extern const char kFabricEventMsgEventLifetimeDisabled[];
extern const char kFabricEventMsgOpEnqueue[];
extern const char kFabricEventMsgOpComplete[];
extern const char kFabricEventMsgOpFallback[];
extern const char kFabricEventMsgOpError[];

struct FabricEvent {
  FabricSeqNo       seq{0};
  std::uint64_t     t_ns{0};

  int               primary_device{-1};
  int               other_device{-1};

  FabricEventKind   kind{FabricEventKind::kOpEnqueue};
  FabricEventLevel  level{FabricEventLevel::kInfo};

  FabricOpId        op_id{0};
  std::uint64_t     numel{0};
  std::uint64_t     bytes{0};

  std::uint32_t     reason_raw{0};
  const char*       message{nullptr};
};

struct FabricEventSnapshot {
  std::uint64_t            base_seq{0};
  std::uint64_t            next_seq{0};
  std::uint64_t            dropped_total{0};
  std::size_t              capacity{0};
  std::vector<FabricEvent> events;
};

FabricEventsMode get_fabric_events_mode() noexcept;
void             set_fabric_events_mode(FabricEventsMode mode) noexcept;

// Internal: true if set_fabric_events_mode was called at least once in this process.
bool fabric_events_mode_was_explicitly_set() noexcept;

inline bool fabric_events_enabled() noexcept {
  return get_fabric_events_mode() != FabricEventsMode::kOff;
}

// Best-effort recording of a Fabric diagnostic event.
//
// This function must never throw and must never impact Fabric correctness.
void record_fabric_event(FabricEvent&& ev) noexcept;

// Snapshot the current Fabric event ring.
FabricEventSnapshot fabric_events_snapshot(std::uint64_t min_seq,
                                          std::size_t max_events);

// Best-effort host-level wait until the event ring reaches target_seq.
//
// Returns false immediately if events mode is OFF at entry or becomes OFF while waiting.
bool fabric_events_wait_for_seq(std::uint64_t target_seq,
                                std::chrono::milliseconds timeout);

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS

// Test-only helper: reset the ring to an empty epoch.
void reset_fabric_events_for_tests();

// Test-only failure injection: make record_fabric_event_unchecked fail on the
// Nth record attempt after this call (N is 1-based).
//
// N <= 0 disables injection. Calling this resets the internal record-attempt
// counter to 0.
void debug_fail_fabric_event_record_on_n_for_testing(int n) noexcept;

// Reset the internal record failure injection state (disables injection and
// resets the counter).
void debug_reset_fabric_event_record_injection_for_testing() noexcept;

#endif

}}} // namespace vbt::cuda::fabric
