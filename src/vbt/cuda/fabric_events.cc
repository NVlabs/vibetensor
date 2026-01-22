// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/cuda/fabric_events.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "vbt/cuda/fabric_state.h"
#include "vbt/logging/logging.h"

namespace vbt { namespace cuda { namespace fabric {

const char kFabricEventMsgTest[]                 = "test";
const char kFabricEventMsgEventsModeChanged[]    = "events_mode_changed";
const char kFabricEventMsgModeChanged[]          = "mode_changed";
const char kFabricEventMsgEventLifetimeEnabled[] = "event_lifetime_enabled";
const char kFabricEventMsgEventLifetimeDisabled[] = "event_lifetime_disabled";
const char kFabricEventMsgOpEnqueue[]            = "op_enqueue";
const char kFabricEventMsgOpComplete[]           = "op_complete";
const char kFabricEventMsgOpFallback[]           = "op_fallback";
const char kFabricEventMsgOpError[]              = "op_error";

namespace {

constexpr std::size_t kFabricEventsMinCapacity = 64;
constexpr std::size_t kFabricEventsMaxCapacity = 16384;
constexpr std::size_t kFabricEventsDefaultCapacity = 4096;

std::atomic<FabricEventsMode> g_mode{FabricEventsMode::kOff};
std::atomic<bool> g_mode_explicit{false};
std::atomic<bool> g_record_failure_warned{false};

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
std::atomic<int> g_fail_record_on_n{0};
std::atomic<int> g_record_call_index{0};

static inline bool should_fail_(std::atomic<int>& which,
                                std::atomic<int>& counter) noexcept {
  const int n = which.load(std::memory_order_relaxed);
  if (n <= 0) return false;
  const int cur = counter.fetch_add(1, std::memory_order_relaxed) + 1;
  return cur == n;
}
#endif

struct FabricEventRing {
  std::mutex              mtx;
  std::condition_variable cv;

  std::vector<FabricEvent> buf;
  std::uint64_t           base_seq{0};
  std::uint64_t           next_seq{0};
  std::uint64_t           dropped_total{0};
};

FabricEventRing& ring_singleton() {
  static FabricEventRing r;
  return r;
}

static inline std::uint64_t steady_time_ns() noexcept {
  using namespace std::chrono;
  return static_cast<std::uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

static inline void atomic_update_max(std::atomic<std::uint64_t>& a,
                                     std::uint64_t v) noexcept {
  std::uint64_t cur = a.load(std::memory_order_relaxed);
  while (v > cur &&
         !a.compare_exchange_weak(cur, v,
                                  std::memory_order_relaxed,
                                  std::memory_order_relaxed)) {
    // Retry until we update the max.
  }
}

static inline std::size_t clamp_capacity(std::size_t cap) noexcept {
  if (cap < kFabricEventsMinCapacity) return kFabricEventsMinCapacity;
  if (cap > kFabricEventsMaxCapacity) return kFabricEventsMaxCapacity;
  return cap;
}

static inline void bump_event_failures_counter() noexcept {
  if (FabricState* fs = try_get_fabric_state_if_initialized()) {
    fs->stats.event_failures_total.fetch_add(1, std::memory_order_relaxed);
  }
}

static void record_fabric_event_unchecked(FabricEvent&& ev) noexcept {
  FabricEventRing& r = ring_singleton();

  try {
    std::unique_lock<std::mutex> lock(r.mtx);

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
    if (should_fail_(g_fail_record_on_n, g_record_call_index)) {
      throw std::runtime_error("injected fabric_events record failure");
    }
#endif

    if (r.buf.empty()) {
      r.buf.resize(clamp_capacity(kFabricEventsDefaultCapacity));
    }

    // Assign seq and timestamp at record time.
    ev.seq = r.next_seq;
    ev.t_ns = steady_time_ns();
    ++r.next_seq;

    const std::size_t cap = r.buf.size();
    if (cap == 0) {
      // Should never happen (capacity is clamped), but keep this path no-throw.
      return;
    }

    const std::size_t idx = static_cast<std::size_t>(ev.seq % cap);
    r.buf[idx] = ev;

    std::uint64_t occ = r.next_seq - r.base_seq;
    if (occ > static_cast<std::uint64_t>(cap)) {
      const std::uint64_t overflow = occ - static_cast<std::uint64_t>(cap);
      r.base_seq += overflow;
      r.dropped_total += overflow;
      occ = static_cast<std::uint64_t>(cap);
    }

    // Best-effort mirroring into Fabric stats.
    if (FabricState* fs = try_get_fabric_state_if_initialized()) {
      atomic_update_max(fs->stats.event_queue_len_peak, occ);
      fs->stats.event_dropped_total.store(r.dropped_total, std::memory_order_relaxed);
    }

    lock.unlock();
    r.cv.notify_all();
  } catch (...) {
    bump_event_failures_counter();

    bool expected = false;
    if (g_record_failure_warned.compare_exchange_strong(
            expected, true,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      try {
        VBT_LOG(WARNING)
            << "[Fabric] fabric_events: failed to record event (suppressed further warnings)";
      } catch (...) {
        // Best-effort only.
      }
    }
  }
}

static inline std::uint64_t add_u64_sat(std::uint64_t a, std::uint64_t b) noexcept {
  if (b > (std::numeric_limits<std::uint64_t>::max() - a)) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return a + b;
}

}  // namespace

FabricEventsMode get_fabric_events_mode() noexcept {
  return g_mode.load(std::memory_order_relaxed);
}

bool fabric_events_mode_was_explicitly_set() noexcept {
  return g_mode_explicit.load(std::memory_order_relaxed);
}

void set_fabric_events_mode(FabricEventsMode mode) noexcept {
  g_mode_explicit.store(true, std::memory_order_relaxed);
  const FabricEventsMode prev = g_mode.load(std::memory_order_relaxed);
  if (prev == mode) {
    // Still notify to ensure any waiting threads can re-check mode.
    ring_singleton().cv.notify_all();
    return;
  }

  // Emit a mode-change event while the old mode is still active.
  if (prev != FabricEventsMode::kOff) {
    FabricEvent ev;
    ev.kind = FabricEventKind::kEventsModeChanged;
    ev.level = FabricEventLevel::kInfo;
    ev.message = kFabricEventMsgEventsModeChanged;
    record_fabric_event_unchecked(std::move(ev));
  }

  g_mode.store(mode, std::memory_order_relaxed);

  if (FabricState* fs = try_get_fabric_state_if_initialized()) {
    fs->config.events_mode_raw.store(static_cast<std::uint8_t>(mode),
                                     std::memory_order_relaxed);
  }

  // Wake any waiters so they can observe the new mode.
  ring_singleton().cv.notify_all();

  // Emit a mode-change event after enabling.
  if (mode != FabricEventsMode::kOff) {
    FabricEvent ev;
    ev.kind = FabricEventKind::kEventsModeChanged;
    ev.level = FabricEventLevel::kInfo;
    ev.message = kFabricEventMsgEventsModeChanged;
    record_fabric_event_unchecked(std::move(ev));
  }
}

void record_fabric_event(FabricEvent&& ev) noexcept {
  if (!fabric_events_enabled()) return;
  record_fabric_event_unchecked(std::move(ev));
}

FabricEventSnapshot fabric_events_snapshot(std::uint64_t min_seq,
                                          std::size_t max_events) {
  FabricEventRing& r = ring_singleton();
  std::unique_lock<std::mutex> lock(r.mtx);

  if (r.buf.empty()) {
    r.buf.resize(clamp_capacity(kFabricEventsDefaultCapacity));
  }

  FabricEventSnapshot snap;
  snap.base_seq = r.base_seq;
  snap.next_seq = r.next_seq;
  snap.dropped_total = r.dropped_total;
  snap.capacity = r.buf.size();

  const std::uint64_t base = r.base_seq;
  const std::uint64_t next = r.next_seq;

  std::uint64_t start = min_seq;
  if (start < base) start = base;
  if (start > next) start = next;

  std::uint64_t end = start;
  if (max_events > 0) {
    end = std::min<std::uint64_t>(next,
                                 add_u64_sat(start, static_cast<std::uint64_t>(max_events)));
  }

  const std::size_t cap = r.buf.size();
  snap.events.reserve(static_cast<std::size_t>(end - start));
  for (std::uint64_t seq = start; seq < end; ++seq) {
    snap.events.push_back(r.buf[static_cast<std::size_t>(seq % cap)]);
  }

  return snap;
}

bool fabric_events_wait_for_seq(std::uint64_t target_seq,
                                std::chrono::milliseconds timeout) {
  if (get_fabric_events_mode() == FabricEventsMode::kOff) {
    return false;
  }

  FabricEventRing& r = ring_singleton();
  std::unique_lock<std::mutex> lock(r.mtx);

  if (r.buf.empty()) {
    r.buf.resize(clamp_capacity(kFabricEventsDefaultCapacity));
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;

  while (true) {
    if (get_fabric_events_mode() == FabricEventsMode::kOff) {
      return false;
    }

    const std::uint64_t base = r.base_seq;
    const std::uint64_t next = r.next_seq;

    if (target_seq <= base || target_seq <= next) {
      return true;
    }

    if (timeout.count() == 0) {
      return false;
    }

    if (r.cv.wait_until(lock, deadline) == std::cv_status::timeout) {
      // One last check after timeout.
      if (get_fabric_events_mode() == FabricEventsMode::kOff) {
        return false;
      }
      const std::uint64_t base2 = r.base_seq;
      const std::uint64_t next2 = r.next_seq;
      return (target_seq <= base2 || target_seq <= next2);
    }
  }
}

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS

void reset_fabric_events_for_tests() {
  FabricEventRing& r = ring_singleton();
  std::unique_lock<std::mutex> lock(r.mtx);

  if (r.buf.empty()) {
    r.buf.resize(clamp_capacity(kFabricEventsDefaultCapacity));
  }

  r.base_seq = 0;
  r.next_seq = 0;
  r.dropped_total = 0;
  for (auto& ev : r.buf) {
    ev = FabricEvent{};
  }

  if (FabricState* fs = try_get_fabric_state_if_initialized()) {
    fs->stats.event_queue_len_peak.store(0, std::memory_order_relaxed);
    fs->stats.event_dropped_total.store(0, std::memory_order_relaxed);
  }

  lock.unlock();
  r.cv.notify_all();
}

void debug_fail_fabric_event_record_on_n_for_testing(int n) noexcept {
  g_record_call_index.store(0, std::memory_order_relaxed);
  g_fail_record_on_n.store(n, std::memory_order_relaxed);
}

void debug_reset_fabric_event_record_injection_for_testing() noexcept {
  g_fail_record_on_n.store(0, std::memory_order_relaxed);
  g_record_call_index.store(0, std::memory_order_relaxed);
}

#endif  // VBT_INTERNAL_TESTS

}}}  // namespace vbt::cuda::fabric
