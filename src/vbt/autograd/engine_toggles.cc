// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/engine_toggles.h"
#include <atomic>

namespace vbt { namespace autograd {

static std::atomic<AutogradDeviceMode> g_device_mode{AutogradDeviceMode::SingleDevice};
static std::atomic<bool> g_multithreading_enabled{false};
static bool g_view_replay_enabled = false;
static std::atomic<bool> g_streaming_backwards{false};

AutogradDeviceMode get_device_mode() noexcept {
  return g_device_mode.load(std::memory_order_relaxed);
}

void set_device_mode(AutogradDeviceMode mode) noexcept {
  g_device_mode.store(mode, std::memory_order_relaxed);
}

bool is_multithreading_enabled() noexcept {
  return g_multithreading_enabled.load(std::memory_order_relaxed);
}

void set_multithreading_enabled(bool v) noexcept {
  g_multithreading_enabled.store(v, std::memory_order_relaxed);
}

bool is_view_replay_enabled() noexcept { return g_view_replay_enabled; }

void set_view_replay_enabled(bool v) noexcept {
  g_view_replay_enabled = v;
}

bool is_streaming_backwards_enabled() noexcept {
#if VBT_WITH_CUDA
  return g_streaming_backwards.load(std::memory_order_relaxed);
#else
  // On CPU-only builds, the CUDA autograd toggle is a stub that always
  // reports "disabled"; the engine never consults it.
  return false;
#endif
}

void set_streaming_backwards_enabled(bool enabled) noexcept {
#if VBT_WITH_CUDA
  g_streaming_backwards.store(enabled, std::memory_order_relaxed);
#else
  (void)enabled;  // no-op when CUDA support is not compiled in
#endif
}

// engine; any future use requires a dedicated design and perf review.

}} // namespace vbt::autograd
