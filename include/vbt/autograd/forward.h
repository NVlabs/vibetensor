// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace vbt { namespace autograd {

struct AutogradMeta;

// Thread-local forward-mode and backward state.
// - g_current_forward_level: id of the active forward-AD level in this thread
//   (-1 when no level is active).
// - g_next_forward_level: monotonically increasing id generator used to
//   assign distinct ids to successive levels.
// - g_backward_depth: counts nested invocations of run_backward; forward-mode
//   is considered disabled while this is > 0.
extern thread_local std::int64_t g_current_forward_level;
extern thread_local std::int64_t g_next_forward_level;
extern thread_local std::int32_t g_backward_depth;

// Open and close a forward-AD level for the current thread.
//
// enter_forward_ad_level()
//   - Throws if inference-mode is enabled.
//   - Otherwise assigns and returns a new level id and marks it active.
//
// exit_forward_ad_level(level_id)
//   - Throws if there is no active level or the id mismatches.
//   - Clears any tangents recorded for that level and marks it inactive.
std::int64_t enter_forward_ad_level();
void         exit_forward_ad_level(std::int64_t level_id);

// Introspection helpers used by Python bindings and boxed fallback.
bool         is_in_backward() noexcept;
std::int64_t current_forward_ad_level() noexcept;

// Return true when forward-mode should participate in boxed fallback.
// This is the *only* gate that tangent propagation logic consults.
bool         is_forward_ad_enabled() noexcept;

// RAII guard that tracks nested calls into run_backward(). When at least one
// guard is live on the current thread, is_in_backward() returns true and
// is_forward_ad_enabled() returns false.
struct BackwardGuard {
  BackwardGuard();
  ~BackwardGuard();

  BackwardGuard(const BackwardGuard&) = delete;
  BackwardGuard& operator=(const BackwardGuard&) = delete;
};

// Internal hook used by set_forward_grad() to register AutogradMeta instances
// that carry tangents in the current level so they can be cleared lexically
// on exit.
void register_forward_grad_meta(AutogradMeta& meta);

}} // namespace vbt::autograd
