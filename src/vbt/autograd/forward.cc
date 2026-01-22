// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/forward.h"

#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"

#include <stdexcept>
#include <vector>

namespace vbt { namespace autograd {

thread_local std::int64_t g_current_forward_level = -1;
thread_local std::int64_t g_next_forward_level = 0;
thread_local std::int32_t g_backward_depth = 0;

namespace {

struct ForwardADContext {
  std::vector<vbt::core::intrusive_ptr<AutogradMeta>> metas_with_tangents;
};

thread_local ForwardADContext g_forward_ad_ctx;

} // anonymous

std::int64_t enter_forward_ad_level() {
  if (InferenceMode::is_enabled()) {
    throw std::runtime_error(
        "enter_forward_ad_level: cannot open a forward AD level inside inference_mode");
  }
  if (g_current_forward_level >= 0) {
    throw std::runtime_error(
        "enter_forward_ad_level: nested forward AD levels are not supported");
  }

  const std::int64_t level = g_next_forward_level++;
  g_current_forward_level = level;
  // Start with a clean slate for this level.
  g_forward_ad_ctx.metas_with_tangents.clear();
  return level;
}

void exit_forward_ad_level(std::int64_t level_id) {
  if (g_current_forward_level < 0 || g_current_forward_level != level_id) {
    throw std::runtime_error(
        "exit_forward_ad_level: mismatched forward AD level id");
  }

  // Lexical cleanup: drop tangents recorded for this level.
  for (auto& meta : g_forward_ad_ctx.metas_with_tangents) {
    if (!meta) continue;
    if (meta->forward_grad_.has_grad && meta->forward_grad_.level == level_id) {
      meta->forward_grad_.grad.reset();
      meta->forward_grad_.has_grad = false;
      meta->forward_grad_.level = -1;
    }
  }
  g_forward_ad_ctx.metas_with_tangents.clear();
  g_current_forward_level = -1;
}

bool is_in_backward() noexcept {
  return g_backward_depth > 0;
}

std::int64_t current_forward_ad_level() noexcept {
  return g_current_forward_level;
}

bool is_forward_ad_enabled() noexcept {
  return g_current_forward_level >= 0 &&
         !is_in_backward() &&
         !InferenceMode::is_enabled();
}

BackwardGuard::BackwardGuard() {
  ++g_backward_depth;
}

BackwardGuard::~BackwardGuard() {
  if (g_backward_depth <= 0) {
    // Defensive: keep the depth non-negative even if misuse sneaks in.
    g_backward_depth = 0;
    return;
  }
  --g_backward_depth;
}

void register_forward_grad_meta(AutogradMeta& meta) {
  // Only track metas when a level is active.
  if (g_current_forward_level < 0) {
    return;
  }

  // Best-effort deduplication: avoid unbounded growth when the same tensor
  // receives new tangents repeatedly.
  for (const auto& existing : g_forward_ad_ctx.metas_with_tangents) {
    if (existing.get() == &meta) {
      return;
    }
  }

  g_forward_ad_ctx.metas_with_tangents.emplace_back(&meta, /*add_ref=*/true);
}

}} // namespace vbt::autograd
