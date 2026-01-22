// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>
#include <vector>
#include <string>
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/saved_variable.h"
#include "vbt/core/indexing.h"

namespace vbt { namespace autograd {

// RAII guard to skip autograd fallback within Dispatcher::callBoxed while in scope.
struct SkipAutogradGuard {
  bool prev_;
  SkipAutogradGuard() noexcept
    : prev_(vbt::dispatch::Dispatcher::tls_skip_autograd_) {
    vbt::dispatch::Dispatcher::tls_skip_autograd_ = true;
  }
  ~SkipAutogradGuard() noexcept {
    vbt::dispatch::Dispatcher::tls_skip_autograd_ = prev_;
  }
  SkipAutogradGuard(const SkipAutogradGuard&) = delete;
  SkipAutogradGuard& operator=(const SkipAutogradGuard&) = delete;
};

static_assert(noexcept(SkipAutogradGuard()), "SkipAutogradGuard ctor must be noexcept");
static_assert(std::is_nothrow_destructible_v<SkipAutogradGuard>, "SkipAutogradGuard dtor must be noexcept");

// GradMode TLS and guards.
extern thread_local bool tls_grad_enabled;
extern thread_local bool tls_inference_mode_enabled;
struct NoGradGuard {
  bool prev_;
  NoGradGuard() noexcept : prev_(tls_grad_enabled) { tls_grad_enabled = false; }
  ~NoGradGuard() noexcept { tls_grad_enabled = prev_; }
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;
};
struct EnableGradGuard {
  bool prev_;
  EnableGradGuard() noexcept : prev_(tls_grad_enabled) { tls_grad_enabled = true; }
  ~EnableGradGuard() noexcept { tls_grad_enabled = prev_; }
  EnableGradGuard(const EnableGradGuard&) = delete;
  EnableGradGuard& operator=(const EnableGradGuard&) = delete;
};
struct GradMode {
  static bool is_enabled() noexcept { return tls_grad_enabled; }
  static void set_enabled(bool v) noexcept { tls_grad_enabled = v; }
};
struct InferenceMode {
  static bool is_enabled() noexcept { return tls_inference_mode_enabled; }
  static void set_enabled(bool v) noexcept { tls_inference_mode_enabled = v; }
};
static_assert(noexcept(NoGradGuard()), "NoGradGuard ctor must be noexcept");
static_assert(std::is_nothrow_destructible_v<NoGradGuard>, "NoGradGuard dtor must be noexcept");
static_assert(noexcept(EnableGradGuard()), "EnableGradGuard ctor must be noexcept");
static_assert(std::is_nothrow_destructible_v<EnableGradGuard>, "EnableGradGuard dtor must be noexcept");

// Utilities
bool any_requires_grad(const vbt::dispatch::BoxedStack& s, uint8_t in_arity) noexcept;

// Forward autograd boxed-only fallback with ctx (OperatorEntry* via ctx)
void autograd_fallback_ctx(void* ctx, vbt::dispatch::BoxedStack& s);

// Registration helper results
struct RegisterResults {
  bool relu{false};
  bool add{false};
  bool mul{false};
  bool index{false};
  bool embedding{false};
  bool any() const noexcept { return relu || add || mul || index || embedding; }
};

// Install boxed-only autograd fallbacks for vt::{relu,add,mul,index,embedding}
RegisterResults register_autograd_fallbacks();

// Build an in-place backward node for the given op name using snapshots.
vbt::core::intrusive_ptr<Node> build_inplace_backward_node(const char* op_fqname,
                                                           const std::vector<SavedVariable>& snaps);

// Autograd indexing v2 flag (design/ad/ad_index.md)
//
// Default: OFF (false). Tests may override this within a process.
bool autograd_indexing_v2_enabled() noexcept;
void set_autograd_indexing_v2_enabled_for_tests(bool enabled) noexcept;

// Default: OFF (false) even when autograd_indexing_v2 is enabled.
// Optional micro-flag for negative-stride basic-index view backward.
bool autograd_indexing_v2_negstride_enabled() noexcept;
void set_autograd_indexing_v2_negstride_enabled_for_tests(bool enabled) noexcept;

// Create a backward node for basic indexing views.
//
// Negative-stride views are only supported when
// autograd_indexing_v2_negstride_enabled() is true.
//
// The returned node expects a single incoming gradient (for the view) and
// produces a single gradient (for the base).
vbt::core::intrusive_ptr<Node> make_basic_index_view_backward_node(
    const vbt::core::TensorImpl& base,
    vbt::core::indexing::IndexSpec spec);

// Create a backward node for basic indexing setitem (overwrite semantics).
//
// The returned node expects a single incoming gradient (for the updated tensor)
// and produces gradients for (self, value).
vbt::core::intrusive_ptr<Node> make_basic_index_put_backward_node(
    const vbt::core::TensorImpl& self,
    const vbt::core::TensorImpl& value,
    vbt::core::indexing::IndexSpec spec);

// Create a backward node for advanced index_put_ / vt::index_put.
//
// The returned node expects a single incoming gradient (for the updated tensor)
// and produces gradients for (self, value).
vbt::core::intrusive_ptr<Node> make_index_put_backward_node(
    const vbt::core::TensorImpl& self,
    const vbt::core::TensorImpl& value,
    vbt::core::indexing::IndexSpec spec,
    bool accumulate);

}} // namespace vbt::autograd
