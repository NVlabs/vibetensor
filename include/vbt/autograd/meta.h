// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "vbt/core/intrusive_ptr.h"
#include "vbt/autograd/node.h"

namespace vbt { namespace core { class TensorImpl; }}

namespace vbt { namespace autograd {

struct TensorHook;

struct ForwardGradSlot {
  std::unique_ptr<vbt::core::TensorImpl> grad;  // owned tangent buffer
  bool has_grad{false};                         // true iff grad is set
  int64_t level{-1};                            // forward level id when grad is valid
};

struct AutogradMeta : public vbt::core::IntrusiveRefcounted {
  bool requires_grad{false};
  bool is_leaf{true};
  int64_t output_nr{0};
  vbt::core::intrusive_ptr<Node> grad_fn; // may be null in V1
  struct ViewMeta {
    bool is_view{false};

    // Owning handle to the *gradient root* TensorImpl for differentiable views.
    // Null for non-differentiable views; in that case gradient_root(view) must
    // return &view and must not touch any base tensor.
    std::shared_ptr<const vbt::core::TensorImpl> base_root;

    vbt::core::intrusive_ptr<Node> weak_grad_fn;
  } view;
  // Leaf accumulation slot: owned copy of gradient when present
  // Thread-safety: grad_ptr/grad_has and hooks are protected by
  // grad_mutex.
  std::unique_ptr<vbt::core::TensorImpl> grad_ptr;
  bool grad_has{false};
  mutable std::mutex grad_mutex;
  ForwardGradSlot forward_grad_;
  std::vector<vbt::core::intrusive_ptr<TensorHook>> hooks;
  ~AutogradMeta();
};

// Returns pointer to AutogradMeta; creates one when create_if_missing==true (and flag enabled).
AutogradMeta* get_autograd_meta(vbt::core::TensorImpl& t, bool create_if_missing);
const AutogradMeta* get_autograd_meta(const vbt::core::TensorImpl& t) noexcept;

bool requires_grad(const vbt::core::TensorImpl& t) noexcept;
void set_requires_grad(vbt::core::TensorImpl& t, bool v);

// View helpers (V1 minimal)
bool is_view(const vbt::core::TensorImpl& t) noexcept;
bool is_leaf(const vbt::core::TensorImpl& t) noexcept;
void as_view(const vbt::core::TensorImpl& base, vbt::core::TensorImpl& out);

// Grad-fn introspection helper
vbt::core::intrusive_ptr<Node>
get_grad_fn(const vbt::core::TensorImpl& t) noexcept;

// Resolve gradient root for a potentially-view tensor. Always returns non-null.
const vbt::core::TensorImpl*
gradient_root(const vbt::core::TensorImpl& t) noexcept;

// Resolve the reverse-mode gradient edge for `t`.
//
// Rule order:
//  1) If !requires_grad(t): return a null edge.
//  2) If t has a local grad_fn: return Edge{grad_fn, output_nr(t)}.
//  3) Otherwise fall back to gradient_root(t):
//     - If the root has a grad_fn: use it.
//     - Else create/reuse an AccumulateGrad sink for the root leaf.
Edge resolve_edge_for_tensor(
    const vbt::core::TensorImpl& t,
    std::unordered_map<const AutogradMeta*, vbt::core::intrusive_ptr<Node>>& sinks);

// Detach helpers
vbt::core::TensorImpl detach_copy(const vbt::core::TensorImpl& t);
void detach_inplace(vbt::core::TensorImpl& t);

// Forward-mode helpers (internal; used by boxed fallback and _C.autograd).
const vbt::core::TensorImpl*
get_forward_grad_view(const vbt::core::TensorImpl& t,
                      int64_t level) noexcept;

vbt::core::TensorImpl
get_forward_grad_copy(const vbt::core::TensorImpl& t,
                      int64_t level);

void set_forward_grad(vbt::core::TensorImpl& t,
                      const vbt::core::TensorImpl& tangent,
                      int64_t level);

void clear_forward_grad(vbt::core::TensorImpl& t) noexcept;

bool has_forward_grad(const vbt::core::TensorImpl& t,
                      int64_t level) noexcept;

bool has_any_forward_grad(const vbt::core::TensorImpl& t) noexcept;

// Tensor hook interface used by AccumulateGrad and Python bindings.
struct TensorHook : public vbt::core::IntrusiveRefcounted {
  virtual ~TensorHook() = default;
  virtual void call(const vbt::core::TensorImpl& grad) = 0;

  void set_removed(bool v = true) noexcept {
    removed_.store(v, std::memory_order_relaxed);
  }
  bool is_removed() const noexcept {
    return removed_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<bool> removed_{false};
};

// Register and retrieve hooks attached to a leaf tensor.
void register_leaf_hook(vbt::core::TensorImpl& leaf,
                        vbt::core::intrusive_ptr<TensorHook> hook);

std::vector<vbt::core::intrusive_ptr<TensorHook>>
get_leaf_hooks(const AutogradMeta& meta);

std::vector<vbt::core::intrusive_ptr<TensorHook>>
get_leaf_hooks(const vbt::core::TensorImpl& leaf);

// Clear stored gradient buffer without affecting requires_grad/is_leaf/view flags.
void clear_tensor_grad(vbt::core::TensorImpl& t);

// In-place wiring: update history on the mutatee (view: install on base; else on self)
void rebase_history(vbt::core::TensorImpl& t, const vbt::core::intrusive_ptr<Node>& node);

}} // namespace vbt::autograd
