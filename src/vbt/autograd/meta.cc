// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/tensor.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/core/type_promotion.h"
#include "vbt/autograd/forward.h"

#include <cstdlib>
#include <stdexcept>

namespace vbt { namespace autograd {
using vbt::core::TensorImpl;

#if !VBT_WITH_AUTOGRAD
AutogradMeta* get_autograd_meta(TensorImpl&, bool) { return nullptr; }
const AutogradMeta* get_autograd_meta(const TensorImpl&) noexcept { return nullptr; }
bool requires_grad(const TensorImpl&) noexcept { return false; }
void set_requires_grad(TensorImpl&, bool) {}
bool is_view(const TensorImpl&) noexcept { return false; }
bool is_leaf(const TensorImpl&) noexcept { return true; }
void as_view(const TensorImpl&, TensorImpl&) {}

vbt::core::intrusive_ptr<Node> get_grad_fn(const TensorImpl&) noexcept {
  return vbt::core::intrusive_ptr<Node>();
}

const TensorImpl* gradient_root(const TensorImpl& t) noexcept {
  return &t;
}

Edge resolve_edge_for_tensor(
    const TensorImpl&,
    std::unordered_map<const AutogradMeta*, vbt::core::intrusive_ptr<Node>>&) {
  return Edge{};
}

vbt::core::TensorImpl detach_copy(const TensorImpl& t) {
  return t;
}

void detach_inplace(TensorImpl&) {}

// Forward-mode stubs when autograd is disabled.
const TensorImpl* get_forward_grad_view(const TensorImpl&, int64_t) noexcept {
  return nullptr;
}

vbt::core::TensorImpl get_forward_grad_copy(const TensorImpl&, int64_t) {
  return TensorImpl();
}

void set_forward_grad(TensorImpl&, const TensorImpl&, int64_t) {}

void clear_forward_grad(TensorImpl&) noexcept {}

bool has_forward_grad(const TensorImpl&, int64_t) noexcept {
  return false;
}

bool has_any_forward_grad(const TensorImpl&) noexcept {
  return false;
}

void register_leaf_hook(TensorImpl&, vbt::core::intrusive_ptr<TensorHook>) {}

std::vector<vbt::core::intrusive_ptr<TensorHook>>
get_leaf_hooks(const AutogradMeta&) {
  return {};
}

std::vector<vbt::core::intrusive_ptr<TensorHook>>
get_leaf_hooks(const TensorImpl&) {
  return {};
}

void clear_tensor_grad(TensorImpl&) {}

void rebase_history(TensorImpl&, const vbt::core::intrusive_ptr<Node>&) {}
AutogradMeta::~AutogradMeta() = default;
#else

extern thread_local bool tls_grad_enabled;
extern thread_local bool tls_inference_mode_enabled;

namespace {

inline bool complex_autograd_enabled_from_env() noexcept {
  const char* complex_raw = std::getenv("VBT_ENABLE_COMPLEX");
  if (!(complex_raw && complex_raw[0] == '1' && complex_raw[1] == '\0')) {
    return false;
  }
  const char* raw = std::getenv("VBT_ENABLE_COMPLEX_AUTOGRAD");
  return raw && raw[0] == '1' && raw[1] == '\0';
}

static constexpr const char* kErrComplexAutogradDisabled =
    "complex autograd is disabled; set VBT_ENABLE_COMPLEX_AUTOGRAD=1";

inline void clear_forward_grad_slot(AutogradMeta& m) noexcept {
  m.forward_grad_.grad.reset();
  m.forward_grad_.has_grad = false;
  m.forward_grad_.level = -1;
}

} // anonymous

AutogradMeta* get_autograd_meta(TensorImpl& t, bool create_if_missing) {
  auto* p = t._autograd().get();
  if (!p && create_if_missing) {
    t._autograd() = vbt::core::make_intrusive<AutogradMeta>();
    p = t._autograd().get();
  }
  return p;
}
const AutogradMeta* get_autograd_meta(const TensorImpl& t) noexcept { return t._autograd().get(); }
bool requires_grad(const TensorImpl& t) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  return m ? (m->requires_grad || static_cast<bool>(m->grad_fn)) : false;
}
void set_requires_grad(TensorImpl& t, bool v) {
  if (v && vbt::core::is_complex(t.dtype()) &&
      !complex_autograd_enabled_from_env()) {
    throw std::runtime_error(kErrComplexAutogradDisabled);
  }

  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/v);
  if (m) m->requires_grad = v;
}

bool is_view(const TensorImpl& t) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  return m ? m->view.is_view : false;
}

bool is_leaf(const TensorImpl& t) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  if (!m) return true; // no meta → leaf by default
  return m->is_leaf;
}

void as_view(const TensorImpl& base, TensorImpl& out) {
  // Guard against identity view ops that return `self` by value (e.g.
  // squeeze(dim) when no squeeze occurs). In those cases base/out can share the
  // same AutogradMeta; treating it as a view would create a self-reference
  // cycle and can break gradient_root termination.
  if (&base == &out) {
    return;
  }

  AutogradMeta* mo = get_autograd_meta(out, /*create_if_missing=*/true);
  if (!mo) return;

  const AutogradMeta* mb = get_autograd_meta(base);
  if (mb && mo == mb) {
    return;
  }

  const bool graph_enabled = tls_grad_enabled && !tls_inference_mode_enabled;
  const bool diff = requires_grad(base) && graph_enabled;

  // If this is a differentiable view, compute the root handle first (so that if
  // allocation throws, we leave `out` unchanged).
  std::shared_ptr<const TensorImpl> root_handle;
  vbt::core::intrusive_ptr<Node> weak_grad_fn;
  if (diff) {
    if (mb && mb->view.is_view && mb->view.base_root) {
      root_handle = mb->view.base_root;
    } else {
      root_handle = std::make_shared<TensorImpl>(base);
    }
    if (mb) {
      weak_grad_fn = mb->grad_fn;
    }
  }

  // Clear any stale history on out before marking as a view.
  mo->grad_fn.reset();
  mo->output_nr = 0;
  mo->requires_grad = false;
  mo->is_leaf = true;

  mo->view.is_view = true;
  mo->view.base_root = std::move(root_handle);
  mo->view.weak_grad_fn = std::move(weak_grad_fn);

  // Non-differentiable view: keep is_view=true but do not retain the base.
  if (!diff) {
    return;
  }

  mo->requires_grad = true;
  mo->is_leaf = false;
}

vbt::core::intrusive_ptr<Node> get_grad_fn(const TensorImpl& t) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  if (!m || !m->grad_fn) {
    return vbt::core::intrusive_ptr<Node>();
  }
  return m->grad_fn;
}

const TensorImpl* gradient_root(const TensorImpl& t) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  if (!m) return &t;
  if (!m->view.is_view) return &t;
  if (!m->view.base_root) return &t;
  return m->view.base_root.get();
}

Edge resolve_edge_for_tensor(
    const TensorImpl& t,
    std::unordered_map<const AutogradMeta*, vbt::core::intrusive_ptr<Node>>& sinks) {
  if (!requires_grad(t)) {
    return Edge{};
  }

  const AutogradMeta* mt = get_autograd_meta(t);
  if (!mt) {
    throw std::logic_error(
        "resolve_edge_for_tensor: tensor requires grad but has no AutogradMeta");
  }

  if (auto fn = get_grad_fn(t)) {
    return Edge{fn, static_cast<std::uint32_t>(mt->output_nr)};
  }

  const TensorImpl* root = gradient_root(t);
  if (!root) {
    throw std::logic_error(
        "resolve_edge_for_tensor: internal gradient_root returned null");
  }

  const AutogradMeta* mr = get_autograd_meta(*root);
  if (!mr) {
    throw std::logic_error(
        "resolve_edge_for_tensor: internal gradient root has no AutogradMeta");
  }

  if (auto fn = get_grad_fn(*root)) {
    return Edge{fn, static_cast<std::uint32_t>(mr->output_nr)};
  }

  auto it = sinks.find(mr);
  if (it != sinks.end()) {
    return Edge{it->second, 0u};
  }

  auto acc_impl = vbt::core::make_intrusive<AccumulateGrad>(
      const_cast<AutogradMeta*>(mr));
  // Best-effort: tag CUDA leaf metadata for correct stream accounting.
  _tag_accumulategrad_cuda_leaf(*acc_impl, *root);

  vbt::core::intrusive_ptr<Node> sink(acc_impl.get(), /*add_ref=*/true);
  sinks.emplace(mr, sink);
  return Edge{sink, 0u};
}

vbt::core::TensorImpl detach_copy(const TensorImpl& t) {
  TensorImpl out = t;
  // Clear autograd metadata on the copy so it is fully detached.
  out._autograd().reset();
  return out;
}

void detach_inplace(TensorImpl& t) {
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/false);
  if (!m) return;
  m->grad_fn.reset();
  m->view.is_view = false;
  m->view.base_root.reset();
  m->view.weak_grad_fn.reset();
  m->requires_grad = false;
  m->is_leaf = true;
  m->output_nr = 0;
  // Intentionally leave grad_ptr/grad_has untouched for inspection, but clear any forward tangent.
  clear_forward_grad_slot(*m);
}

const TensorImpl* get_forward_grad_view(const TensorImpl& t,
                                        int64_t level) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  if (!m) return nullptr;
  const ForwardGradSlot& slot = m->forward_grad_;
  if (!slot.has_grad || !slot.grad || slot.level != level) {
    return nullptr;
  }
  return slot.grad.get();
}

vbt::core::TensorImpl
get_forward_grad_copy(const TensorImpl& t,
                      int64_t level) {
  const TensorImpl* view = get_forward_grad_view(t, level);
  if (!view) {
    return vbt::core::TensorImpl();
  }
  vbt::core::TensorImpl out = *view;
  // Best-effort: ensure the copy does not require grad.
  try {
    set_requires_grad(out, false);
  } catch (...) {
  }
  return out;
}

void set_forward_grad(TensorImpl& t,
                      const TensorImpl& tangent,
                      int64_t level) {
  if (level < 0) {
    throw std::runtime_error(
        "set_forward_grad: level id must be >= 0");
  }

  // Domain: CPU float32, matching shape/device.
  if (!(t.device().type == kDLCPU && tangent.device().type == kDLCPU)) {
    throw std::runtime_error(
        "set_forward_grad: primal and tangent must be CPU tensors");
  }
  if (t.dtype() != vbt::core::ScalarType::Float32 ||
      tangent.dtype() != vbt::core::ScalarType::Float32) {
    throw std::runtime_error(
        "set_forward_grad: primal and tangent must be Float32");
  }
  if (t.sizes() != tangent.sizes()) {
    throw std::runtime_error(
        "set_forward_grad: primal and tangent must have matching sizes");
  }

  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/true);
  if (!m) {
    throw std::runtime_error(
        "set_forward_grad: failed to create AutogradMeta");
  }

  // Clone tangent into an owned TensorImpl. This keeps the stored tangent
  // independent of later in-place mutations on the input.
  vbt::core::TensorImpl stored = vbt::core::clone_cpu(tangent);
  try {
    set_requires_grad(stored, false);
  } catch (...) {
  }

  m->forward_grad_.grad = std::make_unique<TensorImpl>(std::move(stored));
  m->forward_grad_.has_grad = true;
  m->forward_grad_.level = level;

  register_forward_grad_meta(*m);
}

void clear_forward_grad(TensorImpl& t) noexcept {
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/false);
  if (!m) return;
  clear_forward_grad_slot(*m);
}

bool has_forward_grad(const TensorImpl& t, int64_t level) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  if (!m) return false;
  const ForwardGradSlot& slot = m->forward_grad_;
  return slot.has_grad && slot.grad && slot.level == level;
}

bool has_any_forward_grad(const TensorImpl& t) noexcept {
  const AutogradMeta* m = get_autograd_meta(t);
  if (!m) return false;
  const ForwardGradSlot& slot = m->forward_grad_;
  return slot.has_grad && slot.grad != nullptr;
}

void register_leaf_hook(TensorImpl& leaf, vbt::core::intrusive_ptr<TensorHook> hook) {
  if (!hook) return;
  AutogradMeta* m = get_autograd_meta(leaf, /*create_if_missing=*/true);
  if (!m) return;
  std::lock_guard<std::mutex> lk(m->grad_mutex);
  m->hooks.push_back(std::move(hook));
}

std::vector<vbt::core::intrusive_ptr<TensorHook>>
get_leaf_hooks(const AutogradMeta& meta) {
  std::vector<vbt::core::intrusive_ptr<TensorHook>> out;
  {
    std::lock_guard<std::mutex> lk(meta.grad_mutex);
    out = meta.hooks;
  }
  return out;
}

std::vector<vbt::core::intrusive_ptr<TensorHook>>
get_leaf_hooks(const TensorImpl& leaf) {
  std::vector<vbt::core::intrusive_ptr<TensorHook>> out;
  const AutogradMeta* m = get_autograd_meta(leaf);
  if (!m) return out;
  {
    std::lock_guard<std::mutex> lk(m->grad_mutex);
    out = m->hooks;
  }
  return out;
}

void clear_tensor_grad(TensorImpl& t) {
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/false);
  if (!m) return;
  std::lock_guard<std::mutex> lk(m->grad_mutex);
  m->grad_ptr.reset();
  m->grad_has = false;
}

void rebase_history(TensorImpl& t, const vbt::core::intrusive_ptr<Node>& node) {
  if (!node) return;
  AutogradMeta* m = get_autograd_meta(t, /*create_if_missing=*/true);
  if (!m) return;
  // Minimal V1: install on self only
  m->grad_fn = node;
  m->is_leaf = false;
  m->output_nr = 0;
}

AutogradMeta::~AutogradMeta() = default;
#endif

}} // namespace vbt::autograd
