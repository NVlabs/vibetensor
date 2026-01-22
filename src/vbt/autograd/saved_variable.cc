// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/autograd/saved_variable.h"

#include "vbt/autograd/meta.h"
#include "vbt/autograd/wrapper.h"
#include "vbt/autograd/detail/stats_internal.h"

#include <stdexcept>

namespace {
constexpr const char* kSvNotInit = "SavedVariable: not initialized";
constexpr const char* kSvVerMismatch =
    "SavedVariable unpack: version mismatch (in-place modification)";
}

namespace vbt { namespace autograd {

using vbt::core::TensorImpl;
namespace nb = nanobind;

// Use an indirection to avoid running nanobind::object destructors
// after the Python interpreter has shut down; this intentionally
// leaks a small amount of per-thread state at process exit.
static thread_local SavedTensorHookState* g_saved_hooks_tls = nullptr;

SavedTensorHookState& saved_tensor_hooks_tls() {
  if (!g_saved_hooks_tls) {
    g_saved_hooks_tls = new SavedTensorHookState();
  }
  return *g_saved_hooks_tls;
}

// Helpers for detecting tensors in hook payloads. These are intentionally
// best-effort and tolerate missing modules.
static bool is_vibetensor_tensor(const nb::handle& obj) {
  try {
    return nb::isinstance<TensorImpl>(obj);
  } catch (...) {
    return false;
  }
}

static bool is_torch_tensor(const nb::handle& obj) {
  try {
    nb::object torch_mod = nb::module_::import_("torch");
    nb::handle tensor_type = torch_mod.attr("Tensor");
    return nb::isinstance(obj, tensor_type);
  } catch (...) {
    return false;
  }
}

SavedVariable::SavedVariable(const TensorImpl& t)
  : saved_(t),
    saved_version_(t.version()),
    initialized_(true) {
  SavedTensorHookState& st = saved_tensor_hooks_tls();
  if (st.disabled || st.stack.empty()) {
    return; // hooks disabled or not installed
  }

  const SavedTensorHookPair& top = st.stack.back();
  pack_hook_ = top.pack;
  unpack_hook_ = top.unpack;
  hook_trusted_builtin_ = top.trusted_builtin;

  if (!pack_hook_.is_valid()) {
    return;
  }

  // already CPU float32, so no additional casting is required.
  TensorImpl arg = detach_copy(t);

  try {
    NoGradGuard ng;            // ensure hooks run with autograd disabled
    nb::gil_scoped_acquire gil; // acquire GIL before calling into Python

    nb::object packed = pack_hook_(nb::cast(arg));

    if (!hook_trusted_builtin_) {
      // Enforce that user hooks do not keep live tensors.
      if (is_vibetensor_tensor(packed) || is_torch_tensor(packed)) {
        _stats_saved_tensors_hook_violations(1);
        throw std::runtime_error(
            "saved_tensors_hooks: user pack_hook must not return a tensor");
      }
    }

    hook_packed_ = std::move(packed);
    has_hook_payload_ = true;
    _stats_saved_tensors_packed(1);
  } catch (...) {
    // Do not leave a partially-initialized payload on error.
    has_hook_payload_ = false;
    hook_packed_.reset();
    throw;
  }
}

TensorImpl SavedVariable::unpack() const {
  if (!initialized_) {
    throw std::logic_error(kSvNotInit);
  }
  if (saved_.version() != saved_version_) {
    throw std::runtime_error(kSvVerMismatch);
  }

  TensorImpl result = saved_;

  if (has_hook_payload_ && unpack_hook_.is_valid()) {
    try {
      NoGradGuard ng;
      nb::gil_scoped_acquire gil;
      unpack_hook_(hook_packed_);  // ignore return value
      _stats_saved_tensors_unpacked(1);
    } catch (...) {
      // Exceptions propagate to the caller of the op that triggered unpack.
      throw;
    }
  }

  return result;
}

}} // namespace vbt::autograd
