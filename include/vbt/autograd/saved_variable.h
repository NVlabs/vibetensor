// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <nanobind/nanobind.h>

#include "vbt/core/tensor.h"

namespace vbt { namespace autograd {

// Saved-tensor hook pair stored on a thread-local stack.
struct SavedTensorHookPair {
  nanobind::object pack;   // Python callable
  nanobind::object unpack; // Python callable
  bool trusted_builtin{false};
};

struct SavedTensorHookState {
  std::vector<SavedTensorHookPair> stack;
  bool disabled{false};
  std::string disabled_error;
};

// Thread-local access to saved-tensor hook state.
SavedTensorHookState& saved_tensor_hooks_tls();

class SavedVariable {
 public:
  SavedVariable() = default;
  explicit SavedVariable(const vbt::core::TensorImpl& t);

  bool is_initialized() const noexcept { return initialized_; }
  vbt::core::TensorImpl unpack() const; // throws on mismatch/not-initialized

 private:
  vbt::core::TensorImpl saved_{}; // shallow copy; shares VersionCounter
  int64_t saved_version_{0};
  bool initialized_{false};

  // Saved-tensor hook payload.
  bool has_hook_payload_{false};
  nanobind::object hook_packed_;
  nanobind::object pack_hook_;
  nanobind::object unpack_hook_;
  bool hook_trusted_builtin_{false};
};

}} // namespace vbt::autograd
