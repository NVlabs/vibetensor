// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "vbt/plugin/vbt_plugin.h"

namespace vbt {
namespace core {
class TensorImpl;
}  // namespace core
}  // namespace vbt

namespace vbt {
namespace dispatch {
namespace plugin {

// Minimal loader API for plugins (no unloading/hardening).
vt_status load_library(const char* path) noexcept;
const char* get_last_error() noexcept;
void set_last_error(const char* msg) noexcept;

// Diagnostics helpers used by tests.
std::vector<std::string> loaded_libraries();
bool is_library_loaded(const std::string& path);

namespace detail {

// Helper used by TI-backed plugin helpers to obtain a TensorImpl reference
// from an opaque vt_tensor handle. Throws std::invalid_argument on null or
// malformed handles.
::vbt::core::TensorImpl& require_tensor_impl(vt_tensor h,
                                             const char* arg_name);

// Test-only helper: create a borrowed vt_tensor handle for the given
// TensorImpl. The returned handle does not own the TensorImpl and is valid
// only for the duration of the call that uses it.
vt_tensor make_borrowed_handle_for_tests(const ::vbt::core::TensorImpl& t);

// Test-only helper: destroy a borrowed handle created with
// make_borrowed_handle_for_tests. Does not delete the underlying TensorImpl.
void destroy_borrowed_handle_for_tests(vt_tensor h);

// Helper used by TI-backed plugin helpers to write to the plugin TLS
// error channel. Thin wrapper over plugin::set_last_error.
inline void set_last_error_helper(const char* msg) {
  ::vbt::dispatch::plugin::set_last_error(msg);
}

}  // namespace detail

}  // namespace plugin
}  // namespace dispatch
}  // namespace vbt
