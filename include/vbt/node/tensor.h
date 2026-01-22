// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <node_api.h>

#include "vbt/core/tensor.h"
#include "vbt/node/external_memory.h"

namespace vbt::node {

// external-memory accounting metadata so we can keep V8 informed about
// Storage-backed allocations.
struct JsTensor {
  vbt::core::TensorImpl impl;                       // underlying tensor value
  std::size_t nbytes{0};                            // cached storage size
  std::shared_ptr<ExternalMemoryOwner> owner;       // shared external owner
  bool is_owner{false};                             // true if this wrapper
                                                    // performs adjust calls
};

// Wrap a heap-allocated JsTensor* into a JS object whose finalizer
// deletes the pointer. The object also exposes sizes(), dtype(),
// and device() methods implemented in C++.
napi_value WrapJsTensor(napi_env env, JsTensor* jt);

// Non-throwing variant used in async completions.
// Lifetime rules:
//  - On success: returns true, writes *out to a JS object, and attaches jt via
//    JsTensorFinalizer (caller MUST NOT delete jt).
//  - On failure: returns false and deletes jt; caller MUST NOT delete jt.
bool TryWrapJsTensor(napi_env env, JsTensor* jt, napi_value* out);

// Helper that constructs a JsTensor from a TensorImpl, attaches/reuses an
// ExternalMemoryOwner, performs +bytes accounting when needed, and wraps it
// into a JS object. Used by synchronous entrypoints.
napi_value WrapTensorImplAsJsTensor(napi_env env,
                                    vbt::core::TensorImpl impl);

// Non-throwing variant used by async completions that already own a
// TensorImpl result.
bool TryWrapTensorImplAsJsTensor(napi_env env,
                                 vbt::core::TensorImpl impl,
                                 napi_value* out);

// Validate and unwrap a tensor handle produced by this addon.
// On success: writes *out and returns true.
// On failure: sets a JS TypeError and returns false.
bool UnwrapJsTensor(napi_env env, napi_value value, JsTensor** out);

} // namespace vbt::node
