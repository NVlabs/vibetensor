// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

#include <node_api.h>

struct DLManagedTensor;  // fwd from DLPack

namespace vbt {
namespace node {

// Lifecycle states for a DLPack capsule owned by the Node addon.
//  - kNew:      freshly created by ToDlpack; never passed to importer.
//  - kImported: logically handed off to an in-flight import job.
//  - kConsumed: successfully imported or consumed on an error path.
//  - kErrored:  terminal error; deleter already run.
enum class DlpackState { kNew, kImported, kConsumed, kErrored };

// Per-capsule owner that enforces one-shot semantics and mediates calls to
// the provider's deleter. The mutex guards access to state/mt so that the
// finalizer, main thread, and worker threads can coordinate safely.
struct NodeDlpackOwner {
  DlpackState      state{DlpackState::kNew};
  DLManagedTensor* mt{nullptr};
  std::mutex       mu;
};

// Import options passed from JS for DLPack imports.
//  - `copy` controls alias vs copy for CUDA capsules.
//  - `device` is a non-negative logical device index (0 == CPU).
struct ImportOpts {
  bool has_copy{false};
  bool copy{false};              // last copy value; default false (overridden for CUDA)

  bool has_target_device{false};
  int  target_device_id{0};      // logical device index from opts.device (0 == CPU)

  bool has_expected_device{false};
  int  expected_device_type{0};  // reserved for provider metadata (future use)
  int  expected_device_id{0};
};

// Synchronous exporter: validates a Tensor handle and returns a DlpackCapsule
// JS object that wraps a NodeDlpackOwner* via napi_wrap.
napi_value ToDlpack(napi_env env, napi_callback_info info);

// Asynchronous importer: accepts a DlpackCapsule and returns a Promise<Tensor>
// that resolves to a new Tensor on success or rejects with a mapped Error on
napi_value FromDlpackCapsuleAsync(napi_env env, napi_callback_info info);

// Attaches dlpack-related exports (toDlpack, fromDlpackCapsule) to the addon
// exports object. Returns nullptr on error with a pending JS exception.
napi_value RegisterDlpackBindings(napi_env env, napi_value exports);

}  // namespace node
}  // namespace vbt
