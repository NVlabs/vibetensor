// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <node_api.h>

#include "vbt/core/tensor.h"

namespace vbt {
namespace node {

// Owner for external memory accounting of a single Storage. Each owner
// corresponds to at most one underlying Storage per (is_cuda, device, data)
// key and is shared across all JsTensor aliases of that Storage.
struct ExternalMemoryOwner {
  std::size_t bytes{0};                 // clamped storage size in bytes
  std::atomic<bool> accounted{false};   // whether +bytes was applied to V8
  std::atomic<bool> released{false};    // whether owner has been released
};

// Global aggregate stats exposed to JS via _external_memory_stats().
struct ExternalMemStats {
  std::atomic<std::uint64_t> owners_alive{0};
  std::atomic<std::uint64_t> bytes_accounted{0};
};

extern ExternalMemStats g_ext_stats;

// Key used to deduplicate owners across aliases of the same Storage.
//
// Allocator docs guarantee pointer uniqueness per device but not across
// devices, so we key on (is_cuda, device index, data pointer).
struct ExternalOwnerKey {
  bool is_cuda;      // true for CUDA devices, false for CPU/other
  int32_t device;    // device index when is_cuda == true, -1 otherwise
  void* data;        // Storage::data(); nullptr for zero-size storages
};

struct ExternalOwnerKeyHash {
  std::size_t operator()(const ExternalOwnerKey& k) const noexcept;
};

struct ExternalOwnerKeyEq {
  bool operator()(const ExternalOwnerKey& a,
                  const ExternalOwnerKey& b) const noexcept;
};

struct ExternalOwnerRegistry {
  std::mutex mu;
  std::unordered_map<ExternalOwnerKey,
                     std::weak_ptr<ExternalMemoryOwner>,
                     ExternalOwnerKeyHash,
                     ExternalOwnerKeyEq>
      by_key;
};

extern ExternalOwnerRegistry g_ext_registry;

// Helper for safely computing storage size in bytes from a TensorImpl. This
// clamps extremely large values to INT64_MAX to avoid overflowing
// napi_adjust_external_memory's int64_t delta parameter.
std::size_t safe_storage_bytes(const vbt::core::TensorImpl& impl);

// Attach or reuse an ExternalMemoryOwner for the Storage underlying `impl`.
// Returns (owner, is_new_owner).
std::pair<std::shared_ptr<ExternalMemoryOwner>, bool>
AttachExternalOwnerForStorage(const vbt::core::TensorImpl& impl,
                              bool is_cuda_device,
                              int32_t device_index);

// Apply +bytes to V8 external memory accounting for a newly created owner.
// Must be called on the main JS thread; aborts the process via
// napi_fatal_error on failure.
void AccountExternalBytes(napi_env env, ExternalMemoryOwner& owner);

// Release an owner and, if bytes were previously accounted, apply -bytes to
// V8 external memory accounting. Must be called on the main JS thread;
// aborts the process on fatal accounting errors.
void ReleaseExternalOwner(napi_env env, ExternalMemoryOwner& owner);

// N-API entrypoint exposed as addon._external_memory_stats().
napi_value ExternalMemoryStatsNapi(napi_env env, napi_callback_info info);

}  // namespace node
}  // namespace vbt
