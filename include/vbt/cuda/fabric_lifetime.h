// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/core/storage.h"
#include "vbt/cuda/stream.h"

namespace vbt { namespace cuda { namespace fabric {

using DeviceIndex = vbt::cuda::DeviceIndex;
using StreamId    = std::uint64_t;

struct FabricStorageSets {
  std::vector<vbt::core::StoragePtr> primary_storages; // storages on Dp
  std::vector<vbt::core::StoragePtr> remote_storages;  // storages on Dr
};

struct ProducerStreamRef {
  DeviceIndex device{-1};
  Stream      stream{Stream::UNCHECKED, 0u, 0};
};

struct PrimaryRemoteFencePlan {
  DeviceIndex primary_device{-1};
  DeviceIndex remote_device{-1};

  std::vector<ProducerStreamRef> primary_producers; // on Dp
  std::vector<ProducerStreamRef> remote_producers;  // on Dr
};

enum class FabricFailureKind {
  kNone = 0,
  kMetadataFailure,
  kResourceFailure,
};

struct FabricRuntimeError {
  bool        is_metadata_failure{false};
  bool        is_resource_failure{false};
  std::string message;

  std::runtime_error as_exception() const { return std::runtime_error(message); }
};

struct FencePlanBuildResult {
  PrimaryRemoteFencePlan             plan;
  FabricFailureKind                  kind{FabricFailureKind::kNone};
  std::optional<FabricRuntimeError>  error;

  bool ok() const noexcept { return kind == FabricFailureKind::kNone; }
};

struct FenceExecutionResult {
  FabricFailureKind                  kind{FabricFailureKind::kNone};
  std::optional<FabricRuntimeError>  error;

  bool ok() const noexcept { return kind == FabricFailureKind::kNone; }
};

// Acquire the per-device Fabric compute stream (Sp).
//
// Notes:
// - This is a long-lived per-device stream from the non-blocking pool.
// - Throws std::runtime_error on CUDA stream creation errors.
Stream get_fabric_compute_stream(DeviceIndex primary_device);

// Acquire the per-device Fabric proxy stream on a remote device (Sproxy_r).
//
// Notes:
// - This is a long-lived per-device non-blocking stream that never runs
//   kernels; it is used only for cross-device waits and Allocator::record_stream
//   on remote storages participating in Fabric ops.
// - Throws std::runtime_error on CUDA stream creation errors.
Stream get_fabric_proxy_stream(DeviceIndex remote_device);

// Build a deduplicated producer fence plan from Storage-level producer metadata.
FencePlanBuildResult build_primary_remote_fence_plan(
    const FabricStorageSets& storages,
    DeviceIndex              primary_device,
    DeviceIndex              remote_device);

// Execute producer -> compute fencing by recording pooled events on each producer
// stream and waiting on Sp.
FenceExecutionResult execute_primary_remote_fence_plan(
    const PrimaryRemoteFencePlan& plan,
    Stream                        compute_stream);

#if defined(VBT_INTERNAL_TESTS) && VBT_INTERNAL_TESTS
struct FabricFenceDebugCounters {
  std::uint64_t num_producer_events_primary{0};
  std::uint64_t num_producer_events_remote{0};
  std::uint64_t num_event_record_calls{0};
  std::uint64_t num_stream_wait_calls{0};
};

FabricFenceDebugCounters debug_get_fabric_fence_counters_for_testing() noexcept;
void debug_reset_fabric_fence_counters_for_testing() noexcept;

// Test hooks: if enabled, fail the Nth producer event operation.
// N is 1-based; 0 disables the hook.
void debug_fail_fence_event_pool_get_on_n_for_testing(int n) noexcept;
void debug_fail_fence_event_record_on_n_for_testing(int n) noexcept;
void debug_fail_fence_stream_wait_on_n_for_testing(int n) noexcept;
#endif  // VBT_INTERNAL_TESTS

}}} // namespace vbt::cuda::fabric
