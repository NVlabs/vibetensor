// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <node_api.h>

namespace vbt {
namespace node {

// Synchronous snapshot of process-global Fabric stats.
//
// Exported from the addon as `fabricStatsSnapshot()`.
napi_value FabricStatsSnapshotNapi(napi_env env, napi_callback_info info);

// Synchronous snapshot of the process-global Fabric event ring.
//
// Exported from the addon as `fabricEventsSnapshot(minSeq, maxEvents)`.
napi_value FabricEventsSnapshotNapi(napi_env env, napi_callback_info info);

}  // namespace node
}  // namespace vbt
