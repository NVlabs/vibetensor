// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>

#include <node_api.h>

#include "vbt/core/tensor.h"

namespace vbt {
namespace node {

// Error kinds surfaced by callOp*; mapped to JS Error/TypeError
// with stable .code tags (see dispatcher_errors.h).
enum class CallOpErrorKind {
  None,
  InvalidArg,
  Runtime,
  UnknownOp,
  BadAlloc,
  AsyncWorkFailure,
  Unknown,
};

// Parsed inputs for a dispatcher call (sync or async).
struct ParsedCallInputs {
  std::string op_name;
  std::vector<vbt::core::TensorImpl> inputs;
};

// Async job state used by CallOp/CallOpNoOverride workers.
struct CallOpJob {
  // Immutable C++ state (worker-safe)
  std::string op_name;                       // e.g., "vt::add"
  std::vector<vbt::core::TensorImpl> inputs; // copies; hold Storage refs

  // Result state (set on worker, consumed on main thread)
  bool success{false};
  vbt::core::TensorImpl output;              // single-output v1
  CallOpErrorKind error_kind{CallOpErrorKind::None};
  std::string error_message;                 // set when !success

  // JS lifetime tokens (main-thread only)
  napi_deferred deferred{nullptr};           // Promise handle
  napi_async_work work{nullptr};
  std::vector<napi_ref> js_arg_refs;         // keep JS tensor handles alive

  // Diagnostics
  std::uint64_t job_id{0};
};

// Parse and validate (name, args) pair for dispatcher calls. On failure,
// sets a JS exception and returns std::nullopt.
std::optional<ParsedCallInputs>
ParseCallOpInputs(napi_env env, napi_value js_name, napi_value js_args);

// Global inflight counter and cap shared across all async Node jobs
// (dispatcher, CUDA runtime waits, future DLPack helpers).
extern std::atomic<std::uint32_t> g_inflight_ops;
extern std::uint32_t g_max_inflight_ops;

struct AsyncCategoryStatsNative {
  std::atomic<std::uint64_t> started{0};
  std::atomic<std::uint64_t> completed{0};
  std::atomic<std::uint64_t> failed{0};
  std::atomic<std::uint64_t> bytes{0};  // H2D/D2H only
};

struct DebugAsyncStatsNative {
  std::atomic<std::uint32_t> peak_inflight{0};

  std::atomic<std::uint64_t> dispatcher_total{0};
  std::atomic<std::uint64_t> dispatcher_failed{0};

  AsyncCategoryStatsNative h2d;
  AsyncCategoryStatsNative d2h;
};

extern DebugAsyncStatsNative g_async_stats;

// Inflight tracking helpers used by tests and overlay diagnostics.
std::uint32_t max_inflight_ops();
std::uint32_t current_inflight_ops();

// Update global async peak-inflight metric shared across dispatcher,
// DLPack imports, and CUDA helpers.
void UpdateAsyncPeakInflight(std::uint32_t current);

// Node-API entrypoints exported from the addon.
napi_value CallOpSync(napi_env env, napi_callback_info info);
napi_value CallOp(napi_env env, napi_callback_info info);
napi_value CallOpNoOverride(napi_env env, napi_callback_info info);

// Debug-only stats surfaces (not re-exported from public JS API).
napi_value GetDispatcherStats(napi_env env, napi_callback_info info);
napi_value GetAsyncStats(napi_env env, napi_callback_info info);

} // namespace node
} // namespace vbt
