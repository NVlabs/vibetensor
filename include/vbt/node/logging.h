// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>

#include <node_api.h>

namespace vbt::node {

enum class LogLevel : std::uint8_t { kDebug = 0, kInfo = 1, kWarn = 2, kError = 3 };

enum class LogCategory : std::uint16_t {
  kDispatcher,
  kCudaRuntime,
  kCudaAllocator,
  kDlpack,
  kH2D,
  kD2H,
  kExternalMemory,
};

struct LogEntryNative {
  std::uint64_t ts_millis;  // coarse timestamp for ordering
  LogLevel level;
  LogCategory category;
  std::string message;      // sanitized
  std::map<std::string, std::string> data;  // small, sanitized key/value map
};

struct LogState {
  std::mutex mu;
  bool enabled = false;          // any logging enabled at all
  LogLevel min_level = LogLevel::kInfo;
  std::uint32_t category_mask = 0xFFFFu;  // all categories by default
  std::size_t capacity = 1024;   // clamped to [64, 8192]
  std::deque<LogEntryNative> queue;
  std::uint64_t dropped_total = 0;  // total entries dropped due to overflow
};

extern LogState g_log_state;

// Fast-path helper: lock-free flag used by producers to skip formatting and
// locking when logging is disabled.
bool IsLoggingEnabled() noexcept;

// Configure logging state from JS. Must be called on the main thread.
void SetLoggingEnabledFromJs(napi_env env,
                             bool enabled,
                             LogLevel min_level,
                             std::uint32_t category_mask,
                             std::size_t capacity);

// Producer helper used from any thread. When logging is disabled this returns
// immediately. When enabled it pushes a sanitized entry into the bounded queue.
void LogIfEnabled(LogLevel level,
                  LogCategory category,
                  std::string message,
                  std::map<std::string, std::string> data = {});

// N-API entrypoint used by JS overlay to enable/disable logging. Signature:
//   _setLoggingEnabled(enabled: boolean, level: string, mask: number): void
napi_value SetLoggingEnabledNapi(napi_env env, napi_callback_info info);

// N-API entrypoint that drains up to maxEntries log entries into a JS array of
// { ts, level, category, message, data } objects.
napi_value DrainLogsNapi(napi_env env, napi_callback_info info);

// Initialize logging state from VBT_NODE_LOG_LEVEL at addon init. This
// enables native-only logging without requiring a JS sink.
void InitLoggingFromEnv(napi_env env);

} // namespace vbt::node
