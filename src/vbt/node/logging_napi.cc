// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/logging.h"

#include <node_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>

#include "vbt/node/cuda_napi.h"
#include "vbt/node/errors.h"
#include "vbt/node/util.h"

namespace vbt::node {

LogState g_log_state{};

namespace {

std::atomic<bool> g_logging_enabled{false};

using Clock = std::chrono::steady_clock;
static const auto kStartTime = Clock::now();

std::uint64_t NowMillis() {
  auto delta = Clock::now() - kStartTime;
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(delta).count());
}

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

LogLevel ParseLogLevel(const std::string& in) {
  std::string s = ToLower(in);
  if (s == "debug") return LogLevel::kDebug;
  if (s == "info") return LogLevel::kInfo;
  if (s == "warn" || s == "warning") return LogLevel::kWarn;
  if (s == "error" || s == "err") return LogLevel::kError;
  return LogLevel::kInfo;
}

const char* LogLevelToString(LogLevel lvl) {
  switch (lvl) {
    case LogLevel::kDebug: return "debug";
    case LogLevel::kInfo:  return "info";
    case LogLevel::kWarn:  return "warn";
    case LogLevel::kError: return "error";
  }
  return "info";
}

const char* LogCategoryToString(LogCategory cat) {
  switch (cat) {
    case LogCategory::kDispatcher:     return "dispatcher";
    case LogCategory::kCudaRuntime:    return "cuda-runtime";
    case LogCategory::kCudaAllocator:  return "cuda-allocator";
    case LogCategory::kDlpack:         return "dlpack";
    case LogCategory::kH2D:            return "h2d";
    case LogCategory::kD2H:            return "d2h";
    case LogCategory::kExternalMemory: return "external-memory";
  }
  return "dispatcher";
}

std::uint32_t CategoryBit(LogCategory cat) {
  return 1u << static_cast<std::uint16_t>(cat);
}

// Local copy of the addon instance-data helpers used in cuda_napi.cc.

static CudaRuntimeState* GetCudaRuntimeStateLocal(napi_env env) {
  AddonData* data = nullptr;
  napi_status st =
      napi_get_instance_data(env, reinterpret_cast<void**>(&data));
  if (st != napi_ok || !data) {
    return nullptr;
  }
  return &data->cuda_rt;
}

static bool IsOnMainThreadLocal(napi_env env) {
  CudaRuntimeState* rt = GetCudaRuntimeStateLocal(env);
  if (!rt) return false;
  return IsOnMainThreadFromState(*rt, std::this_thread::get_id());
}

static bool EnsureOnMainThreadLocal(napi_env env, const char* fn_name) {
  if (!IsOnMainThreadLocal(env)) {
    std::string msg = std::string(fn_name ? fn_name : "logging") +
                      ": must be called on the main VibeTensor JS thread";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return false;
  }
  return true;
}

std::string SanitizeString(const std::string& in) {
  std::string out = SanitizePointers(in);
  out = SanitizePaths(out);
  return out;
}

}  // namespace

bool IsLoggingEnabled() noexcept {
  return g_logging_enabled.load(std::memory_order_relaxed);
}

void InitLoggingFromEnv(napi_env env) {
  const char* env_level = std::getenv("VBT_NODE_LOG_LEVEL");
  if (!env_level || *env_level == '\0') {
    return;
  }

  std::string raw(env_level);
  std::string lowered = ToLower(raw);

  LogLevel level = LogLevel::kInfo;
  bool valid = false;
  if (lowered == "debug") {
    level = LogLevel::kDebug;
    valid = true;
  } else if (lowered == "info") {
    level = LogLevel::kInfo;
    valid = true;
  } else if (lowered == "warn" || lowered == "warning") {
    level = LogLevel::kWarn;
    valid = true;
  } else if (lowered == "error" || lowered == "err") {
    level = LogLevel::kError;
    valid = true;
  }

  if (!valid) {
    return;
  }

  // Use default capacity and category mask; capacity == 0 preserves the
  // existing queue configuration.
  SetLoggingEnabledFromJs(env,
                          /*enabled=*/true,
                          level,
                          /*category_mask=*/0xFFFFu,
                          /*capacity=*/0);
}

void SetLoggingEnabledFromJs(napi_env /*env*/,
                             bool enabled,
                             LogLevel min_level,
                             std::uint32_t category_mask,
                             std::size_t capacity) {
  // Use existing capacity when 0 is passed.
  if (capacity == 0) {
    capacity = g_log_state.capacity;
  }
  if (capacity < 64) capacity = 64;
  if (capacity > 8192) capacity = 8192;

  std::lock_guard<std::mutex> lock(g_log_state.mu);
  g_log_state.enabled = enabled;
  g_log_state.min_level = min_level;
  g_log_state.category_mask = category_mask;
  g_log_state.capacity = capacity;

  if (!enabled) {
    g_log_state.queue.clear();
  } else if (g_log_state.queue.size() > g_log_state.capacity) {
    std::size_t over = g_log_state.queue.size() - g_log_state.capacity;
    while (over-- > 0 && !g_log_state.queue.empty()) {
      g_log_state.queue.pop_front();
      ++g_log_state.dropped_total;
    }
  }

  g_logging_enabled.store(enabled, std::memory_order_relaxed);
}

void LogIfEnabled(LogLevel level,
                  LogCategory category,
                  std::string message,
                  std::map<std::string, std::string> data) {
  if (!IsLoggingEnabled()) return;

  const std::uint32_t bit = CategoryBit(category);

  // Sanitize eagerly; this is cheap relative to CUDA work and only happens
  // when logging is globally enabled.
  LogEntryNative entry;
  entry.ts_millis = NowMillis();
  entry.level = level;
  entry.category = category;
  entry.message = SanitizeString(message);

  for (auto& kv : data) {
    std::string key = SanitizeString(kv.first);
    std::string val = SanitizeString(kv.second);
    entry.data.emplace(std::move(key), std::move(val));
  }

  std::lock_guard<std::mutex> lock(g_log_state.mu);
  if (!g_log_state.enabled) return;
  if (static_cast<int>(level) < static_cast<int>(g_log_state.min_level)) {
    return;
  }
  if ((g_log_state.category_mask & bit) == 0u) {
    return;
  }

  if (g_log_state.queue.size() >= g_log_state.capacity) {
    if (!g_log_state.queue.empty()) {
      g_log_state.queue.pop_front();
      ++g_log_state.dropped_total;
    }
  }

  g_log_state.queue.push_back(std::move(entry));
}

napi_value SetLoggingEnabledNapi(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThreadLocal(env, "_setLoggingEnabled")) return nullptr;

  size_t argc = 3;
  napi_value args[3];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "SetLoggingEnabled/get_cb_info");

  if (argc < 2) {
    ThrowErrorWithCode(
        env,
        "_setLoggingEnabled(enabled, level, categories?) expects at least 2 arguments",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  // enabled: boolean
  napi_valuetype t_enabled;
  st = napi_typeof(env, args[0], &t_enabled);
  if (!CheckNapiOkImpl(env, st, "SetLoggingEnabled/enabled_typeof")) {
    return nullptr;
  }
  if (t_enabled != napi_boolean) {
    ThrowErrorWithCode(env,
                       "_setLoggingEnabled: enabled must be a boolean",
                       "EINVAL",
                       /*type_error=*/true);
    return nullptr;
  }
  bool enabled = false;
  st = napi_get_value_bool(env, args[0], &enabled);
  if (!CheckNapiOkImpl(env, st, "SetLoggingEnabled/enabled_get")) {
    return nullptr;
  }

  // level: string
  napi_valuetype t_level;
  st = napi_typeof(env, args[1], &t_level);
  if (!CheckNapiOkImpl(env, st, "SetLoggingEnabled/level_typeof")) {
    return nullptr;
  }
  if (t_level != napi_string) {
    ThrowErrorWithCode(env,
                       "_setLoggingEnabled: level must be a string",
                       "EINVAL",
                       /*type_error=*/true);
    return nullptr;
  }

  bool ok = false;
  std::string level_str = detail::GetStringUtf8(
      env, args[1], "SetLoggingEnabled/level_string", &ok);
  if (!ok) return nullptr;  // error already thrown
  LogLevel level = ParseLogLevel(level_str);

  // categories?: string[]
  std::uint32_t mask = 0xFFFFu;  // default: all categories
  if (argc >= 3) {
    napi_value js_cats = args[2];
    napi_valuetype t_cats;
    st = napi_typeof(env, js_cats, &t_cats);
    if (!CheckNapiOkImpl(env, st, "SetLoggingEnabled/categories_typeof")) {
      return nullptr;
    }
    if (t_cats != napi_undefined && t_cats != napi_null) {
      bool is_array = false;
      st = napi_is_array(env, js_cats, &is_array);
      if (!CheckNapiOkImpl(env, st, "SetLoggingEnabled/categories_is_array")) {
        return nullptr;
      }
      if (!is_array) {
        ThrowErrorWithCode(env,
                           "_setLoggingEnabled: categories must be an array of strings",
                           "EINVAL",
                           /*type_error=*/true);
        return nullptr;
      }

      mask = 0u;
      uint32_t length = 0;
      st = napi_get_array_length(env, js_cats, &length);
      if (!CheckNapiOkImpl(env, st,
                           "SetLoggingEnabled/categories_get_length")) {
        return nullptr;
      }
      for (uint32_t i = 0; i < length; ++i) {
        napi_value elem;
        st = napi_get_element(env, js_cats, i, &elem);
        if (!CheckNapiOkImpl(env, st,
                             "SetLoggingEnabled/categories_get_element")) {
          return nullptr;
        }

        napi_valuetype t_elem;
        st = napi_typeof(env, elem, &t_elem);
        if (!CheckNapiOkImpl(env, st,
                             "SetLoggingEnabled/categories_elem_typeof")) {
          return nullptr;
        }
        if (t_elem != napi_string) {
          ThrowErrorWithCode(env,
                             "_setLoggingEnabled: category names must be strings",
                             "EINVAL",
                             /*type_error=*/true);
          return nullptr;
        }

        bool ok_cat = false;
        std::string cat_str = detail::GetStringUtf8(
            env, elem, "SetLoggingEnabled/category_string", &ok_cat);
        if (!ok_cat) return nullptr;
        std::string lc = ToLower(cat_str);

        LogCategory cat;
        if (lc == "dispatcher") {
          cat = LogCategory::kDispatcher;
        } else if (lc == "cuda-runtime") {
          cat = LogCategory::kCudaRuntime;
        } else if (lc == "cuda-allocator") {
          cat = LogCategory::kCudaAllocator;
        } else if (lc == "dlpack") {
          cat = LogCategory::kDlpack;
        } else if (lc == "h2d") {
          cat = LogCategory::kH2D;
        } else if (lc == "d2h") {
          cat = LogCategory::kD2H;
        } else if (lc == "external-memory") {
          cat = LogCategory::kExternalMemory;
        } else {
          ThrowErrorWithCode(env,
                             "_setLoggingEnabled: unknown log category",
                             "EINVAL",
                             /*type_error=*/true);
          return nullptr;
        }
        mask |= CategoryBit(cat);
      }

      if (mask == 0u) {
        // If all categories were filtered out, fall back to "all".
        mask = 0xFFFFu;
      }
    }
  }

  SetLoggingEnabledFromJs(env, enabled, level, mask, /*capacity=*/0);

  napi_value undef;
  st = napi_get_undefined(env, &undef);
  CHECK_NAPI_OK(env, st, "SetLoggingEnabled/get_undefined");
  return undef;
}

napi_value DrainLogsNapi(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThreadLocal(env, "_drainLogs")) return nullptr;

  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "DrainLogs/get_cb_info");

  std::uint32_t max_entries = 256;
  if (argc >= 1) {
    napi_valuetype t;
    st = napi_typeof(env, args[0], &t);
    if (!CheckNapiOkImpl(env, st, "DrainLogs/arg_typeof")) {
      return nullptr;
    }
    if (t != napi_undefined && t != napi_null) {
      if (t != napi_number) {
        ThrowErrorWithCode(env,
                           "_drainLogs: maxEntries must be a number",
                           "EINVAL",
                           /*type_error=*/true);
        return nullptr;
      }
      double dv = 0.0;
      st = napi_get_value_double(env, args[0], &dv);
      if (!CheckNapiOkImpl(env, st, "DrainLogs/get_value_double")) {
        return nullptr;
      }
      if (std::isfinite(dv)) {
        if (dv < 1.0) dv = 1.0;
        if (dv > 1024.0) dv = 1024.0;
        max_entries = static_cast<std::uint32_t>(dv);
      }
    }
  }

  std::deque<LogEntryNative> drained;
  {
    std::lock_guard<std::mutex> lock(g_log_state.mu);
    const std::size_t n = std::min<std::size_t>(max_entries,
                                                g_log_state.queue.size());
    for (std::size_t i = 0; i < n; ++i) {
      drained.push_back(std::move(g_log_state.queue.front()));
      g_log_state.queue.pop_front();
    }
  }

  napi_value arr;
  st = napi_create_array_with_length(env, static_cast<uint32_t>(drained.size()),
                                     &arr);
  CHECK_NAPI_OK(env, st, "DrainLogs/create_array");

  uint32_t index = 0;
  for (auto& e : drained) {
    napi_value obj;
    st = napi_create_object(env, &obj);
    CHECK_NAPI_OK(env, st, "DrainLogs/create_object");

    // ts
    napi_value ts;
    st = napi_create_double(env, static_cast<double>(e.ts_millis), &ts);
    CHECK_NAPI_OK(env, st, "DrainLogs/create_ts");
    st = napi_set_named_property(env, obj, "ts", ts);
    CHECK_NAPI_OK(env, st, "DrainLogs/set_ts");

    // level
    napi_value lvl;
    const char* lvl_str = LogLevelToString(e.level);
    st = napi_create_string_utf8(env, lvl_str, NAPI_AUTO_LENGTH, &lvl);
    CHECK_NAPI_OK(env, st, "DrainLogs/create_level");
    st = napi_set_named_property(env, obj, "level", lvl);
    CHECK_NAPI_OK(env, st, "DrainLogs/set_level");

    // category
    napi_value cat;
    const char* cat_str = LogCategoryToString(e.category);
    st = napi_create_string_utf8(env, cat_str, NAPI_AUTO_LENGTH, &cat);
    CHECK_NAPI_OK(env, st, "DrainLogs/create_category");
    st = napi_set_named_property(env, obj, "category", cat);
    CHECK_NAPI_OK(env, st, "DrainLogs/set_category");

    // message
    napi_value msg;
    st = napi_create_string_utf8(env, e.message.c_str(),
                                 static_cast<size_t>(e.message.size()), &msg);
    CHECK_NAPI_OK(env, st, "DrainLogs/create_message");
    st = napi_set_named_property(env, obj, "message", msg);
    CHECK_NAPI_OK(env, st, "DrainLogs/set_message");

    // data (small object of string:string pairs)
    napi_value data_obj;
    st = napi_create_object(env, &data_obj);
    CHECK_NAPI_OK(env, st, "DrainLogs/create_data");
    for (const auto& kv : e.data) {
      napi_value v;
      st = napi_create_string_utf8(env, kv.second.c_str(),
                                   static_cast<size_t>(kv.second.size()), &v);
      CHECK_NAPI_OK(env, st, "DrainLogs/create_data_value");
      st = napi_set_named_property(env, data_obj, kv.first.c_str(), v);
      CHECK_NAPI_OK(env, st, "DrainLogs/set_data_value");
    }
    st = napi_set_named_property(env, obj, "data", data_obj);
    CHECK_NAPI_OK(env, st, "DrainLogs/set_data");

    st = napi_set_element(env, arr, index++, obj);
    CHECK_NAPI_OK(env, st, "DrainLogs/set_element");
  }

  return arr;
}

}  // namespace vbt::node
