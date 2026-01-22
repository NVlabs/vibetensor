// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/dispatcher.h"

#include <node_api.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/core/device.h"
#include "vbt/node/dispatcher_errors.h"
#include "vbt/node/tensor.h"
#include "vbt/node/util.h"
#include "vbt/node/logging.h"
#include "vbt/node/cuda_napi.h"

namespace vbt {
namespace node {

std::atomic<std::uint64_t> g_next_job_id{1};
std::atomic<std::uint32_t> g_inflight_ops{0};
std::uint32_t g_max_inflight_ops = 1024;  // overridden from VBT_NODE_MAX_INFLIGHT_OPS
DebugAsyncStatsNative g_async_stats{};

void UpdateAsyncPeakInflight(std::uint32_t current) {
  std::uint32_t prev_async =
      g_async_stats.peak_inflight.load(std::memory_order_relaxed);
  while (current > prev_async &&
         !g_async_stats.peak_inflight.compare_exchange_weak(
             prev_async, current, std::memory_order_relaxed,
             std::memory_order_relaxed)) {
    // prev_async is updated with the latest peak_inflight inside CAS.
  }
}

namespace {

using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;

bool g_allow_sync_danger = false;          // from VBT_NODE_ALLOW_SYNC_DANGER
std::int64_t g_sync_numel_limit = 100000;  // SYNC_NUMEL_LIMIT (may be env-overridden later)

struct DispatcherStatsRaw {
  // total_calls counts async calls that passed validation and inflight
  // checks and were successfully queued. failed_calls counts any such
  // call whose Promise was ultimately rejected (including async-work and
  // result-wrapping failures).
  std::atomic<std::uint64_t> total_calls{0};   // async calls that passed validation + cap
  std::atomic<std::uint64_t> failed_calls{0};  // async calls whose Promise was rejected
  std::atomic<std::uint32_t> peak_inflight{0}; // max inflight seen
} g_stats;

std::once_flag g_env_once;

bool IsTruthyEnv(const char* v) {
  if (!v || *v == '\0') return false;
  std::string s(v);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s == "1" || s == "true" || s == "yes";
}

void InitEnvConfigOnce() {
  std::call_once(g_env_once, [] {
    // Max inflight ops
    const char* inflight_str = std::getenv("VBT_NODE_MAX_INFLIGHT_OPS");
    std::uint32_t max_ops = 1024;
    if (inflight_str && *inflight_str) {
      char* end = nullptr;
      unsigned long v = std::strtoul(inflight_str, &end, 10);
      if (end != inflight_str && v <= std::numeric_limits<std::uint32_t>::max()) {
        max_ops = static_cast<std::uint32_t>(v);
        if (max_ops == 0) {
          max_ops = 1;  // clamp to at least 1
        }
      } else {
        max_ops = 1;  // invalid → clamp
      }
    }
    g_max_inflight_ops = max_ops;

    // Sync danger flag
    const char* sync_str = std::getenv("VBT_NODE_ALLOW_SYNC_DANGER");
    g_allow_sync_danger = IsTruthyEnv(sync_str);

    // NOTE: g_sync_numel_limit is intentionally fixed for now; an env
    // override can be added later without breaking API.
  });
}

CudaRuntimeState* GetCudaRuntimeStateLocal(napi_env env) {
  AddonData* data = nullptr;
  napi_status st =
      napi_get_instance_data(env, reinterpret_cast<void**>(&data));
  if (st != napi_ok || !data) {
    return nullptr;
  }
  return &data->cuda_rt;
}

bool IsOnMainThreadLocal(napi_env env) {
  CudaRuntimeState* rt = GetCudaRuntimeStateLocal(env);
  if (!rt) return false;
  return IsOnMainThreadFromState(*rt, std::this_thread::get_id());
}

bool EnsureOnMainThreadLocal(napi_env env, const char* fn_name) {
  if (!IsOnMainThreadLocal(env)) {
    std::string msg = std::string(fn_name ? fn_name : "async") +
                      ": must be called on the main VibeTensor JS thread";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return false;
  }
  return true;
}

void UpdatePeakInflight(std::uint32_t current) {
  std::uint32_t prev = g_stats.peak_inflight.load(std::memory_order_relaxed);
  while (current > prev &&
         !g_stats.peak_inflight.compare_exchange_weak(
             prev, current, std::memory_order_relaxed,
             std::memory_order_relaxed)) {
    // prev is updated with the latest peak_inflight inside CAS.
  }

  // Also update the global async peak_inflight metric shared with
  // CUDA helpers and DLPack imports.
  UpdateAsyncPeakInflight(current);
}

// Small RAII helper used when async setup fails after inflight++.
void RejectAndCleanup(napi_env env,
                      CallOpJob* job,
                      CallOpErrorKind kind,
                      const std::string& message) {
  if (!job) return;

  job->error_kind = kind;
  job->error_message = message;

  napi_value err = MakeDispatchError(env, kind, job->op_name, job->error_message);
  if (job->deferred != nullptr) {
    napi_reject_deferred(env, job->deferred, err);
  }

  // Update stats and inflight counters.
  g_stats.failed_calls.fetch_add(1, std::memory_order_relaxed);
  g_async_stats.dispatcher_failed.fetch_add(1, std::memory_order_relaxed);
  g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);

  // Drop JS references and async work.
  for (napi_ref ref : job->js_arg_refs) {
    if (ref) {
      napi_delete_reference(env, ref);
    }
  }
  job->js_arg_refs.clear();

  if (job->work != nullptr) {
    napi_delete_async_work(env, job->work);
    job->work = nullptr;
  }

  delete job;
}

} // namespace

std::optional<ParsedCallInputs>
ParseCallOpInputs(napi_env env, napi_value js_name, napi_value js_args) {
  ParsedCallInputs out;

  // Validate name
  if (!js_name) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, out.op_name,
                       "callOp: name must be a non-empty string");
    return std::nullopt;
  }

  napi_valuetype t_name;
  napi_status st = napi_typeof(env, js_name, &t_name);
  if (!CheckNapiOkImpl(env, st, "ParseCallOpInputs/name_typeof")) {
    return std::nullopt;  // error already set
  }
  if (t_name != napi_string) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, out.op_name,
                       "callOp: name must be a string");
    return std::nullopt;
  }

  bool ok = false;
  out.op_name = detail::GetStringUtf8(env, js_name,
                                      "ParseCallOpInputs/name_string", &ok);
  if (!ok) {
    // GetStringUtf8 already set a generic Error; surface as-is.
    return std::nullopt;
  }
  if (out.op_name.empty()) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, out.op_name,
                       "callOp: name must be non-empty");
    return std::nullopt;
  }

  // Validate args array
  bool is_array = false;
  st = napi_is_array(env, js_args, &is_array);
  if (!CheckNapiOkImpl(env, st, "ParseCallOpInputs/args_is_array")) {
    return std::nullopt;
  }
  if (!is_array) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, out.op_name,
                       "callOp: args must be an array of Tensor handles");
    return std::nullopt;
  }

  std::uint32_t length = 0;
  st = napi_get_array_length(env, js_args, &length);
  if (!CheckNapiOkImpl(env, st, "ParseCallOpInputs/get_array_length")) {
    return std::nullopt;
  }

  out.inputs.clear();
  out.inputs.reserve(length);

  for (std::uint32_t i = 0; i < length; ++i) {
    napi_value elem;
    st = napi_get_element(env, js_args, i, &elem);
    if (!CheckNapiOkImpl(env, st, "ParseCallOpInputs/get_element")) {
      return std::nullopt;
    }

    // Validate tensor handle. We replicate UnwrapJsTensor logic here but
    // attach a stable .code to the error.
    JsTensor* jt = nullptr;
    st = napi_unwrap(env, elem, reinterpret_cast<void**>(&jt));
    if (st != napi_ok || jt == nullptr) {
      ThrowDispatchError(env, CallOpErrorKind::InvalidArg, out.op_name,
                         "expected Tensor handle");
      return std::nullopt;
    }

    out.inputs.push_back(jt->impl);
  }

  if (out.inputs.empty()) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, out.op_name,
                       "callOp: args must contain at least one Tensor");
    return std::nullopt;
  }

  return out;
}

std::uint32_t max_inflight_ops() {
  InitEnvConfigOnce();
  return g_max_inflight_ops;
}

std::uint32_t current_inflight_ops() {
  return g_inflight_ops.load(std::memory_order_acquire);
}

namespace {

using ExecuteFn = void (*)(napi_env, void*);

napi_value CallOpImpl(napi_env env,
                      napi_callback_info info,
                      ExecuteFn worker_fn) {
  InitEnvConfigOnce();

  size_t argc = 3;
  napi_value args[3];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CallOpImpl/get_cb_info");

  if (argc < 2) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, "",
                       "callOp(name, args, opts?) expects at least 2 arguments");
    return nullptr;
  }

  napi_value js_name = args[0];
  napi_value js_args = args[1];

  auto parsed_opt = ParseCallOpInputs(env, js_name, js_args);
  if (!parsed_opt.has_value()) {
    return nullptr;  // exception already pending
  }

  ParsedCallInputs parsed = std::move(*parsed_opt);

  // Enforce inflight cap.
  const std::uint32_t prev = g_inflight_ops.fetch_add(1, std::memory_order_acq_rel);
  const std::uint32_t current = prev + 1;
  if (current > g_max_inflight_ops) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    g_async_stats.dispatcher_failed.fetch_add(1, std::memory_order_relaxed);
    LogIfEnabled(LogLevel::kWarn,
                 LogCategory::kDispatcher,
                 "callOp: too many inflight ops",
                 {{"op", parsed.op_name}});
    ThrowDispatchError(env, CallOpErrorKind::Runtime, parsed.op_name,
                       "too many inflight ops (see VBT_NODE_MAX_INFLIGHT_OPS)");
    return nullptr;
  }

  UpdatePeakInflight(current);

  // Allocate job and Promise.
  auto* job = new CallOpJob();
  job->op_name = std::move(parsed.op_name);
  job->inputs = std::move(parsed.inputs);
  job->job_id = g_next_job_id.fetch_add(1, std::memory_order_relaxed);

  napi_value promise;
  st = napi_create_promise(env, &job->deferred, &promise);
  if (st != napi_ok) {
    // Treat this as an internal invariant violation.
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    napi_fatal_error("vbt.node.CallOpImpl", NAPI_AUTO_LENGTH,
                     "napi_create_promise failed", NAPI_AUTO_LENGTH);
    return nullptr;  // unreachable
  }

  // Create JS references for each arg to keep them alive during the async op.
  bool is_array = false;
  st = napi_is_array(env, js_args, &is_array);
  if (st != napi_ok || !is_array) {
    RejectAndCleanup(env, job, CallOpErrorKind::AsyncWorkFailure,
                     "callOp: args array became invalid while scheduling async work");
    return promise;
  }

  std::uint32_t length = 0;
  st = napi_get_array_length(env, js_args, &length);
  if (st != napi_ok) {
    RejectAndCleanup(env, job, CallOpErrorKind::AsyncWorkFailure,
                     "callOp: failed to read args length while scheduling async work");
    return promise;
  }

  job->js_arg_refs.reserve(length);
  for (std::uint32_t i = 0; i < length; ++i) {
    napi_value elem;
    st = napi_get_element(env, js_args, i, &elem);
    if (st != napi_ok) {
      RejectAndCleanup(env, job, CallOpErrorKind::AsyncWorkFailure,
                       "callOp: failed to capture arg reference");
      return promise;
    }

    napi_ref ref;
    st = napi_create_reference(env, elem, 1, &ref);
    if (st != napi_ok) {
      RejectAndCleanup(env, job, CallOpErrorKind::AsyncWorkFailure,
                       "callOp: failed to create reference for arg");
      return promise;
    }
    job->js_arg_refs.push_back(ref);
  }

  // Create and queue async work.
  napi_value resource_name;
  st = napi_create_string_utf8(env, "vbt_callOp", NAPI_AUTO_LENGTH, &resource_name);
  if (st != napi_ok) {
    RejectAndCleanup(env, job, CallOpErrorKind::AsyncWorkFailure,
                     "callOp: failed to create async resource name");
    return promise;
  }

  st = napi_create_async_work(env,
                              nullptr,             // async_resource
                              resource_name,
                              worker_fn,
                              /*complete_cb=*/[](napi_env env, napi_status status, void* data) {
                                auto* job = static_cast<CallOpJob*>(data);

                                // Decrement inflight and drop JS references first.
                                g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);

                                for (napi_ref ref : job->js_arg_refs) {
                                  if (ref) {
                                    napi_delete_reference(env, ref);
                                  }
                                }
                                job->js_arg_refs.clear();

                                if (job->work != nullptr) {
                                  napi_delete_async_work(env, job->work);
                                  job->work = nullptr;
                                }

                                // If the worker failed at the N-API level, convert
                                // this into an AsyncWorkFailure error.
                                if (status != napi_ok &&
                                    job->error_kind == CallOpErrorKind::None) {
                                  job->success = false;
                                  job->error_kind = CallOpErrorKind::AsyncWorkFailure;
                                  job->error_message = "async worker failed";
                                }

                                if (job->success) {
                                  // Happy path: wrap the output tensor.
                                  napi_value js_tensor = nullptr;
                                  if (!TryWrapTensorImplAsJsTensor(env,
                                                                   std::move(job->output),
                                                                   &js_tensor)) {
                                    // Wrapping failed; convert to async failure.
                                    g_stats.failed_calls.fetch_add(1, std::memory_order_relaxed);
                                    napi_value err = MakeDispatchError(
                                        env, CallOpErrorKind::AsyncWorkFailure,
                                        job->op_name,
                                        "failed to wrap tensor for JS result");
                                    napi_reject_deferred(env, job->deferred, err);
                                  } else {
                                    LogIfEnabled(LogLevel::kDebug,
                                                 LogCategory::kDispatcher,
                                                 "callOp: async success",
                                                 {{"op", job->op_name}});
                                    napi_resolve_deferred(env, job->deferred, js_tensor);
                                  }
                                } else {
                                  // Error path: reject with mapped error.
                                  g_stats.failed_calls.fetch_add(1, std::memory_order_relaxed);
                                  g_async_stats.dispatcher_failed.fetch_add(1, std::memory_order_relaxed);
                                  napi_value err = MakeDispatchError(
                                      env, job->error_kind, job->op_name,
                                      job->error_message);
                                  LogIfEnabled(LogLevel::kWarn,
                                               LogCategory::kDispatcher,
                                               "callOp: async failure",
                                               {{"op", job->op_name}});
                                  napi_reject_deferred(env, job->deferred, err);
                                }

                                delete job;
                              },
                              job,
                              &job->work);
  if (st != napi_ok) {
    RejectAndCleanup(env, job, CallOpErrorKind::AsyncWorkFailure,
                     "callOp: failed to create async work");
    return promise;
  }

  st = napi_queue_async_work(env, job->work);
  if (st != napi_ok) {
    RejectAndCleanup(env, job, CallOpErrorKind::AsyncWorkFailure,
                     "callOp: failed to queue async work");
    return promise;
  }

  // Async call successfully scheduled.
  g_stats.total_calls.fetch_add(1, std::memory_order_relaxed);
  g_async_stats.dispatcher_total.fetch_add(1, std::memory_order_relaxed);
  return promise;
}

void ExecuteCallCommon(CallOpJob* job, bool redispatch) {
  if (!job) return;

  BoxedStack& stack = job->inputs;

  try {
    if (redispatch) {
      Dispatcher::instance().redispatchBoxed(job->op_name, stack);
    } else {
      Dispatcher::instance().callBoxed(job->op_name, stack);
    }

    if (stack.size() != 1) {
      job->success = false;
      job->error_kind = CallOpErrorKind::Runtime;
      job->error_message =
          "Node: multi-output/not-supported output from kernel";
      return;
    }

    job->success = true;
    job->output = std::move(stack[0]);
  } catch (const std::invalid_argument& e) {
    job->success = false;
    job->error_kind = CallOpErrorKind::InvalidArg;
    job->error_message = e.what() ? e.what() : "invalid argument";
  } catch (const std::bad_alloc& e) {
    job->success = false;
    job->error_kind = CallOpErrorKind::BadAlloc;
    job->error_message = e.what() ? e.what() : "bad alloc";
  } catch (const std::runtime_error& e) {
    job->success = false;
    const char* what = e.what();
    std::string msg = what ? std::string(what) : std::string("runtime error");
    if (msg.rfind("unknown op: ", 0) == 0) {
      job->error_kind = CallOpErrorKind::UnknownOp;
    } else {
      job->error_kind = CallOpErrorKind::Runtime;
    }
    job->error_message = std::move(msg);
  } catch (...) {
    job->success = false;
    job->error_kind = CallOpErrorKind::Unknown;
    job->error_message = "unknown internal error";
  }
}

void ExecuteCallOp(napi_env /*env*/, void* data) {
  auto* job = static_cast<CallOpJob*>(data);
  ExecuteCallCommon(job, /*redispatch=*/false);
}

void ExecuteCallOpNoOverride(napi_env /*env*/, void* data) {
  auto* job = static_cast<CallOpJob*>(data);
  ExecuteCallCommon(job, /*redispatch=*/true);
}

} // namespace

napi_value CallOp(napi_env env, napi_callback_info info) {
  return CallOpImpl(env, info, &ExecuteCallOp);
}

napi_value CallOpNoOverride(napi_env env, napi_callback_info info) {
  return CallOpImpl(env, info, &ExecuteCallOpNoOverride);
}

napi_value CallOpSync(napi_env env, napi_callback_info info) {
  InitEnvConfigOnce();

  if (!g_allow_sync_danger) {
    ThrowDispatchError(env, CallOpErrorKind::Runtime, "",
                       "callOpSync is disabled; set VBT_NODE_ALLOW_SYNC_DANGER=1");
    return nullptr;
  }

  size_t argc = 3;
  napi_value args[3];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CallOpSync/get_cb_info");

  if (argc < 2) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, "",
                       "callOpSync(name, args, opts?) expects at least 2 arguments");
    return nullptr;
  }

  auto parsed_opt = ParseCallOpInputs(env, args[0], args[1]);
  if (!parsed_opt.has_value()) {
    return nullptr;
  }

  ParsedCallInputs parsed = std::move(*parsed_opt);

  // Total numel limit
  std::int64_t total_numel = 0;
  for (const auto& t : parsed.inputs) {
    const std::int64_t n = t.numel();
    if (n < 0) continue;
    if (n > std::numeric_limits<std::int64_t>::max() - total_numel) {
      total_numel = std::numeric_limits<std::int64_t>::max();
      break;
    }
    total_numel += n;

    if (t.device().type == vbt::core::Device::cuda().type) {
      ThrowDispatchError(env, CallOpErrorKind::Runtime, parsed.op_name,
                         "callOpSync: CUDA tensors are not supported");
      return nullptr;
    }
  }

  if (total_numel > g_sync_numel_limit) {
    ThrowDispatchError(env, CallOpErrorKind::Runtime, parsed.op_name,
                       "callOpSync: total numel exceeds sync limit");
    return nullptr;
  }

  BoxedStack& stack = parsed.inputs;

  try {
    Dispatcher::instance().callBoxed(parsed.op_name, stack);

    if (stack.size() != 1) {
      ThrowDispatchError(env, CallOpErrorKind::Runtime, parsed.op_name,
                         "Node: multi-output/not-supported output from kernel");
      return nullptr;
    }

    return WrapTensorImplAsJsTensor(env, std::move(stack[0]));
  } catch (const std::invalid_argument& e) {
    ThrowDispatchError(env, CallOpErrorKind::InvalidArg, parsed.op_name,
                       e.what() ? e.what() : "invalid argument");
    return nullptr;
  } catch (const std::bad_alloc& e) {
    ThrowDispatchError(env, CallOpErrorKind::BadAlloc, parsed.op_name,
                       e.what() ? e.what() : "bad alloc");
    return nullptr;
  } catch (const std::runtime_error& e) {
    const char* what = e.what();
    std::string msg = what ? std::string(what) : std::string("runtime error");
    CallOpErrorKind kind =
        (msg.rfind("unknown op: ", 0) == 0) ? CallOpErrorKind::UnknownOp
                                           : CallOpErrorKind::Runtime;
    ThrowDispatchError(env, kind, parsed.op_name, msg);
    return nullptr;
  } catch (...) {
    ThrowDispatchError(env, CallOpErrorKind::Unknown, parsed.op_name,
                       "unknown internal error");
    return nullptr;
  }
}

napi_value GetDispatcherStats(napi_env env, napi_callback_info /*info*/) {
  InitEnvConfigOnce();

  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  CHECK_NAPI_OK(env, st, "GetDispatcherStats/create_object");

  auto set_int = [&](const char* name, std::uint64_t v) {
    napi_value n;
    napi_status s = napi_create_int64(env, static_cast<std::int64_t>(v), &n);
    CHECK_NAPI_OK(env, s, "GetDispatcherStats/create_int64");
    s = napi_set_named_property(env, obj, name, n);
    CHECK_NAPI_OK(env, s, "GetDispatcherStats/set_named_property");
  };

  set_int("totalCalls", g_stats.total_calls.load(std::memory_order_relaxed));
  set_int("failedCalls", g_stats.failed_calls.load(std::memory_order_relaxed));
  set_int("inflight", g_inflight_ops.load(std::memory_order_acquire));
  set_int("peakInflight", g_stats.peak_inflight.load(std::memory_order_relaxed));
  set_int("maxInflight", g_max_inflight_ops);

  return obj;
}

napi_value GetAsyncStats(napi_env env, napi_callback_info /*info*/) {
  if (!EnsureOnMainThreadLocal(env, "_debugAsyncStats")) {
    return nullptr;
  }

  InitEnvConfigOnce();

  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  CHECK_NAPI_OK(env, st, "GetAsyncStats/create_object");

  auto set_double = [&](napi_value target,
                        const char* name,
                        double value,
                        const char* context) -> bool {
    napi_value v;
    napi_status s = napi_create_double(env, value, &v);
    if (!CheckNapiOkImpl(env, s, context)) {
      return false;
    }
    s = napi_set_named_property(env, target, name, v);
    if (!CheckNapiOkImpl(env, s, context)) {
      return false;
    }
    return true;
  };

  const std::uint32_t inflight =
      g_inflight_ops.load(std::memory_order_acquire);
  const std::uint32_t peak =
      g_async_stats.peak_inflight.load(std::memory_order_relaxed);
  const std::uint32_t max_inflight = g_max_inflight_ops;

  if (!set_double(obj,
                  "inflight",
                  static_cast<double>(inflight),
                  "GetAsyncStats/set_inflight")) {
    return nullptr;
  }
  if (!set_double(obj,
                  "peakInflight",
                  static_cast<double>(peak),
                  "GetAsyncStats/set_peakInflight")) {
    return nullptr;
  }
  if (!set_double(obj,
                  "maxInflight",
                  static_cast<double>(max_inflight),
                  "GetAsyncStats/set_maxInflight")) {
    return nullptr;
  }

  // dispatcher: { totalCalls, failedCalls }
  napi_value dispatcher;
  st = napi_create_object(env, &dispatcher);
  CHECK_NAPI_OK(env, st, "GetAsyncStats/create_dispatcher");
  if (!set_double(
          dispatcher,
          "totalCalls",
          static_cast<double>(
              g_async_stats.dispatcher_total.load(std::memory_order_relaxed)),
          "GetAsyncStats/set_dispatcher_total") ||
      !set_double(
          dispatcher,
          "failedCalls",
          static_cast<double>(
              g_async_stats.dispatcher_failed.load(std::memory_order_relaxed)),
          "GetAsyncStats/set_dispatcher_failed")) {
    return nullptr;
  }
  st = napi_set_named_property(env, obj, "dispatcher", dispatcher);
  CHECK_NAPI_OK(env, st, "GetAsyncStats/set_dispatcher");

  auto make_category = [&](const AsyncCategoryStatsNative& src,
                           const char* context) -> napi_value {
    napi_value cat;
    napi_status s = napi_create_object(env, &cat);
    CHECK_NAPI_OK(env, s, "GetAsyncStats/create_category");

    if (!set_double(
            cat,
            "started",
            static_cast<double>(src.started.load(std::memory_order_relaxed)),
            context) ||
        !set_double(
            cat,
            "completed",
            static_cast<double>(src.completed.load(std::memory_order_relaxed)),
            context) ||
        !set_double(
            cat,
            "failed",
            static_cast<double>(src.failed.load(std::memory_order_relaxed)),
            context) ||
        !set_double(
            cat,
            "bytes",
            static_cast<double>(src.bytes.load(std::memory_order_relaxed)),
            context)) {
      return nullptr;
    }

    return cat;
  };

  napi_value h2d = make_category(g_async_stats.h2d, "GetAsyncStats/h2d");
  if (!h2d) return nullptr;

  napi_value d2h = make_category(g_async_stats.d2h, "GetAsyncStats/d2h");
  if (!d2h) return nullptr;

  st = napi_set_named_property(env, obj, "h2d", h2d);
  CHECK_NAPI_OK(env, st, "GetAsyncStats/set_h2d");
  st = napi_set_named_property(env, obj, "d2h", d2h);
  CHECK_NAPI_OK(env, st, "GetAsyncStats/set_d2h");

  return obj;
}

} // namespace node
} // namespace vbt
