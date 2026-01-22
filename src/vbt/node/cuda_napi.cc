// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/cuda_napi.h"

#include <node_api.h>

#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "vbt/cuda/device.h"
#include "vbt/cuda/event.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/allocator.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor.h"
#include "vbt/core/checked_math.h"
#include "vbt/node/dispatcher.h"
#include "vbt/node/errors.h"
#include "vbt/node/tensor.h"
#include "vbt/node/util.h"
#include "vbt/node/logging.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace vbt {
namespace node {

void AddonDataFinalizer(napi_env /*env*/, void* data, void* /*hint*/) {
  auto* addon = static_cast<AddonData*>(data);
  delete addon;
}

void ThrowCudaRuntimeErrorSimple(napi_env env,
                                 const char* message,
                                 const char* code) {
  const char* effective_code = code ? code : "ERUNTIME";
  std::string msg = message ? std::string(message) : std::string();
  msg = SanitizePointers(msg);
  msg = SanitizePaths(msg);
  const bool is_type_error = std::strcmp(effective_code, "EINVAL") == 0;
  ThrowErrorWithCode(env, msg, effective_code, is_type_error);
}

namespace {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::checked_mul_i64;
using vbt::cuda::Allocator;
using vbt::cuda::DeviceIndex;
using vbt::cuda::DeviceStats;
using vbt::cuda::MemorySegmentSnapshot;
using vbt::cuda::snapshot_memory_segments;

CudaRuntimeState* GetCudaRuntimeState(napi_env env) {
  AddonData* data = nullptr;
  napi_status st =
      napi_get_instance_data(env, reinterpret_cast<void**>(&data));
  if (st != napi_ok || !data) {
    return nullptr;
  }
  return &data->cuda_rt;
}

bool IsOnMainThread(napi_env env) {
  CudaRuntimeState* rt = GetCudaRuntimeState(env);
  if (!rt) return false;
  return IsOnMainThreadFromState(*rt, std::this_thread::get_id());
}

bool EnsureOnMainThread(napi_env env, const char* fn_name) {
  if (!IsOnMainThread(env)) {
    std::string msg = std::string(fn_name ? fn_name : "cuda") +
                      ": must be called on the main VibeTensor JS thread";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return false;
  }
  return true;
}

CudaRuntimeState* RequireCudaRuntimeState(napi_env env,
                                          const char* fn_name) {
  CudaRuntimeState* rt = GetCudaRuntimeState(env);
  if (!rt || !rt->initialized) {
    std::string msg = std::string(fn_name ? fn_name : "cuda") +
                      ": addon state not initialized";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return nullptr;
  }
  return rt;
}

bool EnsureCudaAvailable(napi_env env,
                         const char* fn_name,
                         CudaRuntimeState** out_rt) {
  CudaRuntimeState* rt = RequireCudaRuntimeState(env, fn_name);
  if (!rt) return false;
  if (!rt->has_cuda || rt->device_count <= 0) {
    std::string msg = std::string(fn_name ? fn_name : "cuda") +
                      ": CUDA not available";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ENOCUDA");
    return false;
  }
  if (out_rt) *out_rt = rt;
  return true;
}

bool ParseDeviceIndex(napi_env env,
                      napi_value js_idx,
                      int device_count,
                      const char* fn_name,
                      int* out_idx) {
  if (!out_idx) return false;

  napi_valuetype t;
  napi_status st = napi_typeof(env, js_idx, &t);
  if (!CheckNapiOkImpl(env, st, "CudaSetDevice/typeof")) return false;

  if (t != napi_number) {
    std::string msg = std::string(fn_name ? fn_name : "cuda.setDevice") +
                      ": index must be a finite integer";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "EINVAL");
    return false;
  }

  double dv = 0.0;
  st = napi_get_value_double(env, js_idx, &dv);
  if (!CheckNapiOkImpl(env, st, "CudaSetDevice/get_value_double")) {
    return false;
  }

  if (!std::isfinite(dv)) {
    std::string msg = std::string(fn_name ? fn_name : "cuda.setDevice") +
                      ": index must be finite";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "EINVAL");
    return false;
  }

  long long ll = static_cast<long long>(dv);
  if (static_cast<double>(ll) != dv || ll < 0 || ll >= device_count) {
    std::string msg = std::string(fn_name ? fn_name : "cuda.setDevice") +
                      ": device index out of range";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "EINVAL");
    return false;
  }

  *out_idx = static_cast<int>(ll);
  return true;
}

struct ParsedStatsDeviceArg {
  bool cpu_only{false};       // true if device_count == 0
  bool all_devices{false};    // true when aggregating across all devices
  int32_t device{-1};         // single device index when !all_devices
};

std::optional<ParsedStatsDeviceArg>
ParseStatsDeviceArg(napi_env env, napi_value js_device) {
  ParsedStatsDeviceArg out{};

  CudaRuntimeState* rt = RequireCudaRuntimeState(env, "_cudaMemoryStats");
  if (!rt) return std::nullopt;

  const int32_t device_count = rt->device_count;
  out.cpu_only = (device_count == 0);

  // Handle undefined/null/omitted → all devices.
  if (js_device == nullptr) {
    out.all_devices = true;
    out.device = -1;
    return out;
  }

  napi_valuetype vt;
  napi_status st = napi_typeof(env, js_device, &vt);
  if (!CheckNapiOkImpl(env, st, "ParseStatsDeviceArg/typeof")) {
    return std::nullopt;
  }

  if (vt == napi_undefined || vt == napi_null) {
    out.all_devices = true;
    out.device = -1;
    return out;
  }

  if (vt != napi_number) {
    ThrowCudaRuntimeErrorSimple(
        env,
        "device must be a number or undefined",
        "EINVAL");
    return std::nullopt;
  }

  double d = 0.0;
  st = napi_get_value_double(env, js_device, &d);
  if (!CheckNapiOkImpl(env, st, "ParseStatsDeviceArg/get_value_double")) {
    return std::nullopt;
  }

  if (!std::isfinite(d) || std::floor(d) != d) {
    ThrowCudaRuntimeErrorSimple(
        env,
        "device must be a finite integer",
        "EINVAL");
    return std::nullopt;
  }

  const int64_t idx64 = static_cast<int64_t>(d);
  if (idx64 < 0) {
    ThrowCudaRuntimeErrorSimple(
        env,
        "device index must be >= 0",
        "EINVAL");
    return std::nullopt;
  }

  const int32_t idx = static_cast<int32_t>(idx64);

  if (out.cpu_only) {
    // Any non-negative index is accepted; stats code will synthesize zeros.
    out.all_devices = false;
    out.device = idx;
    return out;
  }

  if (idx >= device_count) {
    ThrowCudaRuntimeErrorSimple(
        env,
        "device index out of range",
        "ERUNTIME");
    return std::nullopt;
  }

  out.all_devices = false;
  out.device = idx;
  return out;
}

// Helper: standard contiguous row-major strides for a shape.
std::vector<int64_t> make_contig_strides(const std::vector<int64_t>& sizes) {
  std::vector<int64_t> st(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    st[idx] = acc;
    const int64_t dim = sizes[idx];
    if (dim != 0) {
      // Best-effort overflow handling; mirror tensor.cc behavior.
      const int64_t next = acc * dim;
      acc = next;
    }
  }
  return st;
}

struct H2DJob {
  // Immutable worker inputs
  void* host_data{nullptr};
  std::size_t nbytes{0};
  std::vector<int64_t> sizes;
  ScalarType dtype{ScalarType::Float32};
  int device_index{0};

  // Worker results
  bool success{false};
  TensorImpl result;
  std::string error_message;
  std::string error_code{"ERUNTIME"};

  // JS-side handles
  napi_deferred deferred{nullptr};
  napi_async_work work{nullptr};
  napi_ref src_ref{nullptr};  // keeps src TypedArray alive
};

struct D2HJob {
  // Immutable worker inputs
  TensorImpl tensor;
  void* host_data{nullptr};
  std::size_t nbytes{0};

  // Worker results
  bool success{false};
  std::string error_message;
  std::string error_code{"ERUNTIME"};

  // JS-side handles
  napi_deferred deferred{nullptr};
  napi_async_work work{nullptr};
  napi_ref array_ref{nullptr};  // keeps result TypedArray alive
};

struct CudaStreamState {
  vbt::cuda::Stream stream;  // RAII wrapper
  int device_index;          // cached for debugging/telemetry
};

struct CudaEventState {
  std::optional<vbt::cuda::Event> event;  // constructed lazily on first record
  bool created{false};
  bool enable_timing{false};
};

using CudaStreamHandleNative = std::shared_ptr<CudaStreamState>;
using CudaEventHandleNative  = std::shared_ptr<CudaEventState>;

void CudaStreamFinalizer(napi_env env, void* data, void* /*hint*/) {
#ifdef VBT_NODE_CUDA_DEBUG
  if (CudaRuntimeState* rt = GetCudaRuntimeState(env)) {
    rt->stream_instances.fetch_sub(1, std::memory_order_relaxed);
  }
#else
  (void)env;
#endif
  auto* box = static_cast<CudaStreamHandleNative*>(data);
  delete box;
}

void CudaEventFinalizer(napi_env env, void* data, void* /*hint*/) {
#ifdef VBT_NODE_CUDA_DEBUG
  if (CudaRuntimeState* rt = GetCudaRuntimeState(env)) {
    rt->event_instances.fetch_sub(1, std::memory_order_relaxed);
  }
#else
  (void)env;
#endif
  auto* box = static_cast<CudaEventHandleNative*>(data);
  delete box;
}

bool UnwrapStreamHandle(napi_env env,
                        napi_value this_arg,
                        CudaStreamHandleNative** out) {
  if (!out) return false;
  void* data = nullptr;
  napi_status st = napi_unwrap(env, this_arg, &data);
  if (st != napi_ok || !data) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.Stream: expected Stream handle", "EINVAL");
    return false;
  }
  auto* box = static_cast<CudaStreamHandleNative*>(data);
  if (!box || !*box) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.Stream: internal handle missing", "ERUNTIME");
    return false;
  }
  *out = box;
  return true;
}

bool UnwrapEventHandle(napi_env env,
                       napi_value this_arg,
                       CudaEventHandleNative** out) {
  if (!out) return false;
  void* data = nullptr;
  napi_status st = napi_unwrap(env, this_arg, &data);
  if (st != napi_ok || !data) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.Event: expected Event handle", "EINVAL");
    return false;
  }
  auto* box = static_cast<CudaEventHandleNative*>(data);
  if (!box || !*box) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.Event: internal handle missing", "ERUNTIME");
    return false;
  }
  *out = box;
  return true;
}

// ==== Stream class ==========================================================

napi_value CudaStreamConstructor(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Stream.create")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.Stream.create", &rt)) {
    return nullptr;
  }

  napi_value this_arg;
  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaStreamConstructor/get_cb_info");

  int device_index = 0;
  if (argc >= 1) {
    if (!ParseDeviceIndex(env, args[0], rt->device_count,
                          "cuda.Stream.create", &device_index)) {
      return nullptr;
    }
  }

  CudaStreamHandleNative native;
  try {
    vbt::cuda::DeviceIndex dev_idx =
        static_cast<vbt::cuda::DeviceIndex>(device_index);
    native = std::make_shared<CudaStreamState>(
        CudaStreamState{vbt::cuda::getStreamFromPool(false, dev_idx),
                        device_index});
  } catch (const std::bad_alloc& e) {
    std::string msg = e.what() ? e.what() : "allocation failed";
    ThrowCudaRuntimeErrorSimple(
        env, ("cuda.Stream.create: " + msg).c_str(), "EOOM");
    return nullptr;
  } catch (const std::exception& e) {
    std::string msg = e.what() ? e.what() : "runtime error";
    ThrowCudaRuntimeErrorSimple(
        env, ("cuda.Stream.create: " + msg).c_str(), "ERUNTIME");
    return nullptr;
  } catch (...) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.Stream.create: unknown internal error", "EUNKNOWN");
    return nullptr;
  }

  auto* box = new CudaStreamHandleNative(std::move(native));

#ifdef VBT_NODE_CUDA_DEBUG
  if (rt) {
    rt->stream_instances.fetch_add(1, std::memory_order_relaxed);
  }
#endif

  st = napi_wrap(env, this_arg, box, CudaStreamFinalizer, nullptr, nullptr);
  if (st != napi_ok) {
    delete box;
    CHECK_NAPI_OK(env, st, "CudaStreamConstructor/wrap");
    return nullptr;  // CHECK_NAPI_OK already threw.
  }

  return this_arg;
}

napi_value CudaStreamDeviceIndex(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Stream.deviceIndex")) return nullptr;

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaStreamDeviceIndex/get_cb_info");

  CudaStreamHandleNative* box = nullptr;
  if (!UnwrapStreamHandle(env, this_arg, &box)) return nullptr;

  napi_value out;
  st = napi_create_int32(env, (*box)->device_index, &out);
  CHECK_NAPI_OK(env, st, "CudaStreamDeviceIndex/create_int32");
  return out;
}

napi_value CudaStreamQuery(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Stream.query")) return nullptr;

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaStreamQuery/get_cb_info");

  CudaStreamHandleNative* box = nullptr;
  if (!UnwrapStreamHandle(env, this_arg, &box)) return nullptr;

  bool ready = (*box)->stream.query();

  napi_value out;
  st = napi_get_boolean(env, ready, &out);
  CHECK_NAPI_OK(env, st, "CudaStreamQuery/get_boolean");
  return out;
}

napi_value CudaStreamToString(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Stream.toString")) return nullptr;

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaStreamToString/get_cb_info");

  CudaStreamHandleNative* box = nullptr;
  if (!UnwrapStreamHandle(env, this_arg, &box)) return nullptr;

  std::string repr = vbt::cuda::to_string((*box)->stream);

  napi_value out;
  st = napi_create_string_utf8(env, repr.c_str(), repr.size(), &out);
  CHECK_NAPI_OK(env, st, "CudaStreamToString/create_string");
  return out;
}

struct StreamSyncJob {
  CudaStreamHandleNative stream;
  napi_deferred deferred{nullptr};
  napi_async_work work{nullptr};
  bool success{false};
  std::string error_message;
  std::string error_code{"ERUNTIME"};
};

napi_value CudaStreamSynchronize(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Stream.synchronize")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.Stream.synchronize", &rt)) {
    return nullptr;
  }

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaStreamSynchronize/get_cb_info");

  CudaStreamHandleNative* box = nullptr;
  if (!UnwrapStreamHandle(env, this_arg, &box)) return nullptr;

  // Ensure env config (VBT_NODE_MAX_INFLIGHT_OPS) has been read.
  (void)max_inflight_ops();

  const std::uint32_t prev =
      g_inflight_ops.fetch_add(1, std::memory_order_acq_rel);
  const std::uint32_t current = prev + 1;
  if (prev >= g_max_inflight_ops) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    ThrowCudaRuntimeErrorSimple(
        env,
        "cuda.Stream.synchronize: too many inflight ops (see VBT_NODE_MAX_INFLIGHT_OPS)",
        "ERUNTIME");
    return nullptr;
  }

  UpdateAsyncPeakInflight(current);

  auto* job = new StreamSyncJob();
  job->stream = *box;

  napi_value promise;
  st = napi_create_promise(env, &job->deferred, &promise);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    napi_fatal_error("vbt.node.CudaStreamSynchronize", NAPI_AUTO_LENGTH,
                     "napi_create_promise failed", NAPI_AUTO_LENGTH);
    return nullptr;
  }

  napi_value resource_name;
  st = napi_create_string_utf8(env, "vbt_cuda_stream_synchronize",
                               NAPI_AUTO_LENGTH, &resource_name);
  if (!CheckNapiOkImpl(env, st,
                       "CudaStreamSynchronize/create_resource_name")) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    return nullptr;
  }

  st = napi_create_async_work(
      env,
      nullptr,
      resource_name,
      [](napi_env /*env*/, void* data) {
        auto* job = static_cast<StreamSyncJob*>(data);
        try {
          job->stream->stream.synchronize();
          job->success = true;
        } catch (const std::bad_alloc& e) {
          job->success = false;
          job->error_code = "EOOM";
          job->error_message = e.what() ? e.what() : "allocation failed";
        } catch (const std::runtime_error& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = e.what() ? e.what() : "runtime error";
        } catch (const std::exception& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = e.what() ? e.what() : "error";
        } catch (...) {
          job->success = false;
          job->error_code = "EUNKNOWN";
          job->error_message = "unknown internal error";
        }
      },
      [](napi_env env, napi_status status, void* data) {
        auto* job = static_cast<StreamSyncJob*>(data);

        g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);

        if (job->work != nullptr) {
          napi_delete_async_work(env, job->work);
          job->work = nullptr;
        }

        if (status != napi_ok && job->success) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = "async worker failed";
        }

        if (job->success) {
          napi_value undef;
          napi_get_undefined(env, &undef);
          napi_resolve_deferred(env, job->deferred, undef);
        } else {
          std::string msg =
              "cuda.Stream.synchronize: " + job->error_message;
          napi_value err = MakeErrorWithCode(
              env, msg, job->error_code.c_str(),
              job->error_code == "EINVAL");
          napi_reject_deferred(env, job->deferred, err);
        }

        delete job;
      },
      job,
      &job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    CHECK_NAPI_OK(env, st, "CudaStreamSynchronize/create_async_work");
    return promise;  // CHECK_NAPI_OK already threw.
  }

  st = napi_queue_async_work(env, job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    CHECK_NAPI_OK(env, st, "CudaStreamSynchronize/queue_async_work");
    return promise;
  }

  return promise;
}

// ==== Event class ===========================================================

napi_value CudaEventConstructor(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Event.create")) return nullptr;

  CudaRuntimeState* rt =
      RequireCudaRuntimeState(env, "cuda.Event.create");
  if (!rt) return nullptr;

  napi_value this_arg;
  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaEventConstructor/get_cb_info");

  bool enable_timing = false;
  if (argc >= 1) {
    napi_valuetype t;
    st = napi_typeof(env, args[0], &t);
    if (!CheckNapiOkImpl(env, st, "CudaEventConstructor/opts_typeof")) {
      return nullptr;
    }
    if (t == napi_object) {
      napi_value v;
      st = napi_get_named_property(env, args[0], "enableTiming", &v);
      if (st == napi_ok) {
        napi_valuetype vt;
        st = napi_typeof(env, v, &vt);
        if (!CheckNapiOkImpl(env, st,
                             "CudaEventConstructor/enableTiming_typeof")) {
          return nullptr;
        }
        if (vt != napi_undefined && vt != napi_null) {
          bool b = false;
          st = napi_get_value_bool(env, v, &b);
          if (!CheckNapiOkImpl(env, st,
                               "CudaEventConstructor/get_enableTiming")) {
            return nullptr;
          }
          enable_timing = b;
        }
      }
    }
  }

  auto native = std::make_shared<CudaEventState>();
  native->enable_timing = enable_timing;

#ifdef VBT_NODE_CUDA_DEBUG
  rt->event_instances.fetch_add(1, std::memory_order_relaxed);
#endif

  auto* box = new CudaEventHandleNative(std::move(native));

  st = napi_wrap(env, this_arg, box, CudaEventFinalizer, nullptr, nullptr);
  if (st != napi_ok) {
    delete box;
    CHECK_NAPI_OK(env, st, "CudaEventConstructor/wrap");
    return nullptr;
  }

  return this_arg;
}

napi_value CudaEventIsCreated(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Event.isCreated")) return nullptr;

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaEventIsCreated/get_cb_info");

  CudaEventHandleNative* box = nullptr;
  if (!UnwrapEventHandle(env, this_arg, &box)) return nullptr;

  bool created = (*box)->created;

  napi_value out;
  st = napi_get_boolean(env, created, &out);
  CHECK_NAPI_OK(env, st, "CudaEventIsCreated/get_boolean");
  return out;
}

napi_value CudaEventQuery(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Event.query")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.Event.query", &rt)) {
    return nullptr;
  }

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaEventQuery/get_cb_info");

  CudaEventHandleNative* box = nullptr;
  if (!UnwrapEventHandle(env, this_arg, &box)) return nullptr;

  bool done = true;
  if ((*box)->event.has_value()) {
    done = (*box)->event->query();
  }

  napi_value out;
  st = napi_get_boolean(env, done, &out);
  CHECK_NAPI_OK(env, st, "CudaEventQuery/get_boolean");
  return out;
}

napi_value CudaEventToString(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Event.toString")) return nullptr;

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaEventToString/get_cb_info");

  CudaEventHandleNative* box = nullptr;
  if (!UnwrapEventHandle(env, this_arg, &box)) return nullptr;

  const CudaEventState& state = **box;
  std::string repr = "cuda.Event(created=";
  repr += state.created ? "true" : "false";
  repr += ", enableTiming=";
  repr += state.enable_timing ? "true" : "false";
  repr += ")";

  napi_value out;
  st = napi_create_string_utf8(env, repr.c_str(), repr.size(), &out);
  CHECK_NAPI_OK(env, st, "CudaEventToString/create_string");
  return out;
}

struct EventSyncJob {
  CudaEventHandleNative event;
  napi_deferred deferred{nullptr};
  napi_async_work work{nullptr};
  bool success{false};
  std::string error_message;
  std::string error_code{"ERUNTIME"};
};

napi_value CudaEventSynchronize(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.Event.synchronize")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.Event.synchronize", &rt)) {
    return nullptr;
  }

  napi_value this_arg;
  size_t argc = 0;
  napi_status st =
      napi_get_cb_info(env, info, &argc, nullptr, &this_arg, nullptr);
  CHECK_NAPI_OK(env, st, "CudaEventSynchronize/get_cb_info");

  CudaEventHandleNative* box = nullptr;
  if (!UnwrapEventHandle(env, this_arg, &box)) return nullptr;

  CudaEventState& state = **box;

  // resolves immediately.
  if (!state.created || !state.event.has_value()) {
    napi_deferred deferred;
    napi_value promise;
    st = napi_create_promise(env, &deferred, &promise);
    if (st != napi_ok) {
      napi_fatal_error("vbt.node.CudaEventSynchronize", NAPI_AUTO_LENGTH,
                       "napi_create_promise failed", NAPI_AUTO_LENGTH);
      return nullptr;
    }
    napi_value undef;
    napi_get_undefined(env, &undef);
    napi_resolve_deferred(env, deferred, undef);
    return promise;
  }

  // Ensure env config (VBT_NODE_MAX_INFLIGHT_OPS) has been read.
  (void)max_inflight_ops();

  const std::uint32_t prev =
      g_inflight_ops.fetch_add(1, std::memory_order_acq_rel);
  const std::uint32_t current = prev + 1;
  if (prev >= g_max_inflight_ops) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    ThrowCudaRuntimeErrorSimple(
        env,
        "cuda.Event.synchronize: too many inflight ops (see VBT_NODE_MAX_INFLIGHT_OPS)",
        "ERUNTIME");
    return nullptr;
  }

  UpdateAsyncPeakInflight(current);

  auto* job = new EventSyncJob();
  job->event = *box;

  napi_value promise;
  st = napi_create_promise(env, &job->deferred, &promise);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    napi_fatal_error("vbt.node.CudaEventSynchronize", NAPI_AUTO_LENGTH,
                     "napi_create_promise failed", NAPI_AUTO_LENGTH);
    return nullptr;
  }

  napi_value resource_name;
  st = napi_create_string_utf8(env, "vbt_cuda_event_synchronize",
                               NAPI_AUTO_LENGTH, &resource_name);
  if (!CheckNapiOkImpl(env, st,
                       "CudaEventSynchronize/create_resource_name")) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    return nullptr;
  }

  st = napi_create_async_work(
      env,
      nullptr,
      resource_name,
      [](napi_env /*env*/, void* data) {
        auto* job = static_cast<EventSyncJob*>(data);
        try {
          job->event->event->synchronize();
          job->success = true;
        } catch (const std::bad_alloc& e) {
          job->success = false;
          job->error_code = "EOOM";
          job->error_message = e.what() ? e.what() : "allocation failed";
        } catch (const std::runtime_error& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = e.what() ? e.what() : "runtime error";
        } catch (const std::exception& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = e.what() ? e.what() : "error";
        } catch (...) {
          job->success = false;
          job->error_code = "EUNKNOWN";
          job->error_message = "unknown internal error";
        }
      },
      [](napi_env env, napi_status status, void* data) {
        auto* job = static_cast<EventSyncJob*>(data);

        g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);

        if (job->work != nullptr) {
          napi_delete_async_work(env, job->work);
          job->work = nullptr;
        }

        if (status != napi_ok && job->success) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = "async worker failed";
        }

        if (job->success) {
          napi_value undef;
          napi_get_undefined(env, &undef);
          napi_resolve_deferred(env, job->deferred, undef);
        } else {
          std::string msg =
              "cuda.Event.synchronize: " + job->error_message;
          napi_value err = MakeErrorWithCode(
              env, msg, job->error_code.c_str(),
              job->error_code == "EINVAL");
          napi_reject_deferred(env, job->deferred, err);
        }

        delete job;
      },
      job,
      &job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    CHECK_NAPI_OK(env, st, "CudaEventSynchronize/create_async_work");
    return promise;
  }

  st = napi_queue_async_work(env, job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    CHECK_NAPI_OK(env, st, "CudaEventSynchronize/queue_async_work");
    return promise;
  }

  return promise;
}

}  // namespace

// ==== Public addon entrypoints ==============================================

static void AccumulateDeviceStats(vbt::cuda::DeviceStats& agg,
                                  const vbt::cuda::DeviceStats& st) {
  agg.allocated_bytes_all_current     += st.allocated_bytes_all_current;
  agg.reserved_bytes_all_current      += st.reserved_bytes_all_current;
  agg.max_allocated_bytes_all         += st.max_allocated_bytes_all;
  agg.max_reserved_bytes_all          += st.max_reserved_bytes_all;
  agg.requested_bytes_all_current     += st.requested_bytes_all_current;
  agg.max_requested_bytes_all         += st.max_requested_bytes_all;
  agg.num_alloc_retries               += st.num_alloc_retries;
  agg.num_ooms                        += st.num_ooms;
  agg.num_device_alloc                += st.num_device_alloc;
  agg.num_device_free                 += st.num_device_free;
  agg.tolerance_fills_count           += st.tolerance_fills_count;
  agg.tolerance_fills_bytes           += st.tolerance_fills_bytes;
  agg.deferred_flush_attempts         += st.deferred_flush_attempts;
  agg.deferred_flush_successes        += st.deferred_flush_successes;
  agg.num_prev_owner_fences           += st.num_prev_owner_fences;
  agg.inactive_split_blocks_all       += st.inactive_split_blocks_all;
  agg.inactive_split_bytes_all        += st.inactive_split_bytes_all;
  agg.fraction_cap_breaches           += st.fraction_cap_breaches;
  agg.fraction_cap_misfires           += st.fraction_cap_misfires;
  agg.gc_passes                       += st.gc_passes;
  agg.gc_reclaimed_bytes              += st.gc_reclaimed_bytes;
  agg.graphs_pools_created            += st.graphs_pools_created;
  agg.graphs_pools_released           += st.graphs_pools_released;
  agg.graphs_pools_active             += st.graphs_pools_active;
}

static bool SetDoubleProperty(napi_env env,
                              napi_value obj,
                              const char* name,
                              double value,
                              const char* context) {
  napi_value v;
  napi_status st = napi_create_double(env, value, &v);
  if (!CheckNapiOkImpl(env, st, context)) {
    return false;
  }
  st = napi_set_named_property(env, obj, name, v);
  if (!CheckNapiOkImpl(env, st, context)) {
    return false;
  }
  return true;
}

static napi_value BuildNestedStatsObject(
    napi_env env,
    const vbt::cuda::DeviceStats& stats) {
  napi_value result;
  napi_status st = napi_create_object(env, &result);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/create_object");

  // Gauge families: allocated_bytes, reserved_bytes, requested_bytes.
  auto make_family = [&](std::uint64_t current,
                         std::uint64_t peak,
                         const char* context) -> napi_value {
    napi_value fam;
    napi_value all;
    napi_status st_local = napi_create_object(env, &fam);
    CHECK_NAPI_OK(env, st_local, context);
    st_local = napi_create_object(env, &all);
    CHECK_NAPI_OK(env, st_local, context);

    if (!SetDoubleProperty(env, all, "current",
                           static_cast<double>(current), context)) {
      return nullptr;
    }
    if (!SetDoubleProperty(env, all, "peak",
                           static_cast<double>(peak), context)) {
      return nullptr;
    }
    if (!SetDoubleProperty(env, all, "allocated", 0.0, context)) {
      return nullptr;
    }
    if (!SetDoubleProperty(env, all, "freed", 0.0, context)) {
      return nullptr;
    }

    st_local = napi_set_named_property(env, fam, "all", all);
    CHECK_NAPI_OK(env, st_local, context);
    return fam;
  };

  napi_value fam_alloc = make_family(stats.allocated_bytes_all_current,
                                     stats.max_allocated_bytes_all,
                                     "CudaMemoryStatsAsNested/allocated_bytes");
  if (!fam_alloc) return nullptr;

  napi_value fam_reserved = make_family(stats.reserved_bytes_all_current,
                                        stats.max_reserved_bytes_all,
                                        "CudaMemoryStatsAsNested/reserved_bytes");
  if (!fam_reserved) return nullptr;

  napi_value fam_requested = make_family(stats.requested_bytes_all_current,
                                         stats.max_requested_bytes_all,
                                         "CudaMemoryStatsAsNested/requested_bytes");
  if (!fam_requested) return nullptr;

  st = napi_set_named_property(env, result, "allocated_bytes", fam_alloc);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/set_allocated_bytes");
  st = napi_set_named_property(env, result, "reserved_bytes", fam_reserved);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/set_reserved_bytes");
  st = napi_set_named_property(env, result, "requested_bytes", fam_requested);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/set_requested_bytes");

  // device_stats.aggregated with scalar counters.
  napi_value dev_stats;
  st = napi_create_object(env, &dev_stats);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/create_device_stats");

  napi_value aggregated;
  st = napi_create_object(env, &aggregated);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/create_aggregated");

  auto add_agg = [&](const char* name, std::uint64_t value) -> bool {
    return SetDoubleProperty(env, aggregated, name,
                             static_cast<double>(value),
                             "CudaMemoryStatsAsNested/device_stats");
  };

  if (!add_agg("num_alloc_retries", stats.num_alloc_retries)) return nullptr;
  if (!add_agg("num_ooms", stats.num_ooms)) return nullptr;
  if (!add_agg("num_device_alloc", stats.num_device_alloc)) return nullptr;
  if (!add_agg("num_device_free", stats.num_device_free)) return nullptr;
  if (!add_agg("tolerance_fills_count", stats.tolerance_fills_count)) return nullptr;
  if (!add_agg("tolerance_fills_bytes", stats.tolerance_fills_bytes)) return nullptr;
  if (!add_agg("deferred_flush_attempts", stats.deferred_flush_attempts)) return nullptr;
  if (!add_agg("deferred_flush_successes", stats.deferred_flush_successes)) return nullptr;
  if (!add_agg("num_prev_owner_fences", stats.num_prev_owner_fences)) return nullptr;
  if (!add_agg("inactive_split_blocks_all", stats.inactive_split_blocks_all)) return nullptr;
  if (!add_agg("inactive_split_bytes_all", stats.inactive_split_bytes_all)) return nullptr;
  if (!add_agg("fraction_cap_breaches", stats.fraction_cap_breaches)) return nullptr;
  if (!add_agg("fraction_cap_misfires", stats.fraction_cap_misfires)) return nullptr;
  if (!add_agg("gc_passes", stats.gc_passes)) return nullptr;
  if (!add_agg("gc_reclaimed_bytes", stats.gc_reclaimed_bytes)) return nullptr;
  if (!add_agg("graphs_pools_created", stats.graphs_pools_created)) return nullptr;
  if (!add_agg("graphs_pools_released", stats.graphs_pools_released)) return nullptr;
  if (!add_agg("graphs_pools_active", stats.graphs_pools_active)) return nullptr;

  st = napi_set_named_property(env, dev_stats, "aggregated", aggregated);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/set_device_stats_aggregated");
  st = napi_set_named_property(env, result, "device_stats", dev_stats);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/set_device_stats");

  return result;
}

static napi_value BuildFlatStatsObject(
    napi_env env,
    const vbt::cuda::DeviceStats& stats) {
  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  CHECK_NAPI_OK(env, st, "CudaMemoryStats/create_object");

  auto add = [&](const char* name, std::uint64_t value) -> bool {
    return SetDoubleProperty(env, obj, name,
                             static_cast<double>(value),
                             "CudaMemoryStats/set_double");
  };

  if (!add("allocated_bytes.all.current", stats.allocated_bytes_all_current)) return nullptr;
  if (!add("allocated_bytes.all.peak", stats.max_allocated_bytes_all)) return nullptr;
  if (!add("allocated_bytes.all.allocated", 0)) return nullptr;
  if (!add("allocated_bytes.all.freed", 0)) return nullptr;

  if (!add("reserved_bytes.all.current", stats.reserved_bytes_all_current)) return nullptr;
  if (!add("reserved_bytes.all.peak", stats.max_reserved_bytes_all)) return nullptr;
  if (!add("reserved_bytes.all.allocated", 0)) return nullptr;
  if (!add("reserved_bytes.all.freed", 0)) return nullptr;

  if (!add("requested_bytes.all.current", stats.requested_bytes_all_current)) return nullptr;
  if (!add("requested_bytes.all.peak", stats.max_requested_bytes_all)) return nullptr;
  if (!add("requested_bytes.all.allocated", 0)) return nullptr;
  if (!add("requested_bytes.all.freed", 0)) return nullptr;

  if (!add("num_alloc_retries", stats.num_alloc_retries)) return nullptr;
  if (!add("num_ooms", stats.num_ooms)) return nullptr;
  if (!add("num_device_alloc", stats.num_device_alloc)) return nullptr;
  if (!add("num_device_free", stats.num_device_free)) return nullptr;
  if (!add("tolerance_fills_count", stats.tolerance_fills_count)) return nullptr;
  if (!add("tolerance_fills_bytes", stats.tolerance_fills_bytes)) return nullptr;
  if (!add("deferred_flush_attempts", stats.deferred_flush_attempts)) return nullptr;
  if (!add("deferred_flush_successes", stats.deferred_flush_successes)) return nullptr;
  if (!add("num_prev_owner_fences", stats.num_prev_owner_fences)) return nullptr;
  if (!add("inactive_split_blocks_all", stats.inactive_split_blocks_all)) return nullptr;
  if (!add("inactive_split_bytes_all", stats.inactive_split_bytes_all)) return nullptr;
  if (!add("fraction_cap_breaches", stats.fraction_cap_breaches)) return nullptr;
  if (!add("fraction_cap_misfires", stats.fraction_cap_misfires)) return nullptr;
  if (!add("gc_passes", stats.gc_passes)) return nullptr;
  if (!add("gc_reclaimed_bytes", stats.gc_reclaimed_bytes)) return nullptr;
  if (!add("graphs_pools_created", stats.graphs_pools_created)) return nullptr;
  if (!add("graphs_pools_released", stats.graphs_pools_released)) return nullptr;
  if (!add("graphs_pools_active", stats.graphs_pools_active)) return nullptr;

  return obj;
}

napi_value HasCuda(napi_env env, napi_callback_info /*info*/) {
  if (!EnsureOnMainThread(env, "cuda.isAvailable")) return nullptr;

  CudaRuntimeState* rt = RequireCudaRuntimeState(env, "cuda.isAvailable");
  if (!rt) return nullptr;

#ifdef VBT_NODE_CUDA_DEBUG
  rt->runtime_queries.fetch_add(1, std::memory_order_relaxed);
#endif

  napi_value out;
  napi_status st = napi_get_boolean(env, rt->has_cuda, &out);
  CHECK_NAPI_OK(env, st, "HasCuda/get_boolean");
  return out;
}

napi_value CudaDeviceCount(napi_env env, napi_callback_info /*info*/) {
  if (!EnsureOnMainThread(env, "cuda.deviceCount")) return nullptr;

  CudaRuntimeState* rt = RequireCudaRuntimeState(env, "cuda.deviceCount");
  if (!rt) return nullptr;

#ifdef VBT_NODE_CUDA_DEBUG
  rt->runtime_queries.fetch_add(1, std::memory_order_relaxed);
#endif

  napi_value out;
  napi_status st = napi_create_int32(env, rt->device_count, &out);
  CHECK_NAPI_OK(env, st, "CudaDeviceCount/create_int32");
  return out;
}

napi_value CudaCurrentDevice(napi_env env, napi_callback_info /*info*/) {
  if (!EnsureOnMainThread(env, "cuda.currentDevice")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.currentDevice", &rt)) {
    return nullptr;
  }

  int dev = -1;
#if VBT_WITH_CUDA
  cudaError_t st_cuda = cudaGetDevice(&dev);
  if (st_cuda != cudaSuccess) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.currentDevice: failed to query current device",
        "ERUNTIME");
    return nullptr;
  }
#else
  (void)rt;
  ThrowCudaRuntimeErrorSimple(
      env, "cuda.currentDevice: CUDA support not built", "ENOCUDA");
  return nullptr;
#endif

  if (dev < 0 || dev >= rt->device_count) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.currentDevice: driver returned invalid device index",
        "ERUNTIME");
    return nullptr;
  }

#ifdef VBT_NODE_CUDA_DEBUG
  rt->runtime_queries.fetch_add(1, std::memory_order_relaxed);
#endif

  napi_value out;
  napi_status st = napi_create_int32(env, dev, &out);
  CHECK_NAPI_OK(env, st, "CudaCurrentDevice/create_int32");
  return out;
}

napi_value CudaSetDevice(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.setDevice")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.setDevice", &rt)) {
    return nullptr;
  }

  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CudaSetDevice/get_cb_info");

  if (argc < 1) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.setDevice: index argument is required", "EINVAL");
    return nullptr;
  }

  int idx = 0;
  if (!ParseDeviceIndex(env, args[0], rt->device_count, "cuda.setDevice",
                        &idx)) {
    return nullptr;
  }

#if VBT_WITH_CUDA
  cudaError_t st_cuda = cudaSetDevice(idx);
  if (st_cuda != cudaSuccess) {
    ThrowCudaRuntimeErrorSimple(
        env, "cuda.setDevice: CUDA driver error", "ERUNTIME");
    return nullptr;
  }
#else
  (void)idx;
  ThrowCudaRuntimeErrorSimple(
      env, "cuda.setDevice: CUDA support not built", "ENOCUDA");
  return nullptr;
#endif

#ifdef VBT_NODE_CUDA_DEBUG
  rt->set_device_calls.fetch_add(1, std::memory_order_relaxed);
#endif

  napi_value undef;
  st = napi_get_undefined(env, &undef);
  CHECK_NAPI_OK(env, st, "CudaSetDevice/get_undefined");
  return undef;
}

napi_value CudaMemoryStatsAsNested(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "_cudaMemoryStatsAsNested")) return nullptr;

  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CudaMemoryStatsAsNested/get_cb_info");

  napi_value js_device = (argc >= 1) ? args[0] : nullptr;

  auto parsed_opt = ParseStatsDeviceArg(env, js_device);
  if (!parsed_opt.has_value()) {
    return nullptr;
  }
  ParsedStatsDeviceArg parsed = *parsed_opt;

  vbt::cuda::DeviceStats stats{};

  if (!parsed.cpu_only) {
    try {
      if (parsed.all_devices) {
        CudaRuntimeState* rt =
            RequireCudaRuntimeState(env, "_cudaMemoryStatsAsNested");
        if (!rt) return nullptr;
        const int32_t device_count = rt->device_count;
        vbt::cuda::DeviceStats agg{};
        bool have = false;
        for (int32_t i = 0; i < device_count; ++i) {
          vbt::cuda::DeviceStats cur =
              Allocator::get(static_cast<DeviceIndex>(i)).getDeviceStats();
          if (!have) {
            agg = cur;
            have = true;
          } else {
            AccumulateDeviceStats(agg, cur);
          }
        }
        if (have) {
          stats = agg;
        }
      } else {
        stats = Allocator::get(static_cast<DeviceIndex>(parsed.device))
                    .getDeviceStats();
      }
    } catch (const std::bad_alloc& e) {
      std::string msg = e.what() ? e.what() : "allocation failed";
      std::string full_msg = std::string("_cudaMemoryStatsAsNested: ") + msg;
      ThrowCudaRuntimeErrorSimple(
          env,
          full_msg.c_str(),
          "EOOM");
      return nullptr;
    } catch (const std::exception& e) {
      std::string msg = e.what() ? e.what() : "runtime error";
      std::string full_msg = std::string("_cudaMemoryStatsAsNested: ") + msg;
      ThrowCudaRuntimeErrorSimple(
          env,
          full_msg.c_str(),
          "ERUNTIME");
      return nullptr;
    } catch (...) {
      ThrowCudaRuntimeErrorSimple(
          env,
          "_cudaMemoryStatsAsNested: unknown internal error",
          "ERUNTIME");
      return nullptr;
    }
  }

  napi_value result = BuildNestedStatsObject(env, stats);
  return result;
}

napi_value CudaMemoryStats(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "_cudaMemoryStats")) return nullptr;

  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CudaMemoryStats/get_cb_info");

  napi_value js_device = (argc >= 1) ? args[0] : nullptr;

  auto parsed_opt = ParseStatsDeviceArg(env, js_device);
  if (!parsed_opt.has_value()) {
    return nullptr;
  }
  ParsedStatsDeviceArg parsed = *parsed_opt;

  vbt::cuda::DeviceStats stats{};

  if (!parsed.cpu_only) {
    try {
      if (parsed.all_devices) {
        CudaRuntimeState* rt =
            RequireCudaRuntimeState(env, "_cudaMemoryStats");
        if (!rt) return nullptr;
        const int32_t device_count = rt->device_count;
        vbt::cuda::DeviceStats agg{};
        bool have = false;
        for (int32_t i = 0; i < device_count; ++i) {
          vbt::cuda::DeviceStats cur =
              Allocator::get(static_cast<DeviceIndex>(i)).getDeviceStats();
          if (!have) {
            agg = cur;
            have = true;
          } else {
            AccumulateDeviceStats(agg, cur);
          }
        }
        if (have) {
          stats = agg;
        }
      } else {
        stats = Allocator::get(static_cast<DeviceIndex>(parsed.device))
                    .getDeviceStats();
      }
    } catch (const std::bad_alloc& e) {
      std::string msg = e.what() ? e.what() : "allocation failed";
      std::string full_msg = std::string("_cudaMemoryStats: ") + msg;
      ThrowCudaRuntimeErrorSimple(
          env,
          full_msg.c_str(),
          "EOOM");
      return nullptr;
    } catch (const std::exception& e) {
      std::string msg = e.what() ? e.what() : "runtime error";
      std::string full_msg = std::string("_cudaMemoryStats: ") + msg;
      ThrowCudaRuntimeErrorSimple(
          env,
          full_msg.c_str(),
          "ERUNTIME");
      return nullptr;
    } catch (...) {
      ThrowCudaRuntimeErrorSimple(
          env,
          "_cudaMemoryStats: unknown internal error",
          "ERUNTIME");
      return nullptr;
    }
  }

  napi_value result = BuildFlatStatsObject(env, stats);
  return result;
}

napi_value CudaMemorySnapshot(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "_cudaMemorySnapshot")) return nullptr;

  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CudaMemorySnapshot/get_cb_info");

  napi_value js_device = (argc >= 1) ? args[0] : nullptr;

  auto parsed_opt = ParseStatsDeviceArg(env, js_device);
  if (!parsed_opt.has_value()) {
    return nullptr;
  }
  ParsedStatsDeviceArg parsed = *parsed_opt;

  napi_value arr;
  st = napi_create_array(env, &arr);
  CHECK_NAPI_OK(env, st, "CudaMemorySnapshot/create_array");

  if (parsed.cpu_only) {
    // CPU-only builds synthesize an empty snapshot without touching CUDA.
    return arr;
  }

  try {
    std::optional<DeviceIndex> filter;
    if (!parsed.all_devices) {
      filter = static_cast<DeviceIndex>(parsed.device);
    }

    std::vector<MemorySegmentSnapshot> snaps = snapshot_memory_segments(filter);
    for (std::size_t i = 0; i < snaps.size(); ++i) {
      const MemorySegmentSnapshot& seg = snaps[i];

      napi_value obj;
      st = napi_create_object(env, &obj);
      CHECK_NAPI_OK(env, st, "CudaMemorySnapshot/create_segment");

      napi_value dev;
      st = napi_create_int32(env, static_cast<int32_t>(seg.device), &dev);
      CHECK_NAPI_OK(env, st, "CudaMemorySnapshot/set_device");
      st = napi_set_named_property(env, obj, "device", dev);
      CHECK_NAPI_OK(env, st, "CudaMemorySnapshot/set_device_prop");

      if (!SetDoubleProperty(env, obj, "poolId",
                             static_cast<double>(seg.pool_id),
                             "CudaMemorySnapshot/set_poolId")) {
        return nullptr;
      }
      if (!SetDoubleProperty(env, obj, "bytesReserved",
                             static_cast<double>(seg.bytes_reserved),
                             "CudaMemorySnapshot/set_bytesReserved")) {
        return nullptr;
      }
      if (!SetDoubleProperty(env, obj, "bytesActive",
                             static_cast<double>(seg.bytes_active),
                             "CudaMemorySnapshot/set_bytesActive")) {
        return nullptr;
      }
      if (!SetDoubleProperty(env, obj, "blocks",
                             static_cast<double>(seg.blocks),
                             "CudaMemorySnapshot/set_blocks")) {
        return nullptr;
      }

      st = napi_set_element(env, arr, static_cast<uint32_t>(i), obj);
      CHECK_NAPI_OK(env, st, "CudaMemorySnapshot/set_element");
    }
  } catch (const std::bad_alloc& e) {
    std::string msg = e.what() ? e.what() : "allocation failed";
    std::string full_msg = std::string("_cudaMemorySnapshot: ") + msg;
    ThrowCudaRuntimeErrorSimple(
        env,
        full_msg.c_str(),
        "EOOM");
    return nullptr;
  } catch (const std::exception& e) {
    std::string msg = e.what() ? e.what() : "runtime error";
    std::string full_msg = std::string("_cudaMemorySnapshot: ") + msg;
    ThrowCudaRuntimeErrorSimple(
        env,
        full_msg.c_str(),
        "ERUNTIME");
    return nullptr;
  } catch (...) {
    ThrowCudaRuntimeErrorSimple(
        env,
        "_cudaMemorySnapshot: unknown internal error",
        "ERUNTIME");
    return nullptr;
  }

  return arr;
}

napi_value CudaH2DAsync(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.h2d")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.h2d", &rt)) {
    return nullptr;
  }

  // Ensure env config (VBT_NODE_MAX_INFLIGHT_OPS) has been read.
  (void)max_inflight_ops();

  size_t argc = 3;
  napi_value args[3];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CudaH2DAsync/get_cb_info");

  if (argc < 3) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: (src, sizes, opts) arguments are required",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  napi_value js_src = args[0];
  napi_value js_sizes = args[1];
  napi_value js_opts = args[2];

  // Validate src TypedArray.
  bool is_typedarray = false;
  st = napi_is_typedarray(env, js_src, &is_typedarray);
  if (!CheckNapiOkImpl(env, st, "CudaH2DAsync/src_is_typedarray")) {
    return nullptr;
  }
  if (!is_typedarray) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: src must be a TypedArray",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  napi_typedarray_type ta_type;
  size_t ta_length = 0;
  void* host_data = nullptr;
  napi_value ab;
  size_t byte_offset = 0;
  st = napi_get_typedarray_info(env,
                                js_src,
                                &ta_type,
                                &ta_length,
                                &host_data,
                                &ab,
                                &byte_offset);
  if (!CheckNapiOkImpl(env, st, "CudaH2DAsync/get_typedarray_info")) {
    return nullptr;
  }

  // Parse sizes.
  std::vector<int64_t> sizes;
  if (!ParseSizes(env, js_sizes, &sizes)) {
    // ParseSizes already threw a TypeError.
    return nullptr;
  }

  // Validate opts object.
  napi_valuetype t_opts;
  st = napi_typeof(env, js_opts, &t_opts);
  if (!CheckNapiOkImpl(env, st, "CudaH2DAsync/opts_typeof")) {
    return nullptr;
  }
  if (t_opts != napi_object) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: options must be an object",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  // opts.dtype (required).
  napi_value js_dtype;
  st = napi_get_named_property(env, js_opts, "dtype", &js_dtype);
  if (st != napi_ok) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: opts.dtype is required",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  napi_valuetype t_dtype;
  st = napi_typeof(env, js_dtype, &t_dtype);
  if (!CheckNapiOkImpl(env, st, "CudaH2DAsync/dtype_typeof")) {
    return nullptr;
  }
  if (t_dtype == napi_undefined || t_dtype == napi_null) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: opts.dtype is required",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }
  if (t_dtype != napi_string) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: dtype must be a string",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  bool ok = false;
  std::string dtype_token =
      detail::GetStringUtf8(env, js_dtype, "CudaH2DAsync/dtype_string", &ok);
  if (!ok) return nullptr;

  ScalarType dtype = ScalarType::Float32;
  if (dtype_token == "float32") {
    dtype = ScalarType::Float32;
  } else if (dtype_token == "int32") {
    dtype = ScalarType::Int32;
  } else if (dtype_token == "int64") {
    dtype = ScalarType::Int64;
  } else if (dtype_token == "bool") {
    dtype = ScalarType::Bool;
  } else {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: unsupported dtype; expected one of {float32,int32,int64,bool}",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  // TypedArray vs dtype compatibility.
  bool dtype_matches = false;
  switch (ta_type) {
    case napi_float32_array:
      dtype_matches = (dtype == ScalarType::Float32);
      break;
    case napi_int32_array:
      dtype_matches = (dtype == ScalarType::Int32);
      break;
    case napi_bigint64_array:
      dtype_matches = (dtype == ScalarType::Int64);
      break;
    case napi_uint8_array:
      dtype_matches = (dtype == ScalarType::Bool);
      break;
    default:
      dtype_matches = false;
      break;
  }
  if (!dtype_matches) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: dtype does not match TypedArray element type",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  // Compute numel and nbytes with overflow checks.
  int64_t numel = 1;
  for (int64_t s : sizes) {
    if (s < 0) {
      ThrowErrorWithCode(
          env,
          "cuda.h2d: sizes entries must be >= 0",
          "EINVAL",
          /*type_error=*/true);
      return nullptr;
    }
    if (s == 0) {
      numel = 0;
      break;
    }
    int64_t tmp = 0;
    if (!checked_mul_i64(numel, s, tmp)) {
      ThrowErrorWithCode(
          env,
          "cuda.h2d: numel overflow",
          "ERUNTIME",
          /*type_error=*/false);
      return nullptr;
    }
    numel = tmp;
  }

  if (static_cast<std::uint64_t>(numel) !=
      static_cast<std::uint64_t>(ta_length)) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: sizes product does not match src.length",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  const int64_t item_b = static_cast<int64_t>(vbt::core::itemsize(dtype));
  int64_t total_bytes = 0;
  if (!checked_mul_i64(numel, item_b, total_bytes)) {
    ThrowErrorWithCode(
        env,
        "cuda.h2d: byte size overflow",
        "ERUNTIME",
        /*type_error=*/false);
    return nullptr;
  }
  std::size_t nbytes = static_cast<std::size_t>(total_bytes);

  // Resolve device index: opts.device (optional) or current device.
  int device_index = -1;

  napi_value js_device;
  st = napi_get_named_property(env, js_opts, "device", &js_device);
  if (st == napi_ok) {
    napi_valuetype t_dev;
    st = napi_typeof(env, js_device, &t_dev);
    if (!CheckNapiOkImpl(env, st, "CudaH2DAsync/device_typeof")) {
      return nullptr;
    }
    if (t_dev != napi_undefined && t_dev != napi_null) {
      if (!ParseDeviceIndex(env,
                            js_device,
                            rt->device_count,
                            "cuda.h2d",
                            &device_index)) {
        return nullptr;
      }
    }
  } else if (st != napi_invalid_arg) {
    if (!CheckNapiOkImpl(env, st, "CudaH2DAsync/device_named_property")) {
      return nullptr;
    }
  }

#if VBT_WITH_CUDA
  if (device_index < 0) {
    int cur = -1;
    cudaError_t st_cuda = cudaGetDevice(&cur);
    if (st_cuda != cudaSuccess) {
      ThrowCudaRuntimeErrorSimple(
          env,
          "cuda.h2d: failed to query current device",
          "ERUNTIME");
      return nullptr;
    }
    if (cur < 0 || cur >= rt->device_count) {
      ThrowCudaRuntimeErrorSimple(
          env,
          "cuda.h2d: driver returned invalid device index",
          "ERUNTIME");
      return nullptr;
    }
    device_index = cur;
  }
#else
  if (device_index < 0) {
    (void)rt;
    ThrowCudaRuntimeErrorSimple(
        env,
        "cuda.h2d: CUDA support not built",
        "ENOCUDA");
    return nullptr;
  }
#endif

  // Enforce inflight cap shared across Node async jobs.
  const std::uint32_t prev =
      g_inflight_ops.fetch_add(1, std::memory_order_acq_rel);
  const std::uint32_t current = prev + 1;
  if (prev >= g_max_inflight_ops) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    g_async_stats.h2d.failed.fetch_add(1, std::memory_order_relaxed);
    ThrowCudaRuntimeErrorSimple(
        env,
        "cuda.h2d: too many inflight ops (see VBT_NODE_MAX_INFLIGHT_OPS)",
        "ERUNTIME");
    return nullptr;
  }

  UpdateAsyncPeakInflight(current);

  // Allocate job and Promise.
  auto* job = new H2DJob();
  job->host_data = host_data;
  job->nbytes = nbytes;
  job->sizes = std::move(sizes);
  job->dtype = dtype;
  job->device_index = device_index;

  napi_value promise;
  st = napi_create_promise(env, &job->deferred, &promise);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    napi_fatal_error("vbt.node.CudaH2DAsync", NAPI_AUTO_LENGTH,
                     "napi_create_promise failed", NAPI_AUTO_LENGTH);
    return nullptr;
  }

  // Hold a reference to src to keep its backing store alive.
  st = napi_create_reference(env, js_src, 1, &job->src_ref);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    ThrowCudaRuntimeErrorSimple(
        env,
        "cuda.h2d: failed to create reference for src TypedArray",
        "ERUNTIME");
    return promise;
  }

  napi_value resource_name;
  st = napi_create_string_utf8(env,
                               "vbt_cuda_h2d",
                               NAPI_AUTO_LENGTH,
                               &resource_name);
  if (!CheckNapiOkImpl(env, st, "CudaH2DAsync/create_resource_name")) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    if (job->src_ref) {
      napi_delete_reference(env, job->src_ref);
      job->src_ref = nullptr;
    }
    delete job;
    return nullptr;
  }

  st = napi_create_async_work(
      env,
      nullptr,
      resource_name,
      [](napi_env /*env*/, void* data) {
        auto* job = static_cast<H2DJob*>(data);
#if VBT_WITH_CUDA
        try {
          int prev_dev = 0;
          cudaError_t st_cuda = cudaGetDevice(&prev_dev);
          if (st_cuda != cudaSuccess) {
            job->success = false;
            job->error_code = "ERUNTIME";
            job->error_message = "cudaGetDevice failed";
            return;
          }

          const int target = job->device_index;
          if (target != prev_dev) {
            st_cuda = cudaSetDevice(target);
            if (st_cuda != cudaSuccess) {
              job->success = false;
              job->error_code = "ERUNTIME";
              job->error_message = "cudaSetDevice failed";
              return;
            }
          }

          auto storage =
              vbt::cuda::new_cuda_storage(job->nbytes, target);
          auto stream = vbt::cuda::getCurrentStream(
              static_cast<vbt::cuda::DeviceIndex>(target));
          vbt::cuda::Allocator& alloc =
              vbt::cuda::Allocator::get(
                  static_cast<vbt::cuda::DeviceIndex>(target));

          if (job->nbytes > 0) {
            cudaError_t st2 = alloc.memcpyAsync(
                storage->data(),
                target,
                job->host_data,
                -1,  // host
                job->nbytes,
                stream,
                /*p2p_enabled=*/false);
            if (st2 != cudaSuccess) {
              const char* msg = cudaGetErrorString(st2);
              job->success = false;
              job->error_code = "ERUNTIME";
              job->error_message =
                  std::string("memcpyAsync failed: ") +
                  (msg ? msg : "");
              if (target != prev_dev) {
                (void)cudaSetDevice(prev_dev);
              }
              return;
            }
            st2 = cudaStreamSynchronize(
                reinterpret_cast<cudaStream_t>(stream.handle()));
            if (st2 != cudaSuccess) {
              const char* msg = cudaGetErrorString(st2);
              job->success = false;
              job->error_code = "ERUNTIME";
              job->error_message =
                  std::string("cudaStreamSynchronize failed: ") +
                  (msg ? msg : "");
              if (target != prev_dev) {
                (void)cudaSetDevice(prev_dev);
              }
              return;
            }
          }

          auto strides = make_contig_strides(job->sizes);
          job->result = TensorImpl(
              storage,
              job->sizes,
              std::move(strides),
              /*storage_offset=*/0,
              job->dtype,
              Device::cuda(target));
          job->success = true;

          if (target != prev_dev) {
            (void)cudaSetDevice(prev_dev);
          }
        } catch (const std::bad_alloc& e) {
          job->success = false;
          job->error_code = "EOOM";
          job->error_message =
              e.what() ? e.what() : "allocation failed";
        } catch (const std::runtime_error& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message =
              e.what() ? e.what() : "runtime error";
        } catch (const std::exception& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message =
              e.what() ? e.what() : "error";
        } catch (...) {
          job->success = false;
          job->error_code = "EUNKNOWN";
          job->error_message = "unknown internal error";
        }
#else
        job->success = false;
        job->error_code = "ENOCUDA";
        job->error_message =
            "CUDA host->device copy requested but CUDA support is not built";
#endif
      },
      [](napi_env env, napi_status status, void* data) {
        auto* job = static_cast<H2DJob*>(data);

        g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);

        if (job->src_ref) {
          napi_delete_reference(env, job->src_ref);
          job->src_ref = nullptr;
        }
        if (job->work != nullptr) {
          napi_delete_async_work(env, job->work);
          job->work = nullptr;
        }

        if (status != napi_ok && job->success) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = "async worker failed";
        }

        if (job->success) {
          napi_value js_tensor = nullptr;
          if (!TryWrapTensorImplAsJsTensor(env, std::move(job->result),
                                           &js_tensor)) {
            g_async_stats.h2d.failed.fetch_add(1, std::memory_order_relaxed);
            napi_value err = MakeErrorWithCode(
                env,
                "cuda.h2d: failed to wrap tensor result",
                "ERUNTIME",
                /*type_error=*/false);
            napi_reject_deferred(env, job->deferred, err);
          } else {
            g_async_stats.h2d.completed.fetch_add(1, std::memory_order_relaxed);
            napi_resolve_deferred(env, job->deferred, js_tensor);
          }
        } else {
          g_async_stats.h2d.failed.fetch_add(1, std::memory_order_relaxed);
          std::string msg = std::string("cuda.h2d: ") + job->error_message;
          napi_value err = MakeErrorWithCode(
              env,
              msg,
              job->error_code.c_str(),
              job->error_code == "EINVAL");
          napi_reject_deferred(env, job->deferred, err);
        }

        delete job;
      },
      job,
      &job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    if (job->src_ref) {
      napi_delete_reference(env, job->src_ref);
      job->src_ref = nullptr;
    }
    delete job;
    CHECK_NAPI_OK(env, st, "CudaH2DAsync/create_async_work");
    return promise;
  }

  st = napi_queue_async_work(env, job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    if (job->src_ref) {
      napi_delete_reference(env, job->src_ref);
      job->src_ref = nullptr;
    }
    if (job->work) {
      napi_delete_async_work(env, job->work);
      job->work = nullptr;
    }
    delete job;
    CHECK_NAPI_OK(env, st, "CudaH2DAsync/queue_async_work");
    return promise;
  }

  LogIfEnabled(LogLevel::kInfo,
               LogCategory::kH2D,
               "cuda.h2d: scheduled async copy",
               {{"bytes", std::to_string(nbytes)},
                {"device", std::to_string(device_index)}});

  g_async_stats.h2d.started.fetch_add(1, std::memory_order_relaxed);
  g_async_stats.h2d.bytes.fetch_add(
      static_cast<std::uint64_t>(nbytes), std::memory_order_relaxed);

  return promise;
}

napi_value CudaD2HAsync(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThread(env, "cuda.d2h")) return nullptr;

  CudaRuntimeState* rt = nullptr;
  if (!EnsureCudaAvailable(env, "cuda.d2h", &rt)) {
    return nullptr;
  }

  // Ensure env config (VBT_NODE_MAX_INFLIGHT_OPS) has been read.
  (void)max_inflight_ops();

  size_t argc = 1;
  napi_value args[1];
  napi_status st =
      napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "CudaD2HAsync/get_cb_info");

  if (argc < 1) {
    ThrowErrorWithCode(
        env,
        "cuda.d2h: tensor handle argument is required",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  JsTensor* jt = nullptr;
  if (!UnwrapJsTensor(env, args[0], &jt)) {
    // UnwrapJsTensor throws TypeError for forged handles.
    return nullptr;
  }

  const TensorImpl& tensor = jt->impl;
  const Device dev = tensor.device();
  if (dev.type != vbt::core::Device::cuda().type) {
    ThrowErrorWithCode(
        env,
        "cuda.d2h: expected CUDA tensor",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  if (!tensor.is_non_overlapping_and_dense()) {
    ThrowErrorWithCode(
        env,
        "cuda.d2h: tensor must be dense-contiguous",
        "EINVAL",
        /*type_error=*/true);
    return nullptr;
  }

  const ScalarType dtype = tensor.dtype();
  napi_typedarray_type ta_type;
  switch (dtype) {
    case ScalarType::Float32:
      ta_type = napi_float32_array;
      break;
    case ScalarType::Int32:
      ta_type = napi_int32_array;
      break;
    case ScalarType::Int64:
      ta_type = napi_bigint64_array;
      break;
    case ScalarType::Bool:
      ta_type = napi_uint8_array;
      break;
    default:
      ThrowErrorWithCode(
          env,
          "cuda.d2h: unsupported dtype for D2H copy",
          "EINVAL",
          /*type_error=*/true);
      return nullptr;
  }

  // Compute nbytes and element length.
  std::vector<int64_t> sizes = tensor.sizes();
  std::size_t nbytes = 0;
  if (!NumelBytes(sizes, dtype, &nbytes)) {
    ThrowErrorWithCode(
        env,
        "cuda.d2h: numel overflow",
        "ERUNTIME",
        /*type_error=*/false);
    return nullptr;
  }

  const std::size_t elem_size =
      static_cast<std::size_t>(vbt::core::itemsize(dtype));
  std::size_t length = 0;
  if (elem_size > 0) {
    length = nbytes / elem_size;
  }

  napi_value ab;
  void* host_data = nullptr;
  st = napi_create_arraybuffer(env, nbytes, &host_data, &ab);
  if (!CheckNapiOkImpl(env, st, "CudaD2HAsync/create_arraybuffer")) {
    return nullptr;
  }

  napi_value js_array;
  st = napi_create_typedarray(env,
                              ta_type,
                              length,
                              ab,
                              /*byte_offset=*/0,
                              &js_array);
  if (!CheckNapiOkImpl(env, st, "CudaD2HAsync/create_typedarray")) {
    return nullptr;
  }

  // Enforce inflight cap shared across Node async jobs.
  const std::uint32_t prev =
      g_inflight_ops.fetch_add(1, std::memory_order_acq_rel);
  const std::uint32_t current = prev + 1;
  if (prev >= g_max_inflight_ops) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    g_async_stats.d2h.failed.fetch_add(1, std::memory_order_relaxed);
    ThrowCudaRuntimeErrorSimple(
        env,
        "cuda.d2h: too many inflight ops (see VBT_NODE_MAX_INFLIGHT_OPS)",
        "ERUNTIME");
    return nullptr;
  }

  UpdateAsyncPeakInflight(current);

  auto* job = new D2HJob();
  job->tensor = tensor;
  job->host_data = host_data;
  job->nbytes = nbytes;

  napi_value promise;
  st = napi_create_promise(env, &job->deferred, &promise);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    napi_fatal_error("vbt.node.CudaD2HAsync", NAPI_AUTO_LENGTH,
                     "napi_create_promise failed", NAPI_AUTO_LENGTH);
    return nullptr;
  }

  // Pin the TypedArray across the async job.
  st = napi_create_reference(env, js_array, 1, &job->array_ref);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    delete job;
    ThrowCudaRuntimeErrorSimple(
        env,
        "cuda.d2h: failed to create reference for result TypedArray",
        "ERUNTIME");
    return promise;
  }

  napi_value resource_name;
  st = napi_create_string_utf8(env,
                               "vbt_cuda_d2h",
                               NAPI_AUTO_LENGTH,
                               &resource_name);
  if (!CheckNapiOkImpl(env, st, "CudaD2HAsync/create_resource_name")) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    if (job->array_ref) {
      napi_delete_reference(env, job->array_ref);
      job->array_ref = nullptr;
    }
    delete job;
    return nullptr;
  }

  st = napi_create_async_work(
      env,
      nullptr,
      resource_name,
      [](napi_env /*env*/, void* data) {
        auto* job = static_cast<D2HJob*>(data);
#if VBT_WITH_CUDA
        try {
          int prev_dev = 0;
          cudaError_t st_cuda = cudaGetDevice(&prev_dev);
          if (st_cuda != cudaSuccess) {
            job->success = false;
            job->error_code = "ERUNTIME";
            job->error_message = "cudaGetDevice failed";
            return;
          }

          const Device dev = job->tensor.device();
          const int src_dev = dev.index;

          cudaError_t st2;
          if (src_dev != prev_dev) {
            st2 = cudaSetDevice(src_dev);
            if (st2 != cudaSuccess) {
              job->success = false;
              job->error_code = "ERUNTIME";
              job->error_message = "cudaSetDevice failed";
              return;
            }
          }

          auto stream = vbt::cuda::getCurrentStream(
              static_cast<vbt::cuda::DeviceIndex>(src_dev));
          vbt::cuda::Allocator& alloc =
              vbt::cuda::Allocator::get(
                  static_cast<vbt::cuda::DeviceIndex>(src_dev));

          if (job->nbytes > 0) {
            st2 = alloc.memcpyAsync(
                job->host_data,
                -1,  // host
                job->tensor.data(),
                src_dev,
                job->nbytes,
                stream,
                /*p2p_enabled=*/false);
            if (st2 != cudaSuccess) {
              const char* msg = cudaGetErrorString(st2);
              job->success = false;
              job->error_code = "ERUNTIME";
              job->error_message =
                  std::string("memcpyAsync failed: ") +
                  (msg ? msg : "");
              if (src_dev != prev_dev) {
                (void)cudaSetDevice(prev_dev);
              }
              return;
            }

            st2 = cudaStreamSynchronize(
                reinterpret_cast<cudaStream_t>(stream.handle()));
            if (st2 != cudaSuccess) {
              const char* msg = cudaGetErrorString(st2);
              job->success = false;
              job->error_code = "ERUNTIME";
              job->error_message =
                  std::string("cudaStreamSynchronize failed: ") +
                  (msg ? msg : "");
              if (src_dev != prev_dev) {
                (void)cudaSetDevice(prev_dev);
              }
              return;
            }
          }

          job->success = true;

          if (src_dev != prev_dev) {
            (void)cudaSetDevice(prev_dev);
          }
        } catch (const std::bad_alloc& e) {
          job->success = false;
          job->error_code = "EOOM";
          job->error_message =
              e.what() ? e.what() : "allocation failed";
        } catch (const std::runtime_error& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message =
              e.what() ? e.what() : "runtime error";
        } catch (const std::exception& e) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message =
              e.what() ? e.what() : "error";
        } catch (...) {
          job->success = false;
          job->error_code = "EUNKNOWN";
          job->error_message = "unknown internal error";
        }
#else
        job->success = false;
        job->error_code = "ENOCUDA";
        job->error_message =
            "CUDA device->host copy requested but CUDA support is not built";
#endif
      },
      [](napi_env env, napi_status status, void* data) {
        auto* job = static_cast<D2HJob*>(data);

        g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);

        if (job->work != nullptr) {
          napi_delete_async_work(env, job->work);
          job->work = nullptr;
        }

        napi_value js_array = nullptr;
        if (job->array_ref) {
          napi_get_reference_value(env, job->array_ref, &js_array);
          napi_delete_reference(env, job->array_ref);
          job->array_ref = nullptr;
        }

        if (status != napi_ok && job->success) {
          job->success = false;
          job->error_code = "ERUNTIME";
          job->error_message = "async worker failed";
        }

        if (job->success) {
          if (!js_array) {
            g_async_stats.d2h.failed.fetch_add(1, std::memory_order_relaxed);
            napi_value err = MakeErrorWithCode(
                env,
                "cuda.d2h: internal error (missing TypedArray)",
                "ERUNTIME",
                /*type_error=*/false);
            napi_reject_deferred(env, job->deferred, err);
          } else {
            g_async_stats.d2h.completed.fetch_add(1, std::memory_order_relaxed);
            napi_resolve_deferred(env, job->deferred, js_array);
          }
        } else {
          g_async_stats.d2h.failed.fetch_add(1, std::memory_order_relaxed);
          std::string msg = std::string("cuda.d2h: ") + job->error_message;
          napi_value err = MakeErrorWithCode(
              env,
              msg,
              job->error_code.c_str(),
              job->error_code == "EINVAL");
          napi_reject_deferred(env, job->deferred, err);
        }

        delete job;
      },
      job,
      &job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    if (job->array_ref) {
      napi_delete_reference(env, job->array_ref);
      job->array_ref = nullptr;
    }
    delete job;
    CHECK_NAPI_OK(env, st, "CudaD2HAsync/create_async_work");
    return promise;
  }

  st = napi_queue_async_work(env, job->work);
  if (st != napi_ok) {
    g_inflight_ops.fetch_sub(1, std::memory_order_acq_rel);
    if (job->array_ref) {
      napi_delete_reference(env, job->array_ref);
      job->array_ref = nullptr;
    }
    if (job->work) {
      napi_delete_async_work(env, job->work);
      job->work = nullptr;
    }
    delete job;
    CHECK_NAPI_OK(env, st, "CudaD2HAsync/queue_async_work");
    return promise;
  }

  LogIfEnabled(LogLevel::kInfo,
               LogCategory::kD2H,
               "cuda.d2h: scheduled async copy",
               {{"bytes", std::to_string(nbytes)}});

  g_async_stats.d2h.started.fetch_add(1, std::memory_order_relaxed);
  g_async_stats.d2h.bytes.fetch_add(
      static_cast<std::uint64_t>(nbytes), std::memory_order_relaxed);

  return promise;
}

napi_value CreateCudaStreamClass(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
      {"deviceIndex", nullptr, CudaStreamDeviceIndex, nullptr, nullptr,
       nullptr, napi_default, nullptr},
      {"query", nullptr, CudaStreamQuery, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"toString", nullptr, CudaStreamToString, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"synchronize", nullptr, CudaStreamSynchronize, nullptr, nullptr,
       nullptr, napi_default, nullptr},
  };

  napi_value ctor;
  napi_status st = napi_define_class(
      env,
      "CudaStream",
      NAPI_AUTO_LENGTH,
      CudaStreamConstructor,
      nullptr,
      sizeof(methods) / sizeof(methods[0]),
      methods,
      &ctor);
  CHECK_NAPI_OK(env, st, "CreateCudaStreamClass/define_class");

  st = napi_set_named_property(env, exports, "CudaStream", ctor);
  CHECK_NAPI_OK(env, st, "CreateCudaStreamClass/set_named_property");

  return ctor;
}

napi_value CreateCudaEventClass(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
      {"isCreated", nullptr, CudaEventIsCreated, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"query", nullptr, CudaEventQuery, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"toString", nullptr, CudaEventToString, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"synchronize", nullptr, CudaEventSynchronize, nullptr, nullptr,
       nullptr, napi_default, nullptr},
  };

  napi_value ctor;
  napi_status st = napi_define_class(
      env,
      "CudaEvent",
      NAPI_AUTO_LENGTH,
      CudaEventConstructor,
      nullptr,
      sizeof(methods) / sizeof(methods[0]),
      methods,
      &ctor);
  CHECK_NAPI_OK(env, st, "CreateCudaEventClass/define_class");

  st = napi_set_named_property(env, exports, "CudaEvent", ctor);
  CHECK_NAPI_OK(env, st, "CreateCudaEventClass/set_named_property");

  return ctor;
}

napi_value RegisterCudaRuntimeBindings(napi_env env, napi_value exports) {
  napi_property_descriptor props[] = {
      {"hasCuda", nullptr, HasCuda, nullptr, nullptr, nullptr, napi_default,
       nullptr},
      {"cudaDeviceCount", nullptr, CudaDeviceCount, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"cudaCurrentDevice", nullptr, CudaCurrentDevice, nullptr, nullptr,
       nullptr, napi_default, nullptr},
      {"cudaSetDevice", nullptr, CudaSetDevice, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"_cudaH2DAsync", nullptr, CudaH2DAsync, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"_cudaD2HAsync", nullptr, CudaD2HAsync, nullptr, nullptr, nullptr,
       napi_default, nullptr},
  };

  napi_status st = napi_define_properties(
      env, exports, sizeof(props) / sizeof(props[0]), props);
  CHECK_NAPI_OK(env, st,
                "RegisterCudaRuntimeBindings/define_properties");

  (void)CreateCudaStreamClass(env, exports);
  (void)CreateCudaEventClass(env, exports);

  return exports;
}

}  // namespace node
}  // namespace vbt
