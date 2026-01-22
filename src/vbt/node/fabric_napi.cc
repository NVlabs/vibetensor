// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/fabric_napi.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>

#include "vbt/cuda/fabric_events.h"
#include "vbt/cuda/fabric_state.h"

#include "vbt/node/cuda_napi.h"  // for AddonData shape + ThrowCudaRuntimeErrorSimple
#include "vbt/node/util.h"

namespace vbt {
namespace node {

namespace {

using vbt::cuda::fabric::FabricEvent;
using vbt::cuda::fabric::FabricEventKind;
using vbt::cuda::fabric::FabricEventLevel;
using vbt::cuda::fabric::FabricEventSnapshot;
using vbt::cuda::fabric::FabricPerDeviceStatsSnapshot;
using vbt::cuda::fabric::FabricStatsSnapshot;

// Local copy of the addon instance-data helpers used in cuda_napi.cc.
// We do not expose these from cuda_napi to keep layering simple.
static CudaRuntimeState* GetCudaRuntimeStateLocal(napi_env env) {
  AddonData* data = nullptr;
  napi_status st = napi_get_instance_data(env,
                                          reinterpret_cast<void**>(&data));
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
    std::string msg = std::string(fn_name ? fn_name : "fabric") +
                      ": must be called on the main VibeTensor JS thread";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return false;
  }
  return true;
}

static napi_value MakeBigIntU64(napi_env env, std::uint64_t v,
                               const char* context) {
  napi_value out;
  napi_status st = napi_create_bigint_uint64(env, v, &out);
  CHECK_NAPI_OK(env, st, context);
  return out;
}

static napi_value MakeInt32(napi_env env, std::int32_t v,
                            const char* context) {
  napi_value out;
  napi_status st = napi_create_int32(env, v, &out);
  CHECK_NAPI_OK(env, st, context);
  return out;
}

static napi_value MakeString(napi_env env, const char* s, const char* context) {
  napi_value out;
  napi_status st = napi_create_string_utf8(env, s ? s : "", NAPI_AUTO_LENGTH, &out);
  CHECK_NAPI_OK(env, st, context);
  return out;
}

static bool SetNamed(napi_env env,
                     napi_value obj,
                     const char* name,
                     napi_value v,
                     const char* context) {
  napi_status st = napi_set_named_property(env, obj, name, v);
  return CheckNapiOkImpl(env, st, context);
}

static const char* FabricEventKindName(FabricEventKind k) {
  switch (k) {
    case FabricEventKind::kOpEnqueue:            return "op_enqueue";
    case FabricEventKind::kOpComplete:           return "op_complete";
    case FabricEventKind::kOpFallback:           return "op_fallback";
    case FabricEventKind::kOpError:              return "op_error";
    case FabricEventKind::kModeChanged:          return "mode_changed";
    case FabricEventKind::kEventLifetimeToggled: return "event_lifetime_toggled";
    case FabricEventKind::kEventsModeChanged:    return "events_mode_changed";
    default:                                     return "unknown";
  }
}

static const char* FabricEventLevelName(FabricEventLevel l) {
  switch (l) {
    case FabricEventLevel::kDebug: return "debug";
    case FabricEventLevel::kInfo:  return "info";
    case FabricEventLevel::kWarn:  return "warn";
    case FabricEventLevel::kError: return "error";
    default:                       return "unknown";
  }
}

static napi_value FabricPerDeviceStatsToJs(napi_env env,
                                          const FabricPerDeviceStatsSnapshot& d) {
  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  CHECK_NAPI_OK(env, st, "FabricPerDeviceStats/create_object");

  if (!SetNamed(env, obj, "device_index",
                MakeInt32(env, static_cast<std::int32_t>(d.device_index),
                          "FabricPerDeviceStats/device_index"),
                "FabricPerDeviceStats/set_device_index")) {
    return nullptr;
  }

  if (!SetNamed(env, obj, "ops_as_primary",
                MakeBigIntU64(env, d.ops_as_primary,
                              "FabricPerDeviceStats/ops_as_primary"),
                "FabricPerDeviceStats/set_ops_as_primary") ||
      !SetNamed(env, obj, "ops_as_remote",
                MakeBigIntU64(env, d.ops_as_remote,
                              "FabricPerDeviceStats/ops_as_remote"),
                "FabricPerDeviceStats/set_ops_as_remote") ||
      !SetNamed(env, obj, "remote_bytes_read",
                MakeBigIntU64(env, d.remote_bytes_read,
                              "FabricPerDeviceStats/remote_bytes_read"),
                "FabricPerDeviceStats/set_remote_bytes_read") ||
      !SetNamed(env, obj, "remote_bytes_written",
                MakeBigIntU64(env, d.remote_bytes_written,
                              "FabricPerDeviceStats/remote_bytes_written"),
                "FabricPerDeviceStats/set_remote_bytes_written")) {
    return nullptr;
  }

  return obj;
}

static napi_value FabricEventToJs(napi_env env, const FabricEvent& e) {
  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  CHECK_NAPI_OK(env, st, "FabricEvent/create_object");

  if (!SetNamed(env, obj, "seq",
                MakeBigIntU64(env, e.seq, "FabricEvent/seq"),
                "FabricEvent/set_seq") ||
      !SetNamed(env, obj, "t_ns",
                MakeBigIntU64(env, e.t_ns, "FabricEvent/t_ns"),
                "FabricEvent/set_t_ns") ||
      !SetNamed(env, obj, "primary_device",
                MakeInt32(env, static_cast<std::int32_t>(e.primary_device),
                          "FabricEvent/primary_device"),
                "FabricEvent/set_primary_device") ||
      !SetNamed(env, obj, "other_device",
                MakeInt32(env, static_cast<std::int32_t>(e.other_device),
                          "FabricEvent/other_device"),
                "FabricEvent/set_other_device")) {
    return nullptr;
  }

  // kind / level are exported as stable snake_case tokens matching Python.
  if (!SetNamed(env, obj, "kind",
                MakeString(env, FabricEventKindName(e.kind),
                           "FabricEvent/kind"),
                "FabricEvent/set_kind") ||
      !SetNamed(env, obj, "level",
                MakeString(env, FabricEventLevelName(e.level),
                           "FabricEvent/level"),
                "FabricEvent/set_level")) {
    return nullptr;
  }

  if (!SetNamed(env, obj, "op_id",
                MakeBigIntU64(env, e.op_id, "FabricEvent/op_id"),
                "FabricEvent/set_op_id") ||
      !SetNamed(env, obj, "numel",
                MakeBigIntU64(env, e.numel, "FabricEvent/numel"),
                "FabricEvent/set_numel") ||
      !SetNamed(env, obj, "bytes",
                MakeBigIntU64(env, e.bytes, "FabricEvent/bytes"),
                "FabricEvent/set_bytes")) {
    return nullptr;
  }

  // reason_raw fits in int32.
  if (!SetNamed(env, obj, "reason_raw",
                MakeInt32(env, static_cast<std::int32_t>(e.reason_raw),
                          "FabricEvent/reason_raw"),
                "FabricEvent/set_reason_raw")) {
    return nullptr;
  }

  napi_value msg;
  if (e.message) {
    msg = MakeString(env, e.message, "FabricEvent/message");
  } else {
    st = napi_get_null(env, &msg);
    CHECK_NAPI_OK(env, st, "FabricEvent/get_null");
  }
  if (!SetNamed(env, obj, "message", msg, "FabricEvent/set_message")) {
    return nullptr;
  }

  return obj;
}

}  // namespace

napi_value FabricStatsSnapshotNapi(napi_env env, napi_callback_info /*info*/) {
  if (!EnsureOnMainThreadLocal(env, "fabricStatsSnapshot")) {
    return nullptr;
  }

  FabricStatsSnapshot s;
  try {
    s = vbt::cuda::fabric::fabric_stats_snapshot();
  } catch (const std::exception& e) {
    std::string msg = std::string("fabricStatsSnapshot: ") + e.what();
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return nullptr;
  } catch (...) {
    ThrowCudaRuntimeErrorSimple(env,
                               "fabricStatsSnapshot: unknown internal error",
                               "ERUNTIME");
    return nullptr;
  }

  napi_value obj;
  napi_status st = napi_create_object(env, &obj);
  CHECK_NAPI_OK(env, st, "FabricStatsSnapshot/create_object");

  auto set_u64 = [&](const char* name, std::uint64_t v, const char* ctx) -> bool {
    return SetNamed(env, obj, name, MakeBigIntU64(env, v, ctx), ctx);
  };

  if (!set_u64("mesh_builds", s.mesh_builds, "FabricStatsSnapshot/mesh_builds") ||
      !set_u64("p2p_pairs_enabled", s.p2p_pairs_enabled,
               "FabricStatsSnapshot/p2p_pairs_enabled") ||
      !set_u64("p2p_pairs_failed", s.p2p_pairs_failed,
               "FabricStatsSnapshot/p2p_pairs_failed") ||
      !set_u64("fabric_ops_attempted", s.fabric_ops_attempted,
               "FabricStatsSnapshot/fabric_ops_attempted") ||
      !set_u64("fabric_ops_hit", s.fabric_ops_hit,
               "FabricStatsSnapshot/fabric_ops_hit") ||
      !set_u64("fabric_ops_fallback", s.fabric_ops_fallback,
               "FabricStatsSnapshot/fabric_ops_fallback") ||
      !set_u64("remote_bytes_read", s.remote_bytes_read,
               "FabricStatsSnapshot/remote_bytes_read") ||
      !set_u64("remote_bytes_written", s.remote_bytes_written,
               "FabricStatsSnapshot/remote_bytes_written") ||
      !set_u64("inflight_ops_current", s.inflight_ops_current,
               "FabricStatsSnapshot/inflight_ops_current") ||
      !set_u64("inflight_ops_peak", s.inflight_ops_peak,
               "FabricStatsSnapshot/inflight_ops_peak") ||
      !set_u64("event_queue_len_peak", s.event_queue_len_peak,
               "FabricStatsSnapshot/event_queue_len_peak") ||
      !set_u64("event_dropped_total", s.event_dropped_total,
               "FabricStatsSnapshot/event_dropped_total") ||
      !set_u64("event_failures_total", s.event_failures_total,
               "FabricStatsSnapshot/event_failures_total") ||
      !set_u64("mode_enable_calls", s.mode_enable_calls,
               "FabricStatsSnapshot/mode_enable_calls") ||
      !set_u64("mode_disable_calls", s.mode_disable_calls,
               "FabricStatsSnapshot/mode_disable_calls") ||
      !set_u64("mode_set_failures", s.mode_set_failures,
               "FabricStatsSnapshot/mode_set_failures")) {
    return nullptr;
  }

  // reasons
  napi_value reasons;
  st = napi_create_object(env, &reasons);
  CHECK_NAPI_OK(env, st, "FabricStatsSnapshot/create_reasons");

  auto set_reason = [&](const char* name, std::uint64_t v, const char* ctx) -> bool {
    napi_value bi = MakeBigIntU64(env, v, ctx);
    napi_status sst = napi_set_named_property(env, reasons, name, bi);
    return CheckNapiOkImpl(env, sst, ctx);
  };

  if (!set_reason("no_p2p", s.reasons.no_p2p, "FabricStatsSnapshot/reasons.no_p2p") ||
      !set_reason("requires_grad", s.reasons.requires_grad,
                  "FabricStatsSnapshot/reasons.requires_grad") ||
      !set_reason("in_backward", s.reasons.in_backward,
                  "FabricStatsSnapshot/reasons.in_backward") ||
      !set_reason("small_tensor", s.reasons.small_tensor,
                  "FabricStatsSnapshot/reasons.small_tensor")) {
    return nullptr;
  }

  if (!SetNamed(env, obj, "reasons", reasons,
                "FabricStatsSnapshot/set_reasons")) {
    return nullptr;
  }

  // per_device
  napi_value arr;
  st = napi_create_array_with_length(env,
                                    static_cast<std::uint32_t>(s.per_device.size()),
                                    &arr);
  CHECK_NAPI_OK(env, st, "FabricStatsSnapshot/create_per_device");

  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(s.per_device.size());
       ++i) {
    napi_value elem = FabricPerDeviceStatsToJs(env, s.per_device[i]);
    if (!elem) return nullptr;
    st = napi_set_element(env, arr, i, elem);
    CHECK_NAPI_OK(env, st, "FabricStatsSnapshot/set_per_device_elem");
  }

  if (!SetNamed(env, obj, "per_device", arr,
                "FabricStatsSnapshot/set_per_device")) {
    return nullptr;
  }

  return obj;
}

napi_value FabricEventsSnapshotNapi(napi_env env, napi_callback_info info) {
  if (!EnsureOnMainThreadLocal(env, "fabricEventsSnapshot")) {
    return nullptr;
  }

  size_t argc = 2;
  napi_value args[2];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/get_cb_info");

  if (argc < 2) {
    ThrowCudaRuntimeErrorSimple(
        env,
        "fabricEventsSnapshot(minSeq, maxEvents) expects 2 arguments",
        "EINVAL");
    return nullptr;
  }

  // minSeq: bigint (uint64)
  napi_valuetype t0;
  st = napi_typeof(env, args[0], &t0);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/minSeq_typeof");
  if (t0 != napi_bigint) {
    ThrowCudaRuntimeErrorSimple(env,
                               "fabricEventsSnapshot: minSeq must be a BigInt",
                               "EINVAL");
    return nullptr;
  }

  bool lossless = false;
  std::uint64_t min_seq = 0;
  st = napi_get_value_bigint_uint64(env, args[0], &min_seq, &lossless);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/get_minSeq");
  if (!lossless) {
    ThrowCudaRuntimeErrorSimple(env,
                               "fabricEventsSnapshot: minSeq must fit in uint64",
                               "EINVAL");
    return nullptr;
  }

  // maxEvents: number (finite int >= 0)
  napi_valuetype t1;
  st = napi_typeof(env, args[1], &t1);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/maxEvents_typeof");
  if (t1 != napi_number) {
    ThrowCudaRuntimeErrorSimple(env,
                               "fabricEventsSnapshot: maxEvents must be a number",
                               "EINVAL");
    return nullptr;
  }

  double max_d = 0.0;
  st = napi_get_value_double(env, args[1], &max_d);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/get_maxEvents");
  if (!std::isfinite(max_d) || max_d < 0.0 || std::floor(max_d) != max_d) {
    ThrowCudaRuntimeErrorSimple(env,
                               "fabricEventsSnapshot: maxEvents must be a non-negative integer",
                               "EINVAL");
    return nullptr;
  }

  const double max_u32 =
      static_cast<double>(std::numeric_limits<std::uint32_t>::max());
  if (max_d > max_u32) {
    ThrowCudaRuntimeErrorSimple(env,
                               "fabricEventsSnapshot: maxEvents is too large",
                               "EINVAL");
    return nullptr;
  }

  const std::size_t max_events =
      static_cast<std::size_t>(static_cast<std::uint32_t>(max_d));

  FabricEventSnapshot s;
  try {
    s = vbt::cuda::fabric::fabric_events_snapshot(min_seq, max_events);
  } catch (const std::exception& e) {
    std::string msg = std::string("fabricEventsSnapshot: ") + e.what();
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return nullptr;
  } catch (...) {
    ThrowCudaRuntimeErrorSimple(env,
                               "fabricEventsSnapshot: unknown internal error",
                               "ERUNTIME");
    return nullptr;
  }

  napi_value obj;
  st = napi_create_object(env, &obj);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/create_object");

  if (!SetNamed(env, obj, "base_seq",
                MakeBigIntU64(env, s.base_seq, "FabricEventsSnapshot/base_seq"),
                "FabricEventsSnapshot/set_base_seq") ||
      !SetNamed(env, obj, "next_seq",
                MakeBigIntU64(env, s.next_seq, "FabricEventsSnapshot/next_seq"),
                "FabricEventsSnapshot/set_next_seq") ||
      !SetNamed(env, obj, "dropped_total",
                MakeBigIntU64(env, s.dropped_total, "FabricEventsSnapshot/dropped_total"),
                "FabricEventsSnapshot/set_dropped_total")) {
    return nullptr;
  }

  // capacity is bounded; expose as a Number for convenience.
  napi_value cap_val;
  st = napi_create_double(env, static_cast<double>(s.capacity), &cap_val);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/create_capacity");
  if (!SetNamed(env, obj, "capacity", cap_val,
                "FabricEventsSnapshot/set_capacity")) {
    return nullptr;
  }

  napi_value arr;
  st = napi_create_array_with_length(
      env, static_cast<std::uint32_t>(s.events.size()), &arr);
  CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/create_events");

  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(s.events.size());
       ++i) {
    napi_value ev = FabricEventToJs(env, s.events[i]);
    if (!ev) return nullptr;
    st = napi_set_element(env, arr, i, ev);
    CHECK_NAPI_OK(env, st, "FabricEventsSnapshot/set_event");
  }

  if (!SetNamed(env, obj, "events", arr,
                "FabricEventsSnapshot/set_events")) {
    return nullptr;
  }

  return obj;
}

}  // namespace node
}  // namespace vbt
