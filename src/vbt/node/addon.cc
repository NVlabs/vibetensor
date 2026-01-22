// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <node_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "vbt/logging/logging.h"
#include "vbt/cuda/device.h"
#include "vbt/node/logging.h"
#include "vbt/node/tensor.h"
#include "vbt/node/util.h"
#include "vbt/node/dispatcher.h"
#include "vbt/node/cuda_napi.h"
#include "vbt/node/dlpack_napi.h"
#include "vbt/node/external_memory.h"
#include "vbt/node/fabric_napi.h"

extern "C" void vbt_register_default_kernels();
#if VBT_WITH_CUDA
extern "C" void vbt_register_cuda_elementwise_kernels();
extern "C" void vbt_register_fabric_kernels();
#endif
extern "C" void vbt_register_reduction_dim_kernels();

namespace vbt::node {

napi_value Zeros(napi_env env, napi_callback_info info);
napi_value DummyAdd(napi_env env, napi_callback_info info);
napi_value ScalarInt64(napi_env env, napi_callback_info info);
napi_value ScalarBool(napi_env env, napi_callback_info info);
napi_value ScalarFloat32(napi_env env, napi_callback_info info);

}  // namespace vbt::node

namespace {

napi_value Init(napi_env env, napi_value exports) {
  // Initialize logging and register default kernels. This mirrors the Python
  vbt::InitLogging(std::nullopt);
  vbt_register_default_kernels();
#if VBT_WITH_CUDA
  vbt_register_cuda_elementwise_kernels();
  vbt_register_fabric_kernels();
#endif
  // Extra kernels used by the Node.js MNIST demo (CUDA reductions with scalar metadata args).
  vbt_register_reduction_dim_kernels();

  // Per-env addon state for CUDA runtime basics.
  auto* data = new vbt::node::AddonData();
  data->cuda_rt.main_thread_id = std::this_thread::get_id();

  int count = 0;
#if VBT_WITH_CUDA
  try {
    count = vbt::cuda::device_count();
  } catch (...) {
    count = 0;
  }
#else
  count = 0;
#endif
  data->cuda_rt.device_count = count;
  data->cuda_rt.has_cuda = (count > 0);
  data->cuda_rt.initialized = true;

  napi_status st =
      napi_set_instance_data(env, data, vbt::node::AddonDataFinalizer, nullptr);
  if (st != napi_ok) {
    vbt::node::ThrowCudaRuntimeErrorSimple(
        env,
        "vibetensor: failed to initialize addon instance data",
        "ERUNTIME");
    return nullptr;
  }

  // Configure native logging from environment, if requested.
  vbt::node::InitLoggingFromEnv(env);

  // Register CUDA runtime bindings (hasCuda/deviceCount/currentDevice/streams/events).
  exports = vbt::node::RegisterCudaRuntimeBindings(env, exports);
  if (exports == nullptr) {
    return nullptr;  // exception already pending
  }

  // Register DLPack bindings (toDlpack/fromDlpackCapsule).
  exports = vbt::node::RegisterDlpackBindings(env, exports);
  if (exports == nullptr) {
    return nullptr;  // exception already pending
  }

  napi_property_descriptor props[] = {
      {"zeros", nullptr, vbt::node::Zeros, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"scalarInt64", nullptr, vbt::node::ScalarInt64, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"scalarBool", nullptr, vbt::node::ScalarBool, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"scalarFloat32", nullptr, vbt::node::ScalarFloat32, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"_dummyAdd", nullptr, vbt::node::DummyAdd, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"callOpSync", nullptr, vbt::node::CallOpSync, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"callOp", nullptr, vbt::node::CallOp, nullptr, nullptr, nullptr,
       napi_default, nullptr},
      {"callOpNoOverride", nullptr, vbt::node::CallOpNoOverride, nullptr,
       nullptr, nullptr, napi_default, nullptr},
      {"fabricStatsSnapshot", nullptr, vbt::node::FabricStatsSnapshotNapi,
       nullptr, nullptr, nullptr, napi_default, nullptr},
      {"fabricEventsSnapshot", nullptr, vbt::node::FabricEventsSnapshotNapi,
       nullptr, nullptr, nullptr, napi_default, nullptr},
      {"_debugDispatcherStats", nullptr, vbt::node::GetDispatcherStats, nullptr,
       nullptr, nullptr, napi_default, nullptr},
      {"_debugAsyncStats", nullptr, vbt::node::GetAsyncStats, nullptr,
       nullptr, nullptr, napi_default, nullptr},
      {"_external_memory_stats", nullptr, vbt::node::ExternalMemoryStatsNapi,
       nullptr, nullptr, nullptr, napi_default, nullptr},
      {"_cudaMemoryStatsAsNested", nullptr, vbt::node::CudaMemoryStatsAsNested,
       nullptr, nullptr, nullptr, napi_default, nullptr},
      {"_cudaMemoryStats", nullptr, vbt::node::CudaMemoryStats,
       nullptr, nullptr, nullptr, napi_default, nullptr},
      {"_cudaMemorySnapshot", nullptr, vbt::node::CudaMemorySnapshot,
       nullptr, nullptr, nullptr, napi_default, nullptr},
      {"_setLoggingEnabled", nullptr, vbt::node::SetLoggingEnabledNapi,
       nullptr, nullptr, nullptr, napi_default, nullptr},
      {"_drainLogs", nullptr, vbt::node::DrainLogsNapi,
       nullptr, nullptr, nullptr, napi_default, nullptr},
  };

  st = napi_define_properties(
      env, exports, sizeof(props) / sizeof(props[0]), props);
  CHECK_NAPI_OK(env, st, "Init/define_properties");

  return exports;
}

}  // namespace

NAPI_MODULE_INIT(/*env, exports*/) { return Init(env, exports); }

namespace vbt::node {

napi_value DummyAdd(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  napi_status st = napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
  CHECK_NAPI_OK(env, st, "DummyAdd/get_cb_info");

  if (argc != 2) {
    return ThrowTypeError(env, "_dummyAdd(a,b) expects 2 arguments");
  }

  double a = 0.0;
  double b = 0.0;
  if (napi_get_value_double(env, args[0], &a) != napi_ok ||
      napi_get_value_double(env, args[1], &b) != napi_ok) {
    return ThrowTypeError(env, "_dummyAdd: expected two numbers");
  }

  napi_value out;
  st = napi_create_double(env, a + b, &out);
  CHECK_NAPI_OK(env, st, "DummyAdd/create_double");
  return out;
}

}  // namespace vbt::node
