// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <thread>

#include <node_api.h>

namespace vbt {
namespace node {

// Per-environment CUDA runtime state, stored in AddonData and cached per
struct CudaRuntimeState {
  bool initialized{false};
  bool has_cuda{false};
  int  device_count{0};
  std::thread::id main_thread_id;  // env's main JS thread

#ifdef VBT_NODE_CUDA_DEBUG
  std::atomic<std::uint64_t> runtime_queries{0};   // isAvailable/deviceCount/currentDevice
  std::atomic<std::uint64_t> set_device_calls{0};  // successful setDevice calls
  std::atomic<std::uint64_t> stream_instances{0};  // live CudaStreamState instances
  std::atomic<std::uint64_t> event_instances{0};   // live CudaEventState instances
#endif
};

// Pure helper used by tests and internal code to check main-thread affinity
// without requiring a napi_env. Returns true only when the runtime has been
// initialized and the provided thread id matches main_thread_id.
inline bool IsOnMainThreadFromState(const CudaRuntimeState& rt,
                                    const std::thread::id& current_thread) {
  if (!rt.initialized) return false;
  return current_thread == rt.main_thread_id;
}

struct AddonData {
  CudaRuntimeState cuda_rt;
  // Future: tensor, DLPack, etc.
};

// Finalizer passed to napi_set_instance_data; deletes AddonData*.
void AddonDataFinalizer(napi_env env, void* data, void* hint);

// Low-level CUDA runtime helpers exposed from the addon. These are
// main-thread only and perform no JS argument validation; surface-level
// validation lives in the TS overlay.
napi_value HasCuda(napi_env env, napi_callback_info info);
napi_value CudaDeviceCount(napi_env env, napi_callback_info info);
napi_value CudaCurrentDevice(napi_env env, napi_callback_info info);
napi_value CudaSetDevice(napi_env env, napi_callback_info info);

napi_value CudaMemoryStatsAsNested(napi_env env, napi_callback_info info);
napi_value CudaMemoryStats(napi_env env, napi_callback_info info);
napi_value CudaMemorySnapshot(napi_env env, napi_callback_info info);

// Defines the CudaStream and CudaEvent JS classes and attaches the
// constructors to the exports object (as exports.CudaStream / CudaEvent).
napi_value CreateCudaStreamClass(napi_env env, napi_value exports);
napi_value CreateCudaEventClass(napi_env env, napi_value exports);

// Registers all CUDA runtime bindings (functions and classes) on exports.
napi_value RegisterCudaRuntimeBindings(napi_env env, napi_value exports);

// addon._cudaH2DAsync / addon._cudaD2HAsync and are thin wrappers
// around napi_async_work jobs that perform CUDA memcpy operations on
// libuv worker threads.
napi_value CudaH2DAsync(napi_env env, napi_callback_info info);
napi_value CudaD2HAsync(napi_env env, napi_callback_info info);

// Simple helper used across the addon to throw CUDA runtime errors with a
// stable .code tag. The message is sanitized internally.
void ThrowCudaRuntimeErrorSimple(napi_env env,
                                 const char* message,
                                 const char* code);

}  // namespace node
}  // namespace vbt
