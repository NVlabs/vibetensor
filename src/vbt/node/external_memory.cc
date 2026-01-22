// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/node/external_memory.h"

#include <climits>
#include <thread>

#include "vbt/core/device.h"
#include "vbt/node/cuda_napi.h"  // for CudaRuntimeState/AddOnData shape
#include "vbt/node/util.h"

namespace vbt {
namespace node {

ExternalMemStats g_ext_stats{};
ExternalOwnerRegistry g_ext_registry{};

namespace {

using vbt::core::Device;
using vbt::core::TensorImpl;

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
    std::string msg = std::string(fn_name ? fn_name : "vbt_napi") +
                      ": must be called on the main VibeTensor JS thread";
    ThrowCudaRuntimeErrorSimple(env, msg.c_str(), "ERUNTIME");
    return false;
  }
  return true;
}

}  // namespace

std::size_t safe_storage_bytes(const TensorImpl& impl) {
  auto storage = impl.storage();
  if (!storage) {
    // Allocator invariants: data()==nullptr, nbytes()==0.
    return 0;
  }
  std::size_t raw = storage->nbytes();
  constexpr std::size_t kMax = static_cast<std::size_t>(INT64_MAX);
  if (raw > kMax) {
    // Clamp rather than overflow int64_t; treat as "at least INT64_MAX".
    return kMax;
  }
  return raw;
}

std::size_t ExternalOwnerKeyHash::operator()(const ExternalOwnerKey& k) const
    noexcept {
  std::size_t h = std::hash<bool>{}(k.is_cuda);
  h ^= (std::hash<int32_t>{}(k.device) + 0x9e3779b97f4a7c15ULL + (h << 6) +
        (h >> 2));
  h ^= (std::hash<void*>{}(k.data) + 0x9e3779b97f4a7c15ULL + (h << 6) +
        (h >> 2));
  return h;
}

bool ExternalOwnerKeyEq::operator()(const ExternalOwnerKey& a,
                                    const ExternalOwnerKey& b) const noexcept {
  return a.is_cuda == b.is_cuda && a.device == b.device && a.data == b.data;
}

std::pair<std::shared_ptr<ExternalMemoryOwner>, bool>
AttachExternalOwnerForStorage(const TensorImpl& impl,
                              bool is_cuda_device,
                              int32_t device_index) {
  const std::size_t bytes = safe_storage_bytes(impl);

  // Zero-size storages are not keyed; they always get independent owners.
  void* data = nullptr;
  if (auto storage = impl.storage()) {
    data = storage->data();
  }

  if (data == nullptr || bytes == 0) {
    auto owner = std::make_shared<ExternalMemoryOwner>();
    owner->bytes = 0;
    g_ext_stats.owners_alive.fetch_add(1, std::memory_order_relaxed);
    return {std::move(owner), true};
  }

  ExternalOwnerKey key{is_cuda_device, device_index, data};

  std::lock_guard<std::mutex> guard(g_ext_registry.mu);
  auto it = g_ext_registry.by_key.find(key);
  if (it != g_ext_registry.by_key.end()) {
    if (auto existing = it->second.lock()) {
      return {std::move(existing), false};
    }
  }

  auto owner = std::make_shared<ExternalMemoryOwner>();
  owner->bytes = bytes;
  g_ext_registry.by_key[key] = owner;
  g_ext_stats.owners_alive.fetch_add(1, std::memory_order_relaxed);
  return {std::move(owner), true};
}

void AccountExternalBytes(napi_env env, ExternalMemoryOwner& owner) {
  if (owner.bytes == 0) return;
  const bool was = owner.accounted.exchange(true, std::memory_order_acq_rel);
  if (was) return;

  const int64_t delta = static_cast<int64_t>(owner.bytes);
  int64_t result = 0;
  napi_status st = napi_adjust_external_memory(env, delta, &result);
  if (st != napi_ok) {
    const napi_extended_error_info* info = nullptr;
    napi_get_last_error_info(env, &info);
    std::string msg = "napi_adjust_external_memory(+bytes) failed";
    if (info && info->error_message) {
      msg += ": ";
      msg += info->error_message;
    }
    // Treat as fatal; we must not leave the process in a partially-accounted
    // state relative to V8's view of external memory.
    napi_fatal_error("vbt_napi", NAPI_AUTO_LENGTH, msg.c_str(),
                     NAPI_AUTO_LENGTH);
  }
  g_ext_stats.bytes_accounted.fetch_add(owner.bytes,
                                        std::memory_order_relaxed);
}

void ReleaseExternalOwner(napi_env env, ExternalMemoryOwner& owner) {
  const bool prev_released =
      owner.released.exchange(true, std::memory_order_acq_rel);
  if (prev_released) return;  // already released

  if (owner.bytes > 0) {
    const bool was_accounted =
        owner.accounted.exchange(true, std::memory_order_acq_rel);
    if (was_accounted) {
      const int64_t delta = -static_cast<int64_t>(owner.bytes);
      int64_t result = 0;
      napi_status st = napi_adjust_external_memory(env, delta, &result);
      if (st == napi_ok) {
        g_ext_stats.bytes_accounted.fetch_sub(owner.bytes,
                                              std::memory_order_relaxed);
      } else {
        // Fatal: we must not underflow bytes_accounted relative to V8's view.
        napi_fatal_error("vbt_napi", NAPI_AUTO_LENGTH,
                         "napi_adjust_external_memory(-bytes) failed",
                         NAPI_AUTO_LENGTH);
      }
    }
  }

  g_ext_stats.owners_alive.fetch_sub(1, std::memory_order_relaxed);
}

napi_value ExternalMemoryStatsNapi(napi_env env, napi_callback_info /*info*/) {
  if (!EnsureOnMainThreadLocal(env, "_external_memory_stats")) {
    return nullptr;
  }

  const std::uint64_t owners =
      g_ext_stats.owners_alive.load(std::memory_order_relaxed);
  const std::uint64_t bytes =
      g_ext_stats.bytes_accounted.load(std::memory_order_relaxed);

  napi_value result;
  napi_value owners_val;
  napi_value bytes_val;
  napi_status st = napi_create_object(env, &result);
  CHECK_NAPI_OK(env, st, "ExternalMemoryStats/create_object");
  st = napi_create_double(env, static_cast<double>(owners), &owners_val);
  CHECK_NAPI_OK(env, st, "ExternalMemoryStats/create_owners");
  st = napi_create_double(env, static_cast<double>(bytes), &bytes_val);
  CHECK_NAPI_OK(env, st, "ExternalMemoryStats/create_bytes");
  st = napi_set_named_property(env, result, "ownersAlive", owners_val);
  CHECK_NAPI_OK(env, st, "ExternalMemoryStats/set_owners");
  st = napi_set_named_property(env, result, "bytesAccounted", bytes_val);
  CHECK_NAPI_OK(env, st, "ExternalMemoryStats/set_bytes");
  return result;
}

}  // namespace node
}  // namespace vbt
