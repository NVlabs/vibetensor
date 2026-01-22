// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <stddef.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#include <dlpack/dlpack.h>

#include "vbt/plugin/vbt_plugin.h"

#include "cutlass/cutlass.h"

// NOTE: These experimental ring_allreduce headers are vendored under
// ring_allreduce_plugin/94_blackwell_ring_allreduce/cutlass/experimental/...
#include "cutlass/experimental/distributed/collective/ring_allreduce_host.hpp"
#include "cutlass/experimental/distributed/collective/ring_allreduce_kernel_sm100.cuh"

namespace {

using cutlass::distributed::collective::RingAllreduceDeviceAtomicU32;
using cutlass::distributed::collective::RingAllreduceError;
using cutlass::distributed::collective::RingAllreduceHostResult;
using cutlass::distributed::collective::RingAllreduceParams;
using cutlass::distributed::collective::RingAllreduceP2POptions;
using cutlass::distributed::collective::RingAllreduceSystemAtomicU32;
using cutlass::distributed::collective::RingAllreduceTiling;
using cutlass::distributed::collective::validate_ring_allreduce_host_tiling;
using cutlass::distributed::collective::validate_ring_p2p_caps_and_enable_peer_access;

static const vbt_host_api* g_host = nullptr;

static inline void set_last_error(const std::string& msg) {
  if (g_host && g_host->set_last_error) {
    g_host->set_last_error(msg.c_str());
  }
}

static inline bool dl_dtype_is_float32(DLDataType dt) {
  return dt.lanes == 1 && dt.code == static_cast<uint8_t>(kDLFloat) &&
         dt.bits == 32;
}

static inline bool dl_dtype_is_int64(DLDataType dt) {
  return dt.lanes == 1 && dt.code == static_cast<uint8_t>(kDLInt) &&
         dt.bits == 64;
}

static inline bool device_is_cuda(DLDevice d) {
  return d.device_type == kDLCUDA;
}

static inline bool device_is_cpu(DLDevice d) { return d.device_type == kDLCPU; }

static inline cudaStream_t as_cuda_stream(vt_stream s) {
  return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(s));
}

static inline const char* ring_allreduce_error_to_string(RingAllreduceError e) {
  switch (e) {
    case RingAllreduceError::kOk:
      return "kOk";
    case RingAllreduceError::kInvalidParams:
      return "kInvalidParams";
    case RingAllreduceError::kTimeout:
      return "kTimeout";
    case RingAllreduceError::kAbortObserved:
      return "kAbortObserved";
    default:
      return "<unknown>";
  }
}

struct SavedCudaDevice {
  int saved = -1;

  SavedCudaDevice() {
    cudaError_t st = cudaGetDevice(&saved);
    if (st != cudaSuccess) {
      (void)cudaGetLastError();
      saved = -1;
    }
  }

  ~SavedCudaDevice() {
    if (saved >= 0) {
      (void)cudaSetDevice(saved);
      (void)cudaGetLastError();
    }
  }

  SavedCudaDevice(const SavedCudaDevice&) = delete;
  SavedCudaDevice& operator=(const SavedCudaDevice&) = delete;
};

static void format_host_result(std::string* out,
                               const char* what,
                               RingAllreduceHostResult const& r) {
  if (!out) return;
  std::string msg;
  msg.reserve(256);
  msg += what;
  msg += " failed: status=";
  msg += cutlassGetStatusString(r.status);
  msg += " cuda_error=";
  msg += cudaGetErrorString(r.cuda_error);
  if (r.device_a >= 0) {
    msg += " device_a=";
    msg += std::to_string(r.device_a);
  }
  if (r.device_b >= 0) {
    msg += " device_b=";
    msg += std::to_string(r.device_b);
  }
  if (r.error_reason) {
    msg += " reason='";
    msg += r.error_reason;
    msg += "'";
  }
  *out = std::move(msg);
}

// ------- Workspace atomics helpers (from NVIDIA benchmark) -------------------

struct RingAllreduceRankAtomics {
  RingAllreduceSystemAtomicU32* self_rs_ready = nullptr;
  RingAllreduceSystemAtomicU32* self_ag_ready = nullptr;
  uint32_t flags_len = 0u;

  RingAllreduceSystemAtomicU32* self_abort = nullptr;
  RingAllreduceSystemAtomicU32* self_error = nullptr;

  RingAllreduceDeviceAtomicU32* self_tiles_finished = nullptr;

  RingAllreduceSystemAtomicU32* self_barrier_gather_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_gather_status = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_token = nullptr;
  RingAllreduceSystemAtomicU32* self_barrier_release_status = nullptr;
};

__global__ void construct_rank_atomics_kernel(RingAllreduceRankAtomics a) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_rs_ready) {
      new (a.self_rs_ready + idx) RingAllreduceSystemAtomicU32{};
    }
    if (a.self_ag_ready) {
      new (a.self_ag_ready + idx) RingAllreduceSystemAtomicU32{};
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    new (a.self_abort) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_error) {
    new (a.self_error) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_tiles_finished) {
    new (a.self_tiles_finished) RingAllreduceDeviceAtomicU32{};
  }

  if (a.self_barrier_gather_token) {
    new (a.self_barrier_gather_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_barrier_gather_status) {
    new (a.self_barrier_gather_status) RingAllreduceSystemAtomicU32{};
  }

  if (a.self_barrier_release_token) {
    new (a.self_barrier_release_token) RingAllreduceSystemAtomicU32{};
  }
  if (a.self_barrier_release_status) {
    new (a.self_barrier_release_status) RingAllreduceSystemAtomicU32{};
  }
}

__global__ void reset_rank_atomics_init_kernel(RingAllreduceRankAtomics a,
                                              uint32_t epoch) {
  // Tokens are reset to 0 between runs. Keep epoch in the signature for future
  // per-epoch reset logic.
  (void)epoch;

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_rs_ready) {
      a.self_rs_ready[idx].store(0u, cuda::memory_order_relaxed);
    }
    if (a.self_ag_ready) {
      a.self_ag_ready[idx].store(0u, cuda::memory_order_relaxed);
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_abort) {
    a.self_abort->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_error) {
    a.self_error->store(static_cast<uint32_t>(RingAllreduceError::kOk),
                        cuda::memory_order_relaxed);
  }
  if (a.self_tiles_finished) {
    a.self_tiles_finished->store(0u, cuda::memory_order_relaxed);
  }

  if (a.self_barrier_gather_token) {
    a.self_barrier_gather_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_barrier_gather_status) {
    a.self_barrier_gather_status->store(static_cast<uint32_t>(RingAllreduceError::kOk),
                                        cuda::memory_order_relaxed);
  }

  if (a.self_barrier_release_token) {
    a.self_barrier_release_token->store(0u, cuda::memory_order_relaxed);
  }
  if (a.self_barrier_release_status) {
    a.self_barrier_release_status->store(static_cast<uint32_t>(RingAllreduceError::kOk),
                                         cuda::memory_order_relaxed);
  }
}

__global__ void destroy_rank_atomics_kernel(RingAllreduceRankAtomics a) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < a.flags_len) {
    if (a.self_ag_ready) {
      a.self_ag_ready[idx].~RingAllreduceSystemAtomicU32();
    }
    if (a.self_rs_ready) {
      a.self_rs_ready[idx].~RingAllreduceSystemAtomicU32();
    }
  }

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (a.self_barrier_release_status) {
    a.self_barrier_release_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_barrier_release_token) {
    a.self_barrier_release_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_barrier_gather_status) {
    a.self_barrier_gather_status->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_barrier_gather_token) {
    a.self_barrier_gather_token->~RingAllreduceSystemAtomicU32();
  }

  if (a.self_tiles_finished) {
    a.self_tiles_finished->~RingAllreduceDeviceAtomicU32();
  }
  if (a.self_error) {
    a.self_error->~RingAllreduceSystemAtomicU32();
  }
  if (a.self_abort) {
    a.self_abort->~RingAllreduceSystemAtomicU32();
  }
}

// ------- Ring selection ------------------------------------------------------

static bool pair_ok_dry_run(int a, int b) {
  int pair[2] = {a, b};
  RingAllreduceP2POptions opts;
  opts.enable_peer_access = false;
  opts.require_native_atomics = true;
  auto r = validate_ring_p2p_caps_and_enable_peer_access(2, pair, opts);
  return r.ok();
}

static bool find_ring_perm(const std::vector<int>& devices,
                           std::vector<int>* out_perm,
                           std::string* out_err) {
  if (!out_perm) return false;
  out_perm->clear();
  const int n = static_cast<int>(devices.size());
  if (n <= 0) return false;
  if (n == 1) {
    out_perm->push_back(0);
    return true;
  }

  std::vector<std::vector<bool>> ok(n, std::vector<bool>(n, false));
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      bool pij = pair_ok_dry_run(devices[i], devices[j]);
      ok[i][j] = pij;
      ok[j][i] = pij;
    }
  }

  std::vector<int> path;
  path.reserve(n);
  std::vector<bool> used(n, false);

  auto dfs = [&](auto&& self) -> bool {
    if (static_cast<int>(path.size()) == n) {
      return ok[path.back()][path.front()];
    }

    int last = path.back();
    for (int cand = 0; cand < n; ++cand) {
      if (used[cand]) continue;
      if (!ok[last][cand]) continue;
      used[cand] = true;
      path.push_back(cand);
      if (self(self)) return true;
      path.pop_back();
      used[cand] = false;
    }
    return false;
  };

  for (int start = 0; start < n; ++start) {
    std::fill(used.begin(), used.end(), false);
    path.clear();
    used[start] = true;
    path.push_back(start);
    if (dfs(dfs)) {
      *out_perm = path;
      return true;
    }
  }

  if (out_err) {
    *out_err = "no valid P2P ring found for provided devices";
  }
  return false;
}

// ------- Host watchdog -------------------------------------------------------

static bool wait_for_events(const std::vector<int>& devices,
                            const std::vector<cudaEvent_t>& done_events,
                            int watchdog_ms,
                            std::string* out_err) {
  const int world_size = static_cast<int>(devices.size());
  auto const host_timeout = std::chrono::milliseconds(watchdog_ms);
  auto deadline = std::chrono::steady_clock::now() + host_timeout;

  std::vector<bool> done(world_size, false);
  while (true) {
    bool all_done = true;
    for (int rank = 0; rank < world_size; ++rank) {
      if (done[rank]) continue;
      all_done = false;
      (void)cudaSetDevice(devices[rank]);
      cudaError_t q = cudaEventQuery(done_events[rank]);
      if (q == cudaSuccess) {
        done[rank] = true;
        continue;
      }
      if (q != cudaErrorNotReady) {
        if (out_err) {
          *out_err = std::string("cudaEventQuery failed: ") +
                     cudaGetErrorString(q);
        }
        return false;
      }
    }

    if (all_done) {
      return true;
    }

    if (std::chrono::steady_clock::now() > deadline) {
      if (out_err) {
        *out_err = "watchdog timeout";
      }
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// ------- Core op impl --------------------------------------------------------

static vt_status ring_allreduce_impl(int world_size,
                                    vt_stream /*s*/, // stream for arg0 device (unused)
                                    const vt_tensor* args,
                                    size_t nargs,
                                    vt_tensor* out) {
  if (!g_host) {
    set_last_error("ring_allreduce: host API unavailable");
    return VT_STATUS_INTERNAL;
  }
  if (!out) {
    set_last_error("ring_allreduce: out is NULL");
    return VT_STATUS_INVALID_ARG;
  }
  *out = nullptr;
  if (!args) {
    set_last_error("ring_allreduce: args is NULL");
    return VT_STATUS_INVALID_ARG;
  }

  SavedCudaDevice saved_device_guard;

  const size_t expected_nargs = static_cast<size_t>(2 * world_size + 1);
  if (nargs != expected_nargs) {
    set_last_error("ring_allreduce: arity mismatch");
    return VT_STATUS_INVALID_ARG;
  }

  // Layout: out0..out{W-1}, in0..in{W-1}, tpl_i64_cpu
  const size_t tpl_index = nargs - 1;
  vt_tensor tpl = args[tpl_index];
  if (!tpl) {
    set_last_error("ring_allreduce: tpl_i64_cpu is NULL");
    return VT_STATUS_INVALID_ARG;
  }

  DLDevice tpl_dev = g_host->tensor_device(tpl);
  DLDataType tpl_dt = g_host->tensor_dtype(tpl);
  if (!device_is_cpu(tpl_dev) || !dl_dtype_is_int64(tpl_dt) ||
      g_host->tensor_ndim(tpl) != 0) {
    set_last_error("ring_allreduce: tpl_i64_cpu must be a CPU int64 scalar (0-d)");
    return VT_STATUS_INVALID_ARG;
  }

  // Gather per-rank tensors.
  std::vector<vt_tensor> out_t(world_size);
  std::vector<vt_tensor> in_t(world_size);
  for (int r = 0; r < world_size; ++r) {
    out_t[r] = args[static_cast<size_t>(r)];
    in_t[r] = args[static_cast<size_t>(world_size + r)];
    if (!out_t[r] || !in_t[r]) {
      set_last_error("ring_allreduce: NULL tensor arg");
      return VT_STATUS_INVALID_ARG;
    }
  }

  // Validate dtypes/devices and shape agreement.
  std::vector<int> devices(world_size);
  std::vector<cudaStream_t> streams(world_size);
  std::vector<float*> out_ptr(world_size);
  std::vector<const float*> in_ptr(world_size);

  int64_t expected_numel = -1;
  size_t expected_ndim = 0;
  const int64_t* expected_sizes = nullptr;

  for (int r = 0; r < world_size; ++r) {
    DLDevice d_out = g_host->tensor_device(out_t[r]);
    DLDevice d_in = g_host->tensor_device(in_t[r]);
    if (!device_is_cuda(d_out) || !device_is_cuda(d_in)) {
      set_last_error("ring_allreduce: all input/output tensors must be CUDA");
      return VT_STATUS_INVALID_ARG;
    }
    if (d_out.device_id != d_in.device_id) {
      set_last_error("ring_allreduce: out_r and in_r must be on same CUDA device");
      return VT_STATUS_INVALID_ARG;
    }

    DLDataType dt_out = g_host->tensor_dtype(out_t[r]);
    DLDataType dt_in = g_host->tensor_dtype(in_t[r]);
    if (!dl_dtype_is_float32(dt_out) || !dl_dtype_is_float32(dt_in)) {
      set_last_error("ring_allreduce: only float32 is supported");
      return VT_STATUS_INVALID_ARG;
    }

    if (!g_host->tensor_is_contiguous(out_t[r]) ||
        !g_host->tensor_is_contiguous(in_t[r])) {
      set_last_error("ring_allreduce: requires contiguous tensors");
      return VT_STATUS_INVALID_ARG;
    }

    size_t ndim_out = g_host->tensor_ndim(out_t[r]);
    size_t ndim_in = g_host->tensor_ndim(in_t[r]);
    if (ndim_out != ndim_in) {
      set_last_error("ring_allreduce: out_r and in_r ndim mismatch");
      return VT_STATUS_INVALID_ARG;
    }
    const int64_t* sizes_out = g_host->tensor_sizes(out_t[r]);
    const int64_t* sizes_in = g_host->tensor_sizes(in_t[r]);
    for (size_t i = 0; i < ndim_out; ++i) {
      if (sizes_out[i] != sizes_in[i]) {
        set_last_error("ring_allreduce: out_r and in_r size mismatch");
        return VT_STATUS_INVALID_ARG;
      }
    }

    int64_t numel = g_host->tensor_numel(out_t[r]);
    if (numel != g_host->tensor_numel(in_t[r])) {
      set_last_error("ring_allreduce: out_r and in_r numel mismatch");
      return VT_STATUS_INVALID_ARG;
    }

    if (expected_numel < 0) {
      expected_numel = numel;
      expected_ndim = ndim_out;
      expected_sizes = sizes_out;
    } else {
      if (numel != expected_numel || ndim_out != expected_ndim) {
        set_last_error("ring_allreduce: shape mismatch across ranks");
        return VT_STATUS_INVALID_ARG;
      }
      for (size_t i = 0; i < ndim_out; ++i) {
        if (sizes_out[i] != expected_sizes[i]) {
          set_last_error("ring_allreduce: size mismatch across ranks");
          return VT_STATUS_INVALID_ARG;
        }
      }
    }

    // Validate compute capability (need SM103).
    int dev_index = d_out.device_id;
    cudaDeviceProp prop{};
    cudaError_t st = cudaGetDeviceProperties(&prop, dev_index);
    if (st != cudaSuccess) {
      set_last_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(st));
      return VT_STATUS_RUNTIME_ERROR;
    }
    if (!(prop.major == 10 && prop.minor == 3)) {
      set_last_error("ring_allreduce: requires SM103 (compute capability 10.3)");
      return VT_STATUS_UNSUPPORTED;
    }

    devices[r] = dev_index;

    // Current stream per device (0 is default stream).
    vt_stream vs = g_host->current_cuda_stream
                       ? g_host->current_cuda_stream(static_cast<int32_t>(dev_index))
                       : 0;
    streams[r] = as_cuda_stream(vs);

    // Pointers.
    const void* out_data = g_host->tensor_data(out_t[r]);
    const void* in_data = g_host->tensor_data(in_t[r]);
    if (numel > 0 && (!out_data || !in_data)) {
      set_last_error("ring_allreduce: tensor has no data");
      return VT_STATUS_INVALID_ARG;
    }
    out_ptr[r] = reinterpret_cast<float*>(const_cast<void*>(out_data));
    in_ptr[r] = reinterpret_cast<const float*>(in_data);
  }

  // Device uniqueness.
  for (int i = 0; i < world_size; ++i) {
    for (int j = i + 1; j < world_size; ++j) {
      if (devices[i] == devices[j]) {
        set_last_error("ring_allreduce: duplicate device in ranks");
        return VT_STATUS_INVALID_ARG;
      }
    }
  }

  // Empty is a fast no-op.
  if (expected_numel == 0) {
    vt_status st = g_host->tensor_new_dense_like(tpl, out);
    if (st != VT_STATUS_OK) {
      set_last_error("ring_allreduce: failed to allocate status tensor");
      return st;
    }
    int64_t* p = reinterpret_cast<int64_t*>(g_host->tensor_mutable_data(*out));
    if (!p) {
      set_last_error("ring_allreduce: failed to get mutable_data for status tensor");
      return VT_STATUS_INTERNAL;
    }
    *p = 0;
    return VT_STATUS_OK;
  }

  // Find a P2P ring order for the provided devices.
  std::vector<int> perm;
  std::string ring_err;
  if (!find_ring_perm(devices, &perm, &ring_err)) {
    set_last_error(ring_err);
    return VT_STATUS_RUNTIME_ERROR;
  }

  std::vector<int> ring_devices(world_size);
  std::vector<cudaStream_t> ring_streams(world_size);
  std::vector<float*> ring_out_ptr(world_size);
  std::vector<const float*> ring_in_ptr(world_size);
  for (int r = 0; r < world_size; ++r) {
    int idx = perm[r];
    ring_devices[r] = devices[idx];
    ring_streams[r] = streams[idx];
    ring_out_ptr[r] = out_ptr[idx];
    ring_in_ptr[r] = in_ptr[idx];
  }

  // Enable peer access for ring neighbors.
  auto p2p = validate_ring_p2p_caps_and_enable_peer_access(
      world_size, ring_devices.data());
  if (!p2p.ok()) {
    std::string msg;
    format_host_result(&msg, "validate_ring_p2p_caps_and_enable_peer_access", p2p);
    set_last_error(msg);
    return VT_STATUS_RUNTIME_ERROR;
  }

  // Compute tiling. Match the upstream benchmark defaults:
  //   --num_channels=1 --tile_elems=256
  // The benchmark notes `tile_elems <= 256` for warp-specialized SMEM.
  constexpr int32_t kNumChannels = 5;
  constexpr uint32_t kTileElems = 14336;

  RingAllreduceTiling tiling{};
  auto tiling_r = validate_ring_allreduce_host_tiling(
      static_cast<uint64_t>(expected_numel),
      world_size,
      kNumChannels,
      kTileElems,
      &tiling,
      ring_devices.data());
  if (!tiling_r.ok()) {
    std::string msg;
    format_host_result(&msg, "validate_ring_allreduce_host_tiling", tiling_r);
    set_last_error(msg);
    return VT_STATUS_RUNTIME_ERROR;
  }

  int watchdog_ms = 20'000;
  if (const char* env = std::getenv("VBT_RING_ALLREDUCE_WATCHDOG_MS")) {
    if (*env != '\0') {
      errno = 0;
      char* end = nullptr;
      long long v = std::strtoll(env, &end, 10);
      if (errno != 0 || end == env || *end != '\0') {
        set_last_error("VBT_RING_ALLREDUCE_WATCHDOG_MS must be an integer (ms)");
        return VT_STATUS_INVALID_ARG;
      }
      if (v <= 0) {
        set_last_error("VBT_RING_ALLREDUCE_WATCHDOG_MS must be > 0");
        return VT_STATUS_INVALID_ARG;
      }
      if (v > 3'600'000) {
        v = 3'600'000;
      }
      watchdog_ms = static_cast<int>(v);
    }
  }

  uint64_t flags_len_u64 =
      static_cast<uint64_t>(world_size) * static_cast<uint64_t>(tiling.num_tiles_total);
  if (flags_len_u64 > static_cast<uint64_t>(UINT32_MAX)) {
    set_last_error("ring_allreduce: flags_len overflow");
    return VT_STATUS_NOMEM;
  }
  uint32_t flags_len = static_cast<uint32_t>(flags_len_u64);
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((flags_len_u64 + kThreads - 1) / kThreads);
  if (blocks == 0) blocks = 1;

  // Per-rank allocations.
  std::vector<RingAllreduceRankAtomics> atomics(world_size);
  std::vector<uint32_t*> device_status(world_size, nullptr);
  std::vector<cudaEvent_t> done_events(world_size, nullptr);

  bool alloc_ok = true;

  for (int rank = 0; rank < world_size; ++rank) {
    cudaError_t st = cudaSetDevice(ring_devices[rank]);
    if (st != cudaSuccess) {
      set_last_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(st));
      return VT_STATUS_RUNTIME_ERROR;
    }

    atomics[rank].flags_len = flags_len;

    st = cudaMalloc(reinterpret_cast<void**>(&device_status[rank]), sizeof(uint32_t));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    st = cudaEventCreateWithFlags(&done_events[rank], cudaEventDisableTiming);
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_rs_ready),
                    sizeof(RingAllreduceSystemAtomicU32) * flags_len);
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }
    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_ag_ready),
                    sizeof(RingAllreduceSystemAtomicU32) * flags_len);
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_abort),
                    sizeof(RingAllreduceSystemAtomicU32));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }
    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_error),
                    sizeof(RingAllreduceSystemAtomicU32));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_tiles_finished),
                    sizeof(RingAllreduceDeviceAtomicU32));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_gather_token),
                    sizeof(RingAllreduceSystemAtomicU32));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }
    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_gather_status),
                    sizeof(RingAllreduceSystemAtomicU32));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }
    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_release_token),
                    sizeof(RingAllreduceSystemAtomicU32));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }
    st = cudaMalloc(reinterpret_cast<void**>(&atomics[rank].self_barrier_release_status),
                    sizeof(RingAllreduceSystemAtomicU32));
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    // Initialize status to invalid, then construct and reset atomics.
    cudaStream_t stream = ring_streams[rank];
    st = cudaMemsetAsync(device_status[rank], 0xFF, sizeof(uint32_t), stream);
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    construct_rank_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics[rank]);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }

    reset_rank_atomics_init_kernel<<<blocks, kThreads, 0, stream>>>(atomics[rank], 1u);
    st = cudaGetLastError();
    if (st != cudaSuccess) {
      alloc_ok = false;
      break;
    }
  }

  if (!alloc_ok) {
    // Best-effort cleanup.
    for (int rank = 0; rank < world_size; ++rank) {
      (void)cudaSetDevice(ring_devices[rank]);
      // Don't synchronize here; allocation failure implies nothing running.
      if (atomics[rank].self_barrier_release_status) (void)cudaFree(atomics[rank].self_barrier_release_status);
      if (atomics[rank].self_barrier_release_token) (void)cudaFree(atomics[rank].self_barrier_release_token);
      if (atomics[rank].self_barrier_gather_status) (void)cudaFree(atomics[rank].self_barrier_gather_status);
      if (atomics[rank].self_barrier_gather_token) (void)cudaFree(atomics[rank].self_barrier_gather_token);
      if (atomics[rank].self_tiles_finished) (void)cudaFree(atomics[rank].self_tiles_finished);
      if (atomics[rank].self_error) (void)cudaFree(atomics[rank].self_error);
      if (atomics[rank].self_abort) (void)cudaFree(atomics[rank].self_abort);
      if (atomics[rank].self_ag_ready) (void)cudaFree(atomics[rank].self_ag_ready);
      if (atomics[rank].self_rs_ready) (void)cudaFree(atomics[rank].self_rs_ready);
      if (device_status[rank]) (void)cudaFree(device_status[rank]);
      if (done_events[rank]) (void)cudaEventDestroy(done_events[rank]);
    }

    set_last_error("ring_allreduce: OOM or allocation failure");
    return VT_STATUS_NOMEM;
  }

  auto cleanup_allocations = [&]() {
    for (int rank = 0; rank < world_size; ++rank) {
      (void)cudaSetDevice(ring_devices[rank]);
      cudaStream_t stream = ring_streams[rank];

      // Best-effort destroy of device-side atomic objects before free.
      destroy_rank_atomics_kernel<<<blocks, kThreads, 0, stream>>>(atomics[rank]);
      (void)cudaGetLastError();
      (void)cudaStreamSynchronize(stream);

      if (atomics[rank].self_barrier_release_status) {
        (void)cudaFree(atomics[rank].self_barrier_release_status);
        atomics[rank].self_barrier_release_status = nullptr;
      }
      if (atomics[rank].self_barrier_release_token) {
        (void)cudaFree(atomics[rank].self_barrier_release_token);
        atomics[rank].self_barrier_release_token = nullptr;
      }
      if (atomics[rank].self_barrier_gather_status) {
        (void)cudaFree(atomics[rank].self_barrier_gather_status);
        atomics[rank].self_barrier_gather_status = nullptr;
      }
      if (atomics[rank].self_barrier_gather_token) {
        (void)cudaFree(atomics[rank].self_barrier_gather_token);
        atomics[rank].self_barrier_gather_token = nullptr;
      }

      if (atomics[rank].self_tiles_finished) {
        (void)cudaFree(atomics[rank].self_tiles_finished);
        atomics[rank].self_tiles_finished = nullptr;
      }

      if (atomics[rank].self_error) {
        (void)cudaFree(atomics[rank].self_error);
        atomics[rank].self_error = nullptr;
      }
      if (atomics[rank].self_abort) {
        (void)cudaFree(atomics[rank].self_abort);
        atomics[rank].self_abort = nullptr;
      }

      if (atomics[rank].self_ag_ready) {
        (void)cudaFree(atomics[rank].self_ag_ready);
        atomics[rank].self_ag_ready = nullptr;
      }
      if (atomics[rank].self_rs_ready) {
        (void)cudaFree(atomics[rank].self_rs_ready);
        atomics[rank].self_rs_ready = nullptr;
      }

      if (device_status[rank]) {
        (void)cudaFree(device_status[rank]);
        device_status[rank] = nullptr;
      }

      if (done_events[rank]) {
        (void)cudaEventDestroy(done_events[rank]);
        done_events[rank] = nullptr;
      }
    }
  };

  // Stage copy in -> out.
  uint64_t max_elems_u64 = static_cast<uint64_t>(SIZE_MAX) / sizeof(float);
  if (expected_numel < 0 || static_cast<uint64_t>(expected_numel) > max_elems_u64) {
    set_last_error("ring_allreduce: tensor is too large (byte size overflow)");
    cleanup_allocations();
    return VT_STATUS_INVALID_ARG;
  }
  const size_t nbytes = static_cast<size_t>(expected_numel) * sizeof(float);
  for (int rank = 0; rank < world_size; ++rank) {
    (void)cudaSetDevice(ring_devices[rank]);
    cudaStream_t stream = ring_streams[rank];

    if (ring_out_ptr[rank] != const_cast<float*>(ring_in_ptr[rank])) {
      cudaError_t st = cudaMemcpyAsync(ring_out_ptr[rank],
                                      ring_in_ptr[rank],
                                      nbytes,
                                      cudaMemcpyDeviceToDevice,
                                      stream);
      if (st != cudaSuccess) {
        set_last_error(std::string("cudaMemcpyAsync in->out failed: ") + cudaGetErrorString(st));
        cleanup_allocations();
        return VT_STATUS_RUNTIME_ERROR;
      }
    }
  }

  // Cross-device "inputs ready" fence: ensure all per-rank initialization and
  // staging copies have completed before any rank can read peer buffers.
  std::vector<cudaEvent_t> ready_events(world_size, nullptr);
  for (int rank = 0; rank < world_size; ++rank) {
    (void)cudaSetDevice(ring_devices[rank]);
    cudaError_t st = cudaEventCreateWithFlags(&ready_events[rank], cudaEventDisableTiming);
    if (st != cudaSuccess) {
      set_last_error(std::string("cudaEventCreateWithFlags failed: ") + cudaGetErrorString(st));
      cleanup_allocations();
      return VT_STATUS_RUNTIME_ERROR;
    }

    st = cudaEventRecord(ready_events[rank], ring_streams[rank]);
    if (st != cudaSuccess) {
      set_last_error(std::string("cudaEventRecord(ready) failed: ") + cudaGetErrorString(st));
      for (int i = 0; i <= rank; ++i) {
        (void)cudaSetDevice(ring_devices[i]);
        if (ready_events[i]) {
          (void)cudaEventDestroy(ready_events[i]);
          ready_events[i] = nullptr;
        }
      }
      cleanup_allocations();
      return VT_STATUS_RUNTIME_ERROR;
    }
  }

  for (int rank = 0; rank < world_size; ++rank) {
    (void)cudaSetDevice(ring_devices[rank]);
    for (int peer = 0; peer < world_size; ++peer) {
      cudaError_t st = cudaStreamWaitEvent(ring_streams[rank], ready_events[peer], 0);
      if (st != cudaSuccess) {
        set_last_error(std::string("cudaStreamWaitEvent(ready) failed: ") + cudaGetErrorString(st));
        for (int i = 0; i < world_size; ++i) {
          (void)cudaSetDevice(ring_devices[i]);
          if (ready_events[i]) {
            (void)cudaEventDestroy(ready_events[i]);
            ready_events[i] = nullptr;
          }
        }
        cleanup_allocations();
        return VT_STATUS_RUNTIME_ERROR;
      }
    }
  }

  // Events can be destroyed after waits are enqueued (CUDA defers the actual
  // release until all streams are done with them).
  for (int rank = 0; rank < world_size; ++rank) {
    (void)cudaSetDevice(ring_devices[rank]);
    if (ready_events[rank]) {
      (void)cudaEventDestroy(ready_events[rank]);
      ready_events[rank] = nullptr;
    }
  }

  // Prepare params.
  std::vector<RingAllreduceParams<float, 8>> params(world_size);
  for (int rank = 0; rank < world_size; ++rank) {
    RingAllreduceParams<float, 8> p{};
    p.world_size = world_size;
    p.rank = rank;
    p.epoch = 1u;

    p.count = static_cast<uint64_t>(expected_numel);
    p.num_channels = kNumChannels;

    p.tile_elems = tiling.tile_elems;
    p.num_chunks_total = tiling.num_chunks_total;
    p.max_chunk_elems = tiling.max_chunk_elems;
    p.tiles_per_chunk = tiling.tiles_per_chunk;
    p.num_tiles_total = tiling.num_tiles_total;

    // Hang-resistant defaults.
    p.timeout_iters = 1u << 18;
    p.timeout_cycles = 0;
    p.poll_sleep_start = 0;
    p.poll_sleep_ns = 0;

    p.self_data = ring_out_ptr[rank];

    p.self_rs_ready = atomics[rank].self_rs_ready;
    p.self_ag_ready = atomics[rank].self_ag_ready;
    p.self_abort = atomics[rank].self_abort;
    p.self_error = atomics[rank].self_error;

    p.self_tiles_finished = atomics[rank].self_tiles_finished;

    p.self_barrier_gather_token = atomics[rank].self_barrier_gather_token;
    p.self_barrier_gather_status = atomics[rank].self_barrier_gather_status;
    p.self_barrier_release_token = atomics[rank].self_barrier_release_token;
    p.self_barrier_release_status = atomics[rank].self_barrier_release_status;

    for (int peer = 0; peer < world_size; ++peer) {
      p.peer_data[peer] = ring_out_ptr[peer];
      p.peer_rs_ready[peer] = atomics[peer].self_rs_ready;
      p.peer_ag_ready[peer] = atomics[peer].self_ag_ready;
      p.peer_abort[peer] = atomics[peer].self_abort;

      p.peer_barrier_gather_token[peer] = atomics[peer].self_barrier_gather_token;
      p.peer_barrier_gather_status[peer] = atomics[peer].self_barrier_gather_status;
      p.peer_barrier_release_token[peer] = atomics[peer].self_barrier_release_token;
      p.peer_barrier_release_status[peer] = atomics[peer].self_barrier_release_status;
    }

    // Debug hooks disabled.
    p.debug_abort_rank = 0xffff'ffffu;
    p.debug_abort_ag_step = 0u;
    p.debug_abort_before_ag_publish = 0u;
    p.debug_abort_after_ag_publish = 0u;

    p.debug_release_delay_rank = 0xffff'ffffu;
    p.debug_release_delay_iters = 0u;

    p.debug_jitter_seed = 0u;
    p.debug_jitter_max_iters = 0u;
    p.debug_jitter_mask = 0u;

    params[rank] = p;
  }

  // Reset scalars (status/abort/error/tiles_finished/barriers) for this run.
  for (int rank = 0; rank < world_size; ++rank) {
    (void)cudaSetDevice(ring_devices[rank]);
    cudaStream_t stream = ring_streams[rank];

    (void)cudaMemsetAsync(device_status[rank], 0xFF, sizeof(uint32_t), stream);

    (void)cudaMemsetAsync(atomics[rank].self_abort, 0,
                          sizeof(RingAllreduceSystemAtomicU32), stream);
    (void)cudaMemsetAsync(atomics[rank].self_error, 0,
                          sizeof(RingAllreduceSystemAtomicU32), stream);
    (void)cudaMemsetAsync(atomics[rank].self_tiles_finished, 0,
                          sizeof(RingAllreduceDeviceAtomicU32), stream);

    (void)cudaMemsetAsync(atomics[rank].self_barrier_gather_token, 0,
                          sizeof(RingAllreduceSystemAtomicU32), stream);
    (void)cudaMemsetAsync(atomics[rank].self_barrier_gather_status, 0,
                          sizeof(RingAllreduceSystemAtomicU32), stream);
    (void)cudaMemsetAsync(atomics[rank].self_barrier_release_token, 0,
                          sizeof(RingAllreduceSystemAtomicU32), stream);
    (void)cudaMemsetAsync(atomics[rank].self_barrier_release_status, 0,
                          sizeof(RingAllreduceSystemAtomicU32), stream);
  }

  // Once any ring kernel is launched, avoid freeing allocations on error paths
  // to prevent potential UAF if kernels are still running on any device.
  bool any_ring_launched = false;

  // Launch kernels.
  for (int rank = 0; rank < world_size; ++rank) {
    (void)cudaSetDevice(ring_devices[rank]);
    cudaStream_t stream = ring_streams[rank];

    cutlass::distributed::collective::ring_allreduce_sm100<float><<<
        tiling.num_tiles_total, 256, 0, stream>>>(params[rank], device_status[rank]);

    cudaError_t st = cudaGetLastError();
    if (st != cudaSuccess) {
      set_last_error(std::string("ring_allreduce kernel launch failed: ") +
                     cudaGetErrorString(st));
      if (!any_ring_launched) {
        cleanup_allocations();
      }
      return VT_STATUS_RUNTIME_ERROR;
    }

    any_ring_launched = true;

    st = cudaEventRecord(done_events[rank], stream);
    if (st != cudaSuccess) {
      set_last_error(std::string("cudaEventRecord failed: ") + cudaGetErrorString(st));
      // Do not attempt cleanup; kernel may still be running.
      return VT_STATUS_RUNTIME_ERROR;
    }
  }

  // Wait for completion with watchdog.
  std::string wait_err;
  if (!wait_for_events(ring_devices, done_events, watchdog_ms, &wait_err)) {
    set_last_error(std::string("ring_allreduce: ") + wait_err);
    // Do not attempt cleanup; kernels may still be running.
    return VT_STATUS_RUNTIME_ERROR;
  }

  bool status_ok = true;
  std::string status_err;

  // Status check.
  for (int rank = 0; rank < world_size; ++rank) {
    uint32_t st_val = 0;
    (void)cudaSetDevice(ring_devices[rank]);
    cudaError_t st = cudaMemcpy(&st_val,
                                device_status[rank],
                                sizeof(uint32_t),
                                cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
      status_ok = false;
      status_err =
          std::string("cudaMemcpy(status) failed: ") + cudaGetErrorString(st);
      break;
    }

    RingAllreduceError e = static_cast<RingAllreduceError>(st_val);
    if (e != RingAllreduceError::kOk) {
      status_ok = false;
      status_err = std::string("ring_allreduce status rank") +
                   std::to_string(rank) + ": " +
                   ring_allreduce_error_to_string(e);
      break;
    }
  }

  cleanup_allocations();

  if (!status_ok) {
    set_last_error(status_err);
    return VT_STATUS_RUNTIME_ERROR;
  }

  // Return status=0 as a CPU int64 scalar.
  vt_status st = g_host->tensor_new_dense_like(tpl, out);
  if (st != VT_STATUS_OK) {
    set_last_error("ring_allreduce: failed to allocate status tensor");
    return st;
  }
  int64_t* p = reinterpret_cast<int64_t*>(g_host->tensor_mutable_data(*out));
  if (!p) {
    set_last_error("ring_allreduce: failed to get mutable_data for status tensor");
    return VT_STATUS_INTERNAL;
  }
  *p = 0;
  return VT_STATUS_OK;
}

static vt_status ring_allreduce_ws2(vt_stream s,
                                   const vt_tensor* args,
                                   size_t nargs,
                                   vt_tensor* out) {
  return ring_allreduce_impl(/*world_size=*/2, s, args, nargs, out);
}

static vt_status ring_allreduce_ws4(vt_stream s,
                                   const vt_tensor* args,
                                   size_t nargs,
                                   vt_tensor* out) {
  return ring_allreduce_impl(/*world_size=*/4, s, args, nargs, out);
}

static vt_status ring_allreduce_ws8(vt_stream s,
                                   const vt_tensor* args,
                                   size_t nargs,
                                   vt_tensor* out) {
  return ring_allreduce_impl(/*world_size=*/8, s, args, nargs, out);
}

} // namespace

extern "C" uint32_t vbt_plugin_get_abi_version(void) {
  return VBT_PLUGIN_ABI_VERSION;
}

extern "C" vt_status vbt_plugin_init(const struct vbt_host_api* host,
                                     struct vbt_plugin_api* out_api) {
  if (!host || !out_api) return VT_STATUS_INVALID_ARG;
  g_host = host;
  out_api->abi_version = VBT_PLUGIN_ABI_VERSION;
  out_api->name = "vbt_ring_allreduce";

  vt_status st = VT_STATUS_OK;

  if (host->register_library) {
    vt_status stl = host->register_library("vbt_dist");
    if (st == VT_STATUS_OK && stl != VT_STATUS_OK) st = stl;
  }

  if (host->def) {
    // ws2: out0,out1,in0,in1,tpl
    vt_status s2 = host->def(
        "vbt_dist::ring_allreduce_ws2(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    if (st == VT_STATUS_OK && s2 != VT_STATUS_OK) st = s2;

    // ws4: out0..out3,in0..in3,tpl
    vt_status s4 = host->def(
        "vbt_dist::ring_allreduce_ws4(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    if (st == VT_STATUS_OK && s4 != VT_STATUS_OK) st = s4;

    // ws8: out0..out7,in0..in7,tpl
    vt_status s8 = host->def(
        "vbt_dist::ring_allreduce_ws8(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
    if (st == VT_STATUS_OK && s8 != VT_STATUS_OK) st = s8;
  }

  if (host->register_kernel_boxed) {
    vt_status k2 = host->register_kernel_boxed(
        "vbt_dist::ring_allreduce_ws2", kDLCUDA, &ring_allreduce_ws2);
    if (st == VT_STATUS_OK && k2 != VT_STATUS_OK) st = k2;

    vt_status k4 = host->register_kernel_boxed(
        "vbt_dist::ring_allreduce_ws4", kDLCUDA, &ring_allreduce_ws4);
    if (st == VT_STATUS_OK && k4 != VT_STATUS_OK) st = k4;

    vt_status k8 = host->register_kernel_boxed(
        "vbt_dist::ring_allreduce_ws8", kDLCUDA, &ring_allreduce_ws8);
    if (st == VT_STATUS_OK && k8 != VT_STATUS_OK) st = k8;
  } else {
    set_last_error("register_kernel_boxed unavailable");
    return VT_STATUS_UNSUPPORTED;
  }

  // Device policy (dispatcher v2, ABI v1.4). Best-effort: if unavailable,
  // mixed-device calls will fail with a dispatcher error.
  if (host->set_device_policy) {
    // ws2 arity=5: [0]=out0 (dispatch), [1]=out1, [2]=in0, [3]=in1, [4]=tpl
    vt_device_constraint c2[4] = {
        {1, VT_CONSTRAINT_DEFER_TO_KERNEL, {0, 0, 0, 0, 0, 0}},
        {2, VT_CONSTRAINT_DEFER_TO_KERNEL, {0, 0, 0, 0, 0, 0}},
        {3, VT_CONSTRAINT_DEFER_TO_KERNEL, {0, 0, 0, 0, 0, 0}},
        {4, VT_CONSTRAINT_CPU_I64_SCALAR_0D, {0, 0, 0, 0, 0, 0}},
    };
    vt_status p2 = host->set_device_policy(
        "vbt_dist::ring_allreduce_ws2",
        VT_DEVICE_POLICY_MASKED_SAME_DEVICE,
        /*dispatch_arg_mask=*/1ULL << 0,
        c2,
        /*nconstraints=*/4,
        /*allow_undefined_mask=*/0ULL);
    if (st == VT_STATUS_OK && p2 != VT_STATUS_OK) st = p2;

    // ws4 arity=9: [0]=out0 (dispatch), [1..7]=defer, [8]=tpl
    vt_device_constraint c4[8];
    for (int i = 0; i < 7; ++i) {
      c4[i] = vt_device_constraint{static_cast<uint8_t>(i + 1), VT_CONSTRAINT_DEFER_TO_KERNEL, {0, 0, 0, 0, 0, 0}};
    }
    c4[7] = vt_device_constraint{8, VT_CONSTRAINT_CPU_I64_SCALAR_0D, {0, 0, 0, 0, 0, 0}};
    vt_status p4 = host->set_device_policy(
        "vbt_dist::ring_allreduce_ws4",
        VT_DEVICE_POLICY_MASKED_SAME_DEVICE,
        /*dispatch_arg_mask=*/1ULL << 0,
        c4,
        /*nconstraints=*/8,
        /*allow_undefined_mask=*/0ULL);
    if (st == VT_STATUS_OK && p4 != VT_STATUS_OK) st = p4;

    // ws8 arity=17: [0]=out0 (dispatch), [1..15]=defer, [16]=tpl
    vt_device_constraint c8[16];
    for (int i = 0; i < 15; ++i) {
      c8[i] = vt_device_constraint{static_cast<uint8_t>(i + 1), VT_CONSTRAINT_DEFER_TO_KERNEL, {0, 0, 0, 0, 0, 0}};
    }
    c8[15] = vt_device_constraint{16, VT_CONSTRAINT_CPU_I64_SCALAR_0D, {0, 0, 0, 0, 0, 0}};
    vt_status p8 = host->set_device_policy(
        "vbt_dist::ring_allreduce_ws8",
        VT_DEVICE_POLICY_MASKED_SAME_DEVICE,
        /*dispatch_arg_mask=*/1ULL << 0,
        c8,
        /*nconstraints=*/16,
        /*allow_undefined_mask=*/0ULL);
    if (st == VT_STATUS_OK && p8 != VT_STATUS_OK) st = p8;
  }

  return st;
}
