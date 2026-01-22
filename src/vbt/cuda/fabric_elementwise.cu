// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "vbt/cuda/fabric_addmul_decision.h"
#include "vbt/cuda/fabric_lifetime.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/event.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/fabric_state.h"
#include "vbt/cuda/fabric_events.h"
#include "vbt/logging/logging.h"

#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor.h"
#include "vbt/core/tensor_iter.h"
#include "vbt/core/tensor_iterator/cuda.h"

#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/kernel_function.h"

#ifndef VBT_WITH_CUDA
#  error "VBT_WITH_CUDA must be defined (0/1)"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#ifdef __has_include
#  if __has_include(<cuda_bf16.h>)
#    include <cuda_bf16.h>
#    define VBT_CUDA_HAS_BF16 1
#  else
#    define VBT_CUDA_HAS_BF16 0
#  endif
#else
#  define VBT_CUDA_HAS_BF16 0
#endif
#endif

namespace {

using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::TensorImpl;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::OptionalTensorImplRef;
using vbt::core::IterOperandRole;
using vbt::core::DeviceStrideMeta;

#if VBT_WITH_CUDA
using vbt::cuda::DeviceGuard;
using vbt::cuda::Stream;
using vbt::cuda::Event;
#endif

static inline void launch_bounds_and_grid(std::int64_t N,
                                         dim3& grid,
                                         dim3& block) {
#if VBT_WITH_CUDA
  const int threads = 256;
  block = dim3(threads);
  std::int64_t blocks = (N + threads - 1) / threads;
  if (blocks <= 0) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  grid = dim3(static_cast<unsigned int>(blocks));
#else
  (void)N;
  (void)grid;
  (void)block;
#endif
}

static inline bool should_use_int32_index(std::int64_t N) noexcept {
  return N <= static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
}

#if VBT_WITH_CUDA

struct AddOp {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a + b;
  }
};

struct MulOp {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};

// FP16/BF16 math is performed in float for parity with existing CUDA elementwise.
__device__ inline float load_as_float(const __half* p) { return __half2float(*p); }
__device__ inline void store_from_float(__half* p, float x) { *p = __float2half_rn(x); }

#if VBT_CUDA_HAS_BF16
__device__ inline float load_as_float(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ inline void store_from_float(__nv_bfloat16* p, float x) { *p = __float2bfloat16_rn(x); }
#endif

// Strided elementwise (out, a, b) with DeviceStrideMeta for each operand.
// Note: output may be non-contiguous; out_meta controls the write location.

template <typename T, typename Op, typename index_t>
__global__ void fabric_binary_kernel(index_t N,
                                    DeviceStrideMeta out_meta,
                                    DeviceStrideMeta a_meta,
                                    DeviceStrideMeta b_meta,
                                    T* out,
                                    const T* a,
                                    const T* b,
                                    Op op) {
  index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);
  for (; i < N; i += stride) {
    const std::int64_t li = static_cast<std::int64_t>(i);
    const std::int64_t off_o = vbt::core::compute_offset_elems(li, out_meta);
    const std::int64_t off_a = vbt::core::compute_offset_elems(li, a_meta);
    const std::int64_t off_b = vbt::core::compute_offset_elems(li, b_meta);
    out[off_o] = op(a[off_a], b[off_b]);
  }
}

template <typename S, typename Op, typename index_t>
__global__ void fabric_binary_kernel_fp16bf16(index_t N,
                                             DeviceStrideMeta out_meta,
                                             DeviceStrideMeta a_meta,
                                             DeviceStrideMeta b_meta,
                                             S* out,
                                             const S* a,
                                             const S* b,
                                             Op op) {
  index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);
  for (; i < N; i += stride) {
    const std::int64_t li = static_cast<std::int64_t>(i);
    const std::int64_t off_o = vbt::core::compute_offset_elems(li, out_meta);
    const std::int64_t off_a = vbt::core::compute_offset_elems(li, a_meta);
    const std::int64_t off_b = vbt::core::compute_offset_elems(li, b_meta);
    const float ax = load_as_float(a + off_a);
    const float bx = load_as_float(b + off_b);
    store_from_float(out + off_o, op(ax, bx));
  }
}

template <typename T, typename index_t>
__global__ void clone_strided_to_contig_kernel(T* out,
                                              const T* base_in,
                                              DeviceStrideMeta in_meta,
                                              index_t N) {
  index_t i = static_cast<index_t>(blockIdx.x * blockDim.x + threadIdx.x);
  index_t stride = static_cast<index_t>(blockDim.x * gridDim.x);
  for (; i < N; i += stride) {
    const std::int64_t li = static_cast<std::int64_t>(i);
    const std::int64_t off = vbt::core::compute_offset_elems(li, in_meta);
    out[i] = base_in[off];
  }
}

#endif // VBT_WITH_CUDA

static inline TensorImpl make_cuda_dense_out_like_shape(const TensorImpl& like,
                                                        int device_index) {
#if VBT_WITH_CUDA
  Device dev = Device::cuda(device_index);
  const std::int64_t N = like.numel();
  const std::size_t item_b = vbt::core::itemsize(like.dtype());
  const std::size_t nbytes = item_b * static_cast<std::size_t>(N);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, device_index);

  // contiguous strides
  std::vector<std::int64_t> strides(like.sizes().size());
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(strides.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    std::int64_t dim = like.sizes()[static_cast<std::size_t>(i)];
    if (dim == 0) dim = 1;
    acc *= dim;
  }
  return TensorImpl(storage,
                    like.sizes(),
                    std::move(strides),
                    /*storage_offset=*/0,
                    like.dtype(),
                    dev);
#else
  (void)like;
  (void)device_index;
  throw std::runtime_error("CUDA not built");
#endif
}

static inline std::int64_t extract_cpu_scalar_int64(const TensorImpl& t,
                                                    const char* op,
                                                    const char* argname) {
  if (t.device().type != kDLCPU || t.numel() != 1 || t.dtype() != ScalarType::Int64) {
    throw std::runtime_error(std::string("[Fabric] ") + op + ": " + argname +
                             " must be a CPU scalar int64 tensor");
  }
  const void* p = t.data();
  if (!p) {
    throw std::runtime_error(std::string("[Fabric] ") + op + ": " + argname + " has no data");
  }
  return *static_cast<const std::int64_t*>(p);
}

#if VBT_WITH_CUDA

static inline void fence_remote_current_stream_to_primary(const TensorImpl& t,
                                                          int primary_device,
                                                          const Stream& primary_stream) {
  if (t.device().type != kDLCUDA) return;
  if (t.numel() == 0) return;
  const int dev = static_cast<int>(t.device().index);
  if (dev == primary_device) return;

  const Stream remote_stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(dev));
  Event ev(false);
  ev.record(remote_stream);
  ev.wait(primary_stream);
}

static inline void synchronize_device_current_stream(int device_index,
                                                     const char* op_fqname,
                                                     const char* context) {
  DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(device_index));
  const Stream s = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(device_index));
  cudaError_t st = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(s.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": cudaStreamSynchronize failed (" +
                             (context ? context : "unknown") + ", device=" +
                             std::to_string(device_index) + "): " + (msg ? msg : ""));
  }
}

static inline bool ensure_peer_access_or_throw(int primary,
                                               int remote,
                                               const char* op_fqname) {
  (void)op_fqname;
  int can01 = 0;
  int can10 = 0;
  cudaError_t st01 = cudaDeviceCanAccessPeer(&can01, primary, remote);
  cudaError_t st10 = cudaDeviceCanAccessPeer(&can10, remote, primary);
  if (st01 != cudaSuccess || st10 != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  if (!(can01 && can10)) {
    return false;
  }

  cudaError_t e01 = vbt::cuda::Allocator::enablePeerAccess(primary, remote);
  cudaError_t e10 = vbt::cuda::Allocator::enablePeerAccess(remote, primary);
  if (e01 != cudaSuccess || e10 != cudaSuccess) {
    // Treat as non-fatal (caller may copy-fallback).
    (void)cudaGetLastError();
    return false;
  }
  return true;
}

static inline TensorImpl clone_to_contig_same_device(const TensorImpl& src) {
  if (src.device().type != kDLCUDA) {
    throw std::runtime_error("[Fabric] clone_to_contig_same_device: expected CUDA tensor");
  }
  if (src.is_contiguous()) {
    return src;
  }

  DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(src.device().index));

  TensorImpl out = make_cuda_dense_out_like_shape(src, static_cast<int>(src.device().index));
  const std::int64_t N = src.numel();
  if (N == 0) {
    return out;
  }

  // Build TI to export striding metadata for src.
  TensorIterConfig cfg;
  cfg.add_output(OptionalTensorImplRef(&out, /*defined=*/true),
                 IterOperandRole::WriteOnly,
                 /*allow_resize=*/false);
  cfg.add_input(src);
  cfg.check_mem_overlap(false);
  cfg.check_all_same_dtype(true);
  cfg.check_all_same_device(true);

  TensorIter iter = cfg.build();

  DeviceStrideMeta m_in{};
  iter.export_device_meta(/*operand_index=*/1, &m_in, vbt::core::kTensorIterCudaMaxNdim);

  dim3 grid, block;
  launch_bounds_and_grid(N, grid, block);
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(src.device().index));

  const bool use32 = should_use_int32_index(N);

  if (src.dtype() == ScalarType::Float32) {
    if (use32) {
      clone_strided_to_contig_kernel<float, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<float*>(out.data()), static_cast<const float*>(src.data()), m_in, static_cast<std::int32_t>(N));
    } else {
      clone_strided_to_contig_kernel<float, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<float*>(out.data()), static_cast<const float*>(src.data()), m_in, static_cast<std::int64_t>(N));
    }
  } else if (src.dtype() == ScalarType::Int64) {
    if (use32) {
      clone_strided_to_contig_kernel<long long, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<long long*>(out.data()), static_cast<const long long*>(src.data()), m_in, static_cast<std::int32_t>(N));
    } else {
      clone_strided_to_contig_kernel<long long, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<long long*>(out.data()), static_cast<const long long*>(src.data()), m_in, static_cast<std::int64_t>(N));
    }
  } else if (src.dtype() == ScalarType::Float16) {
    if (use32) {
      clone_strided_to_contig_kernel<__half, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<__half*>(out.data()), static_cast<const __half*>(src.data()), m_in, static_cast<std::int32_t>(N));
    } else {
      clone_strided_to_contig_kernel<__half, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<__half*>(out.data()), static_cast<const __half*>(src.data()), m_in, static_cast<std::int64_t>(N));
    }
  }
#if VBT_CUDA_HAS_BF16
  else if (src.dtype() == ScalarType::BFloat16) {
    if (use32) {
      clone_strided_to_contig_kernel<__nv_bfloat16, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(src.data()), m_in, static_cast<std::int32_t>(N));
    } else {
      clone_strided_to_contig_kernel<__nv_bfloat16, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<__nv_bfloat16*>(out.data()), static_cast<const __nv_bfloat16*>(src.data()), m_in, static_cast<std::int64_t>(N));
    }
  }
#endif
  else {
    throw std::runtime_error("[Fabric] clone_to_contig_same_device: unsupported dtype");
  }

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    throw std::runtime_error(std::string("[Fabric] clone_to_contig_same_device: kernel launch failed: ") + (msg ? msg : ""));
  }

  // Record output stream for allocator safety.
  vbt::cuda::record_stream(out.storage(), stream);
  return out;
}

static inline TensorImpl copy_to_primary_contig(const TensorImpl& src,
                                               int primary_device,
                                               const char* op_fqname) {
  if (src.device().type != kDLCUDA) {
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname +
                             ": copy fallback expects CUDA tensor inputs");
  }

  const int src_dev = static_cast<int>(src.device().index);
  if (src_dev == primary_device) {
    return src;
  }

  // Ensure the source is contiguous on its owning device, without requiring P2P.
  TensorImpl src_contig = clone_to_contig_same_device(src);

  // Ensure producer -> consumer ordering for the cross-device copy.
  // This must work even when peer access is unavailable, so we use a host-side
  // stream synchronize on the remote device.
  synchronize_device_current_stream(src_dev, op_fqname, "copy fallback pre-copy");

  // Copy the contiguous buffer to primary.
  DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(primary_device));

  TensorImpl out = make_cuda_dense_out_like_shape(src_contig, primary_device);
  const std::size_t nbytes = static_cast<std::size_t>(out.itemsize()) *
                             static_cast<std::size_t>(out.numel());
  if (nbytes == 0) {
    return out;
  }

  const Stream primary_stream =
      vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(primary_device));

  vbt::cuda::Allocator& alloc =
      vbt::cuda::Allocator::get(static_cast<vbt::cuda::DeviceIndex>(primary_device));
  cudaError_t st = alloc.memcpyAsync(out.data(), primary_device,
                                    src_contig.data(), static_cast<int>(src_contig.device().index),
                                    nbytes, primary_stream,
                                    /*p2p_enabled=*/false);
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    throw std::runtime_error(
        std::string("[Fabric] ") + op_fqname +
        ": copy fallback memcpyAsync failed (src_dev=" +
        std::to_string(static_cast<int>(src_contig.device().index)) +
        ", dst_dev=" + std::to_string(primary_device) +
        ", bytes=" + std::to_string(nbytes) +
        "): " + (msg ? msg : ""));
  }

  // Correctness-first: keep the temporary remote contiguous buffer alive until
  // the cross-device memcpy has completed.
  cudaError_t st_sync = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(primary_stream.handle()));
  if (st_sync != cudaSuccess) {
    const char* msg = cudaGetErrorString(st_sync);
    throw std::runtime_error(
        std::string("[Fabric] ") + op_fqname +
        ": copy fallback cudaStreamSynchronize failed (dst_dev=" +
        std::to_string(primary_device) +
        "): " + (msg ? msg : ""));
  }

  vbt::cuda::record_stream(out.storage(), primary_stream);
  return out;
}

static inline TensorImpl redispatch_single_device(const char* op,
                                                  const TensorImpl& a,
                                                  const TensorImpl& b) {
  vbt::dispatch::BoxedStack s{a, b};
  vbt::dispatch::Dispatcher::instance().callBoxed(op, s);
  return s[0];
}

static inline void throw_fabric_dispatch_error(const char* op,
                                              vbt::cuda::fabric::FabricAddMulFallbackReason r) {
  using vbt::cuda::fabric::FabricAddMulFallbackReason;
  std::string msg = std::string("[Fabric] ") + op + ": ";
  switch (r) {
    case FabricAddMulFallbackReason::kInvalidComputeDevice:
      msg += "invalid compute_device index";
      break;
    case FabricAddMulFallbackReason::kNotCuda:
      msg += "expects CUDA tensors";
      break;
    case FabricAddMulFallbackReason::kInvalidShapesOrDtypes:
      msg += "requires equal shapes and a supported dtype";
      break;
    case FabricAddMulFallbackReason::kFabricGloballyDisabled:
      msg += "Fabric is disabled or unavailable";
      break;
    case FabricAddMulFallbackReason::kNotInSameCliqueOrNoP2P:
      msg += "operands are not in a usable Fabric clique or lack mutual P2P";
      break;
    case FabricAddMulFallbackReason::kRequiresGrad:
    case FabricAddMulFallbackReason::kInBackward:
      msg += "Fabric ops do not support autograd";
      break;
    case FabricAddMulFallbackReason::kGraphCaptureActive:
      msg += "Fabric ops are not allowed inside CUDA graph capture";
      break;
    default:
      msg += "unsupported Fabric dispatch configuration";
      break;
  }
  throw std::runtime_error(msg);
}

template <typename Op>
static TensorImpl run_fabric_kernel_impl(const char* op_fqname,
                                        const TensorImpl& a,
                                        const TensorImpl& b,
                                        int primary_device) {
  DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(primary_device));
  auto stream = vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(primary_device));

  // Allocate dense output on primary.
  TensorImpl out = make_cuda_dense_out_like_shape(a, primary_device);
  const std::int64_t N = a.numel();
  if (N == 0) {
    return out;
  }

  // Correctness-first: fence remote producers (current streams) before reading
  // remote pointers on the primary stream.
  fence_remote_current_stream_to_primary(a, primary_device, stream);
  fence_remote_current_stream_to_primary(b, primary_device, stream);

  // Build Fabric TI for (out, a, b).
  TensorIter iter = vbt::core::make_fabric_elementwise_2gpu_iter(
      out, a, b, Device::cuda(primary_device));

  // Export metadata for each operand.
  DeviceStrideMeta mo{}, ma{}, mb{};
  iter.export_device_meta(/*operand_index=*/0, &mo, vbt::core::kTensorIterCudaMaxNdim);
  iter.export_device_meta(/*operand_index=*/1, &ma, vbt::core::kTensorIterCudaMaxNdim);
  iter.export_device_meta(/*operand_index=*/2, &mb, vbt::core::kTensorIterCudaMaxNdim);

  dim3 grid, block;
  launch_bounds_and_grid(N, grid, block);

  const bool use32 = should_use_int32_index(N);

  if (a.dtype() == ScalarType::Float32) {
    if (use32) {
      fabric_binary_kernel<float, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<float*>(out.data()),
          static_cast<const float*>(a.data()),
          static_cast<const float*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel<float, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<float*>(out.data()),
          static_cast<const float*>(a.data()),
          static_cast<const float*>(b.data()),
          Op{});
    }
  } else if (a.dtype() == ScalarType::Int64) {
    if (use32) {
      fabric_binary_kernel<long long, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<long long*>(out.data()),
          static_cast<const long long*>(a.data()),
          static_cast<const long long*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel<long long, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<long long*>(out.data()),
          static_cast<const long long*>(a.data()),
          static_cast<const long long*>(b.data()),
          Op{});
    }
  } else if (a.dtype() == ScalarType::Float16) {
    if (use32) {
      fabric_binary_kernel_fp16bf16<__half, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<__half*>(out.data()),
          static_cast<const __half*>(a.data()),
          static_cast<const __half*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel_fp16bf16<__half, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<__half*>(out.data()),
          static_cast<const __half*>(a.data()),
          static_cast<const __half*>(b.data()),
          Op{});
    }
  }
#if VBT_CUDA_HAS_BF16
  else if (a.dtype() == ScalarType::BFloat16) {
    if (use32) {
      fabric_binary_kernel_fp16bf16<__nv_bfloat16, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<__nv_bfloat16*>(out.data()),
          static_cast<const __nv_bfloat16*>(a.data()),
          static_cast<const __nv_bfloat16*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel_fp16bf16<__nv_bfloat16, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<__nv_bfloat16*>(out.data()),
          static_cast<const __nv_bfloat16*>(a.data()),
          static_cast<const __nv_bfloat16*>(b.data()),
          Op{});
    }
  }
#endif
  else {
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": unsupported dtype");
  }

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": kernel launch failed: " + (msg ? msg : ""));
  }

  vbt::cuda::record_stream(out.storage(), stream);
  return out;
}

template <typename Op>
static std::optional<TensorImpl> try_run_fabric_kernel_with_producer_fence(
    const char* op_fqname,
    const TensorImpl& a,
    const TensorImpl& b,
    int primary_device,
    int remote_device,
    bool require_fabric,
    bool use_copy_fallback) {
  DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(primary_device));

  Stream compute_stream = vbt::cuda::getDefaultStream(static_cast<vbt::cuda::DeviceIndex>(primary_device));
  try {
    compute_stream = vbt::cuda::fabric::get_fabric_compute_stream(
        static_cast<vbt::cuda::DeviceIndex>(primary_device));
  } catch (const std::exception& e) {
    if (!require_fabric && use_copy_fallback) {
      return std::nullopt;
    }
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname +
                             ": failed to acquire compute stream: " + e.what());
  }

  // Allocate dense output on primary.
  TensorImpl out = make_cuda_dense_out_like_shape(a, primary_device);
  const std::int64_t N = a.numel();
  if (N == 0) {
    return out;
  }

  vbt::cuda::fabric::FabricStorageSets storages;
  std::unordered_set<const vbt::core::Storage*> seen;
  auto add_unique = [&](std::vector<vbt::core::StoragePtr>& vec,
                        const vbt::core::StoragePtr& s) {
    if (!s || s->nbytes() == 0) return;
    const vbt::core::Storage* key = s.get();
    if (seen.insert(key).second) {
      vec.push_back(s);
    }
  };

  add_unique(storages.primary_storages, out.storage());

  const int a_dev = static_cast<int>(a.device().index);
  const int b_dev = static_cast<int>(b.device().index);

  if (a_dev == primary_device) add_unique(storages.primary_storages, a.storage());
  else add_unique(storages.remote_storages, a.storage());

  if (b_dev == primary_device) add_unique(storages.primary_storages, b.storage());
  else add_unique(storages.remote_storages, b.storage());

  auto plan_r = vbt::cuda::fabric::build_primary_remote_fence_plan(
      storages,
      static_cast<vbt::cuda::DeviceIndex>(primary_device),
      static_cast<vbt::cuda::DeviceIndex>(remote_device));

  if (!plan_r.ok()) {
    if (!require_fabric && use_copy_fallback) {
      return std::nullopt;
    }
    throw plan_r.error ? plan_r.error->as_exception()
                       : std::runtime_error(std::string("[Fabric] ") + op_fqname +
                                            ": producer-metadata failure");
  }

  auto exec_r = vbt::cuda::fabric::execute_primary_remote_fence_plan(plan_r.plan, compute_stream);
  if (!exec_r.ok()) {
    if (!require_fabric && use_copy_fallback) {
      return std::nullopt;
    }
    throw exec_r.error ? exec_r.error->as_exception()
                       : std::runtime_error(std::string("[Fabric] ") + op_fqname +
                                            ": producer-fence failure");
  }

  // Build Fabric TI for (out, a, b).
  TensorIter iter = vbt::core::make_fabric_elementwise_2gpu_iter(
      out, a, b, Device::cuda(primary_device));

  // Export metadata for each operand.
  DeviceStrideMeta mo{}, ma{}, mb{};
  iter.export_device_meta(/*operand_index=*/0, &mo, vbt::core::kTensorIterCudaMaxNdim);
  iter.export_device_meta(/*operand_index=*/1, &ma, vbt::core::kTensorIterCudaMaxNdim);
  iter.export_device_meta(/*operand_index=*/2, &mb, vbt::core::kTensorIterCudaMaxNdim);

  dim3 grid, block;
  launch_bounds_and_grid(N, grid, block);

  const bool use32 = should_use_int32_index(N);

  if (a.dtype() == ScalarType::Float32) {
    if (use32) {
      fabric_binary_kernel<float, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<float*>(out.data()),
          static_cast<const float*>(a.data()),
          static_cast<const float*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel<float, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<float*>(out.data()),
          static_cast<const float*>(a.data()),
          static_cast<const float*>(b.data()),
          Op{});
    }
  } else if (a.dtype() == ScalarType::Int64) {
    if (use32) {
      fabric_binary_kernel<long long, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<long long*>(out.data()),
          static_cast<const long long*>(a.data()),
          static_cast<const long long*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel<long long, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<long long*>(out.data()),
          static_cast<const long long*>(a.data()),
          static_cast<const long long*>(b.data()),
          Op{});
    }
  } else if (a.dtype() == ScalarType::Float16) {
    if (use32) {
      fabric_binary_kernel_fp16bf16<__half, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<__half*>(out.data()),
          static_cast<const __half*>(a.data()),
          static_cast<const __half*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel_fp16bf16<__half, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<__half*>(out.data()),
          static_cast<const __half*>(a.data()),
          static_cast<const __half*>(b.data()),
          Op{});
    }
  }
#if VBT_CUDA_HAS_BF16
  else if (a.dtype() == ScalarType::BFloat16) {
    if (use32) {
      fabric_binary_kernel_fp16bf16<__nv_bfloat16, Op, std::int32_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int32_t>(N), mo, ma, mb,
          static_cast<__nv_bfloat16*>(out.data()),
          static_cast<const __nv_bfloat16*>(a.data()),
          static_cast<const __nv_bfloat16*>(b.data()),
          Op{});
    } else {
      fabric_binary_kernel_fp16bf16<__nv_bfloat16, Op, std::int64_t><<<grid, block, 0, reinterpret_cast<cudaStream_t>(compute_stream.handle())>>>(
          static_cast<std::int64_t>(N), mo, ma, mb,
          static_cast<__nv_bfloat16*>(out.data()),
          static_cast<const __nv_bfloat16*>(a.data()),
          static_cast<const __nv_bfloat16*>(b.data()),
          Op{});
    }
  }
#endif
  else {
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": unsupported dtype");
  }

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": kernel launch failed: " + (msg ? msg : ""));
  }

  // compute stream. We guard all primary storages (inputs + outputs), not
  // just the output tensor.
  for (const auto& S : storages.primary_storages) {
    if (S && S->nbytes() > 0) {
      vbt::cuda::record_stream(S, compute_stream);
    }
  }

  // remote device. We propagate completion of the Fabric kernels on the
  // primary compute stream to the remote proxy stream using a CUDA event,
  // then record allocator usage for all remote storages on that proxy.
  if (!storages.remote_storages.empty()) {
    // By construction (see FabricAddMulIO/classify_fabric_storages in the
    VBT_ASSERT(remote_device >= 0);

    Stream proxy_stream;
    try {
      proxy_stream = vbt::cuda::fabric::get_fabric_proxy_stream(
          static_cast<vbt::cuda::DeviceIndex>(remote_device));

      Event done(/*enable_timing=*/false);
      done.record(compute_stream);
      done.wait(proxy_stream);
    } catch (const std::exception& e) {
      // Guard failure after kernels have been launched: ensure that all work
      // on the primary compute stream has completed before returning so that
      // the allocator can safely reuse remote storages even without remote
      // record_stream guards.
      cudaError_t st_sync = cudaStreamSynchronize(
          reinterpret_cast<cudaStream_t>(compute_stream.handle()));
      if (st_sync != cudaSuccess) {
        const char* sync_msg = cudaGetErrorString(st_sync);
        throw std::runtime_error(
            std::string("[Fabric] ") + op_fqname +
            ": remote lifetime guard setup failed and cudaStreamSynchronize also failed: " +
            e.what() + " / sync_error=" + (sync_msg ? sync_msg : ""));
      }

      throw std::runtime_error(
          std::string("[Fabric] ") + op_fqname +
          ": remote lifetime guard setup failed after kernel launch: " + e.what());
    }

    // vbt::cuda::record_stream (and the underlying Allocator::record_stream)
    // are noexcept by design, so once we reach this point we either arm all
    // remote guards or none of them for this op.
    for (const auto& S : storages.remote_storages) {
      if (S && S->nbytes() > 0) {
        vbt::cuda::record_stream(S, proxy_stream);
      }
    }
  }

  return out;
}

static TensorImpl fabric_addmul_impl(const char* op_fqname,
                                    const TensorImpl& a,
                                    const TensorImpl& b,
                                    int compute_device,
                                    bool require_fabric,
                                    bool use_copy_fallback,
                                    bool is_mul) {
  if (a.device().type != kDLCUDA || b.device().type != kDLCUDA) {
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": expects CUDA tensors");
  }
  if (a.sizes() != b.sizes()) {
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": requires equal shapes");
  }
  if (a.dtype() != b.dtype()) {
    throw std::runtime_error(std::string("[Fabric] ") + op_fqname + ": requires equal dtypes");
  }

  const bool cross_device_inputs = (a.device() != b.device());

  // Single-device case: always behave like the existing single-device op.
  if (!cross_device_inputs) {
    return redispatch_single_device(is_mul ? "vt::mul" : "vt::add", a, b);
  }

  // Global stats live in FabricState.
  auto& fs = vbt::cuda::fabric::fabric_state();
  auto& stats = fs.stats;

  // Host-call backpressure for cross-device inputs (regardless of hit/fallback/error).
  vbt::cuda::fabric::FabricInflightGuard inflight_guard(&stats, cross_device_inputs);
  (void)inflight_guard;

  const std::size_t itemsize = static_cast<std::size_t>(a.itemsize());

  // Decision helper.
  vbt::cuda::fabric::FabricAddMulDecision dec = vbt::cuda::fabric::decide_fabric_addmul_2gpu(
      compute_device, a, b, require_fabric, use_copy_fallback);

  // Events integration: assign a logical op_id and emit enqueue/terminal events.
  const bool events_on = vbt::cuda::fabric::fabric_events_enabled();
  vbt::cuda::fabric::FabricOpId op_id = 0;
  bool terminal_event_recorded = false;
  vbt::cuda::fabric::FabricAddMulDecision dec_for_event = dec;

  auto emit_event = [&](vbt::cuda::fabric::FabricEventKind kind,
                        vbt::cuda::fabric::FabricEventLevel level,
                        const vbt::cuda::fabric::FabricAddMulDecision& d,
                        const char* msg) noexcept {
    if (!events_on) return;
    if (op_id == 0) return;

    vbt::cuda::fabric::FabricEvent ev;
    ev.kind = kind;
    ev.level = level;
    ev.primary_device = d.primary_device;
    ev.other_device = d.other_device;
    ev.op_id = op_id;
    ev.numel = d.numel > 0 ? static_cast<std::uint64_t>(d.numel) : 0ull;
    ev.bytes = vbt::cuda::fabric::compute_fabric_bytes(d, itemsize);
    ev.reason_raw = static_cast<std::uint32_t>(d.reason);
    ev.message = msg;
    vbt::cuda::fabric::record_fabric_event(std::move(ev));
  };

  if (events_on) {
    op_id = fs.next_op_id.fetch_add(1, std::memory_order_relaxed) + 1;
    emit_event(vbt::cuda::fabric::FabricEventKind::kOpEnqueue,
               vbt::cuda::fabric::FabricEventLevel::kInfo,
               dec_for_event,
               vbt::cuda::fabric::kFabricEventMsgOpEnqueue);
  }

  try {
    if (dec.reason == vbt::cuda::fabric::FabricAddMulFallbackReason::kInvalidComputeDevice) {
      // Treat misconfigured compute_device as a configuration error rather than
      // a Fabric decision: we surface a hard error and intentionally leave
      // Fabric stats unchanged.
      throw_fabric_dispatch_error(op_fqname, dec.reason);
    }

  // Dry-run mode: force copy fallback when policy allows.
  const auto mode = fs.config.mode.load(std::memory_order_acquire);
  const bool dry_run = (mode == vbt::cuda::fabric::FabricMode::DryRun);

  if (dry_run && !require_fabric && use_copy_fallback) {
    vbt::cuda::fabric::FabricAddMulDecision dec_stats = dec;
    dec_stats.use_fabric = false;
    dec_stats.use_copy_fallback = true;
    vbt::cuda::fabric::record_fabric_decision_stats(stats, dec_stats, itemsize);

    TensorImpl a_p = copy_to_primary_contig(a, compute_device, op_fqname);
    TensorImpl b_p = copy_to_primary_contig(b, compute_device, op_fqname);
    dec_for_event = dec_stats;
    TensorImpl out = redispatch_single_device(is_mul ? "vt::mul" : "vt::add", a_p, b_p);
    emit_event(vbt::cuda::fabric::FabricEventKind::kOpFallback,
               vbt::cuda::fabric::FabricEventLevel::kInfo,
               dec_for_event,
               vbt::cuda::fabric::kFabricEventMsgOpFallback);
    terminal_event_recorded = true;
    return out;
  }

  // NOTE: record_fabric_decision_stats is the *only* writer for Fabric add/mul
  // decision stats in this function.

  if (!dec.use_fabric && !dec.use_copy_fallback) {
    vbt::cuda::fabric::record_fabric_decision_stats(stats, dec, itemsize);
    throw_fabric_dispatch_error(op_fqname, dec.reason);
  }

  if (dec.use_copy_fallback) {
    vbt::cuda::fabric::record_fabric_decision_stats(stats, dec, itemsize);

    TensorImpl a_p = copy_to_primary_contig(a, dec.primary_device, op_fqname);
    TensorImpl b_p = copy_to_primary_contig(b, dec.primary_device, op_fqname);
    dec_for_event = dec;
    TensorImpl out = redispatch_single_device(is_mul ? "vt::mul" : "vt::add", a_p, b_p);
    emit_event(vbt::cuda::fabric::FabricEventKind::kOpFallback,
               vbt::cuda::fabric::FabricEventLevel::kInfo,
               dec_for_event,
               vbt::cuda::fabric::kFabricEventMsgOpFallback);
    terminal_event_recorded = true;
    return out;
  }

  // Fabric fast path.
  if (!ensure_peer_access_or_throw(dec.primary_device, dec.other_device, op_fqname)) {
    vbt::cuda::fabric::FabricAddMulDecision dec_no_p2p = dec;
    dec_no_p2p.reason = vbt::cuda::fabric::FabricAddMulFallbackReason::kNotInSameCliqueOrNoP2P;
    dec_no_p2p.use_fabric = false;
    dec_no_p2p.use_copy_fallback = (!require_fabric && use_copy_fallback);
    vbt::cuda::fabric::record_fabric_decision_stats(stats, dec_no_p2p, itemsize);
    dec_for_event = dec_no_p2p;

    if (dec_no_p2p.use_copy_fallback) {
      TensorImpl a_p = copy_to_primary_contig(a, dec.primary_device, op_fqname);
      TensorImpl b_p = copy_to_primary_contig(b, dec.primary_device, op_fqname);
      TensorImpl out = redispatch_single_device(is_mul ? "vt::mul" : "vt::add", a_p, b_p);
      emit_event(vbt::cuda::fabric::FabricEventKind::kOpFallback,
                 vbt::cuda::fabric::FabricEventLevel::kInfo,
                 dec_for_event,
                 vbt::cuda::fabric::kFabricEventMsgOpFallback);
      terminal_event_recorded = true;
      return out;
    }
    throw std::runtime_error(
        std::string("[Fabric] ") + op_fqname +
        ": peer access not available for the selected device pair (primary=" +
        std::to_string(dec.primary_device) + ", remote=" +
        std::to_string(dec.other_device) + ")");
  }

  // Event-based lifetime path.
  const bool lifetime_enabled = vbt::cuda::fabric::is_fabric_event_lifetime_enabled();
  if (lifetime_enabled) {
    std::optional<TensorImpl> maybe_out;
    if (is_mul) {
      maybe_out = try_run_fabric_kernel_with_producer_fence<MulOp>(
          op_fqname, a, b,
          dec.primary_device,
          dec.other_device,
          require_fabric,
          use_copy_fallback);
    } else {
      maybe_out = try_run_fabric_kernel_with_producer_fence<AddOp>(
          op_fqname, a, b,
          dec.primary_device,
          dec.other_device,
          require_fabric,
          use_copy_fallback);
    }

    if (!maybe_out.has_value()) {
      vbt::cuda::fabric::FabricAddMulDecision dec_fb = dec;
      dec_fb.use_fabric = false;
      dec_fb.use_copy_fallback = true;
      vbt::cuda::fabric::record_fabric_decision_stats(stats, dec_fb, itemsize);

      TensorImpl a_p = copy_to_primary_contig(a, dec.primary_device, op_fqname);
      TensorImpl b_p = copy_to_primary_contig(b, dec.primary_device, op_fqname);
      dec_for_event = dec_fb;
      TensorImpl out = redispatch_single_device(is_mul ? "vt::mul" : "vt::add", a_p, b_p);
      emit_event(vbt::cuda::fabric::FabricEventKind::kOpFallback,
                 vbt::cuda::fabric::FabricEventLevel::kInfo,
                 dec_for_event,
                 vbt::cuda::fabric::kFabricEventMsgOpFallback);
      terminal_event_recorded = true;
      return out;
    }

    vbt::cuda::fabric::record_fabric_decision_stats(stats, dec, itemsize);
    dec_for_event = dec;
    emit_event(vbt::cuda::fabric::FabricEventKind::kOpComplete,
               vbt::cuda::fabric::FabricEventLevel::kInfo,
               dec_for_event,
               vbt::cuda::fabric::kFabricEventMsgOpComplete);
    terminal_event_recorded = true;
    return *maybe_out;
  }

  // Legacy path: kernels on current stream, plus current-stream remote fence.
  TensorImpl out = is_mul ? run_fabric_kernel_impl<MulOp>(op_fqname, a, b, dec.primary_device)
                          : run_fabric_kernel_impl<AddOp>(op_fqname, a, b, dec.primary_device);

  vbt::cuda::fabric::record_fabric_decision_stats(stats, dec, itemsize);
  dec_for_event = dec;
  emit_event(vbt::cuda::fabric::FabricEventKind::kOpComplete,
             vbt::cuda::fabric::FabricEventLevel::kInfo,
             dec_for_event,
             vbt::cuda::fabric::kFabricEventMsgOpComplete);
  terminal_event_recorded = true;
  return out;
  } catch (...) {
    if (events_on && op_id != 0 && !terminal_event_recorded) {
      emit_event(vbt::cuda::fabric::FabricEventKind::kOpError,
                 vbt::cuda::fabric::FabricEventLevel::kError,
                 dec_for_event,
                 vbt::cuda::fabric::kFabricEventMsgOpError);
      terminal_event_recorded = true;
    }
    throw;
  }
}

#endif // VBT_WITH_CUDA

} // namespace

extern "C" {

#if VBT_WITH_CUDA

TensorImpl vbt_cuda_fabric_add_impl(const TensorImpl& a,
                                   const TensorImpl& b,
                                   int compute_device,
                                   bool require_fabric,
                                   bool use_copy_fallback) {
  return fabric_addmul_impl("vt::fabric_add", a, b, compute_device, require_fabric, use_copy_fallback, /*is_mul=*/false);
}

TensorImpl vbt_cuda_fabric_mul_impl(const TensorImpl& a,
                                   const TensorImpl& b,
                                   int compute_device,
                                   bool require_fabric,
                                   bool use_copy_fallback) {
  return fabric_addmul_impl("vt::fabric_mul", a, b, compute_device, require_fabric, use_copy_fallback, /*is_mul=*/true);
}

static void vbt_cuda_fabric_add_boxed(vbt::dispatch::BoxedStack& s) {
  const int compute_device = static_cast<int>(extract_cpu_scalar_int64(s[2], "vt::fabric_add", "compute_device"));
  const bool require_fabric = extract_cpu_scalar_int64(s[3], "vt::fabric_add", "require_fabric") != 0;
  const bool use_copy_fallback = extract_cpu_scalar_int64(s[4], "vt::fabric_add", "use_copy_fallback") != 0;
  TensorImpl out = vbt_cuda_fabric_add_impl(s[0], s[1], compute_device, require_fabric, use_copy_fallback);
  s.clear();
  s.push_back(out);
}

static void vbt_cuda_fabric_mul_boxed(vbt::dispatch::BoxedStack& s) {
  const int compute_device = static_cast<int>(extract_cpu_scalar_int64(s[2], "vt::fabric_mul", "compute_device"));
  const bool require_fabric = extract_cpu_scalar_int64(s[3], "vt::fabric_mul", "require_fabric") != 0;
  const bool use_copy_fallback = extract_cpu_scalar_int64(s[4], "vt::fabric_mul", "use_copy_fallback") != 0;
  TensorImpl out = vbt_cuda_fabric_mul_impl(s[0], s[1], compute_device, require_fabric, use_copy_fallback);
  s.clear();
  s.push_back(out);
}

void vbt_register_fabric_kernels() {
  auto& D = vbt::dispatch::Dispatcher::instance();

  if (!D.has("vt::fabric_add")) {
    D.registerLibrary("vt");
    D.def("vt::fabric_add(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
  }
  if (!D.has("vt::fabric_mul")) {
    D.registerLibrary("vt");
    D.def("vt::fabric_mul(Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor");
  }

  // Register boxed CUDA kernels.
  if (!D.find("vt::fabric_add").get().cuda_base.has_value()) {
    D.registerCudaKernelFunction("vt::fabric_add",
                                 vbt::dispatch::KernelFunction::makeBoxed(
                                     /*arity=*/5, &vbt_cuda_fabric_add_boxed));
  }
  if (!D.find("vt::fabric_mul").get().cuda_base.has_value()) {
    D.registerCudaKernelFunction("vt::fabric_mul",
                                 vbt::dispatch::KernelFunction::makeBoxed(
                                     /*arity=*/5, &vbt_cuda_fabric_mul_boxed));
  }

  // Mark Fabric allowlist for dispatcher bypass.
  D.mark_fabric_op("vt::fabric_add", /*is_fabric_op=*/true,
                   /*allow_multi_device_fabric=*/true);
  D.mark_fabric_op("vt::fabric_mul", /*is_fabric_op=*/true,
                   /*allow_multi_device_fabric=*/true);
}

#else

TensorImpl vbt_cuda_fabric_add_impl(const TensorImpl& /*a*/, const TensorImpl& /*b*/, int /*compute_device*/, bool /*require_fabric*/, bool /*use_copy_fallback*/) {
  throw std::runtime_error("CUDA not built");
}
TensorImpl vbt_cuda_fabric_mul_impl(const TensorImpl& /*a*/, const TensorImpl& /*b*/, int /*compute_device*/, bool /*require_fabric*/, bool /*use_copy_fallback*/) {
  throw std::runtime_error("CUDA not built");
}

void vbt_register_fabric_kernels() {
  // No-op when CUDA is not built.
}

#endif

} // extern "C"
