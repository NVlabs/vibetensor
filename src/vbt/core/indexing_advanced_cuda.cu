// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vbt/core/indexing.h"
#include "vbt/core/indexing_advanced_stats.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/logging/logging.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <algorithm>
#include <atomic>
#include <mutex>

#include "vbt/core/checked_math.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/tensor_ops.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/cub.h"
#include "vbt/cuda/stream.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/graphs.h"

#ifndef VBT_WITH_CUDA
#  error "indexing_advanced_cuda.cu requires VBT_WITH_CUDA=1"
#endif
static_assert(VBT_WITH_CUDA == 0 || VBT_WITH_CUDA == 1,
              "VBT_WITH_CUDA must be 0 or 1");

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#ifdef __has_include
#  if __has_include(<cuda_bf16.h>)
#    include <cuda_bf16.h>
#    define VBT_INDEX_CUDA_HAS_BF16 1
#  else
#    define VBT_INDEX_CUDA_HAS_BF16 0
#  endif
#else
#  define VBT_INDEX_CUDA_HAS_BF16 0
#endif
#endif

namespace vbt {
namespace core {
namespace indexing {

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::checked_add_i64;
using vbt::core::checked_mul_i64;
using vbt::cuda::DeviceGuard;
using vbt::cuda::getCurrentStream;
using vbt::cuda::DeviceIndex;
using vbt::cuda::Stream;
using detail::CudaBoundsMode;

// cached env configuration. These are thin wrappers so that hot-path
// code does not need to know about the full AdvancedIndexEnvConfig
// structure.
static inline bool cuda_bool_mask_indices_enabled() noexcept {
  return detail::get_advanced_index_env_config().cuda_allow_bool_mask_indices;
}

static inline bool cuda_bool_mask_cub_backend_enabled() noexcept {
  return detail::get_advanced_index_env_config().cuda_bool_mask_use_cub;
}

static inline bool cuda_extended_index_dtypes_enabled() noexcept {
  return detail::get_advanced_index_env_config().cuda_allow_extended_dtypes;
}

static inline bool cuda_cub_index_put_accumulate_enabled() noexcept {
  return detail::get_advanced_index_env_config().cuda_cub_index_put_accumulate;
}

namespace {

// These caps mirror the CPU-side limits in indexing_advanced.cc.
constexpr std::int64_t kAdvIndexMaxIndexNumel  = 10'000'000;   // 1e7
constexpr std::int64_t kAdvIndexMaxResultNumel = 100'000'000;  // 1e8
constexpr std::int64_t kAdvIndexMaxMaskNumel   = 10'000'000;   // 1e7 (mask.numel cap)

static_assert(kAdvIndexMaxResultNumel <=
              static_cast<std::int64_t>(
                  std::numeric_limits<std::int32_t>::max()),
              "CUDA advanced indexing result cap must fit in int32_t for 32-bit loops");

// We keep the CUDA rank cap aligned with TensorIterator's CUDA helper.
inline constexpr int kIndexCudaMaxRank = 25;

inline std::int64_t safe_numel_from_sizes_cuda(
    const std::vector<std::int64_t>& sizes) {
  if (sizes.empty()) {
    return 1;  // scalar
  }
  std::int64_t n = 1;
  for (std::int64_t s : sizes) {
    if (s == 0) {
      return 0;
    }
    std::int64_t tmp = 0;
    if (!checked_mul_i64(n, s, tmp)) {
      // Mirror TensorImpl::numel semantics: treat overflow as zero.
      return 0;
    }
    n = tmp;
  }
  return n;
}

inline bool compute_use32bit_indexing_for_sizes_cuda(
    const std::vector<std::int64_t>& sizes) {
  const std::int64_t R = safe_numel_from_sizes_cuda(sizes);
  if (R == 0) {
    // Empty or overflow-sentinel domain.
    return true;
  }
  return R > 0 &&
         R <= static_cast<std::int64_t>(
                  std::numeric_limits<std::int32_t>::max());
}

inline bool should_use_32bit_for_advanced_index_cuda(
    const AdvancedIndex& info,
    std::int64_t         N) {
  // Precondition (production):
  //   - N is the logical number of result elements for this AdvancedIndex,
  //     computed as safe_numel_from_sizes_cuda(info.result_shape) and N > 0.
  //   - Zero-numel (empty or overflow-sentinel) cases were handled by the
  //     caller via an early return.

  if (!advanced_index_32bit_enabled()) {
    return false;
  }
  if (!info.use32bit_indexing) {
    return false;
  }
  if (N <= 0) {
    return false;  // defensive; callers should have early-returned.
  }

  return N <= static_cast<std::int64_t>(
                 std::numeric_limits<std::int32_t>::max());
}

struct Grid1DConfig {
  dim3 block_dim;
  dim3 grid_dim;
};

#if VBT_WITH_CUDA
struct DeviceCapsEntry {
  int          device_index{0};
  unsigned int max_grid_x{1};
};

static std::mutex                 g_device_caps_mutex;
static std::vector<DeviceCapsEntry> g_device_caps_cache;

static unsigned int get_device_max_grid_x(DeviceIndex dev_index) {
  const int idx = static_cast<int>(dev_index);
  std::lock_guard<std::mutex> lock(g_device_caps_mutex);
  for (const auto& e : g_device_caps_cache) {
    if (e.device_index == idx) {
      return e.max_grid_x;
    }
  }

  cudaDeviceProp prop{};
  unsigned int cap = 1u;
  cudaError_t st = cudaGetDeviceProperties(&prop, idx);
  if (st == cudaSuccess && prop.maxGridSize[0] > 0) {
    cap = static_cast<unsigned int>(prop.maxGridSize[0]);
  }
  g_device_caps_cache.push_back(DeviceCapsEntry{idx, cap});
  return cap;
}
#endif  // VBT_WITH_CUDA

static Grid1DConfig make_1d_grid(std::int64_t N,
                                 int          threads,
                                 DeviceIndex  dev_index) {
#if !VBT_WITH_CUDA
  (void)N;
  (void)threads;
  (void)dev_index;
  throw std::runtime_error(
      "advanced_index_cuda: built without CUDA support");
#else
  const auto& cfg = detail::get_advanced_index_env_config();

#ifndef NDEBUG
  VBT_ASSERT(threads > 0);
  VBT_ASSERT(N >= 0);
#endif

  if (N <= 0) {
    return Grid1DConfig{dim3(static_cast<unsigned int>(threads)), dim3(1u)};
  }

  const unsigned int device_max = get_device_max_grid_x(dev_index);

  unsigned int env_cap = device_max;
  if (cfg.cuda_max_blocks_cap > 0) {
    const std::int64_t capped = std::min<std::int64_t>(
        cfg.cuda_max_blocks_cap,
        static_cast<std::int64_t>(device_max));
    env_cap = static_cast<unsigned int>(
        std::max<std::int64_t>(capped, 1));
  }

  const unsigned int max_blocks = std::max(1u, std::min(device_max, env_cap));

  std::int64_t requested_blocks = (N + threads - 1) / threads;
  if (requested_blocks <= 0) {
    requested_blocks = 1;
  }

  std::int64_t final_blocks_i64 = std::min<std::int64_t>(
      requested_blocks,
      static_cast<std::int64_t>(max_blocks));
  if (final_blocks_i64 <= 0) {
    final_blocks_i64 = 1;
  }

  const unsigned int final_blocks =
      static_cast<unsigned int>(final_blocks_i64);

  return Grid1DConfig{
      dim3(static_cast<unsigned int>(threads)),
      dim3(final_blocks)};
#endif
}

#if VBT_INTERNAL_TESTS
// -1: no override; 0: LegacyHost; 1: DeviceNormalized.
static std::atomic<int> g_test_bounds_mode_override{-1};
#endif

static CudaBoundsMode get_effective_cuda_bounds_mode() {
  const auto& cfg = detail::get_advanced_index_env_config();

  if (cfg.cuda_gpu_bounds_disable) {
    return CudaBoundsMode::LegacyHost;
  }

#if VBT_INTERNAL_TESTS
  const int override = g_test_bounds_mode_override.load(std::memory_order_relaxed);
  if (override == 0) {
    return CudaBoundsMode::LegacyHost;
  }
  if (override == 1) {
    return CudaBoundsMode::DeviceNormalized;
  }
#endif

  return cfg.cuda_bounds_default;
}

struct AdvancedIndexCudaMeta {
  int32_t R{0};            // result rank
  int32_t index_ndim{0};   // number of index dims
  int32_t dims_before{0};  // number of prefix dims in base
  int32_t pad{0};          // reserved

  // Only the first R entries of result_sizes and base_strides are used.
  std::int64_t result_sizes[kIndexCudaMaxRank]{};
  std::int64_t base_strides[kIndexCudaMaxRank]{};
  // Only the first index_ndim entries of index_sizes are used.
  std::int64_t index_sizes[kIndexCudaMaxRank]{};

  // Base storage offset in elements.
  std::int64_t storage_offset{0};
};

struct AdvancedIndexPutCudaMeta {
  int32_t ndim{0};          // rank of result_shape
  int32_t dims_before{0};   // number of prefix dims in src
  int32_t index_ndim{0};    // number of index dims
  int32_t pad{0};           // reserved / future use

  std::int64_t result_sizes[kIndexCudaMaxRank]{};   // logical result sizes
  std::int64_t dst_strides[kIndexCudaMaxRank]{};    // src.strides() in elements
  std::int64_t index_shape[kIndexCudaMaxRank]{};    // index broadcast shape
  std::int64_t value_strides[kIndexCudaMaxRank]{};  // value_b.strides() in elements

  std::int64_t indexed_stride_elems{0};   // stride along advanced dim in elements
  std::int64_t storage_offset_elems{0};   // base.storage_offset() in elements
};

// Allocate a contiguous CPU Int64 tensor with the given sizes.
static TensorImpl make_cpu_int64_tensor(
    const std::vector<std::int64_t>& sizes) {
  using vbt::core::Storage;
  using vbt::core::DataPtr;
  using vbt::core::itemsize;

  const Device dev = Device::cpu();
  const ScalarType dtype = ScalarType::Int64;
  const std::size_t item_b = static_cast<std::size_t>(itemsize(dtype));
  const std::int64_t n = safe_numel_from_sizes_cuda(sizes);
  const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  auto storage = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<std::int64_t> strides(sizes.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return TensorImpl(storage, sizes, strides,
                    /*storage_offset=*/0, dtype, dev);
}

// Allocate a contiguous CUDA tensor with the given sizes and the same
// dtype/device as `like`.
static TensorImpl make_cuda_dense_out(
    const TensorImpl& like,
    const std::vector<std::int64_t>& sizes) {
#if VBT_WITH_CUDA
  using vbt::core::itemsize;

  auto dev = like.device();
  if (dev.type != kDLCUDA) {
    throw std::invalid_argument("index_cuda: expected CUDA tensor for make_cuda_dense_out");
  }

  const ScalarType dtype = like.dtype();
  const std::size_t item_b = static_cast<std::size_t>(itemsize(dtype));
  const std::int64_t n = safe_numel_from_sizes_cuda(sizes);
  const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev.index);

  std::vector<std::int64_t> strides(sizes.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return TensorImpl(storage, sizes, strides,
                    /*storage_offset=*/0, dtype, dev);
#else
  (void)like;
  (void)sizes;
  throw std::runtime_error("index_cuda: built without CUDA support");
#endif
}

// Allocate a contiguous CUDA Int64 tensor with the given sizes on the same
// device as `like`.
static TensorImpl make_cuda_int64_tensor(
    const TensorImpl& like,
    const std::vector<std::int64_t>& sizes) {
#if VBT_WITH_CUDA
  using vbt::core::itemsize;

  auto dev = like.device();
  if (dev.type != kDLCUDA) {
    throw std::invalid_argument("index_cuda: expected CUDA tensor for make_cuda_int64_tensor");
  }

  const ScalarType dtype = ScalarType::Int64;
  const std::size_t item_b = static_cast<std::size_t>(itemsize(dtype));
  const std::int64_t n = safe_numel_from_sizes_cuda(sizes);
  const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev.index);

  std::vector<std::int64_t> strides(sizes.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return TensorImpl(storage, sizes, strides,
                    /*storage_offset=*/0, dtype, dev);
#else
  (void)like;
  (void)sizes;
  throw std::runtime_error("index_cuda: built without CUDA support");
#endif
}

// Allocate a contiguous CUDA Int32 tensor with the given sizes on the same
// device as `like`.
static TensorImpl make_cuda_int32_tensor(
    const TensorImpl& like,
    const std::vector<std::int64_t>& sizes) {
#if VBT_WITH_CUDA
  using vbt::core::itemsize;

  auto dev = like.device();
  if (dev.type != kDLCUDA) {
    throw std::invalid_argument("index_cuda: expected CUDA tensor for make_cuda_int32_tensor");
  }

  const ScalarType dtype = ScalarType::Int32;
  const std::size_t item_b = static_cast<std::size_t>(itemsize(dtype));
  const std::int64_t n = safe_numel_from_sizes_cuda(sizes);
  const std::size_t nbytes = static_cast<std::size_t>(n) * item_b;

  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev.index);

  std::vector<std::int64_t> strides(sizes.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  return TensorImpl(storage, sizes, strides,
                    /*storage_offset=*/0, dtype, dev);
#else
  (void)like;
  (void)sizes;
  throw std::runtime_error("index_cuda: built without CUDA support");
#endif
}

inline bool advanced_indices_are_raw_cuda(const AdvancedIndex& info) {
  if (info.indices.size() != 1u) {
    return false;
  }

  const TensorImpl& idx = info.indices[0];
  const auto dev = idx.device();
  if (dev.type != kDLCUDA) {
    return false;
  }
  if (!(dev == info.src.device())) {
    return false;
  }

  const ScalarType dt = idx.dtype();
  if (dt != ScalarType::Int32 && dt != ScalarType::Int64) {
    return false;
  }

  const auto& ish = info.index_shape;
  const auto& idx_sizes = idx.sizes();
  if (ish.size() != idx_sizes.size()) {
    return false;
  }

  const std::int64_t numel_meta =
      safe_numel_from_sizes_cuda(ish);
  const std::int64_t numel_idx = idx.numel();
  if (numel_meta != numel_idx) {
    return false;
  }

  return true;
}

inline void check_advanced_indices_are_raw_cuda(const AdvancedIndex& info) {
  if (!advanced_indices_are_raw_cuda(info)) {
    throw std::logic_error(
        "CUDA AdvancedIndex invariants violated: expected raw CUDA index tensor");
  }
}

struct NormalizedIndexBuffers {
  TensorImpl   idx_cpu;     // contiguous Int64 on CPU (may be zero-numel)
  TensorImpl   idx_cuda;    // contiguous Int64 on CUDA (same device as src)
  std::int64_t D{0};
  std::int64_t index_numel{0};
};

struct NormalizedIndexBuffersDevice {
  TensorImpl   idx_cuda;    // contiguous Int64 on CUDA (same device as src)
  std::int64_t D{0};
  std::int64_t index_numel{0};
};

struct NormalizedIndexBuffersAny {
  TensorImpl   idx_cuda;    // contiguous Int64 on CUDA (same device as src)
  std::int64_t D{0};
  std::int64_t index_numel{0};
};

static NormalizedIndexBuffers make_normalized_index_buffers(
    const TensorImpl&                index_raw,
    const std::vector<std::int64_t>& idx_sizes,
    std::int64_t                     D,
    bool                             is_read,
    bool                             allow_int32,
    DeviceIndex                      dev_index,
    Stream                           stream) {
#if !VBT_WITH_CUDA
  (void)index_raw;
  (void)idx_sizes;
  (void)D;
  (void)is_read;
  (void)allow_int32;
  (void)dev_index;
  (void)stream;
  throw std::runtime_error(
      "advanced_index_cuda: built without CUDA support");
#else
  NormalizedIndexBuffers bufs{};
  bufs.D = D;
  bufs.index_numel = safe_numel_from_sizes_cuda(idx_sizes);

  // Allocate logical buffers up front so zero-numel still has correct shapes.
  bufs.idx_cpu = make_cpu_int64_tensor(idx_sizes);
  bufs.idx_cuda = make_cuda_int64_tensor(index_raw, idx_sizes);

  if (bufs.index_numel == 0) {
    return bufs;
  }

  const ScalarType idx_dt = index_raw.dtype();
  if (index_raw.device().type != kDLCUDA ||
      index_raw.device().index != static_cast<int>(dev_index)) {
    throw std::invalid_argument(
        is_read ?
        "index: advanced index tensor must be on the same CUDA device as self" :
        "advanced_index_put_cuda: index tensor must be a CUDA tensor on the same device as src");
  }

  TensorImpl index_contig = index_raw;
  if (!index_contig.is_contiguous()) {
    index_contig = vbt::core::clone_cuda(index_contig);
  }

  auto* idx_cpu_data = static_cast<std::int64_t*>(bufs.idx_cpu.data());

  const char* prefix = is_read ? "index: " : "index_put_: ";

  if (idx_dt == ScalarType::Int64) {
    const std::size_t nbytes_idx =
        static_cast<std::size_t>(bufs.index_numel) * sizeof(std::int64_t);
    cudaError_t st = cudaMemcpyAsync(
        idx_cpu_data, index_contig.data(), nbytes_idx,
        cudaMemcpyDeviceToHost,
        reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string(prefix) +
                      errors::kErrCudaAdvCopyD2HFailed + ": ";
      m += (msg ? msg : "");
      throw std::runtime_error(m);
    }
    st = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string(prefix) +
                      errors::kErrCudaAdvSyncFailed + ": ";
      m += (msg ? msg : "");
      throw std::runtime_error(m);
    }
  } else if (idx_dt == ScalarType::Int32 && allow_int32) {
    const std::size_t nbytes_src =
        static_cast<std::size_t>(bufs.index_numel) * sizeof(std::int32_t);
    std::vector<std::int32_t> host(
        static_cast<std::size_t>(bufs.index_numel));
    cudaError_t st = cudaMemcpyAsync(
        host.data(), index_contig.data(), nbytes_src,
        cudaMemcpyDeviceToHost,
        reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string(prefix) +
                      errors::kErrCudaAdvCopyD2HFailed + ": ";
      m += (msg ? msg : "");
      throw std::runtime_error(m);
    }
    st = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string(prefix) +
                      errors::kErrCudaAdvSyncFailed + ": ";
      m += (msg ? msg : "");
      throw std::runtime_error(m);
    }
    for (std::int64_t i = 0; i < bufs.index_numel; ++i) {
      idx_cpu_data[i] = static_cast<std::int64_t>(
          host[static_cast<std::size_t>(i)]);
    }
  } else {
    if (is_read) {
      throw std::invalid_argument(
          "index: CUDA advanced indexing only supports Int64 index tensors");
    } else {
      throw std::invalid_argument(
          "advanced_index_put_cuda: index tensor must be int32 or int64 on CUDA");
    }
  }

  // Host-side bounds check and negative index normalization.
  for (std::int64_t i = 0; i < bufs.index_numel; ++i) {
    std::int64_t v = idx_cpu_data[i];
    if (v < -D || v >= D) {
      throw std::out_of_range(
          std::string(errors::kErrIndexOutOfRange) + " " +
          std::to_string(D));
    }
    if (v < 0) {
      v += D;
    }
    idx_cpu_data[i] = v;
  }

  // Copy normalized indices back to CUDA.
  const std::size_t nbytes_idx =
      static_cast<std::size_t>(bufs.index_numel) * sizeof(std::int64_t);
  cudaError_t st = cudaMemcpyAsync(
      bufs.idx_cuda.data(), bufs.idx_cpu.data(), nbytes_idx,
      cudaMemcpyHostToDevice,
      reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(prefix) +
                    errors::kErrCudaAdvCopyH2DFailed + ": ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  return bufs;
#endif
}

#if VBT_WITH_CUDA
// Device-side bounds check and normalization. This mirrors the host
// semantics in make_normalized_index_buffers (OOB detection via
// kErrIndexOutOfRange and negative-index normalization) but performs
// the work on-device.

template <typename IndexScalar>
__global__ void advanced_index_bounds_normalize_kernel(
    const IndexScalar* __restrict__ idx_in,
    std::int64_t                    D,
    std::int64_t                    N,
    std::int64_t* __restrict__      idx_out,
    int* __restrict__               error_flag) {
  for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N;
       i += blockDim.x * gridDim.x) {
    const std::int64_t v_raw = static_cast<std::int64_t>(idx_in[i]);
    if (v_raw < -D || v_raw >= D) {
      // Any OOB index sets the error flag; host will throw.
      atomicExch(error_flag, 1);
      continue;
    }
    std::int64_t v = v_raw;
    if (v < 0) {
      v += D;
    }
    idx_out[i] = v;
  }
}

// Pack a potentially strided 1D Bool mask into a contiguous {0,1} byte buffer.
// This is used by the legacy CUDA bool-mask indexing path to avoid incorrect
// reads from non-contiguous mask views.
__global__ void pack_strided_bool_mask_to_contig_kernel(
    const std::uint8_t* __restrict__ in,
    std::int64_t                    stride_elems,
    std::uint8_t* __restrict__      out,
    std::int64_t                    N) {
  for (std::int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < N;
       i += blockDim.x * gridDim.x) {
    const std::uint8_t v = in[i * stride_elems];
    out[i] = static_cast<std::uint8_t>(v != 0);
  }
}

static NormalizedIndexBuffersDevice make_normalized_index_buffers_device(
    const TensorImpl&                index_raw,
    const std::vector<std::int64_t>& idx_sizes,
    std::int64_t                     D,
    bool                             is_read,
    bool                             allow_int32,
    DeviceIndex                      dev_index,
    Stream                           stream) {
  NormalizedIndexBuffersDevice bufs{};
  bufs.D = D;
  bufs.index_numel = safe_numel_from_sizes_cuda(idx_sizes);

  // Allocate logical CUDA buffer up front so zero-numel still has
  // correct shape.
  bufs.idx_cuda = make_cuda_int64_tensor(index_raw, idx_sizes);

  if (bufs.index_numel == 0) {
    return bufs;
  }

  const auto dev = index_raw.device();
  const ScalarType idx_dt = index_raw.dtype();
  const char* prefix = is_read ? "index: " : "index_put_: ";

  if (dev.type != kDLCUDA ||
      dev.index != static_cast<int>(dev_index)) {
    throw std::invalid_argument(
        is_read ?
        "index: advanced index tensor must be on the same CUDA device as self" :
        "advanced_index_put_cuda: index tensor must be a CUDA tensor on the same device as src");
  }

  TensorImpl index_contig = index_raw;
  if (!index_contig.is_contiguous()) {
    index_contig = vbt::core::clone_cuda(index_contig);
  }

  auto* idx_out = static_cast<std::int64_t*>(bufs.idx_cuda.data());

  int* d_error = nullptr;
  cudaError_t st_alloc = cudaMalloc(&d_error, sizeof(int));
  if (st_alloc != cudaSuccess) {
    const char* msg = cudaGetErrorString(st_alloc);
    std::string m = std::string(prefix) +
                    "CUDA advanced indexing bounds kernel alloc failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  struct DeviceIntGuard {
    int* ptr;
    ~DeviceIntGuard() {
      if (ptr) {
        (void)cudaFree(ptr);
      }
    }
  } guard{d_error};

  cudaError_t st = cudaMemsetAsync(
      d_error, 0, sizeof(int),
      reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(prefix) +
                    errors::kErrCudaAdvCopyH2DFailed + ": ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  const std::int64_t N = bufs.index_numel;
  const int threads = 256;
  Grid1DConfig cfg = make_1d_grid(N, threads, dev_index);
  dim3 block_dim = cfg.block_dim;
  dim3 grid_dim  = cfg.grid_dim;

  (void)cudaGetLastError();

  if (idx_dt == ScalarType::Int64) {
    advanced_index_bounds_normalize_kernel<std::int64_t><<<
        grid_dim, block_dim, 0,
        reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<const std::int64_t*>(index_contig.data()),
            D,
            N,
            idx_out,
            d_error);
  } else if (idx_dt == ScalarType::Int32 && allow_int32) {
    advanced_index_bounds_normalize_kernel<std::int32_t><<<
        grid_dim, block_dim, 0,
        reinterpret_cast<cudaStream_t>(stream.handle())>>>(
            static_cast<const std::int32_t*>(index_contig.data()),
            D,
            N,
            idx_out,
            d_error);
  } else {
    if (is_read) {
      throw std::invalid_argument(
          "index: CUDA advanced indexing only supports Int64 index tensors");
    } else {
      throw std::invalid_argument(
          "advanced_index_put_cuda: index tensor must be int32 or int64 on CUDA");
    }
  }

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = std::string(prefix) +
                    "CUDA advanced indexing bounds kernel launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  int h_error = 0;
  st = cudaMemcpyAsync(
      &h_error, d_error, sizeof(int),
      cudaMemcpyDeviceToHost,
      reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(prefix) +
                    errors::kErrCudaAdvCopyD2HFailed + ": ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  st = cudaStreamSynchronize(
      reinterpret_cast<cudaStream_t>(stream.handle()));
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(prefix) +
                    errors::kErrCudaAdvSyncFailed + ": ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  if (h_error != 0) {
    throw std::out_of_range(
        std::string(errors::kErrIndexOutOfRange) + " " +
        std::to_string(D));
  }

  return bufs;
}
#endif  // VBT_WITH_CUDA

static NormalizedIndexBuffersAny normalize_indices_cuda(
    const TensorImpl&                index_raw,
    const std::vector<std::int64_t>& idx_sizes,
    std::int64_t                     D,
    bool                             is_read,
    bool                             allow_int32,
    DeviceIndex                      dev_index,
    Stream                           stream) {
  const std::int64_t index_numel =
      safe_numel_from_sizes_cuda(idx_sizes);
  if (index_numel == 0) {
    auto host = make_normalized_index_buffers(
        index_raw, idx_sizes, D,
        is_read, allow_int32,
        dev_index, stream);
    NormalizedIndexBuffersAny any{};
    any.idx_cuda = std::move(host.idx_cuda);
    any.D = host.D;
    any.index_numel = host.index_numel;
    return any;
  }

  const CudaBoundsMode mode = get_effective_cuda_bounds_mode();
  switch (mode) {
    case CudaBoundsMode::LegacyHost: {
      auto host = make_normalized_index_buffers(
          index_raw, idx_sizes, D,
          is_read, allow_int32,
          dev_index, stream);
      NormalizedIndexBuffersAny any{};
      any.idx_cuda = std::move(host.idx_cuda);
      any.D = host.D;
      any.index_numel = host.index_numel;
      return any;
    }
    case CudaBoundsMode::DeviceNormalized: {
#if VBT_WITH_CUDA
      auto dev = make_normalized_index_buffers_device(
          index_raw, idx_sizes, D,
          is_read, allow_int32,
          dev_index, stream);
      NormalizedIndexBuffersAny any{};
      any.idx_cuda = std::move(dev.idx_cuda);
      any.D = dev.D;
      any.index_numel = dev.index_numel;
      return any;
#else
      (void)is_read;
      (void)allow_int32;
      (void)dev_index;
      (void)stream;
      throw std::runtime_error(
          "advanced_index_cuda: built without CUDA support");
#endif
    }
  }
  // Unreachable, but some compilers warn without a default.
  throw std::logic_error("normalize_indices_cuda: unknown CudaBoundsMode");
}

static TensorImpl normalize_index_tensor_cuda_to_int64_read(
    const TensorImpl&                index_raw,
    const std::vector<std::int64_t>& idx_sizes,
    std::int64_t                     D,
    DeviceIndex                      dev_index,
    Stream                           stream) {
  const bool allow_int32 = cuda_extended_index_dtypes_enabled();
  auto any = normalize_indices_cuda(
      index_raw, idx_sizes, D,
      /*is_read=*/true,
      /*allow_int32=*/allow_int32,
      dev_index, stream);
  return any.idx_cuda;
}

static bool is_dense_contiguous(
    const std::vector<std::int64_t>& sizes,
    const std::vector<std::int64_t>& strides) {
  if (sizes.size() != strides.size()) {
    return false;
  }
  if (sizes.empty()) {
    return true;
  }

  std::int64_t expected = 1;
  for (std::size_t i = sizes.size(); i-- > 0;) {
    const std::int64_t sz = sizes[i];
    const std::int64_t st = strides[i];
    if (sz == 0) {
      continue;
    }
    if (st != expected) {
      return false;
    }
    std::int64_t tmp = 0;
    if (!checked_mul_i64(expected, sz, tmp)) {
      return false;
    }
    expected = tmp;
  }
  return true;
}

template <typename scalar_t>
__global__ void advanced_index_gather_1d_kernel(
    const scalar_t* __restrict__ src,        // base data [B_flat, D]
    const std::int64_t* __restrict__ idx,    // normalized indices [K]
    scalar_t* __restrict__ out,              // result [B_flat, K]
    std::int64_t B_flat,
    std::int64_t D,
    std::int64_t K,
    std::int64_t src_storage_offset) {
  const std::int64_t N = B_flat * K;
  for (std::int64_t li = blockIdx.x * blockDim.x + threadIdx.x;
       li < N;
       li += blockDim.x * gridDim.x) {
    std::int64_t linear = li;
    const std::int64_t k = linear % K;
    linear /= K;
    const std::int64_t b = linear;  // 0 <= b < B_flat

    const std::int64_t idx_val = idx[k];  // 0 <= idx_val < D

    const std::int64_t src_off = src_storage_offset + b * D + idx_val;
    const std::int64_t out_off = b * K + k;

    out[out_off] = src[src_off];
  }
}

template <typename scalar_t>
static void launch_advanced_index_gather_1d_kernel(
    const TensorImpl& src,       // logically [B_flat, D], dense contiguous
    const TensorImpl& idx_cuda,  // normalized Int64 CUDA [K]
    TensorImpl&       out,       // dense CUDA [B_flat, K]
    std::int64_t      B_flat,
    std::int64_t      D,
    std::int64_t      K,
    DeviceIndex       dev_index,
    Stream            stream) {
#if !VBT_WITH_CUDA
  (void)src;
  (void)idx_cuda;
  (void)out;
  (void)B_flat;
  (void)D;
  (void)K;
  (void)dev_index;
  (void)stream;
  throw std::runtime_error(
      "advanced_index_cuda: built without CUDA support");
#else
  const std::int64_t N = B_flat * K;
  if (N == 0) {
    return;
  }

  const int threads = 256;
  Grid1DConfig cfg = make_1d_grid(N, threads, dev_index);
  dim3 block_dim = cfg.block_dim;
  dim3 grid_dim  = cfg.grid_dim;

  (void)cudaGetLastError();

  auto* out_data = static_cast<scalar_t*>(out.data());
  const auto* src_data = static_cast<const scalar_t*>(src.storage()->data());
  const auto* idx_data = static_cast<const std::int64_t*>(idx_cuda.data());

  advanced_index_gather_1d_kernel<scalar_t><<<
      grid_dim, block_dim, 0,
      reinterpret_cast<cudaStream_t>(stream.handle())>>>(
          src_data,
          idx_data,
          out_data,
          B_flat,
          D,
          K,
          src.storage_offset());

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = "index: CUDA advanced indexing kernel launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
#endif
}

static bool can_use_1d_gather_fastpath(const AdvancedIndex& info,
                                       const TensorImpl&    idx_cuda) {
  if (info.indices.size() != 1u) {
    return false;
  }

  const auto& index_shape = info.index_shape;
  if (index_shape.size() != 1u) {
    return false;
  }
  const std::int64_t K = index_shape[0];
  if (K <= 0) {
    return false;
  }

  if (idx_cuda.device().type != kDLCUDA ||
      !(idx_cuda.device() == info.src.device())) {
    return false;
  }
  if (idx_cuda.dtype() != ScalarType::Int64) {
    return false;
  }
  if (idx_cuda.sizes().size() != 1u || idx_cuda.sizes()[0] != K) {
    return false;
  }
  if (!idx_cuda.is_contiguous()) {
    return false;
  }

  const std::int64_t B = info.dims_before;
  if (B < 0) {
    return false;
  }
  if (info.dims_after != 0) {
    return false;
  }

  const auto& result_shape = info.result_shape;
  if (static_cast<std::int64_t>(result_shape.size()) != B + 1) {
    return false;
  }
  for (std::int64_t d = 0; d < B; ++d) {
    if (result_shape[static_cast<std::size_t>(d)] <= 0) {
      return false;
    }
  }
  if (result_shape[static_cast<std::size_t>(B)] != K) {
    return false;
  }

  if (info.indexed_sizes.size() != 1u ||
      info.indexed_strides_elems.size() != 1u) {
    return false;
  }
  const std::int64_t D = info.indexed_sizes[0];
  const std::int64_t stride_D = info.indexed_strides_elems[0];
  if (D <= 0 || stride_D != 1) {
    return false;
  }

  const auto& src_sizes = info.src.sizes();
  const auto& src_strides = info.src.strides();
  const std::int64_t R = static_cast<std::int64_t>(src_sizes.size());
  if (R != B + 1) {
    return false;
  }

  std::vector<std::int64_t> base_sizes;
  std::vector<std::int64_t> base_strides;
  base_sizes.reserve(static_cast<std::size_t>(B + 1));
  base_strides.reserve(static_cast<std::size_t>(B + 1));
  for (std::int64_t d = 0; d < B; ++d) {
    base_sizes.push_back(src_sizes[static_cast<std::size_t>(d)]);
    base_strides.push_back(src_strides[static_cast<std::size_t>(d)]);
  }
  base_sizes.push_back(D);
  base_strides.push_back(stride_D);

  if (!is_dense_contiguous(base_sizes, base_strides)) {
    return false;
  }

  return true;
}

// Forward declaration of the generic gather kernel.
template <typename scalar_t, typename IndexT>
__global__ void advanced_index_cuda_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ src,
    const std::int64_t* __restrict__ idx,
    AdvancedIndexCudaMeta meta,
    std::int64_t         advanced_stride_elems,
    IndexT               N);

static void launch_generic_gather_from_advanced_index(
    const AdvancedIndex& info,
    const TensorImpl&    idx_cuda,
    TensorImpl&          out,
    std::int64_t         N,
    DeviceIndex          dev_index,
    Stream               stream) {
#if !VBT_WITH_CUDA
  (void)info;
  (void)idx_cuda;
  (void)out;
  (void)N;
  (void)dev_index;
  (void)stream;
  throw std::runtime_error(
      "advanced_index_cuda: built without CUDA support");
#else
  const auto& result_shape = info.result_shape;
  const auto& src_strides  = info.src.strides();

  AdvancedIndexCudaMeta meta{};
  const std::int64_t R = static_cast<std::int64_t>(result_shape.size());
  if (R > kIndexCudaMaxRank) {
    throw std::runtime_error(
        "index: CUDA advanced indexing result rank exceeds internal limit");
  }

  meta.R = static_cast<int32_t>(R);
  meta.index_ndim = static_cast<int32_t>(
      static_cast<std::int64_t>(info.index_shape.size()));
  meta.dims_before = static_cast<int32_t>(info.dims_before);
  meta.storage_offset = info.src.storage_offset();

  for (int d = 0; d < meta.R; ++d) {
    meta.result_sizes[d] =
        result_shape[static_cast<std::size_t>(d)];
  }
  for (int d = 0; d < meta.dims_before; ++d) {
    meta.base_strides[d] =
        src_strides[static_cast<std::size_t>(d)];
  }
  for (int j = 0; j < meta.index_ndim; ++j) {
    meta.index_sizes[j] =
        info.index_shape[static_cast<std::size_t>(j)];
  }

  const std::int64_t advanced_stride_elems =
      info.indexed_strides_elems[0];

  const bool use32 = should_use_32bit_for_advanced_index_cuda(info, N);

  const int threads = 256;
  Grid1DConfig cfg = make_1d_grid(N, threads, dev_index);
  dim3 block_dim = cfg.block_dim;
  dim3 grid_dim  = cfg.grid_dim;

  (void)cudaGetLastError();

  const ScalarType dt = info.src.dtype();
  const bool extended = cuda_extended_index_dtypes_enabled();
  if (dt == ScalarType::Float32) {
    if (use32) {
      advanced_index_cuda_kernel<float, std::int32_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(out.data()),
              static_cast<const float*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_cuda_kernel<float, std::int64_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(out.data()),
              static_cast<const float*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int64_t>(N));
    }
  } else if (dt == ScalarType::Int64) {
    if (use32) {
      advanced_index_cuda_kernel<long long, std::int32_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<long long*>(out.data()),
              static_cast<const long long*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_cuda_kernel<long long, std::int64_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<long long*>(out.data()),
              static_cast<const long long*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int64_t>(N));
    }
  } else if (dt == ScalarType::Float16 && extended) {
    if (use32) {
      advanced_index_cuda_kernel<__half, std::int32_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<__half*>(out.data()),
              static_cast<const __half*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_cuda_kernel<__half, std::int64_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<__half*>(out.data()),
              static_cast<const __half*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int64_t>(N));
    }
#if VBT_INDEX_CUDA_HAS_BF16
  } else if (dt == ScalarType::BFloat16 && extended) {
    if (use32) {
      advanced_index_cuda_kernel<__nv_bfloat16, std::int32_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<__nv_bfloat16*>(out.data()),
              static_cast<const __nv_bfloat16*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_cuda_kernel<__nv_bfloat16, std::int64_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<__nv_bfloat16*>(out.data()),
              static_cast<const __nv_bfloat16*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int64_t>(N));
    }
#endif
  } else if (dt == ScalarType::Int32 && extended) {
    if (use32) {
      advanced_index_cuda_kernel<std::int32_t, std::int32_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<std::int32_t*>(out.data()),
              static_cast<const std::int32_t*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_cuda_kernel<std::int32_t, std::int64_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<std::int32_t*>(out.data()),
              static_cast<const std::int32_t*>(info.src.storage()->data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int64_t>(N));
    }
  } else {
    throw std::invalid_argument(
        "index: CUDA advanced indexing is only implemented for float32 and int64 tensors");
  }

  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = "index: CUDA advanced indexing kernel launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
#endif
}

// GPU gather kernel for advanced indexing on CUDA. We restrict to the
// of `base` after an optional basic prefix. The result shape is
//   result_shape = base.sizes()[0:dims_before] + index_sizes.

template <typename scalar_t, typename IndexT>
__global__ void advanced_index_cuda_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ src,
    const std::int64_t* __restrict__ idx,
    AdvancedIndexCudaMeta meta,
    std::int64_t advanced_stride_elems,
    IndexT N) {
  IndexT i = blockIdx.x * blockDim.x + threadIdx.x;
  const IndexT stride = blockDim.x * gridDim.x;

  while (i < N) {
    const std::int64_t linear = static_cast<std::int64_t>(i);

    // Decode linear index into coordinates in result_shape.
    std::int64_t coords[kIndexCudaMaxRank];
    std::int64_t tmp = linear;
    for (int d = meta.R - 1; d >= 0; --d) {
      const std::int64_t sz = meta.result_sizes[d];
      std::int64_t c = 0;
      if (sz > 0) {
        c = tmp % sz;
        tmp /= sz;
      }
      coords[d] = c;
    }

    // Base offset within src coming from prefix dims.
    std::int64_t src_off_elems = 0;
    for (int d = 0; d < meta.dims_before; ++d) {
      const std::int64_t stride_d = meta.base_strides[d];
      const std::int64_t c = coords[d];
      src_off_elems += c * stride_d;
    }

    // Linear index into the index tensor from advanced dims.
    std::int64_t idx_linear = 0;
    for (int j = 0; j < meta.index_ndim; ++j) {
      const std::int64_t sz = meta.index_sizes[j];
      const int d = meta.dims_before + j;
      const std::int64_t c = coords[d];
      idx_linear = idx_linear * sz + c;
    }

    const std::int64_t idx_val = idx[idx_linear];  // already normalized
    const std::int64_t adv_off = idx_val * advanced_stride_elems;

    const std::int64_t src_index = meta.storage_offset + src_off_elems + adv_off;

    out[linear] = src[src_index];

    i += stride;
  }
}

// CUDA scatter kernel for advanced index_put_ on CUDA.
template <typename scalar_t, typename IndexT, bool Accumulate>
__global__ void advanced_index_put_cuda_kernel(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    const std::int64_t* __restrict__ idx,
    AdvancedIndexPutCudaMeta meta,
    IndexT total_elements) {
  IndexT linear = blockIdx.x * blockDim.x + threadIdx.x;
  const IndexT stride = blockDim.x * gridDim.x;

  while (linear < total_elements) {
    // Decode linear index into multi-dimensional coordinates.
    std::int64_t coords[kIndexCudaMaxRank];
    IndexT tmp = linear;
    for (int d = meta.ndim - 1; d >= 0; --d) {
      const std::int64_t size_d = meta.result_sizes[d];
      std::int64_t c = 0;
      if (size_d > 0) {
        const IndexT sd = static_cast<IndexT>(size_d);
        c = static_cast<std::int64_t>(tmp % sd);
        tmp /= sd;
      }
      coords[d] = c;
    }

    // Base offset within dst coming from prefix/suffix dims.
    std::int64_t dst_off_elems = 0;
    for (int d = 0; d < meta.ndim; ++d) {
      const std::int64_t stride_d = meta.dst_strides[d];
      if (stride_d == 0) {
        continue;
      }
      const std::int64_t c = coords[d];
      dst_off_elems += c * stride_d;
    }

    // Advanced offset from index tensor.
    std::int64_t idx_linear = 0;
    for (int j = 0; j < meta.index_ndim; ++j) {
      const std::int64_t sz = meta.index_shape[j];
      const int d = meta.dims_before + j;
      const std::int64_t c = coords[d];
      idx_linear = idx_linear * sz + c;
    }
    const std::int64_t idx_val = idx[idx_linear];
    const std::int64_t adv_off_elems = idx_val * meta.indexed_stride_elems;

    const std::int64_t dst_off_total =
        meta.storage_offset_elems + dst_off_elems + adv_off_elems;

    // Offset for value_b based on its strides.
    std::int64_t src_off_elems = 0;
    for (int d = 0; d < meta.ndim; ++d) {
      const std::int64_t stride_d = meta.value_strides[d];
      if (stride_d == 0) {
        continue;
      }
      const std::int64_t c = coords[d];
      src_off_elems += c * stride_d;
    }

    const scalar_t v = src[src_off_elems];

    if constexpr (Accumulate) {
      // accumulate == true, dtype == float32 only (host-enforced).
      atomicAdd(&dst[dst_off_total], v);
    } else {
      dst[dst_off_total] = v;
    }

    linear += stride;
  }
}

// CUDA helper to materialize (key,value) pairs for index_put_(accumulate=true)
// so we can sort+reduce-by-key without atomics.
//
// key: element offset into `dst` (same convention as advanced_index_put_cuda_kernel)
// val: the update value for that offset.
template <typename scalar_t, typename IndexT>
__global__ void advanced_index_put_make_kv_cuda_kernel(
    std::uint64_t* __restrict__ out_keys,
    scalar_t* __restrict__ out_vals,
    const scalar_t* __restrict__ src_vals,
    const std::int64_t* __restrict__ idx,
    AdvancedIndexPutCudaMeta meta,
    IndexT total_elements) {
  IndexT linear = blockIdx.x * blockDim.x + threadIdx.x;
  const IndexT stride = blockDim.x * gridDim.x;

  while (linear < total_elements) {
    // Decode linear index into multi-dimensional coordinates.
    std::int64_t coords[kIndexCudaMaxRank];
    IndexT tmp = linear;
    for (int d = meta.ndim - 1; d >= 0; --d) {
      const std::int64_t size_d = meta.result_sizes[d];
      std::int64_t c = 0;
      if (size_d > 0) {
        const IndexT sd = static_cast<IndexT>(size_d);
        c = static_cast<std::int64_t>(tmp % sd);
        tmp /= sd;
      }
      coords[d] = c;
    }

    // Base offset within dst coming from prefix/suffix dims.
    std::int64_t dst_off_elems = 0;
    for (int d = 0; d < meta.ndim; ++d) {
      const std::int64_t stride_d = meta.dst_strides[d];
      if (stride_d == 0) {
        continue;
      }
      const std::int64_t c = coords[d];
      dst_off_elems += c * stride_d;
    }

    // Advanced offset from index tensor.
    std::int64_t idx_linear = 0;
    for (int j = 0; j < meta.index_ndim; ++j) {
      const std::int64_t sz = meta.index_shape[j];
      const int d = meta.dims_before + j;
      const std::int64_t c = coords[d];
      idx_linear = idx_linear * sz + c;
    }
    const std::int64_t idx_val = idx[idx_linear];
    const std::int64_t adv_off_elems = idx_val * meta.indexed_stride_elems;

    const std::int64_t dst_off_total =
        meta.storage_offset_elems + dst_off_elems + adv_off_elems;

    // Offset for value_b based on its strides.
    std::int64_t src_off_elems = 0;
    for (int d = 0; d < meta.ndim; ++d) {
      const std::int64_t stride_d = meta.value_strides[d];
      if (stride_d == 0) {
        continue;
      }
      const std::int64_t c = coords[d];
      src_off_elems += c * stride_d;
    }

    out_keys[linear] = static_cast<std::uint64_t>(dst_off_total);
    out_vals[linear] = src_vals[src_off_elems];

    linear += stride;
  }
}

template <typename scalar_t, typename IndexT>
__global__ void advanced_index_put_scatter_reduced_kv_cuda_kernel(
    scalar_t* __restrict__ dst,
    const std::uint64_t* __restrict__ unique_keys,
    const scalar_t* __restrict__ reduced_vals,
    const int* __restrict__ d_num_unique,
    IndexT total_items) {
  IndexT linear = blockIdx.x * blockDim.x + threadIdx.x;
  const IndexT stride = blockDim.x * gridDim.x;

  int n_unique_i = 0;
  if (d_num_unique) {
    n_unique_i = *d_num_unique;
    if (n_unique_i < 0) {
      n_unique_i = 0;
    }
  }
  const IndexT n_unique = static_cast<IndexT>(n_unique_i);

  while (linear < total_items) {
    if (linear < n_unique) {
      const std::uint64_t k = unique_keys[linear];
      const scalar_t v = reduced_vals[linear];
      dst[k] += v;
    }

    linear += stride;
  }
}

} // namespace

TensorImpl index_cuda(const TensorImpl& self, const IndexSpec& spec_raw) {
#if !VBT_WITH_CUDA
  (void)self;
  (void)spec_raw;
  throw std::runtime_error("index_cuda: built without CUDA support");
#else
  if (self.device().type != kDLCUDA) {
    throw std::invalid_argument(
        "index: CUDA advanced indexing requires a CUDA tensor");
  }

  const auto& self_sizes = self.sizes();
  const std::int64_t self_dim =
      static_cast<std::int64_t>(self_sizes.size());
  if (self_dim == 0) {
    throw std::invalid_argument(
        "index: advanced indexing is not supported for 0-d tensors on CUDA");
  }

  // Normalize ellipsis and validate dimension counts. This mirrors the
  // CPU path and ensures we have a stable view of prefix/advanced/suffix.
  IndexSpec spec = expand_ellipsis_and_validate(spec_raw, self_dim);

  if (!has_any_advanced(spec)) {
    throw std::logic_error(
        "index_cuda called with basic-only IndexSpec");
  }

  // Classify the first advanced index and count how many we have.
  int first_adv = -1;
  int adv_count = 0;
  const std::int64_t n_items =
      static_cast<std::int64_t>(spec.items.size());
  for (std::int64_t i = 0; i < n_items; ++i) {
    const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
    if (is_advanced_kind(kind)) {
      if (first_adv < 0) {
        first_adv = static_cast<int>(i);
      }
      ++adv_count;
    }
  }

  if (first_adv < 0) {
    throw std::logic_error(
        "index_cuda: advanced spec misclassified (no advanced index)");
  }

  if (adv_count > 1) {
    throw std::invalid_argument(
        "index: advanced indexing with multiple tensor/bool indices is not supported on CUDA");
  }

  const TensorIndex& adv_it =
      spec.items[static_cast<std::size_t>(first_adv)];

  if (adv_it.kind != IndexKind::Tensor || !adv_it.tensor.storage().get()) {
    throw std::invalid_argument(
        "index: advanced indexing pattern is not supported on CUDA");
  }

  TensorImpl index_tensor = adv_it.tensor;

  // Restrict index tensors to Int64 on the same CUDA device as self.
  // destination CUDA device. Bool-mask indices remain CUDA-only.
  if (index_tensor.device().type == kDLCPU) {
    const ScalarType dt_cpu = index_tensor.dtype();
    if (dt_cpu == ScalarType::Bool) {
      throw std::invalid_argument(
          "index: advanced index tensor must be on the same CUDA device as self");
    }

    std::vector<std::int64_t> idx_sizes_cpu(
        index_tensor.sizes().begin(), index_tensor.sizes().end());
    const std::int64_t n_cpu = safe_numel_from_sizes_cuda(idx_sizes_cpu);

    std::size_t nbytes = 0;
    if (dt_cpu == ScalarType::Int64) {
      nbytes = static_cast<std::size_t>(n_cpu) * sizeof(std::int64_t);
    } else if (dt_cpu == ScalarType::Int32) {
      nbytes = static_cast<std::size_t>(n_cpu) * sizeof(std::int32_t);
    } else {
      throw std::invalid_argument(
          "index: CPU index tensor for CUDA must be int32 or int64");
    }

    DeviceIndex dev_index = static_cast<DeviceIndex>(self.device().index);
    DeviceGuard guard(dev_index);
    Stream stream = getCurrentStream(dev_index);

    TensorImpl index_contig = index_tensor;
    if (!index_contig.is_contiguous()) {
      index_contig = vbt::core::clone_cpu(index_contig);
    }

    TensorImpl idx_cuda;
    if (dt_cpu == ScalarType::Int64) {
      idx_cuda = make_cuda_int64_tensor(self, idx_sizes_cpu);
    } else {
      idx_cuda = make_cuda_int32_tensor(self, idx_sizes_cpu);
    }

    if (nbytes > 0) {
      cudaError_t st = cudaMemcpyAsync(
          idx_cuda.data(),
          index_contig.data(),
          nbytes,
          cudaMemcpyHostToDevice,
          reinterpret_cast<cudaStream_t>(stream.handle()));
      if (st != cudaSuccess) {
        const char* msg = cudaGetErrorString(st);
        const char* prefix = "index: ";
        std::string m = std::string(prefix) +
                        errors::kErrCudaAdvCopyH2DFailed + ": ";
        m += (msg ? msg : "");
        throw std::runtime_error(m);
      }
    }

    // Record stream for allocator safety (derived CUDA index tensor).
    vbt::cuda::record_stream(idx_cuda.storage(), stream);

    index_tensor = std::move(idx_cuda);
  }

  if (index_tensor.device().type != kDLCUDA ||
      index_tensor.device().index != self.device().index) {
    throw std::invalid_argument(
        "index: advanced index tensor must be on the same CUDA device as self");
  }

  if (index_tensor.dtype() != ScalarType::Int64) {
    throw std::invalid_argument(
        "index: CUDA advanced indexing only supports Int64 index tensors");
  }

  if (index_tensor.sizes().empty()) {
    throw std::invalid_argument(
        "index: advanced tensor indices must have dim() > 0");
  }

  // Reject any suffix indices after the advanced block on CUDA.
  for (int i = first_adv + 1; i < static_cast<int>(n_items); ++i) {
    const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
    if (is_advanced_kind(kind)) {
      throw std::invalid_argument(
          "index: advanced indexing with multiple tensor/bool indices is not supported on CUDA");
    }
    if (kind == IndexKind::Integer ||
        kind == IndexKind::Slice ||
        kind == IndexKind::None) {
      throw std::invalid_argument(
          "index: advanced indexing with suffix basic indices is not supported on CUDA");
    }
  }

  // Build prefix-only spec and apply basic indexing to obtain the base
  // tensor on which advanced semantics are defined.
  IndexSpec prefix_spec;
  prefix_spec.items.assign(
      spec.items.begin(),
      spec.items.begin() + static_cast<std::ptrdiff_t>(first_adv));

  TensorImpl base = self;
  if (!prefix_spec.items.empty()) {
    base = basic_index(base, prefix_spec);
  }

  const auto& base_sizes = base.sizes();
  const auto& base_strides = base.strides();
  const std::int64_t B =
      static_cast<std::int64_t>(base_sizes.size());
  if (B == 0) {
    throw std::invalid_argument(
        "index: advanced indexing is not supported for 0-d tensors on CUDA");
  }

  const std::int64_t dims_before = B - 1;
  const std::size_t adv_dim =
      static_cast<std::size_t>(B - 1);

  const std::int64_t D = base_sizes[adv_dim];
  if (D < 0) {
    throw std::invalid_argument(
        "index: negative dimension size in CUDA advanced indexing");
  }

  // Index tensor shape and DoS caps.
  std::vector<std::int64_t> idx_sizes(
      index_tensor.sizes().begin(), index_tensor.sizes().end());
  const std::int64_t index_ndim =
      static_cast<std::int64_t>(idx_sizes.size());
  if (index_ndim > kIndexCudaMaxRank) {
    throw std::runtime_error(
        "index: CUDA advanced indexing too many index dims");
  }

  const std::int64_t index_numel =
      safe_numel_from_sizes_cuda(idx_sizes);
  if (index_numel > kAdvIndexMaxIndexNumel) {
    throw std::runtime_error(
        "index: CUDA advanced indexing too many index elements");
  }

  // Result shape = base.sizes() prefix + index shape.
  std::vector<std::int64_t> result_shape;
  result_shape.reserve(static_cast<std::size_t>(dims_before + index_ndim));
  for (std::int64_t d = 0; d < dims_before; ++d) {
    result_shape.push_back(base_sizes[static_cast<std::size_t>(d)]);
  }
  result_shape.insert(result_shape.end(),
                      idx_sizes.begin(), idx_sizes.end());

  const std::int64_t result_numel =
      safe_numel_from_sizes_cuda(result_shape);
  if (result_numel > kAdvIndexMaxResultNumel) {
    throw std::runtime_error(errors::kErrCudaAdvResultTooLarge);
  }

  // Allocate output tensor on CUDA.
  TensorImpl out = make_cuda_dense_out(base, result_shape);
  if (result_numel == 0) {
    // Zero-numel (empty or overflow-sentinel) is a strict no-op: no kernel.
    return out;
  }

  // Ensure the index tensor is contiguous on CUDA.
  if (!index_tensor.is_contiguous()) {
    index_tensor = vbt::core::clone_cuda(index_tensor);
  }

  // Copy indices to CPU for bounds checking and normalization.
  TensorImpl idx_cpu = make_cpu_int64_tensor(idx_sizes);
  const std::size_t nbytes_idx =
      static_cast<std::size_t>(index_numel) * sizeof(std::int64_t);

  DeviceIndex dev_index = static_cast<DeviceIndex>(self.device().index);
  DeviceGuard guard(dev_index);
  Stream stream = getCurrentStream(dev_index);

  // D2H copy of raw Int64 indices.
  {
    cudaError_t st = cudaMemcpyAsync(
        idx_cpu.data(), index_tensor.data(), nbytes_idx,
        cudaMemcpyDeviceToHost,
        reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string("index: ") +
                       errors::kErrCudaAdvCopyD2HFailed + ": ";
      m += (msg ? msg : "");
      throw std::runtime_error(m);
    }
    st = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string("index: ") +
                       errors::kErrCudaAdvSyncFailed + ": ";
      m += (msg ? msg : "");
      throw std::runtime_error(m);
    }
  }

  // Normalize indices in-place on CPU and check bounds.
  auto* idx_data = static_cast<std::int64_t*>(idx_cpu.data());
  for (std::int64_t i = 0; i < index_numel; ++i) {
    std::int64_t v = idx_data[i];
    if (v < -D || v >= D) {
      throw std::out_of_range(
          std::string(errors::kErrIndexOutOfRange) + " " +
          std::to_string(D));
    }
    if (v < 0) {
      v += D;
    }
    idx_data[i] = v;
  }

  // Copy normalized indices back to CUDA into a dedicated Int64 tensor.
  TensorImpl idx_cuda = make_cuda_int64_tensor(base, idx_sizes);
  {
    cudaError_t st = cudaMemcpyAsync(
        idx_cuda.data(), idx_cpu.data(), nbytes_idx,
        cudaMemcpyHostToDevice,
        reinterpret_cast<cudaStream_t>(stream.handle()));
    if (st != cudaSuccess) {
      const char* msg = cudaGetErrorString(st);
      std::string m = std::string("index: ") +
                       errors::kErrCudaAdvCopyH2DFailed + ": ";
      m += (msg ? msg : "");
      throw std::runtime_error(m);
    }
  }

  // Pack metadata for the CUDA kernel.
  AdvancedIndexCudaMeta meta{};
  meta.R = static_cast<int32_t>(result_shape.size());
  meta.index_ndim = static_cast<int32_t>(index_ndim);
  meta.dims_before = static_cast<int32_t>(dims_before);
  meta.storage_offset = base.storage_offset();

  if (meta.R > kIndexCudaMaxRank) {
    throw std::runtime_error(
        "index: CUDA advanced indexing result rank exceeds internal limit");
  }

  for (int d = 0; d < meta.R; ++d) {
    meta.result_sizes[d] =
        result_shape[static_cast<std::size_t>(d)];
  }
  for (int d = 0; d < meta.dims_before; ++d) {
    meta.base_strides[d] =
        base_strides[static_cast<std::size_t>(d)];
  }
  for (int j = 0; j < meta.index_ndim; ++j) {
    meta.index_sizes[j] = idx_sizes[static_cast<std::size_t>(j)];
  }

  const std::int64_t advanced_stride_elems =
      base_strides[adv_dim];

  const std::int64_t N = result_numel;
  const bool domain_hint =
      compute_use32bit_indexing_for_sizes_cuda(result_shape);

  // Launch configuration.
  const int threads = 256;
  Grid1DConfig cfg = make_1d_grid(N, threads, dev_index);
  dim3 block_dim = cfg.block_dim;
  dim3 grid_dim  = cfg.grid_dim;

  const ScalarType dt = base.dtype();
  const bool use32 = advanced_index_32bit_enabled() &&
                     domain_hint &&
                     (N <= static_cast<std::int64_t>(
                              std::numeric_limits<std::int32_t>::max()));

  // Clear any sticky error and launch.
  (void)cudaGetLastError();

  if (dt == ScalarType::Float32) {
    if (use32) {
      advanced_index_cuda_kernel<float, std::int32_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(out.data()),
              static_cast<const float*>(base.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_cuda_kernel<float, std::int64_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(out.data()),
              static_cast<const float*>(base.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int64_t>(N));
    }
  } else if (dt == ScalarType::Int64) {
    if (use32) {
      advanced_index_cuda_kernel<long long, std::int32_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<long long*>(out.data()),
              static_cast<const long long*>(base.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_cuda_kernel<long long, std::int64_t><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<long long*>(out.data()),
              static_cast<const long long*>(base.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              advanced_stride_elems,
              static_cast<std::int64_t>(N));
    }
  } else {
    throw std::invalid_argument(
        "index: CUDA advanced indexing is only implemented for float32 and int64 tensors");
  }

  // Surface any kernel launch failure.
  cudaError_t lc = cudaGetLastError();
  if (lc != cudaSuccess) {
    const char* msg = cudaGetErrorString(lc);
    std::string m = "index: CUDA advanced indexing kernel launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  return out;
#endif
}

namespace cuda_impl {

AdvancedIndexCudaResult make_advanced_index_cuda(
    const TensorImpl& self,
    const IndexSpec&  spec_raw,
    AdvancedIndexCudaMode mode) {
#if !VBT_WITH_CUDA
  (void)self;
  (void)spec_raw;
  (void)mode;
  throw std::runtime_error(
      "index: CUDA advanced indexing is not available when built without CUDA");
#else
  if (self.device().type != kDLCUDA) {
    throw std::invalid_argument(
        "index: CUDA advanced indexing requires a CUDA tensor");
  }

  const auto& self_sizes = self.sizes();
  const std::int64_t self_dim =
      static_cast<std::int64_t>(self_sizes.size());
  if (self_dim == 0) {
    throw std::invalid_argument(
        "index: advanced indexing is not supported for 0-d tensors on CUDA");
  }

  // Normalize ellipsis and validate dimension counts.
  IndexSpec spec = expand_ellipsis_and_validate(spec_raw, self_dim);

  if (!has_any_advanced(spec)) {
    throw std::logic_error(
        "make_advanced_index_cuda called with basic-only IndexSpec");
  }

  // Classify the first advanced index and count how many we have.
  int first_adv = -1;
  int adv_count = 0;
  const std::int64_t n_items =
      static_cast<std::int64_t>(spec.items.size());
  for (std::int64_t i = 0; i < n_items; ++i) {
    const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
    if (is_advanced_kind(kind)) {
      if (first_adv < 0) {
        first_adv = static_cast<int>(i);
      }
      ++adv_count;
    }
  }

  if (first_adv < 0) {
    throw std::logic_error(
        "make_advanced_index_cuda: advanced spec misclassified (no advanced index)");
  }

  if (adv_count > 1) {
    throw std::invalid_argument(
        "index: advanced indexing with multiple tensor/bool indices is not supported on CUDA");
  }

  const TensorIndex& adv_it =
      spec.items[static_cast<std::size_t>(first_adv)];

  if (adv_it.kind != IndexKind::Tensor || !adv_it.tensor.storage().get()) {
    throw std::invalid_argument(
        "index: advanced indexing pattern is not supported on CUDA");
  }

  TensorImpl index_tensor = adv_it.tensor;

  // destination CUDA device. Bool-mask indices remain CUDA-only.
  if (index_tensor.device().type == kDLCPU) {
    const ScalarType dt_cpu = index_tensor.dtype();
    if (dt_cpu == ScalarType::Bool) {
      throw std::invalid_argument(
          "index: advanced index tensor must be on the same CUDA device as self");
    }

    std::vector<std::int64_t> idx_sizes_cpu(
        index_tensor.sizes().begin(), index_tensor.sizes().end());
    const std::int64_t n_cpu = safe_numel_from_sizes_cuda(idx_sizes_cpu);

    std::size_t nbytes = 0;
    if (dt_cpu == ScalarType::Int64) {
      nbytes = static_cast<std::size_t>(n_cpu) * sizeof(std::int64_t);
    } else if (dt_cpu == ScalarType::Int32) {
      nbytes = static_cast<std::size_t>(n_cpu) * sizeof(std::int32_t);
    } else {
      throw std::invalid_argument(
          "index: CPU index tensor for CUDA must be int32 or int64");
    }

    DeviceIndex dev_index = static_cast<DeviceIndex>(self.device().index);
    DeviceGuard guard(dev_index);
    Stream stream = getCurrentStream(dev_index);

    TensorImpl index_contig = index_tensor;
    if (!index_contig.is_contiguous()) {
      index_contig = vbt::core::clone_cpu(index_contig);
    }

    TensorImpl idx_cuda;
    if (dt_cpu == ScalarType::Int64) {
      idx_cuda = make_cuda_int64_tensor(self, idx_sizes_cpu);
    } else {
      idx_cuda = make_cuda_int32_tensor(self, idx_sizes_cpu);
    }

    if (nbytes > 0) {
      cudaError_t st = cudaMemcpyAsync(
          idx_cuda.data(),
          index_contig.data(),
          nbytes,
          cudaMemcpyHostToDevice,
          reinterpret_cast<cudaStream_t>(stream.handle()));
      if (st != cudaSuccess) {
        const char* msg = cudaGetErrorString(st);
        const char* prefix =
            (mode == AdvancedIndexCudaMode::Read) ? "index: " : "index_put_: ";
        std::string m = std::string(prefix) +
                        errors::kErrCudaAdvCopyH2DFailed + ": ";
        m += (msg ? msg : "");
        throw std::runtime_error(m);
      }
    }

    // Record stream for allocator safety (derived CUDA index tensor).
    vbt::cuda::record_stream(idx_cuda.storage(), stream);

    index_tensor = std::move(idx_cuda);
  }

  if (index_tensor.device().type != kDLCUDA ||
      index_tensor.device().index != self.device().index) {
    throw std::invalid_argument(
        "index: advanced index tensor must be on the same CUDA device as self");
  }

  const ScalarType idx_dt = index_tensor.dtype();
  const bool is_bool_index = (idx_dt == ScalarType::Bool);
  const bool extended_dtypes = cuda_extended_index_dtypes_enabled();

  // Bool index tensors are treated as policy-unsupported unless the
  // kErrCudaAdvBoolMaskUnsupported surface.
  if (is_bool_index && !cuda_bool_mask_indices_enabled()) {
    throw std::invalid_argument(
        std::string("index: ") +
        errors::kErrCudaAdvBoolMaskUnsupported);
  }

  if (mode == AdvancedIndexCudaMode::Read) {
    if (!is_bool_index) {
      // Integer-based read path.
      if (!(idx_dt == ScalarType::Int64 ||
            (idx_dt == ScalarType::Int32 && extended_dtypes))) {
        throw std::invalid_argument(
            "index: CUDA advanced indexing only supports Int64 index tensors");
      }
    }
    // Bool indices are validated later once we know the base dim and
    // pattern; at this point we only enforced the B flag above.
  } else {
    // Write path: integer indices are always restricted to Int32/Int64;
    // Bool indices (when enabled) are validated later.
    if (!is_bool_index &&
        !(idx_dt == ScalarType::Int32 || idx_dt == ScalarType::Int64)) {
      throw std::invalid_argument(
          "index: CUDA advanced indexing only supports Int32 and Int64 index tensors");
    }
  }

  if (index_tensor.sizes().empty()) {
    throw std::invalid_argument(
        "index: advanced tensor indices must have dim() > 0");
  }

  // Reject any suffix indices after the advanced block on CUDA.
  for (int i = first_adv + 1; i < static_cast<int>(n_items); ++i) {
    const auto kind = spec.items[static_cast<std::size_t>(i)].kind;
    if (is_advanced_kind(kind)) {
      throw std::invalid_argument(
          "index: advanced indexing with multiple tensor/bool indices is not supported on CUDA");
    }
    if (kind == IndexKind::Integer ||
        kind == IndexKind::Slice ||
        kind == IndexKind::None) {
      throw std::invalid_argument(
          "index: advanced indexing with suffix basic indices is not supported on CUDA");
    }
  }

  // Build prefix-only spec and apply basic indexing to obtain the base
  // tensor on which advanced semantics are defined.
  IndexSpec prefix_spec;
  prefix_spec.items.assign(
      spec.items.begin(),
      spec.items.begin() + static_cast<std::ptrdiff_t>(first_adv));

  TensorImpl base = self;
  if (!prefix_spec.items.empty()) {
    base = basic_index(base, prefix_spec);
  }

  const auto& base_sizes = base.sizes();
  const auto& base_strides = base.strides();
  const std::int64_t B =
      static_cast<std::int64_t>(base_sizes.size());
  if (B == 0) {
    throw std::invalid_argument(
        "index: advanced indexing is not supported for 0-d tensors on CUDA");
  }

  const std::int64_t dims_before = B - 1;
  const std::size_t adv_dim =
      static_cast<std::size_t>(B - 1);

  const std::int64_t D = base_sizes[adv_dim];
  if (D < 0) {
    throw std::invalid_argument(
        "index: negative dimension size in CUDA advanced indexing");
  }

  // Index tensor shape and DoS caps.
  std::vector<std::int64_t> idx_sizes;
  std::int64_t index_ndim = 0;
  std::int64_t index_numel = 0;
  TensorImpl   index_for_info;

  if (is_bool_index) {
    // Enforce narrow Bool-mask dtype allowlist: only float32/int64
    // data are supported for CUDA Bool masks, independent of the D
    // gate.
    const ScalarType data_dt = base.dtype();
    if (!(data_dt == ScalarType::Float32 ||
          data_dt == ScalarType::Int64)) {
      throw std::invalid_argument(
          std::string("index: ") +
          errors::kErrCudaAdvBoolMaskUnsupported);
    }

    // last dimension after an optional basic prefix).
    const auto& mask_sizes_vec = index_tensor.sizes();
    if (mask_sizes_vec.size() != 1u) {
      throw std::invalid_argument(
          std::string("index: ") +
          errors::kErrCudaAdvBoolMaskUnsupported);
    }

    const std::int64_t mask_len = mask_sizes_vec[0];
    if (mask_len != D) {
      throw std::invalid_argument(
          std::string("index: ") +
          errors::kErrCudaAdvBoolMaskUnsupported);
    }

    if (mask_len > kAdvIndexMaxMaskNumel) {
      throw std::runtime_error(errors::kErrAdvIndexTooLarge);
    }

    DeviceIndex dev_index = static_cast<DeviceIndex>(self.device().index);
    DeviceGuard guard(dev_index);
    Stream stream = getCurrentStream(dev_index);

    // Bool-mask indexing performs a scalar D2H sync (shape) and is not capture-safe.
    if (vbt::cuda::currentStreamCaptureStatus(dev_index) !=
        vbt::cuda::CaptureStatus::None) {
      throw std::runtime_error(
          std::string("index: ") + vbt::cuda::kErrAllocatorCaptureDenied);
    }

    const std::size_t mask_nbytes =
        static_cast<std::size_t>(mask_len) * sizeof(std::uint8_t);

    TensorImpl mask_contig = index_tensor;
    TensorImpl packed_mask;
    if (!mask_contig.is_contiguous()) {
      std::vector<std::int64_t> packed_sizes{mask_len};
      packed_mask = make_cuda_dense_out(index_tensor, packed_sizes);

      if (mask_len > 0) {
        const int threads = 256;
        Grid1DConfig cfg = make_1d_grid(mask_len, threads, dev_index);
        dim3 block_dim = cfg.block_dim;
        dim3 grid_dim  = cfg.grid_dim;

        (void)cudaGetLastError();
        pack_strided_bool_mask_to_contig_kernel<<<
            grid_dim, block_dim, 0,
            reinterpret_cast<cudaStream_t>(stream.handle())>>>(
                static_cast<const std::uint8_t*>(index_tensor.data()),
                index_tensor.strides()[0],
                static_cast<std::uint8_t*>(packed_mask.data()),
                mask_len);

        cudaError_t lc = cudaGetLastError();
        if (lc != cudaSuccess) {
          const char* msg = cudaGetErrorString(lc);
          std::string m = std::string("index: ") +
                          "CUDA bool-mask pack kernel launch failed: ";
          m += (msg ? msg : "");
          throw std::runtime_error(m);
        }
      }

      mask_contig = packed_mask;
    }

    const bool use_cub = cuda_bool_mask_cub_backend_enabled();

    if (use_cub) {
      auto& alloc = vbt::cuda::Allocator::get(dev_index);

      if (mask_len > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "index: CUDA bool-mask length exceeds int32 limit");
      }
      const int n = static_cast<int>(mask_len);
      const std::uint8_t* flags01 =
          static_cast<const std::uint8_t*>(mask_contig.data());

      int count_h = 0;
      TensorImpl d_count;
      int* d_count_ptr = nullptr;
      if (mask_len > 0) {
        d_count = make_cuda_int32_tensor(base, std::vector<std::int64_t>{1});
        d_count_ptr = static_cast<int*>(d_count.data());

        vbt::cuda::cub::reduce_sum_u8_as_i32(
            alloc,
            stream,
            flags01,
            n,
            d_count_ptr);

        cudaError_t st = cudaMemcpyAsync(
            &count_h,
            d_count_ptr,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            reinterpret_cast<cudaStream_t>(stream.handle()));
        if (st != cudaSuccess) {
          const char* msg = cudaGetErrorString(st);
          std::string m = std::string("index: ") +
                          errors::kErrCudaAdvCopyD2HFailed + ": ";
          m += (msg ? msg : "");
          throw std::runtime_error(m);
        }
        st = cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(stream.handle()));
        if (st != cudaSuccess) {
          const char* msg = cudaGetErrorString(st);
          std::string m = std::string("index: ") +
                          errors::kErrCudaAdvSyncFailed + ": ";
          m += (msg ? msg : "");
          throw std::runtime_error(m);
        }
      }

      if (count_h < 0) {
        count_h = 0;  // paranoia; sum of u8 flags must be >= 0
      }
      if (static_cast<std::int64_t>(count_h) > kAdvIndexMaxIndexNumel) {
        throw std::runtime_error(errors::kErrAdvIndexTooLarge);
      }

      index_numel = static_cast<std::int64_t>(count_h);
      idx_sizes.assign(1, index_numel);
      index_ndim = 1;

      // Allocate CUDA Int64 index tensor and fill it entirely on device.
      TensorImpl idx_cuda = make_cuda_int64_tensor(base, idx_sizes);
      if (index_numel > 0) {
        if (!d_count_ptr) {
          throw std::runtime_error(
              "index: internal error: missing device count buffer for CUB bool-mask indexing");
        }
        vbt::cuda::cub::select_indices_from_u8_flags_i64(
            alloc,
            stream,
            flags01,
            n,
            static_cast<std::int64_t*>(idx_cuda.data()),
            d_count_ptr);
      }

      // Record streams for allocator safety (inputs + derived indices).
      vbt::cuda::record_stream(index_tensor.storage(), stream);
      vbt::cuda::record_stream(mask_contig.storage(), stream);
      vbt::cuda::record_stream(idx_cuda.storage(), stream);

      index_for_info = std::move(idx_cuda);
    } else {
      // Legacy host scan path: D2H copy full mask, scan on CPU, H2D indices.
      std::vector<std::uint8_t> host_mask(
          static_cast<std::size_t>(mask_len));

      // Track whether the full-mask D2H transfer was invoked (perf cliff).
      detail::record_cuda_bool_mask_d2h_bytes(mask_nbytes);

      {
        cudaError_t st = cudaMemcpyAsync(
            host_mask.data(),
            mask_contig.data(),
            mask_nbytes,
            cudaMemcpyDeviceToHost,
            reinterpret_cast<cudaStream_t>(stream.handle()));
        if (st != cudaSuccess) {
          const char* msg = cudaGetErrorString(st);
          std::string m = std::string("index: ") +
                          errors::kErrCudaAdvCopyD2HFailed + ": ";
          m += (msg ? msg : "");
          throw std::runtime_error(m);
        }
        st = cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(stream.handle()));
        if (st != cudaSuccess) {
          const char* msg = cudaGetErrorString(st);
          std::string m = std::string("index: ") +
                          errors::kErrCudaAdvSyncFailed + ": ";
          m += (msg ? msg : "");
          throw std::runtime_error(m);
        }
      }

      std::vector<std::int64_t> host_indices;
      host_indices.reserve(static_cast<std::size_t>(mask_len));
      for (std::int64_t i = 0; i < mask_len; ++i) {
        if (host_mask[static_cast<std::size_t>(i)] != 0) {
          host_indices.push_back(i);
          if (static_cast<std::int64_t>(host_indices.size()) >
              kAdvIndexMaxIndexNumel) {
            throw std::runtime_error(errors::kErrAdvIndexTooLarge);
          }
        }
      }

      index_numel = static_cast<std::int64_t>(host_indices.size());
      idx_sizes.assign(1, index_numel);
      index_ndim = 1;

      // Allocate CUDA Int64 index tensor and copy indices to device.
      TensorImpl idx_cuda = make_cuda_int64_tensor(base, idx_sizes);
      if (index_numel > 0) {
        const std::size_t idx_nbytes =
            static_cast<std::size_t>(index_numel) * sizeof(std::int64_t);
        cudaError_t st = cudaMemcpyAsync(
            idx_cuda.data(),
            host_indices.data(),
            idx_nbytes,
            cudaMemcpyHostToDevice,
            reinterpret_cast<cudaStream_t>(stream.handle()));
        if (st != cudaSuccess) {
          const char* msg = cudaGetErrorString(st);
          std::string m = std::string("index: ") +
                          errors::kErrCudaAdvCopyH2DFailed + ": ";
          m += (msg ? msg : "");
          throw std::runtime_error(m);
        }
      }

      // Record streams for allocator safety.
      vbt::cuda::record_stream(index_tensor.storage(), stream);
      vbt::cuda::record_stream(mask_contig.storage(), stream);
      vbt::cuda::record_stream(idx_cuda.storage(), stream);

      index_for_info = std::move(idx_cuda);
    }
  } else {
    // Integer tensor index path (existing behavior).
    idx_sizes.assign(
        index_tensor.sizes().begin(), index_tensor.sizes().end());
    index_ndim =
        static_cast<std::int64_t>(idx_sizes.size());
    if (index_ndim > kIndexCudaMaxRank) {
      throw std::runtime_error(
          "index: CUDA advanced indexing too many index dims");
    }

    index_numel =
        safe_numel_from_sizes_cuda(idx_sizes);
    if (index_numel > kAdvIndexMaxIndexNumel) {
      throw std::runtime_error(
          "index: CUDA advanced indexing too many index elements");
    }

    index_for_info = index_tensor;
  }

  // Result shape = base.sizes() prefix + index shape.
  std::vector<std::int64_t> result_shape;
  result_shape.reserve(static_cast<std::size_t>(dims_before + index_ndim));
  for (std::int64_t d = 0; d < dims_before; ++d) {
    result_shape.push_back(base_sizes[static_cast<std::size_t>(d)]);
  }
  result_shape.insert(result_shape.end(),
                      idx_sizes.begin(), idx_sizes.end());

  const std::int64_t result_numel =
      safe_numel_from_sizes_cuda(result_shape);
  if (result_numel > kAdvIndexMaxResultNumel) {
    throw std::runtime_error(errors::kErrCudaAdvResultTooLarge);
  }

  // Build src view for AdvancedIndex: result_shape with zero strides on
  // advanced dims mirroring the CPU AdvancedIndex contract.
  std::vector<std::int64_t> src_sizes = result_shape;
  std::vector<std::int64_t> src_strides(src_sizes.size(), 0);

  for (std::int64_t d = 0; d < dims_before; ++d) {
    src_strides[static_cast<std::size_t>(d)] =
        base_strides[static_cast<std::size_t>(d)];
  }
  for (std::int64_t j = 0; j < index_ndim; ++j) {
    const std::size_t out_d =
        static_cast<std::size_t>(dims_before + j);
    src_strides[out_d] = 0;
  }

  TensorImpl src(
      base.storage(),
      src_sizes,
      src_strides,
      base.storage_offset(),
      base.dtype(),
      base.device());

  AdvancedIndex info;
  info.src = std::move(src);
  info.indices.clear();
  // For integer indices we store the raw CUDA index tensor; for Bool
  // masks we store the derived Int64 CUDA index tensor. In both cases
  // advanced_index_cuda_impl and advanced_index_put_cuda will normalize
  // and materialize indices as needed for kernels.
  info.indices.push_back(index_for_info);
  info.indexed_sizes.clear();
  info.indexed_strides_elems.clear();
  info.indexed_sizes.push_back(D);
  info.indexed_strides_elems.push_back(
      base_strides[adv_dim]);
  info.dims_before = dims_before;
  info.dims_after  = 0;
  info.index_shape = std::move(idx_sizes);
  info.result_shape = std::move(result_shape);
  info.use32bit_indexing =
      compute_use32bit_indexing_for_sizes_cuda(info.result_shape);

  AdvancedIndexCudaResult out;
  out.info = std::move(info);
  // Coarse, shape-only fast-path flag; layout checks live in
  // can_use_1d_gather_fastpath.
  const std::int64_t coarse_N =
      safe_numel_from_sizes_cuda(out.info.result_shape);
  out.can_use_1d_fastpath =
      (coarse_N > 0) &&
      (out.info.indices.size() == 1u) &&
      (out.info.index_shape.size() == 1u) &&
      (out.info.dims_after == 0) &&
      (out.info.indexed_sizes.size() == 1u);

  const bool has_nonempty_domain = (coarse_N > 0);
  detail::record_cuda_32bit_hint(out.info.use32bit_indexing,
                                 has_nonempty_domain);

  return out;
#endif
}

TensorImpl advanced_index_cuda_impl(AdvancedIndex& info,
                                    bool           coarse_can_use_1d_fastpath) {
#if !VBT_WITH_CUDA
  (void)info;
  (void)coarse_can_use_1d_fastpath;
  throw std::runtime_error(
      "index: CUDA advanced indexing is not available when built without CUDA");
#else
  const auto src_dev = info.src.device();
  if (src_dev.type != kDLCUDA) {
    throw std::invalid_argument(
        "advanced_index_cuda_impl: info.src must be a CUDA tensor");
  }

  // Sanity check that CUDA AdvancedIndex metadata still carries the raw
  // CUDA index tensor as designed.
  check_advanced_indices_are_raw_cuda(info);

  const std::int64_t N =
      safe_numel_from_sizes_cuda(info.result_shape);

  TensorImpl out = make_cuda_dense_out(info.src, info.result_shape);
  if (N == 0) {
    // Zero-numel (empty or overflow-sentinel) is a strict no-op.
    return out;
  }

  // Defensive DoS re-checks mirroring the read/write entrypoints.
  const std::int64_t index_numel =
      safe_numel_from_sizes_cuda(info.index_shape);
  if (index_numel > kAdvIndexMaxIndexNumel) {
    throw std::runtime_error(
        "index: CUDA advanced indexing too many index elements");
  }
  if (N > kAdvIndexMaxResultNumel) {
    throw std::runtime_error(errors::kErrCudaAdvResultTooLarge);
  }

  if (info.indices.size() != 1u ||
      info.indexed_sizes.size() != 1u ||
      info.indexed_strides_elems.size() != 1u) {
    throw std::invalid_argument(
        "advanced_index_cuda_impl: inconsistent AdvancedIndex metadata");
  }

  const std::vector<std::int64_t>& idx_sizes = info.index_shape;
  if (idx_sizes.empty()) {
    throw std::invalid_argument(
        "index: advanced tensor indices must have dim() > 0");
  }

  const std::int64_t D = info.indexed_sizes[0];
  if (D < 0) {
    throw std::invalid_argument(
        "index: negative dimension size in CUDA advanced indexing");
  }

  DeviceIndex dev_index = static_cast<DeviceIndex>(src_dev.index);
  DeviceGuard guard(dev_index);
  Stream stream = getCurrentStream(dev_index);

  const TensorImpl& index_raw = info.indices[0];
  if (index_raw.device().type != kDLCUDA ||
      index_raw.device().index != src_dev.index) {
    throw std::invalid_argument(
        "index: advanced index tensor must be on the same CUDA device as self");
  }
  const ScalarType idx_dt = index_raw.dtype();
  const bool allow_int32 = cuda_extended_index_dtypes_enabled();
  if (!(idx_dt == ScalarType::Int64 ||
        (idx_dt == ScalarType::Int32 && allow_int32))) {
    throw std::invalid_argument(
        "index: CUDA advanced indexing only supports Int64 index tensors");
  }

  TensorImpl idx_cuda = normalize_index_tensor_cuda_to_int64_read(
      index_raw, idx_sizes, D, dev_index, stream);

  const ScalarType dt = info.src.dtype();
  const bool dtype_supported =
      (dt == ScalarType::Float32) || (dt == ScalarType::Int64);

  const bool can_consider_fast1d =
      coarse_can_use_1d_fastpath &&
      advanced_index_32bit_enabled() &&
      dtype_supported;

  if (can_consider_fast1d &&
      can_use_1d_gather_fastpath(info, idx_cuda)) {
    const std::int64_t K = idx_sizes[0];
    const std::int64_t B_flat = (K == 0) ? 0 : (N / K);

    // N > 0 is guaranteed here because zero-numel cases returned early.
    detail::record_cuda_fast1d_hit();

    if (dt == ScalarType::Float32) {
      launch_advanced_index_gather_1d_kernel<float>(
          info.src, idx_cuda, out, B_flat, D, K, dev_index, stream);
    } else if (dt == ScalarType::Int64) {
      launch_advanced_index_gather_1d_kernel<long long>(
          info.src, idx_cuda, out, B_flat, D, K, dev_index, stream);
    } else {
      throw std::invalid_argument(
          "index: CUDA advanced indexing is only implemented for float32 and int64 tensors");
    }

    return out;
  }

  // Generic TensorIterator-backed kernel path.
  launch_generic_gather_from_advanced_index(
      info, idx_cuda, out, N, dev_index, stream);

  return out;
#endif
}

#if VBT_WITH_CUDA && VBT_INTERNAL_TESTS
detail::CudaBoundsMode get_effective_cuda_bounds_mode_for_tests() noexcept {
  return get_effective_cuda_bounds_mode();
}

unsigned int get_1d_grid_x_for_tests(std::int64_t N,
                                     int          threads,
                                     int          dev_index_raw) {
  DeviceIndex dev_index = static_cast<DeviceIndex>(dev_index_raw);
  Grid1DConfig cfg = make_1d_grid(N, threads, dev_index);
  return cfg.grid_dim.x;
}

void set_device_max_grid_x_override_for_tests(
    int                         dev_index_raw,
    std::optional<unsigned int> max_grid_x) {
  DeviceIndex dev_index = static_cast<DeviceIndex>(dev_index_raw);
  const int idx = static_cast<int>(dev_index);
  std::lock_guard<std::mutex> lock(g_device_caps_mutex);

  auto it = std::remove_if(
      g_device_caps_cache.begin(),
      g_device_caps_cache.end(),
      [idx](const DeviceCapsEntry& e) { return e.device_index == idx; });
  g_device_caps_cache.erase(it, g_device_caps_cache.end());

  if (max_grid_x.has_value()) {
    unsigned int cap = *max_grid_x;
    if (cap == 0u) {
      cap = 1u;
    }
    g_device_caps_cache.push_back(DeviceCapsEntry{idx, cap});
  }
}
#endif  // VBT_WITH_CUDA && VBT_INTERNAL_TESTS

} // namespace cuda_impl

void advanced_index_put_cuda(AdvancedIndex& info,
                             const TensorImpl& value,
                             bool accumulate) {
#if !VBT_WITH_CUDA
  (void)info;
  (void)value;
  (void)accumulate;
  throw std::runtime_error(
      "advanced_index_put_cuda: built without CUDA support");
#else
  const auto src_dev = info.src.device();
  if (src_dev.type != kDLCUDA) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: info.src must be a CUDA tensor");
  }
  if (value.device().type != kDLCUDA ||
      value.device().index != src_dev.index) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: value must be a CUDA tensor on the same device as src");
  }
  if (value.dtype() != info.src.dtype()) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: dtype mismatch between src and value");
  }

  if (info.indices.size() != 1) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: only a single tensor advanced index is supported on CUDA");
  }

  const std::int64_t result_numel =
      safe_numel_from_sizes_cuda(info.result_shape);
  if (result_numel == 0) {
    return;
  }

  const ScalarType dt = value.dtype();
  const bool extended = cuda_extended_index_dtypes_enabled();
  const bool cub_accum = cuda_cub_index_put_accumulate_enabled();
  if (accumulate) {
    const bool allowed = (dt == ScalarType::Float32) ||
        (cub_accum && dt == ScalarType::Int64);
    if (!allowed) {
      throw std::runtime_error(
          "index_put_: accumulate=true on CUDA is only implemented for float32 (and int64 when the CUB accumulate prototype is enabled)");
    }
  } else {
    const bool base_allowed = (dt == ScalarType::Float32 || dt == ScalarType::Int64);
    const bool extended_dtypes_allowed = extended &&
        (dt == ScalarType::Float16 ||
#if VBT_INDEX_CUDA_HAS_BF16
         dt == ScalarType::BFloat16 ||
#endif
         dt == ScalarType::Int32);
    if (!(base_allowed || extended_dtypes_allowed)) {
      throw std::runtime_error(
          "index_put_: advanced indexing dtype not supported on CUDA");
    }
  }

  const TensorImpl& index_raw = info.indices[0];
  if (index_raw.device().type != kDLCUDA ||
      index_raw.device().index != src_dev.index) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: index tensor must be a CUDA tensor on the same device as src");
  }

  const ScalarType idx_dt = index_raw.dtype();
  if (!(idx_dt == ScalarType::Int32 || idx_dt == ScalarType::Int64)) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: index tensor must be int32 or int64 on CUDA");
  }

  if (info.index_shape.empty()) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: advanced tensor indices must have dim() > 0");
  }

  const std::int64_t index_ndim =
      static_cast<std::int64_t>(info.index_shape.size());
  if (index_ndim > kIndexCudaMaxRank) {
    throw std::runtime_error(
        "index: CUDA advanced indexing too many index dims");
  }

  const std::int64_t index_numel =
      safe_numel_from_sizes_cuda(info.index_shape);
  if (index_numel > kAdvIndexMaxIndexNumel) {
    throw std::runtime_error(
        "index: CUDA advanced indexing too many index elements");
  }

  if (result_numel > kAdvIndexMaxResultNumel) {
    throw std::runtime_error(errors::kErrCudaAdvResultTooLarge);
  }

  TensorImpl value_b;
  try {
    value_b = broadcast_to(
        value,
        std::span<const std::int64_t>(
            info.result_shape.data(), info.result_shape.size()));
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    if (msg.find("shape mismatch") != std::string::npos) {
      throw std::invalid_argument(
          "advanced_index_put_cuda: shape mismatch during broadcast_to");
    }
    throw;
  }

  const auto& dst_sizes = info.src.sizes();
  const auto& dst_strides = info.src.strides();
  const auto& val_sizes = value_b.sizes();
  const auto& val_strides = value_b.strides();

  const std::int64_t R =
      static_cast<std::int64_t>(info.result_shape.size());
  if (R != static_cast<std::int64_t>(dst_sizes.size()) ||
      R != static_cast<std::int64_t>(val_sizes.size())) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: result/value shapes must match src shape");
  }

  const std::size_t K = info.indexed_sizes.size();
  if (K != 1u || info.indices.size() != 1u ||
      info.indexed_strides_elems.size() != 1u) {
    throw std::invalid_argument(
        "advanced_index_put_cuda: inconsistent AdvancedIndex metadata");
  }

  const std::int64_t dims_before = info.dims_before;

  using vbt::cuda::CaptureStatus;
  using vbt::cuda::currentStreamCaptureStatus;

  DeviceIndex dev_index = static_cast<DeviceIndex>(src_dev.index);
  DeviceGuard guard(dev_index);
  auto stream = getCurrentStream(dev_index);

  if (currentStreamCaptureStatus(dev_index) == CaptureStatus::Active) {
    throw std::runtime_error(
        "index_put_: CUDA advanced indexing writes are not supported under CUDA graph capture");
  }

  // Normalize indices on CPU or device according to the current
  // bounds mode and copy back to CUDA as Int64.
  const std::int64_t D =
      info.indexed_sizes[0];
  std::vector<std::int64_t> idx_sizes = info.index_shape;
  NormalizedIndexBuffersAny any = normalize_indices_cuda(
      index_raw, idx_sizes, D,
      /*is_read=*/false,
      /*allow_int32=*/true,
      dev_index, stream);

  TensorImpl idx_cuda = std::move(any.idx_cuda);

  auto record_streams = [&]() {
    vbt::cuda::record_stream(info.src.storage(), stream);
    vbt::cuda::record_stream(value_b.storage(), stream);
    vbt::cuda::record_stream(idx_cuda.storage(), stream);
  };

  AdvancedIndexPutCudaMeta meta{};
  meta.ndim = static_cast<int32_t>(R);
  meta.dims_before = static_cast<int32_t>(dims_before);
  meta.index_ndim = static_cast<int32_t>(index_ndim);
  meta.storage_offset_elems = info.src.storage_offset();
  meta.indexed_stride_elems = info.indexed_strides_elems[0];

  if (meta.ndim > kIndexCudaMaxRank) {
    throw std::runtime_error(
        "index: CUDA advanced indexing result rank exceeds internal limit");
  }

  for (int d = 0; d < meta.ndim; ++d) {
    meta.result_sizes[d] =
        info.result_shape[static_cast<std::size_t>(d)];
    meta.dst_strides[d] =
        dst_strides[static_cast<std::size_t>(d)];
    meta.value_strides[d] =
        val_strides[static_cast<std::size_t>(d)];
  }
  for (int j = 0; j < meta.index_ndim; ++j) {
    meta.index_shape[j] =
        info.index_shape[static_cast<std::size_t>(j)];
  }

  const std::int64_t N = result_numel;
  const bool use32 = should_use_32bit_for_advanced_index_cuda(info, N);

  const int threads = 256;
  Grid1DConfig cfg = make_1d_grid(N, threads, dev_index);
  dim3 block_dim = cfg.block_dim;
  dim3 grid_dim  = cfg.grid_dim;

  (void)cudaGetLastError();

  const ScalarType src_dt = info.src.dtype();
  if (accumulate) {
    // Sort (key,value) pairs by key, reduce duplicates, and scatter once per
    // unique key to avoid atomics.
    if (cub_accum &&
        (src_dt == ScalarType::Float32 || src_dt == ScalarType::Int64)) {
      auto& alloc = vbt::cuda::Allocator::get(dev_index);

      if (N <= static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        const int n_updates = static_cast<int>(N);
        const std::vector<std::int64_t> flat_sizes{N};
        cudaStream_t cu_stream =
            reinterpret_cast<cudaStream_t>(stream.handle());

        TensorImpl keys = make_cuda_int64_tensor(info.src, flat_sizes);
        TensorImpl vals = make_cuda_dense_out(info.src, flat_sizes);

        // Materialize (key,value) pairs.
        if (src_dt == ScalarType::Float32) {
          if (use32) {
            advanced_index_put_make_kv_cuda_kernel<float, std::int32_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<std::uint64_t*>(keys.data()),
                    static_cast<float*>(vals.data()),
                    static_cast<const float*>(value_b.data()),
                    static_cast<const std::int64_t*>(idx_cuda.data()),
                    meta,
                    static_cast<std::int32_t>(N));
          } else {
            advanced_index_put_make_kv_cuda_kernel<float, std::int64_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<std::uint64_t*>(keys.data()),
                    static_cast<float*>(vals.data()),
                    static_cast<const float*>(value_b.data()),
                    static_cast<const std::int64_t*>(idx_cuda.data()),
                    meta,
                    static_cast<std::int64_t>(N));
          }

          cudaError_t lc = cudaGetLastError();
          if (lc != cudaSuccess) {
            const char* msg = cudaGetErrorString(lc);
            std::string m =
                "index_put_: CUDA accumulate key/value kernel launch failed: ";
            m += (msg ? msg : "");
            throw std::runtime_error(m);
          }

          vbt::cuda::cub::radix_sort_pairs_u64_f32(
              alloc,
              stream,
              static_cast<std::uint64_t*>(keys.data()),
              static_cast<float*>(vals.data()),
              n_updates);

          TensorImpl unique_keys = make_cuda_int64_tensor(info.src, flat_sizes);
          TensorImpl reduced_vals = make_cuda_dense_out(info.src, flat_sizes);
          TensorImpl d_num_unique =
              make_cuda_int32_tensor(info.src, std::vector<std::int64_t>{1});

          vbt::cuda::cub::reduce_by_key_sum_u64_f32(
              alloc,
              stream,
              static_cast<const std::uint64_t*>(keys.data()),
              static_cast<const float*>(vals.data()),
              n_updates,
              static_cast<std::uint64_t*>(unique_keys.data()),
              static_cast<float*>(reduced_vals.data()),
              static_cast<int*>(d_num_unique.data()));

          if (use32) {
            advanced_index_put_scatter_reduced_kv_cuda_kernel<float, std::int32_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<float*>(info.src.storage()->data()),
                    static_cast<const std::uint64_t*>(unique_keys.data()),
                    static_cast<const float*>(reduced_vals.data()),
                    static_cast<const int*>(d_num_unique.data()),
                    static_cast<std::int32_t>(N));
          } else {
            advanced_index_put_scatter_reduced_kv_cuda_kernel<float, std::int64_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<float*>(info.src.storage()->data()),
                    static_cast<const std::uint64_t*>(unique_keys.data()),
                    static_cast<const float*>(reduced_vals.data()),
                    static_cast<const int*>(d_num_unique.data()),
                    static_cast<std::int64_t>(N));
          }
        } else {
          // Int64 accumulate path (requires CUB backend).
          if (use32) {
            advanced_index_put_make_kv_cuda_kernel<long long, std::int32_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<std::uint64_t*>(keys.data()),
                    static_cast<long long*>(vals.data()),
                    static_cast<const long long*>(value_b.data()),
                    static_cast<const std::int64_t*>(idx_cuda.data()),
                    meta,
                    static_cast<std::int32_t>(N));
          } else {
            advanced_index_put_make_kv_cuda_kernel<long long, std::int64_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<std::uint64_t*>(keys.data()),
                    static_cast<long long*>(vals.data()),
                    static_cast<const long long*>(value_b.data()),
                    static_cast<const std::int64_t*>(idx_cuda.data()),
                    meta,
                    static_cast<std::int64_t>(N));
          }

          cudaError_t lc = cudaGetLastError();
          if (lc != cudaSuccess) {
            const char* msg = cudaGetErrorString(lc);
            std::string m =
                "index_put_: CUDA accumulate key/value kernel launch failed: ";
            m += (msg ? msg : "");
            throw std::runtime_error(m);
          }

          vbt::cuda::cub::radix_sort_pairs_u64_i64(
              alloc,
              stream,
              static_cast<std::uint64_t*>(keys.data()),
              static_cast<long long*>(vals.data()),
              n_updates);

          TensorImpl unique_keys = make_cuda_int64_tensor(info.src, flat_sizes);
          TensorImpl reduced_vals = make_cuda_dense_out(info.src, flat_sizes);
          TensorImpl d_num_unique =
              make_cuda_int32_tensor(info.src, std::vector<std::int64_t>{1});

          vbt::cuda::cub::reduce_by_key_sum_u64_i64(
              alloc,
              stream,
              static_cast<const std::uint64_t*>(keys.data()),
              static_cast<const long long*>(vals.data()),
              n_updates,
              static_cast<std::uint64_t*>(unique_keys.data()),
              static_cast<long long*>(reduced_vals.data()),
              static_cast<int*>(d_num_unique.data()));

          if (use32) {
            advanced_index_put_scatter_reduced_kv_cuda_kernel<long long, std::int32_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<long long*>(info.src.storage()->data()),
                    static_cast<const std::uint64_t*>(unique_keys.data()),
                    static_cast<const long long*>(reduced_vals.data()),
                    static_cast<const int*>(d_num_unique.data()),
                    static_cast<std::int32_t>(N));
          } else {
            advanced_index_put_scatter_reduced_kv_cuda_kernel<long long, std::int64_t><<<
                grid_dim, block_dim, 0, cu_stream>>>(
                    static_cast<long long*>(info.src.storage()->data()),
                    static_cast<const std::uint64_t*>(unique_keys.data()),
                    static_cast<const long long*>(reduced_vals.data()),
                    static_cast<const int*>(d_num_unique.data()),
                    static_cast<std::int64_t>(N));
          }
        }

        cudaError_t lc2 = cudaGetLastError();
        if (lc2 != cudaSuccess) {
          const char* msg = cudaGetErrorString(lc2);
          std::string m =
              "index_put_: CUDA accumulate scatter kernel launch failed: ";
          m += (msg ? msg : "");
          throw std::runtime_error(m);
        }

        record_streams();

        return;
      }

      // If we reached here, we couldn't run the CUB path due to an
      // unexpected size constraint. Int64 has no atomic fallback.
      if (src_dt == ScalarType::Int64) {
        throw std::runtime_error(
            "index_put_: accumulate=true on CUDA is only implemented for float32");
      }
    }

    // Fallback accumulate path: float32 atomicAdd.
    if (use32) {
      advanced_index_put_cuda_kernel<float, std::int32_t, true><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(info.src.storage()->data()),
              static_cast<const float*>(value_b.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_put_cuda_kernel<float, std::int64_t, true><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(info.src.storage()->data()),
              static_cast<const float*>(value_b.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              static_cast<std::int64_t>(N));
    }
  } else if (src_dt == ScalarType::Float32) {
    if (use32) {
      advanced_index_put_cuda_kernel<float, std::int32_t, false><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(info.src.storage()->data()),
              static_cast<const float*>(value_b.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_put_cuda_kernel<float, std::int64_t, false><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<float*>(info.src.storage()->data()),
              static_cast<const float*>(value_b.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              static_cast<std::int64_t>(N));
    }
  } else if (src_dt == ScalarType::Int64) {
    if (use32) {
      advanced_index_put_cuda_kernel<long long, std::int32_t, false><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<long long*>(info.src.storage()->data()),
              static_cast<const long long*>(value_b.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              static_cast<std::int32_t>(N));
    } else {
      advanced_index_put_cuda_kernel<long long, std::int64_t, false><<<
          grid_dim, block_dim, 0,
          reinterpret_cast<cudaStream_t>(stream.handle())>>>(
              static_cast<long long*>(info.src.storage()->data()),
              static_cast<const long long*>(value_b.data()),
              static_cast<const std::int64_t*>(idx_cuda.data()),
              meta,
              static_cast<std::int64_t>(N));
    }
  } else {
    // Should be unreachable due to earlier dtype grid checks.
    throw std::logic_error("advanced_index_put_cuda: unsupported src dtype");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    const char* msg = cudaGetErrorString(err);
    std::string m =
        "index_put_: CUDA advanced indexing write kernel launch failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  record_streams();
#endif
}

} // namespace indexing
} // namespace core
} // namespace vbt
