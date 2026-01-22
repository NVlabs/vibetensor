// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbt/core/checked_math.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/strided_loop.h"
#include "vbt/core/tensor.h"
#include "vbt/core/write_guard.h"
#include "vbt/cpu/storage.h"
#include "vbt/cuda/guard.h"
#include "vbt/cuda/storage.h"
#include "vbt/cuda/stream.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/allocator.h"
#include "vbt/cuda/cub.h"
#include <cuda_runtime_api.h>
#endif
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::dispatch::BoxedStack;

#if VBT_WITH_CUDA
namespace vbt { namespace ops {
bool embedding_cuda_bounds_check_i64(const std::int64_t* idx,
                                    std::int64_t N,
                                    std::int64_t V,
                                    vbt::cuda::Stream stream,
                                    const char* op_name);
void embedding_cuda_gather_f32(const float* w,
                              std::int64_t ws0,
                              std::int64_t ws1,
                              const std::int64_t* idx,
                              float* out,
                              std::int64_t N,
                              std::int64_t D,
                              vbt::cuda::Stream stream,
                              const char* op_name);
void embedding_cuda_pack_keys_vals_u64_i64(const std::int64_t* idx,
                                          std::uint64_t* keys,
                                          long long* vals,
                                          int n,
                                          vbt::cuda::Stream stream,
                                          const char* op_name);
void embedding_cuda_renorm_f32(const std::uint64_t* unique_keys,
                              const int* d_num_unique,
                              int n_max,
                              float* w,
                              std::int64_t ws0,
                              std::int64_t ws1,
                              std::int64_t D,
                              float max_norm,
                              float p,
                              int* d_mutated,
                              vbt::cuda::Stream stream,
                              const char* op_name);
}}  // namespace vbt::ops
#endif

namespace {

static std::vector<int64_t> make_contig_strides_checked(
    const std::vector<int64_t>& sizes, const char* op_name) {
  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0;
       --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const int64_t dim = sizes[idx];
    if (dim == 0) {
      continue;
    }
    int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(acc, dim, tmp)) {
      throw std::overflow_error(std::string(op_name) + ": stride overflow");
    }
    acc = tmp;
  }
  return strides;
}

static int64_t checked_numel_from_sizes(const std::vector<int64_t>& sizes,
                                       const char* op_name,
                                       const char* what) {
  if (sizes.empty()) {
    return 1;  // scalar
  }
  int64_t n = 1;
  for (int64_t s : sizes) {
    if (s < 0) {
      throw std::invalid_argument(std::string(op_name) + ": " + what +
                                  " size must be >= 0");
    }
    if (s == 0) {
      return 0;
    }
    int64_t tmp = 0;
    if (!vbt::core::checked_mul_i64(n, s, tmp)) {
      throw std::overflow_error(std::string(op_name) + ": " + what +
                                " numel overflow");
    }
    n = tmp;
  }
  return n;
}

static std::size_t checked_num_bytes(int64_t n,
                                    std::size_t itemsize,
                                    const char* op_name,
                                    const char* what) {
  if (n <= 0) {
    return 0;
  }
  int64_t total_i64 = 0;
  const int64_t item_b = static_cast<int64_t>(itemsize);
  if (!vbt::core::checked_mul_i64(n, item_b, total_i64)) {
    throw std::overflow_error(std::string(op_name) + ": " + what +
                              " numel*itemsize overflow");
  }
  if (total_i64 < 0 ||
      static_cast<std::uint64_t>(total_i64) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(op_name) + ": " + what +
                              " byte size overflow");
  }
  return static_cast<std::size_t>(total_i64);
}

static int64_t read_cpu_scalar_int64_0d(const TensorImpl& t,
                                       const char* op_name,
                                       const char* what) {
  if (t.device().type != kDLCPU || t.dtype() != ScalarType::Int64 ||
      !t.sizes().empty()) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " must be a 0-d int64 CPU tensor");
  }
  const void* p = t.data();
  if (!p) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " has no data");
  }
  return *static_cast<const int64_t*>(p);
}

static bool read_cpu_scalar_bool_0d(const TensorImpl& t,
                                   const char* op_name,
                                   const char* what) {
  if (t.device().type != kDLCPU || t.dtype() != ScalarType::Bool ||
      !t.sizes().empty()) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " must be a 0-d bool CPU tensor");
  }
  const void* p = t.data();
  if (!p) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " has no data");
  }
  return (*static_cast<const std::uint8_t*>(p) != 0);
}

static float read_cpu_scalar_float32_0d(const TensorImpl& t,
                                       const char* op_name,
                                       const char* what) {
  if (t.device().type != kDLCPU || t.dtype() != ScalarType::Float32 ||
      !t.sizes().empty()) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " must be a 0-d float32 CPU tensor");
  }
  const void* p = t.data();
  if (!p) {
    throw std::invalid_argument(std::string(op_name) + ": " + what +
                                " has no data");
  }
  return *static_cast<const float*>(p);
}

static void copy_indices_to_contig_int64(TensorImpl& dst,
                                        const TensorImpl& src,
                                        int64_t N,
                                        const char* op_name) {
  if (dst.dtype() != ScalarType::Int64) {
    throw std::invalid_argument(std::string(op_name) +
                                ": internal: idx64 must be int64");
  }
  if (!(src.dtype() == ScalarType::Int32 || src.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument(std::string(op_name) +
                                ": internal: indices must be int32 or int64");
  }
  if (dst.sizes() != src.sizes()) {
    throw std::invalid_argument(std::string(op_name) +
                                ": internal: idx64 size mismatch");
  }

  if (src.data() == nullptr) {
    throw std::invalid_argument(std::string(op_name) + ": indices has no data");
  }

  auto* out = static_cast<int64_t*>(dst.data());
  if (!out) {
    throw std::invalid_argument(std::string(op_name) +
                                ": internal: idx64 has no data");
  }

  auto* st = src.storage().get();
  if (!st) {
    throw std::invalid_argument(std::string(op_name) +
                                ": indices must be defined");
  }
  const auto* storage_base = static_cast<const std::uint8_t*>(st->data());
  if (!storage_base) {
    throw std::invalid_argument(std::string(op_name) + ": indices has no data");
  }
  const std::size_t in_nbytes = st->nbytes();

  const auto& in_sizes = src.sizes();
  const auto& in_strides = src.strides();
  const std::size_t nd = in_sizes.size();

  const std::size_t item_b = src.itemsize();
  const int64_t item_b_i64 = static_cast<int64_t>(item_b);

  std::vector<int64_t> idx(nd, 0);

  for (int64_t linear = 0; linear < N; ++linear) {
    int64_t abs_off_elems = src.storage_offset();

    for (std::size_t d = 0; d < nd; ++d) {
      int64_t term = 0;
      if (!vbt::core::checked_mul_i64(idx[d], in_strides[d], term) ||
          !vbt::core::checked_add_i64(abs_off_elems, term, abs_off_elems)) {
        throw std::overflow_error(std::string(op_name) +
                                  ": indices pointer offset overflow");
      }
    }

    if (abs_off_elems < 0) {
      throw std::out_of_range(std::string(op_name) +
                              ": indices pointer out of range");
    }

    int64_t byte_off_i64 = 0;
    if (!vbt::core::checked_mul_i64(abs_off_elems, item_b_i64, byte_off_i64)) {
      throw std::overflow_error(std::string(op_name) +
                                ": indices pointer offset overflow");
    }
    if (byte_off_i64 < 0) {
      throw std::out_of_range(std::string(op_name) +
                              ": indices pointer out of range");
    }

    const std::size_t byte_off = static_cast<std::size_t>(byte_off_i64);
    if (byte_off > in_nbytes || item_b > (in_nbytes - byte_off)) {
      throw std::out_of_range(std::string(op_name) +
                              ": indices pointer out of range");
    }

    const std::uint8_t* ps = storage_base + byte_off;
    if (src.dtype() == ScalarType::Int64) {
      out[linear] = *reinterpret_cast<const int64_t*>(ps);
    } else {
      out[linear] = static_cast<int64_t>(*reinterpret_cast<const int32_t*>(ps));
    }

    // Increment idx (row-major from last dimension)
    for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nd) - 1; d >= 0; --d) {
      const auto ud = static_cast<std::size_t>(d);
      idx[ud] += 1;
      if (idx[ud] < in_sizes[ud]) {
        break;
      }
      idx[ud] = 0;
    }
  }
}

}  // namespace

static void vt_embedding_cpu_boxed(BoxedStack& s) {
  constexpr const char* op_name = "vt::embedding";

  if (s.size() != 5) {
    throw std::invalid_argument(
        std::string(op_name) + " boxed kernel expected 5 arguments, got " +
        std::to_string(s.size()));
  }

  TensorImpl weight = s[0];
  TensorImpl indices = s[1];
  const TensorImpl padding_idx_t = s[2];
  const TensorImpl scale_grad_by_freq_t = s[3];
  const TensorImpl sparse_t = s[4];

  // Validate scalar args (forward ignores padding_idx, scale_grad_by_freq, sparse).
  (void)read_cpu_scalar_int64_0d(padding_idx_t, op_name, "padding_idx");
  (void)read_cpu_scalar_bool_0d(scale_grad_by_freq_t, op_name,
                                "scale_grad_by_freq");
  (void)read_cpu_scalar_bool_0d(sparse_t, op_name, "sparse");

  // Validate weight
  if (!weight.storage().get()) {
    throw std::invalid_argument(std::string(op_name) + ": weight must be defined");
  }
  if (weight.device().type != kDLCPU) {
    throw std::invalid_argument(
        std::string(op_name) + ": only CPU tensors are supported in this build");
  }
  if (weight.dtype() != ScalarType::Float32) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be float32");
  }
  if (weight.sizes().size() != 2) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be rank 2");
  }
  const int64_t V = weight.sizes()[0];
  const int64_t D = weight.sizes()[1];

  // Validate indices
  if (!indices.storage().get()) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be defined");
  }
  if (indices.device() != weight.device()) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be on the same device as weight");
  }
  if (!(indices.dtype() == ScalarType::Int32 || indices.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be int32 or int64");
  }

  // Compute N (checked) and allocate output.
  const int64_t N = checked_numel_from_sizes(indices.sizes(), op_name, "indices");

  int64_t out_numel = 0;
  if (!vbt::core::checked_mul_i64(N, D, out_numel)) {
    throw std::overflow_error(std::string(op_name) + ": output numel overflow");
  }

  std::vector<int64_t> out_sizes = indices.sizes();
  out_sizes.push_back(D);

  std::vector<int64_t> out_strides;
  if (out_numel == 0) {
    // For empty outputs (N==0 or D==0), strides are irrelevant but must be
    // representable without overflow.
    out_strides.assign(out_sizes.size(), 1);
  } else {
    out_strides = make_contig_strides_checked(out_sizes, op_name);
  }

  const std::size_t out_bytes =
      checked_num_bytes(out_numel, sizeof(float), op_name, "output");
  auto out_storage = vbt::cpu::new_cpu_storage(out_bytes, /*pinned=*/false);
  TensorImpl out(std::move(out_storage), std::move(out_sizes),
                 std::move(out_strides), /*storage_offset=*/0,
                 ScalarType::Float32, weight.device());

  // For empty indices, return an empty output without touching indices/weight.
  if (N == 0) {
    s.clear();
    s.push_back(std::move(out));
    return;
  }

  // Canonicalize indices to contiguous int64.
  const int64_t* idx_ptr = nullptr;
  TensorImpl idx64;

  if (indices.dtype() == ScalarType::Int64 && indices.is_contiguous()) {
    idx_ptr = static_cast<const int64_t*>(indices.data());
  } else {
    const std::size_t idx_bytes =
        checked_num_bytes(N, sizeof(int64_t), op_name, "indices");
    auto idx_storage = vbt::cpu::new_cpu_storage(idx_bytes, /*pinned=*/false);
    std::vector<int64_t> idx_sizes = indices.sizes();
    std::vector<int64_t> idx_strides =
        make_contig_strides_checked(idx_sizes, op_name);
    idx64 = TensorImpl(std::move(idx_storage), std::move(idx_sizes),
                       std::move(idx_strides), /*storage_offset=*/0,
                       ScalarType::Int64, weight.device());

    if (indices.data() == nullptr) {
      throw std::invalid_argument(std::string(op_name) + ": indices has no data");
    }

    bool has_neg_stride = false;
    for (int64_t st : indices.strides()) {
      if (st < 0) {
        has_neg_stride = true;
        break;
      }
    }

    // vbt::core::for_each_1out_1in iterates strided tensors in increasing-address
    // order (lower-corner + abs stride bytes). This is fine for non-negative
    // strides, but it reverses logical order for negative-stride views. Indices
    // order is semantically observable, so handle negative strides explicitly.
    if (has_neg_stride) {
      copy_indices_to_contig_int64(idx64, indices, N, op_name);
    } else if (indices.dtype() == ScalarType::Int64) {
      vbt::core::for_each_1out_1in(idx64, indices,
                                  [](std::uint8_t* pd, std::uint8_t* ps) {
                                    *reinterpret_cast<int64_t*>(pd) =
                                        *reinterpret_cast<const int64_t*>(ps);
                                  });
    } else {
      vbt::core::for_each_1out_1in(idx64, indices,
                                  [](std::uint8_t* pd, std::uint8_t* ps) {
                                    *reinterpret_cast<int64_t*>(pd) =
                                        static_cast<int64_t>(
                                            *reinterpret_cast<const int32_t*>(ps));
                                  });
    }
    idx_ptr = static_cast<const int64_t*>(idx64.data());
  }

  if (!idx_ptr) {
    throw std::invalid_argument(std::string(op_name) + ": indices has no data");
  }

  // Bounds check. Must run even when D == 0.
  for (int64_t i = 0; i < N; ++i) {
    const int64_t idx = idx_ptr[i];
    if (idx < 0 || idx >= V) {
      throw std::out_of_range(
          std::string(op_name) + ": index out of range in self");
    }
  }

  if (D == 0) {
    s.clear();
    s.push_back(std::move(out));
    return;
  }

  // Weight deref guard: protect against as_strided views where V*D overflows
  // and TensorImpl::data() returns nullptr.
  if (V > 0 && D > 0) {
    int64_t wd_numel = 0;
    if (!vbt::core::checked_mul_i64(V, D, wd_numel)) {
      throw std::overflow_error(std::string(op_name) + ": weight numel overflow");
    }
    if (weight.data() == nullptr) {
      throw std::invalid_argument(std::string(op_name) + ": weight has no data");
    }
  }

  float* out_data = static_cast<float*>(out.data());
  const float* w_data = static_cast<const float*>(weight.data());
  if (!out_data || !w_data) {
    throw std::invalid_argument(std::string(op_name) + ": internal: null data");
  }

  const int64_t ws0 = weight.strides()[0];
  const int64_t ws1 = weight.strides()[1];
  if (ws0 < 0 || ws1 < 0) {
    throw std::invalid_argument(std::string(op_name) +
                                ": negative weight strides are not supported");
  }

  auto to_ptrdiff_checked = [&](int64_t x, const char* what) -> std::ptrdiff_t {
    constexpr std::ptrdiff_t pd_min =
        std::numeric_limits<std::ptrdiff_t>::min();
    constexpr std::ptrdiff_t pd_max =
        std::numeric_limits<std::ptrdiff_t>::max();
    if (x < static_cast<int64_t>(pd_min) || x > static_cast<int64_t>(pd_max)) {
      throw std::overflow_error(std::string(op_name) + ": " + what +
                                " pointer offset overflow");
    }
    return static_cast<std::ptrdiff_t>(x);
  };

  for (int64_t i = 0; i < N; ++i) {
    const int64_t idx = idx_ptr[i];

    int64_t w_row_off = 0;
    if (!vbt::core::checked_mul_i64(idx, ws0, w_row_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }

    int64_t out_row_off = 0;
    if (!vbt::core::checked_mul_i64(i, D, out_row_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": output pointer offset overflow");
    }

    const float* w_row = w_data + to_ptrdiff_checked(w_row_off, "weight");
    float* out_row = out_data + to_ptrdiff_checked(out_row_off, "output");

    if (ws1 == 1) {
      std::memcpy(out_row, w_row, static_cast<std::size_t>(D) * sizeof(float));
    } else {
      if (D > 0) {
        int64_t max_in_row = 0;
        if (!vbt::core::checked_mul_i64(ws1, D - 1, max_in_row)) {
          throw std::overflow_error(std::string(op_name) +
                                    ": weight pointer offset overflow");
        }
        int64_t last_off = 0;
        if (!vbt::core::checked_add_i64(w_row_off, max_in_row, last_off)) {
          throw std::overflow_error(std::string(op_name) +
                                    ": weight pointer offset overflow");
        }
      }

      const std::ptrdiff_t step = to_ptrdiff_checked(ws1, "weight");
      const float* wp = w_row;
      for (int64_t j = 0; j < D; ++j) {
        out_row[j] = *wp;
        wp += step;
      }
    }
  }

  s.clear();
  s.push_back(std::move(out));
}

#if VBT_WITH_CUDA
static void vt_embedding_cuda_boxed(BoxedStack& s) {
  constexpr const char* op_name = "vt::embedding";

  if (s.size() != 5) {
    throw std::invalid_argument(
        std::string(op_name) + " boxed kernel expected 5 arguments, got " +
        std::to_string(s.size()));
  }

  TensorImpl weight = s[0];
  TensorImpl indices = s[1];
  const TensorImpl padding_idx_t = s[2];
  const TensorImpl scale_grad_by_freq_t = s[3];
  const TensorImpl sparse_t = s[4];

  // Validate scalar args (forward ignores padding_idx, scale_grad_by_freq, sparse).
  (void)read_cpu_scalar_int64_0d(padding_idx_t, op_name, "padding_idx");
  (void)read_cpu_scalar_bool_0d(scale_grad_by_freq_t, op_name,
                                "scale_grad_by_freq");
  (void)read_cpu_scalar_bool_0d(sparse_t, op_name, "sparse");

  // Validate weight
  if (!weight.storage().get()) {
    throw std::invalid_argument(std::string(op_name) +
                                ": weight must be defined");
  }
  if (weight.device().type != kDLCUDA) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be a CUDA tensor");
  }
  if (weight.dtype() != ScalarType::Float32) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be float32");
  }
  if (weight.sizes().size() != 2) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be rank 2");
  }
  const int64_t V = weight.sizes()[0];
  const int64_t D = weight.sizes()[1];

  // Validate indices
  if (!indices.storage().get()) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be defined");
  }
  if (indices.device() != weight.device()) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be on the same device as weight");
  }
  if (indices.dtype() != ScalarType::Int64) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be int64 on CUDA");
  }
  if (!indices.is_contiguous()) {
    throw std::invalid_argument(
        std::string(op_name) + ": CUDA indices must be contiguous");
  }

  // Compute N (checked) and allocate output.
  const int64_t N = checked_numel_from_sizes(indices.sizes(), op_name, "indices");

  int64_t out_numel = 0;
  if (!vbt::core::checked_mul_i64(N, D, out_numel)) {
    throw std::overflow_error(std::string(op_name) + ": output numel overflow");
  }

  std::vector<int64_t> out_sizes = indices.sizes();
  out_sizes.push_back(D);

  std::vector<int64_t> out_strides;
  if (out_numel == 0) {
    // For empty outputs (N==0 or D==0), strides are irrelevant but must be
    // representable without overflow.
    out_strides.assign(out_sizes.size(), 1);
  } else {
    out_strides = make_contig_strides_checked(out_sizes, op_name);
  }

  const std::size_t out_bytes =
      checked_num_bytes(out_numel, sizeof(float), op_name, "output");

  const int dev_index = weight.device().index;
  vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(dev_index));
  vbt::cuda::Stream stream =
      vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(dev_index));

  auto out_storage =
      vbt::cuda::new_cuda_storage(out_bytes, /*device_index=*/dev_index);
  TensorImpl out(std::move(out_storage), std::move(out_sizes), std::move(out_strides),
                 /*storage_offset=*/0, ScalarType::Float32, weight.device());

  // For empty indices, return an empty output without touching indices/weight.
  if (N == 0) {
    s.clear();
    s.push_back(std::move(out));
    return;
  }

  const int64_t* idx_ptr = static_cast<const int64_t*>(indices.data());
  if (!idx_ptr) {
    throw std::invalid_argument(std::string(op_name) + ": indices has no data");
  }

  // NOTE: CUDA bounds checking requires a device-to-host transfer and a stream
  // synchronize, so vt::embedding is not CUDA-graph-capture safe.
  const bool has_oob =
      vbt::ops::embedding_cuda_bounds_check_i64(
          idx_ptr, N, V, stream, op_name);
  if (has_oob) {
    throw std::out_of_range(
        std::string(op_name) + ": index out of range in self");
  }

  if (D == 0) {
    s.clear();
    s.push_back(std::move(out));
    return;
  }

  // Weight deref guard: protect against as_strided views where V*D overflows
  // and TensorImpl::data() returns nullptr.
  if (V > 0 && D > 0) {
    int64_t wd_numel = 0;
    if (!vbt::core::checked_mul_i64(V, D, wd_numel)) {
      throw std::overflow_error(std::string(op_name) + ": weight numel overflow");
    }
    if (weight.data() == nullptr) {
      throw std::invalid_argument(std::string(op_name) + ": weight has no data");
    }
  }

  const int64_t ws0 = weight.strides()[0];
  const int64_t ws1 = weight.strides()[1];
  if (ws0 < 0 || ws1 < 0) {
    throw std::invalid_argument(std::string(op_name) +
                                ": negative weight strides are not supported");
  }

  // Guard against int64 overflow in row/col offset arithmetic.
  if (V > 0 && D > 0) {
    int64_t max_row_off = 0;
    if (!vbt::core::checked_mul_i64(V - 1, ws0, max_row_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }
    int64_t max_col_off = 0;
    if (!vbt::core::checked_mul_i64(D - 1, ws1, max_col_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }
    int64_t max_off = 0;
    if (!vbt::core::checked_add_i64(max_row_off, max_col_off, max_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }
  }

  const float* w_data = static_cast<const float*>(weight.data());
  float* out_data = static_cast<float*>(out.data());
  if (!w_data || !out_data) {
    throw std::invalid_argument(std::string(op_name) + ": internal: null data");
  }

  vbt::ops::embedding_cuda_gather_f32(
      w_data, ws0, ws1, idx_ptr, out_data, N, D, stream, op_name);

  vbt::cuda::record_stream(out.storage(), stream);
  vbt::cuda::record_stream(weight.storage(), stream);
  vbt::cuda::record_stream(indices.storage(), stream);

  s.clear();
  s.push_back(std::move(out));
}

static void vt_embedding_renorm_cuda_boxed(BoxedStack& s) {
  constexpr const char* op_name = "vt::embedding_renorm_";

  if (s.size() != 4) {
    throw std::invalid_argument(
        std::string(op_name) + " boxed kernel expected 4 arguments, got " +
        std::to_string(s.size()));
  }

  TensorImpl weight = s[0];
  TensorImpl indices = s[1];
  const TensorImpl max_norm_t = s[2];
  const TensorImpl norm_type_t = s[3];

  const float max_norm =
      read_cpu_scalar_float32_0d(max_norm_t, op_name, "max_norm");
  const float p =
      read_cpu_scalar_float32_0d(norm_type_t, op_name, "norm_type");

  if (std::isnan(max_norm) || max_norm <= 0.0f) {
    throw std::invalid_argument(std::string(op_name) + ": max_norm must be > 0");
  }
  if (std::isnan(p) || p <= 0.0f) {
    throw std::invalid_argument(std::string(op_name) + ": norm_type must be > 0");
  }

  // Validate weight
  if (!weight.storage().get()) {
    throw std::invalid_argument(std::string(op_name) + ": weight must be defined");
  }
  if (weight.device().type != kDLCUDA) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be a CUDA tensor");
  }
  if (weight.dtype() != ScalarType::Float32) {
    throw std::invalid_argument(std::string(op_name) + ": weight must be float32");
  }
  if (weight.sizes().size() != 2) {
    throw std::invalid_argument(std::string(op_name) + ": weight must be rank 2");
  }
  const int64_t V = weight.sizes()[0];
  const int64_t D = weight.sizes()[1];

  // Validate indices
  if (!indices.storage().get()) {
    throw std::invalid_argument(std::string(op_name) + ": indices must be defined");
  }
  if (indices.device() != weight.device()) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be on the same device as weight");
  }
  if (indices.dtype() != ScalarType::Int64) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be int64 on CUDA");
  }
  if (!indices.is_contiguous()) {
    throw std::invalid_argument(
        std::string(op_name) + ": CUDA indices must be contiguous");
  }

  // In-place write guard: reject definite internal overlap.
  vbt::core::check_writable(weight);

  // Compute N.
  const int64_t N = checked_numel_from_sizes(indices.sizes(), op_name, "indices");

  // For empty indices, this is a no-op.
  if (N == 0) {
    s.clear();
    s.push_back(std::move(weight));
    return;
  }

  if (N > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        std::string(op_name) + ": CUDA indices numel exceeds int32 limit");
  }
  const int n_updates = static_cast<int>(N);

  const int dev_index = weight.device().index;
  vbt::cuda::DeviceGuard dg(static_cast<vbt::cuda::DeviceIndex>(dev_index));
  vbt::cuda::Stream stream =
      vbt::cuda::getCurrentStream(static_cast<vbt::cuda::DeviceIndex>(dev_index));
  cudaStream_t cu_stream =
      reinterpret_cast<cudaStream_t>(stream.handle());

  const int64_t* idx_ptr = static_cast<const int64_t*>(indices.data());
  if (!idx_ptr) {
    throw std::invalid_argument(std::string(op_name) + ": indices has no data");
  }

  // NOTE: CUDA bounds checking requires a device-to-host transfer and a stream
  // synchronize, so vt::embedding_renorm_ is not CUDA-graph-capture safe.
  const bool has_oob =
      vbt::ops::embedding_cuda_bounds_check_i64(idx_ptr, N, V, stream, op_name);
  if (has_oob) {
    throw std::out_of_range(std::string(op_name) + ": index out of range in self");
  }

  // Nothing to do for empty rows or max_norm=+inf.
  if (D == 0 || (std::isinf(max_norm) && max_norm > 0.0f)) {
    s.clear();
    s.push_back(std::move(weight));
    return;
  }

  // Weight deref guard: mirror vt::embedding and CPU renorm.
  if (V > 0 && D > 0) {
    int64_t wd_numel = 0;
    if (!vbt::core::checked_mul_i64(V, D, wd_numel)) {
      throw std::overflow_error(std::string(op_name) + ": weight numel overflow");
    }
    if (weight.data() == nullptr) {
      throw std::invalid_argument(std::string(op_name) + ": weight has no data");
    }
  }

  float* w_data = static_cast<float*>(weight.data());
  if (!w_data) {
    throw std::invalid_argument(std::string(op_name) + ": internal: null data");
  }

  const int64_t ws0 = weight.strides()[0];
  const int64_t ws1 = weight.strides()[1];
  if (ws0 < 0 || ws1 < 0) {
    throw std::invalid_argument(std::string(op_name) +
                                ": negative weight strides are not supported");
  }

  // Guard against int64 overflow in row/col offset arithmetic.
  if (V > 0 && D > 0) {
    int64_t max_row_off = 0;
    if (!vbt::core::checked_mul_i64(V - 1, ws0, max_row_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }
    int64_t max_col_off = 0;
    if (!vbt::core::checked_mul_i64(D - 1, ws1, max_col_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }
    int64_t max_off = 0;
    if (!vbt::core::checked_add_i64(max_row_off, max_col_off, max_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }
    (void)max_off;
  }

  // Record user storages early for allocator safety.
  vbt::cuda::record_stream(weight.storage(), stream);
  vbt::cuda::record_stream(indices.storage(), stream);

  // Build (key,value) pairs for sort+reduce (dedup indices).
  const std::size_t flat_bytes =
      checked_num_bytes(N, sizeof(int64_t), op_name, "tmp");

  TensorImpl keys(
      vbt::cuda::new_cuda_storage(flat_bytes, /*device_index=*/dev_index),
      std::vector<int64_t>{N},
      std::vector<int64_t>{1},
      /*storage_offset=*/0,
      ScalarType::Int64,
      weight.device());

  TensorImpl vals(
      vbt::cuda::new_cuda_storage(flat_bytes, /*device_index=*/dev_index),
      std::vector<int64_t>{N},
      std::vector<int64_t>{1},
      /*storage_offset=*/0,
      ScalarType::Int64,
      weight.device());

  TensorImpl unique_keys(
      vbt::cuda::new_cuda_storage(flat_bytes, /*device_index=*/dev_index),
      std::vector<int64_t>{N},
      std::vector<int64_t>{1},
      /*storage_offset=*/0,
      ScalarType::Int64,
      weight.device());

  TensorImpl reduced_vals(
      vbt::cuda::new_cuda_storage(flat_bytes, /*device_index=*/dev_index),
      std::vector<int64_t>{N},
      std::vector<int64_t>{1},
      /*storage_offset=*/0,
      ScalarType::Int64,
      weight.device());

  const std::size_t i32_bytes = checked_num_bytes(1, sizeof(int), op_name, "tmp");

  TensorImpl d_num_unique(
      vbt::cuda::new_cuda_storage(i32_bytes, /*device_index=*/dev_index),
      std::vector<int64_t>{1},
      std::vector<int64_t>{1},
      /*storage_offset=*/0,
      ScalarType::Int32,
      weight.device());

  TensorImpl d_mutated(
      vbt::cuda::new_cuda_storage(i32_bytes, /*device_index=*/dev_index),
      std::vector<int64_t>{1},
      std::vector<int64_t>{1},
      /*storage_offset=*/0,
      ScalarType::Int32,
      weight.device());

  cudaError_t st = cudaMemsetAsync(d_mutated.data(), 0, i32_bytes, cu_stream);
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(op_name) + ": cudaMemsetAsync(d_mutated) failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }
  vbt::cuda::record_stream(d_mutated.storage(), stream);

  // Materialize keys/vals.
  vbt::ops::embedding_cuda_pack_keys_vals_u64_i64(
      idx_ptr,
      static_cast<std::uint64_t*>(keys.data()),
      static_cast<long long*>(vals.data()),
      n_updates,
      stream,
      op_name);

  vbt::cuda::record_stream(keys.storage(), stream);
  vbt::cuda::record_stream(vals.storage(), stream);

  // Outputs of reduce_by_key are used on the current stream; record before calling
  // into CUB so we remain exception-safe even on launch failures.
  vbt::cuda::record_stream(unique_keys.storage(), stream);
  vbt::cuda::record_stream(reduced_vals.storage(), stream);
  vbt::cuda::record_stream(d_num_unique.storage(), stream);

  auto& alloc = vbt::cuda::Allocator::get(dev_index);

  vbt::cuda::cub::radix_sort_pairs_u64_i64(
      alloc,
      stream,
      static_cast<std::uint64_t*>(keys.data()),
      static_cast<long long*>(vals.data()),
      n_updates);

  vbt::cuda::cub::reduce_by_key_sum_u64_i64(
      alloc,
      stream,
      static_cast<const std::uint64_t*>(keys.data()),
      static_cast<const long long*>(vals.data()),
      n_updates,
      static_cast<std::uint64_t*>(unique_keys.data()),
      static_cast<long long*>(reduced_vals.data()),
      static_cast<int*>(d_num_unique.data()));

  // Apply renorm per unique row. This kernel reads d_num_unique on device and is
  // safe to launch with n_max = N.
  vbt::ops::embedding_cuda_renorm_f32(
      static_cast<const std::uint64_t*>(unique_keys.data()),
      static_cast<const int*>(d_num_unique.data()),
      n_updates,
      w_data,
      ws0,
      ws1,
      D,
      max_norm,
      p,
      static_cast<int*>(d_mutated.data()),
      stream,
      op_name);

  // Read mutated flag and bump version iff any row was scaled.
  int h_mutated = 0;
  st = cudaMemcpyAsync(&h_mutated, d_mutated.data(), sizeof(int),
                       cudaMemcpyDeviceToHost, cu_stream);
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(op_name) + ": cudaMemcpyAsync(d_mutated) failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  st = cudaStreamSynchronize(cu_stream);
  if (st != cudaSuccess) {
    const char* msg = cudaGetErrorString(st);
    std::string m = std::string(op_name) + ": cudaStreamSynchronize failed: ";
    m += (msg ? msg : "");
    throw std::runtime_error(m);
  }

  if (h_mutated != 0 && weight.numel() > 0) {
    weight.bump_version();
  }

  // Fence remaining temporaries.
  vbt::cuda::record_stream(d_mutated.storage(), stream);
  vbt::cuda::record_stream(keys.storage(), stream);
  vbt::cuda::record_stream(vals.storage(), stream);
  vbt::cuda::record_stream(unique_keys.storage(), stream);
  vbt::cuda::record_stream(reduced_vals.storage(), stream);
  vbt::cuda::record_stream(d_num_unique.storage(), stream);

  s.clear();
  s.push_back(std::move(weight));
}

#endif  // VBT_WITH_CUDA

static void vt_embedding_renorm_cpu_boxed(BoxedStack& s) {
  constexpr const char* op_name = "vt::embedding_renorm_";

  if (s.size() != 4) {
    throw std::invalid_argument(
        std::string(op_name) + " boxed kernel expected 4 arguments, got " +
        std::to_string(s.size()));
  }

  TensorImpl weight = s[0];
  TensorImpl indices = s[1];
  const TensorImpl max_norm_t = s[2];
  const TensorImpl norm_type_t = s[3];

  const float max_norm =
      read_cpu_scalar_float32_0d(max_norm_t, op_name, "max_norm");
  const float p =
      read_cpu_scalar_float32_0d(norm_type_t, op_name, "norm_type");

  if (std::isnan(max_norm) || max_norm <= 0.0f) {
    throw std::invalid_argument(std::string(op_name) + ": max_norm must be > 0");
  }
  if (std::isnan(p) || p <= 0.0f) {
    throw std::invalid_argument(std::string(op_name) + ": norm_type must be > 0");
  }

  // Validate weight
  if (!weight.storage().get()) {
    throw std::invalid_argument(std::string(op_name) + ": weight must be defined");
  }
  if (weight.device().type != kDLCPU) {
    throw std::invalid_argument(
        std::string(op_name) + ": only CPU tensors are supported in this build");
  }
  if (weight.dtype() != ScalarType::Float32) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be float32");
  }
  if (weight.sizes().size() != 2) {
    throw std::invalid_argument(
        std::string(op_name) + ": weight must be rank 2");
  }
  const int64_t V = weight.sizes()[0];
  const int64_t D = weight.sizes()[1];

  // Validate indices
  if (!indices.storage().get()) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be defined");
  }
  if (indices.device() != weight.device()) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be on the same device as weight");
  }
  if (!(indices.dtype() == ScalarType::Int32 ||
        indices.dtype() == ScalarType::Int64)) {
    throw std::invalid_argument(
        std::string(op_name) + ": indices must be int32 or int64");
  }

  // In-place write guard: reject definite internal overlap.
  vbt::core::check_writable(weight);

  // Compute N.
  const int64_t N = checked_numel_from_sizes(indices.sizes(), op_name, "indices");

  // For empty indices, this is a no-op.
  if (N == 0) {
    s.clear();
    s.push_back(std::move(weight));
    return;
  }

  // Canonicalize indices to contiguous int64.
  const int64_t* idx_ptr = nullptr;
  TensorImpl idx64;

  if (indices.dtype() == ScalarType::Int64 && indices.is_contiguous()) {
    idx_ptr = static_cast<const int64_t*>(indices.data());
  } else {
    const std::size_t idx_bytes =
        checked_num_bytes(N, sizeof(int64_t), op_name, "indices");
    auto idx_storage = vbt::cpu::new_cpu_storage(idx_bytes, /*pinned=*/false);
    std::vector<int64_t> idx_sizes = indices.sizes();
    std::vector<int64_t> idx_strides =
        make_contig_strides_checked(idx_sizes, op_name);
    idx64 = TensorImpl(std::move(idx_storage), std::move(idx_sizes),
                       std::move(idx_strides), /*storage_offset=*/0,
                       ScalarType::Int64, weight.device());

    if (indices.data() == nullptr) {
      throw std::invalid_argument(std::string(op_name) + ": indices has no data");
    }

    bool has_neg_stride = false;
    for (int64_t st : indices.strides()) {
      if (st < 0) {
        has_neg_stride = true;
        break;
      }
    }

    // Preserve logical order for negative strides.
    if (has_neg_stride) {
      copy_indices_to_contig_int64(idx64, indices, N, op_name);
    } else if (indices.dtype() == ScalarType::Int64) {
      vbt::core::for_each_1out_1in(idx64, indices,
                                  [](std::uint8_t* pd, std::uint8_t* ps) {
                                    *reinterpret_cast<int64_t*>(pd) =
                                        *reinterpret_cast<const int64_t*>(ps);
                                  });
    } else {
      vbt::core::for_each_1out_1in(idx64, indices,
                                  [](std::uint8_t* pd, std::uint8_t* ps) {
                                    *reinterpret_cast<int64_t*>(pd) =
                                        static_cast<int64_t>(
                                            *reinterpret_cast<const int32_t*>(ps));
                                  });
    }

    idx_ptr = static_cast<const int64_t*>(idx64.data());
  }

  if (!idx_ptr) {
    throw std::invalid_argument(std::string(op_name) + ": indices has no data");
  }

  // Bounds check.
  for (int64_t i = 0; i < N; ++i) {
    const int64_t idx = idx_ptr[i];
    if (idx < 0 || idx >= V) {
      throw std::out_of_range(
          std::string(op_name) + ": index out of range in self");
    }
  }

  // Nothing to do for empty rows or max_norm=+inf.
  if (D == 0 || (std::isinf(max_norm) && max_norm > 0.0f)) {
    s.clear();
    s.push_back(std::move(weight));
    return;
  }

  // Weight deref guard: mirror vt::embedding.
  if (V > 0 && D > 0) {
    int64_t wd_numel = 0;
    if (!vbt::core::checked_mul_i64(V, D, wd_numel)) {
      throw std::overflow_error(std::string(op_name) + ": weight numel overflow");
    }
    if (weight.data() == nullptr) {
      throw std::invalid_argument(std::string(op_name) + ": weight has no data");
    }
  }

  float* w_data = static_cast<float*>(weight.data());
  if (!w_data) {
    throw std::invalid_argument(std::string(op_name) + ": internal: null data");
  }

  const int64_t ws0 = weight.strides()[0];
  const int64_t ws1 = weight.strides()[1];
  if (ws0 < 0 || ws1 < 0) {
    throw std::invalid_argument(std::string(op_name) +
                                ": negative weight strides are not supported");
  }

  auto to_ptrdiff_checked = [&](int64_t x, const char* what) -> std::ptrdiff_t {
    constexpr std::ptrdiff_t pd_min =
        std::numeric_limits<std::ptrdiff_t>::min();
    constexpr std::ptrdiff_t pd_max =
        std::numeric_limits<std::ptrdiff_t>::max();
    if (x < static_cast<int64_t>(pd_min) || x > static_cast<int64_t>(pd_max)) {
      throw std::overflow_error(std::string(op_name) + ": " + what +
                                " pointer offset overflow");
    }
    return static_cast<std::ptrdiff_t>(x);
  };

  // Sort+dedup indices (touch each row at most once).
  std::vector<int64_t> uniq;
  uniq.reserve(static_cast<std::size_t>(N));
  for (int64_t i = 0; i < N; ++i) {
    uniq.push_back(idx_ptr[i]);
  }
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

  const std::ptrdiff_t step = to_ptrdiff_checked(ws1, "weight");
  constexpr float kEps = 1e-7f;
  bool mutated = false;

  for (int64_t idx : uniq) {
    int64_t w_row_off = 0;
    if (!vbt::core::checked_mul_i64(idx, ws0, w_row_off)) {
      throw std::overflow_error(std::string(op_name) +
                                ": weight pointer offset overflow");
    }

    // Validate representability of the last element offset for this row.
    if (ws1 != 1 && D > 0) {
      int64_t max_in_row = 0;
      if (!vbt::core::checked_mul_i64(ws1, D - 1, max_in_row)) {
        throw std::overflow_error(std::string(op_name) +
                                  ": weight pointer offset overflow");
      }
      int64_t last_off = 0;
      if (!vbt::core::checked_add_i64(w_row_off, max_in_row, last_off)) {
        throw std::overflow_error(std::string(op_name) +
                                  ": weight pointer offset overflow");
      }
    }

    float* row = w_data + to_ptrdiff_checked(w_row_off, "weight");

    // Compute p-norm.
    float norm = 0.0f;
    if (std::isinf(p)) {
      float m = 0.0f;
      const float* rp = row;
      for (int64_t j = 0; j < D; ++j) {
        m = std::max(m, static_cast<float>(std::fabs(*rp)));
        rp += step;
      }
      norm = m;
    } else if (p == 1.0f) {
      float acc = 0.0f;
      const float* rp = row;
      for (int64_t j = 0; j < D; ++j) {
        acc += static_cast<float>(std::fabs(*rp));
        rp += step;
      }
      norm = acc;
    } else if (p == 2.0f) {
      float acc = 0.0f;
      const float* rp = row;
      for (int64_t j = 0; j < D; ++j) {
        const float v = *rp;
        acc += v * v;
        rp += step;
      }
      norm = static_cast<float>(std::sqrt(acc));
    } else {
      float acc = 0.0f;
      const float* rp = row;
      for (int64_t j = 0; j < D; ++j) {
        acc += static_cast<float>(std::pow(std::fabs(*rp), p));
        rp += step;
      }
      norm = static_cast<float>(std::pow(acc, 1.0f / p));
    }

    if (norm > max_norm) {
      const float scale = max_norm / (norm + kEps);
      float* wp = row;
      for (int64_t j = 0; j < D; ++j) {
        *wp *= scale;
        wp += step;
      }
      mutated = true;
    }
  }

  if (mutated && weight.numel() > 0) {
    weight.bump_version();
  }

  s.clear();
  s.push_back(std::move(weight));
}

extern "C" void vbt_register_embedding_kernels() {
  using vbt::dispatch::Dispatcher;
  using vbt::dispatch::KernelFunction;

  auto& D = Dispatcher::instance();

  if (!D.has("vt::embedding")) {
    D.registerLibrary("vt");
    D.def(
        "vt::embedding(Tensor weight, Tensor indices, Tensor padding_idx, Tensor scale_grad_by_freq, Tensor sparse) -> Tensor");

    auto kf = KernelFunction::makeBoxed(/*arity=*/5, &vt_embedding_cpu_boxed);
    D.registerCpuKernelFunction("vt::embedding", kf);
#if VBT_WITH_CUDA
    auto kf_cuda =
        KernelFunction::makeBoxed(/*arity=*/5, &vt_embedding_cuda_boxed);
    D.registerCudaKernelFunction("vt::embedding", kf_cuda);
#endif
  }

  if (!D.has("vt::embedding_renorm_")) {
    if (!D.has("vt::embedding")) {
      D.registerLibrary("vt");
    }
    D.def(
        "vt::embedding_renorm_(Tensor weight, Tensor indices, Tensor max_norm, Tensor norm_type) -> Tensor");

    auto kf =
        KernelFunction::makeBoxed(/*arity=*/4, &vt_embedding_renorm_cpu_boxed);
    D.registerCpuKernelFunction("vt::embedding_renorm_", kf);
#if VBT_WITH_CUDA
    auto kf_cuda =
        KernelFunction::makeBoxed(/*arity=*/4, &vt_embedding_renorm_cuda_boxed);
    D.registerCudaKernelFunction("vt::embedding_renorm_", kf_cuda);
#endif
  }
}
