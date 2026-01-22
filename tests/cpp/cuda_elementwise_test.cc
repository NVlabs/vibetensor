// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cmath>

#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor_iterator/cuda.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"

#if VBT_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

using vbt::dispatch::Dispatcher;
using vbt::dispatch::BoxedStack;
using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
extern "C" vbt::core::TensorImpl vbt_cuda_add_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_mul_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_sub_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_div_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_relu_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_eq_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_ne_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_lt_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_gt_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_le_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_ge_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_bitwise_and_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_bitwise_or_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_bitwise_xor_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_logical_and_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_logical_or_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_logical_xor_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_lshift_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_rshift_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_fmod_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_remainder_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_atan2_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_copysign_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_hypot_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_xlogy_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_xlog1py_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_nextafter_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_heaviside_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_abs_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_neg_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_exp_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_log_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_sqrt_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_rsqrt_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_sin_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_cos_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_tanh_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_sigmoid_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_expm1_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_log1p_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_floor_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_ceil_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_trunc_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_round_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_frac_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_reciprocal_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_sign_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_exp2_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_log2_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_sinh_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_cosh_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_tan_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_asin_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_acos_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_atan_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_erf_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_erfc_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_lgamma_impl(const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_pow_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_clamp_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);
extern "C" vbt::core::TensorImpl vbt_cuda_lerp_impl(const vbt::core::TensorImpl&, const vbt::core::TensorImpl&, const vbt::core::TensorImpl&);

extern "C" void vbt_register_default_kernels();
extern "C" void vbt_register_cuda_elementwise_kernels();

static TensorImpl make_cuda_long_tensor_from_host(const std::vector<int64_t>& host);

static TensorImpl make_cuda_tensor_from_host(const std::vector<float>& host) {
#if VBT_WITH_CUDA
  const int dev = 0;
  const std::size_t nbytes = host.size() * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  // Copy host -> device
  cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed");
  std::vector<int64_t> sizes{static_cast<int64_t>(host.size())};
  std::vector<int64_t> strides{1};
  return TensorImpl(storage, sizes, strides, 0, ScalarType::Float32, Device::cuda(dev));
#else
  (void)host; throw std::runtime_error("CUDA not built");
#endif
}

static std::vector<float> copy_cuda_tensor_to_host(const TensorImpl& t) {
#if VBT_WITH_CUDA
  std::vector<float> out(t.numel());
  cudaError_t st = cudaMemcpy(out.data(), t.data(), out.size()*sizeof(float), cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");
  return out;
#else
  (void)t; throw std::runtime_error("CUDA not built");
#endif
}

static std::vector<uint8_t> copy_cuda_bool_tensor_to_host(const TensorImpl& t) {
#if VBT_WITH_CUDA
  std::vector<uint8_t> out(t.numel());
  cudaError_t st = cudaMemcpy(out.data(), t.data(), out.size(), cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");
  return out;
#else
  (void)t; throw std::runtime_error("CUDA not built");
#endif
}

static std::vector<int64_t> copy_cuda_long_tensor_to_host(const TensorImpl& t) {
#if VBT_WITH_CUDA
  std::vector<int64_t> out(t.numel());
  cudaError_t st = cudaMemcpy(out.data(), t.data(), out.size() * sizeof(int64_t), cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");
  return out;
#else
  (void)t; throw std::runtime_error("CUDA not built");
#endif
}

TEST(CUDAElementwise, DenseFastpathFloat32) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  // Ensure op schemas and kernels are registered
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();
  auto& D = Dispatcher::instance();
  try { D.registerLibrary("vt"); } catch (...) {}
  try { D.def("vt::add(Tensor, Tensor) -> Tensor"); } catch (...) {}
  try { D.def("vt::relu(Tensor) -> Tensor"); } catch (...) {}
  try { D.registerCudaKernel("vt::add", &vbt_cuda_add_impl); } catch (...) {}
  try { D.registerCudaKernel("vt::relu", &vbt_cuda_relu_impl); } catch (...) {}
  // Validate registration is visible
  EXPECT_TRUE(D.has("vt::add"));
  EXPECT_TRUE(D.has("vt::relu"));
  const int N = 1024;
  std::vector<float> ha(N), hb(N);
  for (int i = 0; i < N; ++i) { ha[i] = static_cast<float>(i); hb[i] = static_cast<float>(2*i); }
  TensorImpl a = make_cuda_tensor_from_host(ha);
  TensorImpl b = make_cuda_tensor_from_host(hb);

  // Direct call to CUDA unboxed kernel instead of dispatcher due to static init order/linking of registrars
  TensorImpl out_add = vbt_cuda_add_impl(a, b);

  // Synchronize current device to ensure copies below see results
  cudaDeviceSynchronize();
  auto hc = copy_cuda_tensor_to_host(out_add);
  ASSERT_EQ(static_cast<int>(hc.size()), N);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(hc[i], ha[i] + hb[i]);
  }

  // Direct call to CUDA unboxed relu kernel
  TensorImpl out_relu = vbt_cuda_relu_impl(a);
  cudaDeviceSynchronize();
  auto hr = copy_cuda_tensor_to_host(out_relu);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(hr[i], ha[i] > 0.0f ? ha[i] : 0.0f);
  }

  // Scalar tests (Scalar CPU + Tensor CUDA)
  float* scalar_val = new float(10.0f);
  auto scalar_storage = vbt::core::make_intrusive<Storage>(
      DataPtr(scalar_val, [](void* p) noexcept { delete static_cast<float*>(p); }),
      sizeof(float));
  // Scalar tensor: empty sizes/strides
  TensorImpl scalar_cpu(scalar_storage, {}, {}, 0, ScalarType::Float32, Device::cpu());
  
  TensorImpl out_scalar_add = vbt_cuda_add_impl(a, scalar_cpu);
  cudaDeviceSynchronize();
  auto h_scalar_add = copy_cuda_tensor_to_host(out_scalar_add);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(h_scalar_add[i], ha[i] + 10.0f);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, VectorizedFloat32) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();
  
  // Must be multiple of 4 and aligned
  const int N = 4096; 
  std::vector<float> ha(N), hb(N);
  for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 2.0f; }
  
  TensorImpl a = make_cuda_tensor_from_host(ha);
  TensorImpl b = make_cuda_tensor_from_host(hb);
  
  // Verify alignment (usually cudaMalloc returns 256-byte aligned pointers)
  // But make_cuda_tensor_from_host uses new_cuda_storage.
  // Let's assume it is aligned.
  
  TensorImpl out = vbt_cuda_add_impl(a, b);
  cudaDeviceSynchronize();
  
  auto hc = copy_cuda_tensor_to_host(out);
  ASSERT_EQ(hc.size(), N);
  for (int i = 0; i < N; ++i) {
    EXPECT_FLOAT_EQ(hc[i], 3.0f);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, AddAndMulRankLimit25Vs26) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  const int dev = 0;
  const int rank25 = static_cast<int>(vbt::core::kTensorIterCudaMaxNdim);

  std::vector<int64_t> sizes25(static_cast<std::size_t>(rank25), 1);
  sizes25[0] = 2;
  sizes25[1] = 3;
  const std::int64_t N = 6;

  std::vector<int64_t> strides25(static_cast<std::size_t>(rank25));
  std::int64_t stride = 1;
  for (int i = rank25 - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides25[idx] = stride;
    const std::int64_t dim = sizes25[idx];
    stride *= (dim == 0 ? 1 : dim);
  }

  const std::size_t nbytes = static_cast<std::size_t>(N) * sizeof(float);
  auto st_a = vbt::cuda::new_cuda_storage(nbytes, dev);
  auto st_b = vbt::cuda::new_cuda_storage(nbytes, dev);

  std::vector<float> ha(static_cast<std::size_t>(N));
  std::vector<float> hb(static_cast<std::size_t>(N));
  for (std::int64_t i = 0; i < N; ++i) {
    ha[static_cast<std::size_t>(i)] = static_cast<float>(i);
    hb[static_cast<std::size_t>(i)] = static_cast<float>(2 * i);
  }

  cudaError_t st = cudaMemcpy(st_a->data(), ha.data(), nbytes, cudaMemcpyHostToDevice);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaMemcpy(st_b->data(), hb.data(), nbytes, cudaMemcpyHostToDevice);
  ASSERT_EQ(st, cudaSuccess);

  TensorImpl a25(st_a, sizes25, strides25, 0, ScalarType::Float32, Device::cuda(dev));
  TensorImpl b25(st_b, sizes25, strides25, 0, ScalarType::Float32, Device::cuda(dev));

  TensorImpl out_add25 = vbt_cuda_add_impl(a25, b25);
  TensorImpl out_mul25 = vbt_cuda_mul_impl(a25, b25);
  cudaDeviceSynchronize();

  auto h_add25 = copy_cuda_tensor_to_host(out_add25);
  auto h_mul25 = copy_cuda_tensor_to_host(out_mul25);
  ASSERT_EQ(h_add25.size(), ha.size());
  ASSERT_EQ(h_mul25.size(), ha.size());
  for (std::size_t i = 0; i < ha.size(); ++i) {
    EXPECT_FLOAT_EQ(h_add25[i], ha[i] + hb[i]);
    EXPECT_FLOAT_EQ(h_mul25[i], ha[i] * hb[i]);
  }

  // 26D: same numel but rank exceeds CUDA TI practical limit.
  std::vector<int64_t> sizes26(static_cast<std::size_t>(rank25 + 1), 1);
  sizes26[0] = 2;
  sizes26[1] = 3;

  std::vector<int64_t> strides26(static_cast<std::size_t>(rank25 + 1));
  stride = 1;
  for (int i = rank25; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides26[idx] = stride;
    const std::int64_t dim = sizes26[idx];
    stride *= (dim == 0 ? 1 : dim);
  }

  auto st_c = vbt::cuda::new_cuda_storage(nbytes, dev);
  auto st_d = vbt::cuda::new_cuda_storage(nbytes, dev);
  st = cudaMemcpy(st_c->data(), ha.data(), nbytes, cudaMemcpyHostToDevice);
  ASSERT_EQ(st, cudaSuccess);
  st = cudaMemcpy(st_d->data(), hb.data(), nbytes, cudaMemcpyHostToDevice);
  ASSERT_EQ(st, cudaSuccess);

  TensorImpl c26(st_c, sizes26, strides26, 0, ScalarType::Float32, Device::cuda(dev));
  TensorImpl d26(st_d, sizes26, strides26, 0, ScalarType::Float32, Device::cuda(dev));

  bool add_threw = false;
  try {
    (void)vbt_cuda_add_impl(c26, d26);
  } catch (const std::invalid_argument& e) {
    add_threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("supports up to "), std::string::npos);
    EXPECT_NE(msg.find(std::to_string(rank25)), std::string::npos);
  }
  EXPECT_TRUE(add_threw);

  bool mul_threw = false;
  try {
    (void)vbt_cuda_mul_impl(c26, d26);
  } catch (const std::invalid_argument& e) {
    mul_threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("supports up to "), std::string::npos);
    EXPECT_NE(msg.find(std::to_string(rank25)), std::string::npos);
  }
  EXPECT_TRUE(mul_threw);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, StridedFallbackAddFloat32) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  const int rows = 2;
  const int cols = 256;
  const int N = rows * cols;

  // a: shape [rows, cols] with distinct values per element
  std::vector<float> ha(N);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      ha[r * cols + c] = static_cast<float>(r * 1000 + c);
    }
  }

  // b: shape [cols]; broadcast across the leading dimension
  std::vector<float> hb(cols);
  for (int c = 0; c < cols; ++c) {
    hb[c] = 1.5f * static_cast<float>(c);
  }

  const int dev = 0;
  const std::size_t nbytes_a = ha.size() * sizeof(float);
  auto st_a = vbt::cuda::new_cuda_storage(nbytes_a, dev);
  cudaError_t st = cudaMemcpy(st_a->data(), ha.data(), nbytes_a, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed");

  // a: sizes {rows, cols}, strides {cols, 1}; contiguous 2D
  TensorImpl a(st_a, {rows, cols}, {cols, 1}, /*storage_offset=*/0,
               ScalarType::Float32, Device::cuda(dev));

  // b: contiguous [cols] on the same device
  TensorImpl b = make_cuda_tensor_from_host(hb);

  // Direct call to CUDA unboxed add kernel (broadcast + TI-backed path)
  TensorImpl out = vbt_cuda_add_impl(a, b);
  cudaDeviceSynchronize();
  auto hout = copy_cuda_tensor_to_host(out);
  ASSERT_EQ(static_cast<int>(hout.size()), N);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const int idx = r * cols + c;
      EXPECT_FLOAT_EQ(hout[idx], ha[idx] + hb[c]);
    }
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, Comparison) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  const int N = 10;
  std::vector<float> ha(N), hb(N);
  for (int i = 0; i < N; ++i) { ha[i] = (float)i; hb[i] = (float)i; }
  hb[5] = 100.0f; // 5 != 100

  TensorImpl a = make_cuda_tensor_from_host(ha);
  TensorImpl b = make_cuda_tensor_from_host(hb);

  TensorImpl out_eq = vbt_cuda_eq_impl(a, b);
  cudaDeviceSynchronize();

  std::vector<uint8_t> hout(N);
  cudaError_t st = cudaMemcpy(hout.data(), out_eq.data(), N * sizeof(bool), cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy D2H failed");

  for (int i = 0; i < N; ++i) {
      bool expected = (i != 5);
      EXPECT_EQ(static_cast<bool>(hout[i]), expected) << "at index " << i;
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, SubDiv) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  const int N = 10;
  std::vector<float> ha(N), hb(N);
  for (int i = 0; i < N; ++i) { ha[i] = (float)(i + 1) * 10; hb[i] = (float)(i + 1); }
  
  TensorImpl a = make_cuda_tensor_from_host(ha);
  TensorImpl b = make_cuda_tensor_from_host(hb);
  
  // Dense
  TensorImpl out_sub = vbt_cuda_sub_impl(a, b);
  TensorImpl out_div = vbt_cuda_div_impl(a, b);
  cudaDeviceSynchronize();
  
  auto h_sub = copy_cuda_tensor_to_host(out_sub);
  auto h_div = copy_cuda_tensor_to_host(out_div);
  
  for (int i = 0; i < N; ++i) {
      EXPECT_FLOAT_EQ(h_sub[i], ha[i] - hb[i]);
      EXPECT_FLOAT_EQ(h_div[i], ha[i] / hb[i]);
  }
  
  // Scalar
  float* scalar_val = new float(2.0f);
  auto scalar_storage = vbt::core::make_intrusive<Storage>(
      DataPtr(scalar_val, [](void* p) noexcept { delete static_cast<float*>(p); }),
      sizeof(float));
  TensorImpl scalar_cpu(scalar_storage, {}, {}, 0, ScalarType::Float32, Device::cpu());
  
  TensorImpl out_sub_scalar = vbt_cuda_sub_impl(a, scalar_cpu);
  TensorImpl out_div_scalar = vbt_cuda_div_impl(a, scalar_cpu);
  cudaDeviceSynchronize();
  
  auto h_sub_scalar = copy_cuda_tensor_to_host(out_sub_scalar);
  auto h_div_scalar = copy_cuda_tensor_to_host(out_div_scalar);
  
  for (int i = 0; i < N; ++i) {
      EXPECT_FLOAT_EQ(h_sub_scalar[i], ha[i] - 2.0f);
      EXPECT_FLOAT_EQ(h_div_scalar[i], ha[i] / 2.0f);
  }

#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, Int64DivUnsupported) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  std::vector<int64_t> ha = {10};
  std::vector<int64_t> hb = {2};
  TensorImpl a = make_cuda_long_tensor_from_host(ha);
  TensorImpl b = make_cuda_long_tensor_from_host(hb);

  EXPECT_THROW(vbt_cuda_div_impl(a, b), std::runtime_error);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, ComparisonsComprehensive) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();
  
  const int N = 4;
  std::vector<float> ha = {1, 2, 3, 4};
  std::vector<float> hb = {1, 3, 2, 5};
  
  TensorImpl a = make_cuda_tensor_from_host(ha);
  TensorImpl b = make_cuda_tensor_from_host(hb);
  
  // Eq
  auto out_eq = vbt_cuda_eq_impl(a, b);
  cudaDeviceSynchronize();
  auto h_eq = copy_cuda_bool_tensor_to_host(out_eq);
  EXPECT_EQ(h_eq[0], 1); // 1==1
  EXPECT_EQ(h_eq[1], 0); // 2!=3
  
  // Ne
  auto out_ne = vbt_cuda_ne_impl(a, b);
  cudaDeviceSynchronize();
  auto h_ne = copy_cuda_bool_tensor_to_host(out_ne);
  EXPECT_EQ(h_ne[0], 0);
  EXPECT_EQ(h_ne[1], 1);
  
  // Lt
  auto out_lt = vbt_cuda_lt_impl(a, b);
  cudaDeviceSynchronize();
  auto h_lt = copy_cuda_bool_tensor_to_host(out_lt);
  EXPECT_EQ(h_lt[0], 0); // 1<1 F
  EXPECT_EQ(h_lt[1], 1); // 2<3 T
  EXPECT_EQ(h_lt[2], 0); // 3<2 F
  
  // Gt
  auto out_gt = vbt_cuda_gt_impl(a, b);
  cudaDeviceSynchronize();
  auto h_gt = copy_cuda_bool_tensor_to_host(out_gt);
  EXPECT_EQ(h_gt[0], 0);
  EXPECT_EQ(h_gt[1], 0);
  EXPECT_EQ(h_gt[2], 1); // 3>2 T

  // Le
  auto out_le = vbt_cuda_le_impl(a, b);
  cudaDeviceSynchronize();
  auto h_le = copy_cuda_bool_tensor_to_host(out_le);
  EXPECT_EQ(h_le[0], 1); // 1<=1 T
  EXPECT_EQ(h_le[1], 1); // 2<=3 T
  EXPECT_EQ(h_le[2], 0);
  
  // Ge
  auto out_ge = vbt_cuda_ge_impl(a, b);
  cudaDeviceSynchronize();
  auto h_ge = copy_cuda_bool_tensor_to_host(out_ge);
  EXPECT_EQ(h_ge[0], 1);
  EXPECT_EQ(h_ge[1], 0);
  EXPECT_EQ(h_ge[2], 1);

#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

static TensorImpl make_cuda_long_tensor_from_host(const std::vector<int64_t>& host) {
#if VBT_WITH_CUDA
  const int dev = 0;
  const std::size_t nbytes = host.size() * sizeof(int64_t);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  cudaError_t st = cudaMemcpy(storage->data(), host.data(), nbytes, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) throw std::runtime_error("cudaMemcpy H2D failed");
  std::vector<int64_t> sizes{static_cast<int64_t>(host.size())};
  std::vector<int64_t> strides{1};
  return TensorImpl(storage, sizes, strides, 0, ScalarType::Int64, Device::cuda(dev));
#else
  (void)host; throw std::runtime_error("CUDA not built");
#endif
}

TEST(CUDAElementwise, SubDivBroadcast) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  // Setup: A=[2, 3], B=[3] -> broadcast to [2, 3]
  // A = [[0, 1, 2], [3, 4, 5]]
  // B = [10, 20, 30]
  // A - B = [[-10, -19, -28], [-7, -16, -25]]
  
  const int rows = 2;
  const int cols = 3;
  std::vector<float> ha = {0, 1, 2, 3, 4, 5};
  std::vector<float> hb = {10, 20, 30};

  const int dev = 0;
  auto st_a = vbt::cuda::new_cuda_storage(ha.size() * sizeof(float), dev);
  cudaError_t st = cudaMemcpy(st_a->data(), ha.data(), ha.size() * sizeof(float), cudaMemcpyHostToDevice);
  ASSERT_EQ(st, cudaSuccess);
  TensorImpl a(st_a, {rows, cols}, {cols, 1}, 0, ScalarType::Float32, Device::cuda(dev));

  TensorImpl b = make_cuda_tensor_from_host(hb); // [3]

  TensorImpl out = vbt_cuda_sub_impl(a, b);
  cudaDeviceSynchronize();
  auto h_out = copy_cuda_tensor_to_host(out);

  ASSERT_EQ(h_out.size(), 6);
  EXPECT_FLOAT_EQ(h_out[0], -10);
  EXPECT_FLOAT_EQ(h_out[1], -19);
  EXPECT_FLOAT_EQ(h_out[2], -28);
  EXPECT_FLOAT_EQ(h_out[3], -7);
  EXPECT_FLOAT_EQ(h_out[4], -16);
  EXPECT_FLOAT_EQ(h_out[5], -25);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, ComparisonBroadcastAndInt64) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();
  
  // Int64 Broadcast case
  // A = [10, 20] (size 2)
  // B = [[10, 20], [10, 21]] (size 2x2)
  // Check Eq
  
  std::vector<int64_t> ha = {10, 20};
  std::vector<int64_t> hb = {10, 20, 10, 21};
  
  TensorImpl a = make_cuda_long_tensor_from_host(ha); // [2]
  
  const int dev = 0;
  auto st_b = vbt::cuda::new_cuda_storage(hb.size() * sizeof(int64_t), dev);
  cudaError_t st = cudaMemcpy(st_b->data(), hb.data(), hb.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  ASSERT_EQ(st, cudaSuccess);
  TensorImpl b(st_b, {2, 2}, {2, 1}, 0, ScalarType::Int64, Device::cuda(dev));

  // Note: a broadcasts to [[10, 20], [10, 20]]
  // b is [[10, 20], [10, 21]]
  // eq -> [[T, T], [T, F]]

  TensorImpl out = vbt_cuda_eq_impl(a, b);
  cudaDeviceSynchronize();
  auto h_out = copy_cuda_bool_tensor_to_host(out); // size 4
  
  EXPECT_EQ(h_out[0], 1);
  EXPECT_EQ(h_out[1], 1);
  EXPECT_EQ(h_out[2], 1);
  EXPECT_EQ(h_out[3], 0);

#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, ScalarDevicePlacementFix) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() < 1) GTEST_SKIP() << "Need at least 1 CUDA device";
  // To strictly test the fix (out allocated on device of tensor, not scalar),
  // we conceptually want scalar=CPU, tensor=CUDA:0.
  // The buggy code used a.device() which would be CPU for a=scalar, b=tensor.
  // We'll construct `out = scalar + tensor`.
  
  float* scalar_val = new float(10.0f);
  auto scalar_storage = vbt::core::make_intrusive<Storage>(
      DataPtr(scalar_val, [](void* p) noexcept { delete static_cast<float*>(p); }),
      sizeof(float));
  TensorImpl scalar_cpu(scalar_storage, {}, {}, 0, ScalarType::Float32, Device::cpu());

  std::vector<float> hb = {1, 2, 3};
  TensorImpl b = make_cuda_tensor_from_host(hb); // CUDA:0

  // Case 1: scalar + tensor -> out should be on CUDA:0
  TensorImpl out = vbt_cuda_add_impl(scalar_cpu, b);
  EXPECT_EQ(out.device().type, kDLCUDA);
  EXPECT_EQ(out.device().index, 0);
  
  cudaDeviceSynchronize();
  auto h_out = copy_cuda_tensor_to_host(out);
  EXPECT_EQ(h_out.size(), 3);
  EXPECT_FLOAT_EQ(h_out[0], 11.0f);

  // Case 2: tensor + scalar -> out should be on CUDA:0 (this was already correct, but good to verify)
  TensorImpl out2 = vbt_cuda_add_impl(b, scalar_cpu);
  EXPECT_EQ(out2.device().type, kDLCUDA);
  EXPECT_EQ(out2.device().index, 0);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, BitwiseAndShift) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  std::vector<int64_t> a_host = {0xF0, 0x0F, 0xAA};
  std::vector<int64_t> b_host = {0xCC, 0x33, 0x55};

  TensorImpl a = make_cuda_long_tensor_from_host(a_host);
  TensorImpl b = make_cuda_long_tensor_from_host(b_host);

  TensorImpl out_and = vbt_cuda_bitwise_and_impl(a, b);
  TensorImpl out_or  = vbt_cuda_bitwise_or_impl(a, b);
  TensorImpl out_xor = vbt_cuda_bitwise_xor_impl(a, b);
  cudaDeviceSynchronize();

  auto h_and = copy_cuda_long_tensor_to_host(out_and);
  auto h_or  = copy_cuda_long_tensor_to_host(out_or);
  auto h_xor = copy_cuda_long_tensor_to_host(out_xor);

  EXPECT_EQ(h_and[0], 0xC0);
  EXPECT_EQ(h_or[1], 0x3F);
  EXPECT_EQ(h_xor[2], 0xFF);

  // Shifts
  std::vector<int64_t> s_in = {1, 2, 4};
  std::vector<int64_t> s_by = {1, 2, 3};
  TensorImpl t_in = make_cuda_long_tensor_from_host(s_in);
  TensorImpl t_by = make_cuda_long_tensor_from_host(s_by);
  TensorImpl out_lsh = vbt_cuda_lshift_impl(t_in, t_by);
  TensorImpl out_rsh = vbt_cuda_rshift_impl(out_lsh, t_by); // invert by shifting back
  cudaDeviceSynchronize();
  auto h_lsh = copy_cuda_long_tensor_to_host(out_lsh);
  auto h_rsh = copy_cuda_long_tensor_to_host(out_rsh);

  EXPECT_EQ(h_lsh[0], 2);   // 1 << 1
  EXPECT_EQ(h_lsh[1], 8);   // 2 << 2
  EXPECT_EQ(h_lsh[2], 32);  // 4 << 3
  EXPECT_EQ(h_rsh[0], 1);   // back to original
  EXPECT_EQ(h_rsh[1], 2);
  EXPECT_EQ(h_rsh[2], 4);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAElementwise, AdvancedArithmeticOps) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  std::vector<float> ha = {5.5f, -4.0f, 0.0f};
  std::vector<float> hb = {2.0f, 3.0f, 1.0f};
  TensorImpl a = make_cuda_tensor_from_host(ha);
  TensorImpl b = make_cuda_tensor_from_host(hb);

  TensorImpl out_fmod = vbt_cuda_fmod_impl(a, b);
  TensorImpl out_rem  = vbt_cuda_remainder_impl(a, b);
  TensorImpl out_atan2 = vbt_cuda_atan2_impl(a, b);
  TensorImpl out_copysign = vbt_cuda_copysign_impl(a, b);
  TensorImpl out_hypot = vbt_cuda_hypot_impl(a, b);
  TensorImpl out_xlogy = vbt_cuda_xlogy_impl(a, b);
  TensorImpl out_xlog1py = vbt_cuda_xlog1py_impl(a, b);
  TensorImpl out_nextafter = vbt_cuda_nextafter_impl(a, b);
  TensorImpl out_heaviside = vbt_cuda_heaviside_impl(a, b);
  cudaDeviceSynchronize();

  auto h_fmod = copy_cuda_tensor_to_host(out_fmod);
  auto h_rem  = copy_cuda_tensor_to_host(out_rem);
  auto h_atan2 = copy_cuda_tensor_to_host(out_atan2);
  auto h_copysign = copy_cuda_tensor_to_host(out_copysign);
  auto h_hypot = copy_cuda_tensor_to_host(out_hypot);
  auto h_xlogy = copy_cuda_tensor_to_host(out_xlogy);
  auto h_xlog1py = copy_cuda_tensor_to_host(out_xlog1py);
  auto h_nextafter = copy_cuda_tensor_to_host(out_nextafter);
  auto h_heaviside = copy_cuda_tensor_to_host(out_heaviside);

  auto expect_close = [](float a, float b) {
    EXPECT_NEAR(a, b, 1e-5f);
  };

  expect_close(h_fmod[0], std::fmod(5.5f, 2.0f));
  // torch.remainder semantics (modulus with sign of divisor), not IEEE remainder().
  {
    const float x = -4.0f;
    const float y = 3.0f;
    float r = std::fmod(x, y);
    if (r != 0.0f && ((r < 0.0f) != (y < 0.0f))) r += y;
    expect_close(h_rem[1], r);
  }
  expect_close(h_atan2[0], std::atan2(5.5f, 2.0f));
  expect_close(h_copysign[1], std::copysign(-4.0f, 3.0f));
  expect_close(h_hypot[0], std::hypot(5.5f, 2.0f));
  expect_close(h_xlogy[0], 5.5f * std::log(2.0f));
  expect_close(h_xlog1py[0], 5.5f * std::log1p(2.0f));
  // nextafter(5.5, 2.0) is smaller than 5.5
  EXPECT_LT(h_nextafter[0], ha[0]);
  
  // Heaviside:
  // ha[0]=5.5 (pos) -> 1.0
  // ha[1]=-4.0 (neg) -> 0.0
  // ha[2]=0.0 (zero) -> hb[2]=1.0
  expect_close(h_heaviside[0], 1.0f);
  expect_close(h_heaviside[1], 0.0f);
  expect_close(h_heaviside[2], 1.0f);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(CUDAUnary, BasicUnaryOps) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";
  vbt_register_default_kernels();
  vbt_register_cuda_elementwise_kernels();

  std::vector<float> ha = {-2.5f, -1.0f, 0.5f, 2.0f};
  TensorImpl a = make_cuda_tensor_from_host(ha);

  auto run_unary = [&](auto fn) {
    TensorImpl out = fn(a);
    cudaDeviceSynchronize();
    return copy_cuda_tensor_to_host(out);
  };

  auto h_abs = run_unary(vbt_cuda_abs_impl);
  auto h_neg = run_unary(vbt_cuda_neg_impl);
  auto h_exp = run_unary(vbt_cuda_exp_impl);
  auto h_log = run_unary(vbt_cuda_log_impl);
  auto h_sqrt = run_unary(vbt_cuda_sqrt_impl);
  auto h_rsqrt = run_unary(vbt_cuda_rsqrt_impl);
  auto h_sin = run_unary(vbt_cuda_sin_impl);
  auto h_cos = run_unary(vbt_cuda_cos_impl);
  auto h_tanh = run_unary(vbt_cuda_tanh_impl);
  auto h_sigmoid = run_unary(vbt_cuda_sigmoid_impl);
  auto h_expm1 = run_unary(vbt_cuda_expm1_impl);
  auto h_log1p = run_unary(vbt_cuda_log1p_impl);
  auto h_floor = run_unary(vbt_cuda_floor_impl);
  auto h_ceil = run_unary(vbt_cuda_ceil_impl);
  auto h_trunc = run_unary(vbt_cuda_trunc_impl);
  auto h_round = run_unary(vbt_cuda_round_impl);
  auto h_frac = run_unary(vbt_cuda_frac_impl);
  auto h_reciprocal = run_unary(vbt_cuda_reciprocal_impl);
  auto h_sign = run_unary(vbt_cuda_sign_impl);
  auto h_exp2 = run_unary(vbt_cuda_exp2_impl);
  auto h_log2 = run_unary(vbt_cuda_log2_impl);
  auto h_sinh = run_unary(vbt_cuda_sinh_impl);
  auto h_cosh = run_unary(vbt_cuda_cosh_impl);
  auto h_tan = run_unary(vbt_cuda_tan_impl);
  auto h_asin = run_unary(vbt_cuda_asin_impl);
  auto h_acos = run_unary(vbt_cuda_acos_impl);
  auto h_atan = run_unary(vbt_cuda_atan_impl);

  auto expect_close = [](float a, float b) {
    EXPECT_NEAR(a, b, 1e-5f);
  };

  for (std::size_t i = 0; i < ha.size(); ++i) {
    float x = ha[i];
    expect_close(h_abs[i], std::fabs(x));
    expect_close(h_neg[i], -x);
    expect_close(h_exp[i], std::exp(x));
    if (x > 0) expect_close(h_log[i], std::log(x));
    if (x >= 0.0f) {
      expect_close(h_sqrt[i], std::sqrt(x));
    } else {
      EXPECT_TRUE(std::isnan(h_sqrt[i]));
    }
    if (x > 0) expect_close(h_rsqrt[i], 1.0f / std::sqrt(x));
    expect_close(h_sin[i], std::sin(x));
    expect_close(h_cos[i], std::cos(x));
    expect_close(h_tanh[i], std::tanh(x));
    expect_close(h_sigmoid[i], 1.0f / (1.0f + std::exp(-x)));
    expect_close(h_expm1[i], std::expm1(x));
    if (x > -1.0f) expect_close(h_log1p[i], std::log1p(x));
    expect_close(h_floor[i], std::floor(x));
    expect_close(h_ceil[i], std::ceil(x));
    expect_close(h_trunc[i], std::trunc(x));
    expect_close(h_round[i], std::nearbyint(x));
    double ipart = 0.0;
    double frac = std::modf(static_cast<double>(x), &ipart);
    expect_close(h_frac[i], static_cast<float>(frac));
    if (x != 0.0f) expect_close(h_reciprocal[i], 1.0f / x);
    expect_close(h_sign[i], (x > 0) - (x < 0));
    expect_close(h_exp2[i], std::exp2(x));
    if (x > 0) expect_close(h_log2[i], std::log2(x));
    expect_close(h_sinh[i], std::sinh(x));
    expect_close(h_cosh[i], std::cosh(x));
    expect_close(h_tan[i], std::tan(x));
    if (x >= -1.0f && x <= 1.0f) {
      expect_close(h_asin[i], std::asin(x));
      expect_close(h_acos[i], std::acos(x));
    }
    expect_close(h_atan[i], std::atan(x));
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}
