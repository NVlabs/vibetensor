// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <string>

#include "vbt/core/indexing.h"
#include "vbt/core/indexing/index_errors.h"
#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/engine.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/wrapper.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#include <cuda_runtime_api.h>
#endif

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::indexing::IndexSpec;
using vbt::core::indexing::TensorIndex;
using vbt::core::indexing::advanced_indexing_enabled;
using vbt::core::indexing::set_advanced_indexing_enabled_for_tests;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;
namespace idx_errors = vbt::core::indexing::errors;

extern "C" void vbt_register_default_kernels();
extern "C" void vbt_register_indexing_kernels();

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

struct AdvancedIndexingGuard {
  bool prev;
  explicit AdvancedIndexingGuard(bool enabled)
      : prev(advanced_indexing_enabled()) {
    set_advanced_indexing_enabled_for_tests(enabled);
  }
  ~AdvancedIndexingGuard() {
    set_advanced_indexing_enabled_for_tests(prev);
  }
};

struct CudaAutogradGuard {
  bool prev;
  explicit CudaAutogradGuard(bool enabled)
      : prev(vbt::autograd::is_streaming_backwards_enabled()) {
    vbt::autograd::set_streaming_backwards_enabled(enabled);
  }
  ~CudaAutogradGuard() {
    vbt::autograd::set_streaming_backwards_enabled(prev);
  }
};

static TensorImpl make_ones_like_f32(const TensorImpl& like) {
  const auto& sizes = like.sizes();
  std::vector<std::int64_t> sizes_vec(sizes.begin(), sizes.end());
  std::size_t ne = 1;
  for (std::int64_t s : sizes_vec) {
    ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  }
  std::size_t nbytes = ne * sizeof(float);
  auto storage = make_storage_bytes(nbytes);
  std::vector<std::int64_t> strides(sizes_vec.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes_vec.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes_vec[idx];
    acc *= (sz == 0 ? 1 : sz);
  }
  TensorImpl out(storage, sizes_vec, strides, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(out.data());
  for (std::size_t i = 0; i < ne; ++i) {
    p[i] = 1.0f;
  }
  return out;
}

static TensorImpl make_zeros_like_f32(const TensorImpl& like) {
  const auto& sizes = like.sizes();
  std::vector<std::int64_t> sizes_vec(sizes.begin(), sizes.end());
  std::size_t ne = 1;
  for (std::int64_t s : sizes_vec) {
    ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  }
  std::size_t nbytes = ne * sizeof(float);
  auto storage = make_storage_bytes(nbytes);
  std::vector<std::int64_t> strides(sizes_vec.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes_vec.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes_vec[idx];
    acc *= (sz == 0 ? 1 : sz);
  }
  TensorImpl out(storage, sizes_vec, strides, /*storage_offset=*/0,
                 ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(out.data());
  for (std::size_t i = 0; i < ne; ++i) {
    p[i] = 0.0f;
  }
  return out;
}

TEST(IndexingAdvancedEntryPointsAutogradTest, VtIndexGradientMatchesScatter) {
  AdvancedIndexingGuard flag_guard(true);

  vbt_register_default_kernels();
  vbt_register_indexing_kernels();
  (void)vbt::autograd::register_autograd_fallbacks();

  // Base tensor x: 1-D Float32 of length 5 with values 0..4.
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl x(storage, sizes, strides, /*storage_offset=*/0,
               ScalarType::Float32, Device::cpu());
  auto* x_data = static_cast<float*>(x.data());
  for (int i = 0; i < 5; ++i) {
    x_data[i] = static_cast<float>(i);
  }

  vbt::autograd::set_requires_grad(x, true);

  // Index tensor: [0, 3, 4].
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 0;
  idx_data[1] = 3;
  idx_data[2] = 4;

  // Meta tensor mirroring encode_index_meta_tensor() layout.
  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes = static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 0;  // version
  m[1] = 1;  // advanced_kind = Tensor
  m[2] = 0;  // advanced_param
  m[3] = 0;  // prefix_len

  BoxedStack stack;
  stack.push_back(x);
  stack.push_back(idx);
  stack.push_back(meta);

  // Ensure GradMode is enabled so autograd fallback is active.
  vbt::autograd::GradMode::set_enabled(true);

  Dispatcher::instance().callBoxed("vt::index", stack);
  ASSERT_EQ(stack.size(), 1u);
  TensorImpl y = stack[0];

  // Seed gradient of ones like y and run backward through the graph rooted
  // at y's grad_fn (installed by the autograd fallback).
  TensorImpl grad_out = make_ones_like_f32(y);

  auto* meta_y = vbt::autograd::get_autograd_meta(y, /*create_if_missing=*/false);
  ASSERT_NE(meta_y, nullptr);
  ASSERT_TRUE(static_cast<bool>(meta_y->grad_fn));
  auto root = meta_y->grad_fn;

  std::vector<vbt::autograd::OptionalTensor> seeds;
  seeds.resize(static_cast<std::size_t>(root->num_inputs()));
  const std::size_t slot = static_cast<std::size_t>(meta_y->output_nr);
  if (slot < seeds.size()) {
    seeds[slot] = grad_out;
  }

  vbt::autograd::run_backward(root, seeds, {});

  auto* meta_x = vbt::autograd::get_autograd_meta(x, /*create_if_missing=*/false);
  ASSERT_NE(meta_x, nullptr);
  ASSERT_TRUE(meta_x->grad_ptr != nullptr);
  ASSERT_TRUE(meta_x->grad_has);
  const TensorImpl& grad_x = *(meta_x->grad_ptr);

  // Reference gradient: zeros_like(x) with index_put_(..., accumulate=true).
  TensorImpl ref = make_zeros_like_f32(x);
  IndexSpec spec;
  spec.items.emplace_back(TensorIndex(idx));
  vbt::core::indexing::index_put_(ref, spec, grad_out, /*accumulate=*/true);

  ASSERT_EQ(grad_x.sizes().size(), ref.sizes().size());
  for (std::size_t i = 0; i < grad_x.sizes().size(); ++i) {
    EXPECT_EQ(grad_x.sizes()[i], ref.sizes()[i]);
  }

  const float* gx = static_cast<const float*>(grad_x.data());
  const float* gr = static_cast<const float*>(ref.data());
  const std::size_t ne = static_cast<std::size_t>(grad_x.numel());
  for (std::size_t i = 0; i < ne; ++i) {
    EXPECT_FLOAT_EQ(gx[i], gr[i]);
  }
}

TEST(IndexingAdvancedEntryPointsAutogradTest, VtIndexBackwardHonorsFeatureFlag) {
  AdvancedIndexingGuard flag_guard(true);

  vbt_register_default_kernels();
  vbt_register_indexing_kernels();
  (void)vbt::autograd::register_autograd_fallbacks();

  // Simple 1-D Float32 tensor of length 3.
  const std::vector<std::int64_t> sizes{3};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl x(storage, sizes, strides, /*storage_offset=*/0,
               ScalarType::Float32, Device::cpu());

  vbt::autograd::set_requires_grad(x, true);

  // Index tensor [0, 2].
  const std::vector<std::int64_t> idx_sizes{2};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes = static_cast<std::size_t>(2 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl idx(idx_storage, idx_sizes, idx_strides, 0,
                 ScalarType::Int64, Device::cpu());

  auto* idx_data = static_cast<std::int64_t*>(idx.data());
  idx_data[0] = 0;
  idx_data[1] = 2;

  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes = static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 0;
  m[1] = 1;
  m[2] = 0;
  m[3] = 0;

  BoxedStack stack;
  stack.push_back(x);
  stack.push_back(idx);
  stack.push_back(meta);

  vbt::autograd::GradMode::set_enabled(true);

  Dispatcher::instance().callBoxed("vt::index", stack);
  ASSERT_EQ(stack.size(), 1u);
  TensorImpl y = stack[0];

  TensorImpl grad_out = make_ones_like_f32(y);

  auto* meta_y = vbt::autograd::get_autograd_meta(y, /*create_if_missing=*/false);
  ASSERT_NE(meta_y, nullptr);
  ASSERT_TRUE(static_cast<bool>(meta_y->grad_fn));
  auto root = meta_y->grad_fn;

  std::vector<vbt::autograd::OptionalTensor> seeds;
  seeds.resize(static_cast<std::size_t>(root->num_inputs()));
  const std::size_t slot = static_cast<std::size_t>(meta_y->output_nr);
  if (slot < seeds.size()) {
    seeds[slot] = grad_out;
  }

  // Disable advanced indexing before backward; IndexBackwardNode should
  // surface the "advanced indexing disabled" runtime error.
  AdvancedIndexingGuard guard(false);
  try {
    vbt::autograd::run_backward(root, seeds, {});
    FAIL() << "expected runtime_error when advanced indexing is disabled in backward";
  } catch (const std::runtime_error& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find("advanced indexing disabled"), std::string::npos);
  }
}

TEST(IndexingAdvancedEntryPointsAutogradTest, VtIndexGradientMatchesScatterCudaFloat32) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  AdvancedIndexingGuard flag_guard(true);
  CudaAutogradGuard cuda_guard(true);

  vbt_register_default_kernels();
  vbt_register_indexing_kernels();
  (void)vbt::autograd::register_autograd_fallbacks();

  // Base tensor x: 1-D Float32 of length 5 with values 0..4.
  const std::vector<std::int64_t> sizes{5};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(5 * sizeof(float));
  auto storage_cpu = make_storage_bytes(nbytes);
  TensorImpl x_cpu(storage_cpu, sizes, strides, /*storage_offset=*/0,
                   ScalarType::Float32, Device::cpu());
  auto* x_cpu_data = static_cast<float*>(x_cpu.data());
  for (int i = 0; i < 5; ++i) {
    x_cpu_data[i] = static_cast<float>(i);
  }

  vbt::autograd::set_requires_grad(x_cpu, true);

  // CUDA copy of x on device 0.
  int dev = 0;
  auto storage_cuda = vbt::cuda::new_cuda_storage(nbytes, dev);
  TensorImpl x_cuda(storage_cuda, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        x_cuda.data(), x_cpu.data(), nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for x failed";
  }

  vbt::autograd::set_requires_grad(x_cuda, true);

  // Index tensor: [0, 3, 4].
  const std::vector<std::int64_t> idx_sizes{3};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes =
      static_cast<std::size_t>(3 * sizeof(std::int64_t));
  auto idx_storage_cpu = make_storage_bytes(idx_nbytes);
  TensorImpl idx_cpu(idx_storage_cpu, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(idx_cpu.data());
  idx_data[0] = 0;
  idx_data[1] = 3;
  idx_data[2] = 4;

  auto idx_storage_cuda = vbt::cuda::new_cuda_storage(idx_nbytes, dev);
  TensorImpl idx_cuda(idx_storage_cuda, idx_sizes, idx_strides, 0,
                      ScalarType::Int64, Device::cuda(dev));
  {
    cudaError_t st = cudaMemcpy(
        idx_cuda.data(), idx_cpu.data(), idx_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for idx failed";
  }

  // Meta tensor mirroring encode_index_meta_tensor() layout.
  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes =
      static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 0;
  m[1] = 1;
  m[2] = 0;
  m[3] = 0;

  BoxedStack stack;
  stack.push_back(x_cuda);
  stack.push_back(idx_cuda);
  stack.push_back(meta);

  vbt::autograd::GradMode::set_enabled(true);

  Dispatcher::instance().callBoxed("vt::index", stack);
  ASSERT_EQ(stack.size(), 1u);
  TensorImpl y = stack[0];

  // Seed gradient of ones like y on CUDA.
  const auto& y_sizes = y.sizes();
  std::vector<std::int64_t> y_sizes_vec(y_sizes.begin(), y_sizes.end());
  std::size_t ne = 1;
  for (std::int64_t s : y_sizes_vec) {
    ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  }
  const std::size_t grad_nbytes = ne * sizeof(float);
  auto grad_storage_cuda = vbt::cuda::new_cuda_storage(grad_nbytes, dev);

  std::vector<std::int64_t> grad_strides(y_sizes_vec.size(), 0);
  std::int64_t acc = 1;
  for (std::ptrdiff_t i =
           static_cast<std::ptrdiff_t>(y_sizes_vec.size()) - 1;
       i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    grad_strides[idx] = acc;
    const auto sz = y_sizes_vec[idx];
    acc *= (sz == 0 ? 1 : sz);
  }

  TensorImpl grad_out(grad_storage_cuda, y_sizes_vec, grad_strides,
                      /*storage_offset=*/0,
                      ScalarType::Float32,
                      Device::cuda(dev));
  if (grad_nbytes > 0) {
    std::vector<float> host_grad(ne, 1.0f);
    cudaError_t st = cudaMemcpy(
        grad_out.data(), host_grad.data(), grad_nbytes,
        cudaMemcpyHostToDevice);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy H2D for grad_out failed";
  }

  auto* meta_y2 = vbt::autograd::get_autograd_meta(y, /*create_if_missing=*/false);
  ASSERT_NE(meta_y2, nullptr);
  ASSERT_TRUE(static_cast<bool>(meta_y2->grad_fn));
  auto root = meta_y2->grad_fn;

  std::vector<vbt::autograd::OptionalTensor> seeds;
  seeds.resize(static_cast<std::size_t>(root->num_inputs()));
  const std::size_t slot2 = static_cast<std::size_t>(meta_y2->output_nr);
  if (slot2 < seeds.size()) {
    seeds[slot2] = grad_out;
  }

  vbt::autograd::run_backward(root, seeds, {});

  auto* meta_x = vbt::autograd::get_autograd_meta(x_cuda, /*create_if_missing=*/false);
  ASSERT_NE(meta_x, nullptr);
  ASSERT_TRUE(meta_x->grad_ptr != nullptr);
  ASSERT_TRUE(meta_x->grad_has);
  const TensorImpl& grad_x = *(meta_x->grad_ptr);

  // Reference gradient computed on CPU via index_put_(..., accumulate=true).
  TensorImpl ref = make_zeros_like_f32(x_cpu);
  IndexSpec spec_cpu;
  spec_cpu.items.emplace_back(TensorIndex(idx_cpu));
  TensorImpl grad_out_cpu = make_ones_like_f32(vbt::core::indexing::index(x_cpu, spec_cpu));
  vbt::core::indexing::index_put_(ref, spec_cpu, grad_out_cpu, /*accumulate=*/true);

  ASSERT_EQ(grad_x.sizes().size(), ref.sizes().size());
  for (std::size_t i = 0; i < grad_x.sizes().size(); ++i) {
    EXPECT_EQ(grad_x.sizes()[i], ref.sizes()[i]);
  }

  const std::size_t grad_ne = static_cast<std::size_t>(grad_x.numel());
  const std::size_t grad_bytes = grad_ne * sizeof(float);
  std::vector<float> grad_x_host(grad_ne, 0.0f);
  {
    cudaError_t st = cudaMemcpy(
        grad_x_host.data(), grad_x.data(), grad_bytes,
        cudaMemcpyDeviceToHost);
    ASSERT_EQ(st, cudaSuccess) << "cudaMemcpy D2H for grad_x failed";
  }

  const float* ref_data = static_cast<const float*>(ref.data());
  for (std::size_t i = 0; i < grad_ne; ++i) {
    EXPECT_NEAR(grad_x_host[i], ref_data[i], 1e-5f);
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

#if VBT_WITH_AUTOGRAD
TEST(IndexingAdvancedEntryPointsAutogradTest, VtIndexPutAutogradPolicyUsesCanonicalSubstring) {
  AdvancedIndexingGuard flag_guard(true);

  vbt_register_default_kernels();
  vbt_register_indexing_kernels();
  (void)vbt::autograd::register_autograd_fallbacks();

  // Base tensor: 1-D Float32 of length 3 on CPU.
  const std::vector<std::int64_t> sizes{3};
  const std::vector<std::int64_t> strides{1};
  const std::size_t nbytes = static_cast<std::size_t>(3 * sizeof(float));
  auto storage = make_storage_bytes(nbytes);
  TensorImpl self(storage, sizes, strides, /*storage_offset=*/0,
                  ScalarType::Float32, Device::cpu());
  auto* self_data = static_cast<float*>(self.data());
  for (int i = 0; i < 3; ++i) {
    self_data[i] = 0.0f;
  }

  // Enable autograd and mark self as requiring grad.
  vbt::autograd::set_requires_grad(self, true);

  // Index tensor [0, 2].
  const std::vector<std::int64_t> idx_sizes{2};
  const std::vector<std::int64_t> idx_strides{1};
  const std::size_t idx_nbytes =
      static_cast<std::size_t>(2 * sizeof(std::int64_t));
  auto idx_storage = make_storage_bytes(idx_nbytes);
  TensorImpl index_t(idx_storage, idx_sizes, idx_strides, 0,
                     ScalarType::Int64, Device::cpu());
  auto* idx_data = static_cast<std::int64_t*>(index_t.data());
  idx_data[0] = 0;
  idx_data[1] = 2;

  // Value tensor [1.0, 2.0].
  const std::vector<std::int64_t> val_sizes{2};
  const std::vector<std::int64_t> val_strides{1};
  const std::size_t val_nbytes =
      static_cast<std::size_t>(2 * sizeof(float));
  auto val_storage = make_storage_bytes(val_nbytes);
  TensorImpl value(val_storage, val_sizes, val_strides, 0,
                   ScalarType::Float32, Device::cpu());
  auto* val_data = static_cast<float*>(value.data());
  val_data[0] = 1.0f;
  val_data[1] = 2.0f;

  // Meta tensor describing a single tensor advanced index (meta v0).
  const std::vector<std::int64_t> meta_sizes{4};
  const std::vector<std::int64_t> meta_strides{1};
  const std::size_t meta_nbytes =
      static_cast<std::size_t>(4 * sizeof(std::int64_t));
  auto meta_storage = make_storage_bytes(meta_nbytes);
  TensorImpl meta(meta_storage, meta_sizes, meta_strides, 0,
                  ScalarType::Int64, Device::cpu());
  auto* m = static_cast<std::int64_t*>(meta.data());
  m[0] = 0;  // version
  m[1] = 1;  // adv_kind = Tensor
  m[2] = 0;  // adv_param
  m[3] = 0;  // prefix_len

  // 0-d Bool CPU tensor accumulate=False.
  const std::size_t acc_nbytes = sizeof(std::uint8_t);
  auto acc_storage = make_storage_bytes(acc_nbytes);
  std::vector<std::int64_t> acc_sizes;    // 0-d
  std::vector<std::int64_t> acc_strides;  // empty for 0-d
  TensorImpl acc(acc_storage, acc_sizes, acc_strides, 0,
                 ScalarType::Bool, Device::cpu());
  auto* acc_data = static_cast<std::uint8_t*>(acc.data());
  if (acc_data) *acc_data = 0u;

  BoxedStack stack;
  stack.push_back(self);
  stack.push_back(index_t);
  stack.push_back(value);
  stack.push_back(meta);
  stack.push_back(acc);

  // Ensure GradMode is enabled so vt::index_put sees autograd as active.
  vbt::autograd::GradMode::set_enabled(true);

  try {
    Dispatcher::instance().callBoxed("vt::index_put", stack);
    FAIL() << "expected std::runtime_error for vt::index_put autograd policy";
  } catch (const std::runtime_error& ex) {
    const std::string msg = ex.what();
    EXPECT_NE(msg.find(idx_errors::kErrIndexPutAutogradUnsupported),
              std::string::npos);
  }
}
#endif
