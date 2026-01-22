// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/core/tensor_iterator/core.h"
#include "vbt/dispatch/plugin_loader.h"
#include "vbt/plugin/vbt_plugin.h"

#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#endif

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_contiguous_tensor(const std::vector<int64_t>& sizes) {
  const std::size_t nd = sizes.size();
  std::vector<int64_t> strides(nd, 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    const auto sz = sizes[static_cast<std::size_t>(i)];
    acc *= (sz == 0 ? 1 : sz);
  }

  int64_t ne = 1;
  bool any_zero = false;
  for (auto s : sizes) {
    if (s == 0) {
      any_zero = true;
      break;
    }
    ne *= s;
  }
  if (any_zero) {
    ne = 0;
  }

  const std::size_t nbytes = static_cast<std::size_t>(ne) * sizeof(float);
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cpu());
}

#if VBT_WITH_CUDA
static TensorImpl make_contiguous_tensor_cuda(const std::vector<int64_t>& sizes,
                                              int dev = 0) {
  const std::size_t nd = sizes.size();
  std::vector<int64_t> strides(nd, 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = acc;
    const auto sz = sizes[static_cast<std::size_t>(i)];
    acc *= (sz == 0 ? 1 : sz);
  }

  int64_t ne = 1;
  bool any_zero = false;
  for (auto s : sizes) {
    if (s == 0) {
      any_zero = true;
      break;
    }
    ne *= s;
  }
  if (any_zero) {
    ne = 0;
  }

  const std::size_t nbytes = static_cast<std::size_t>(ne) * sizeof(float);
  auto storage = vbt::cuda::new_cuda_storage(nbytes, dev);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0,
                    ScalarType::Float32, Device::cuda(dev));
}
#endif

struct TriAddCtx {
  int64_t tiles{0};
};

static void tri_add_loop(char** data,
                         const int64_t* strides,
                         int64_t size,
                         void* ctx_void) {
  auto* ctx = static_cast<TriAddCtx*>(ctx_void);
  ++ctx->tiles;

  char* out_base = data[0];
  char* a_base   = data[1];
  char* b_base   = data[2];

  const int64_t out_stride = strides[0];
  const int64_t a_stride   = strides[1];
  const int64_t b_stride   = strides[2];

  for (int64_t i = 0; i < size; ++i) {
    auto* out = reinterpret_cast<float*>(out_base + i * out_stride);
    const auto* a = reinterpret_cast<const float*>(a_base + i * a_stride);
    const auto* b = reinterpret_cast<const float*>(b_base + i * b_stride);
    *out = *a + *b;
  }
}

}  // namespace

TEST(TensorIterPluginHandleTest, ElementwiseCpuBuildAndForEach) {
  using vbt::dispatch::plugin::get_last_error;
  using vbt::dispatch::plugin::set_last_error;

  auto out = make_contiguous_tensor({8});
  auto a   = make_contiguous_tensor({8});
  auto b   = make_contiguous_tensor({8});

  auto* out_data = static_cast<float*>(out.data());
  auto* a_data   = static_cast<float*>(a.data());
  auto* b_data   = static_cast<float*>(b.data());

  for (int i = 0; i < 8; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(2 * i);
    out_data[i] = 0.0f;
  }

  vt_tensor out_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out);
  vt_tensor a_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(a);
  vt_tensor b_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(b);

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;
  vt_tensor_iter iter = NULL;

  set_last_error("preset");
  vt_tensor tensors[3] = {out_h, a_h, b_h};
  vt_status st = vt_tensor_iter_build_elementwise(&cfg,
                                                  /*ntensors=*/3,
                                                  tensors,
                                                  &iter);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_NE(iter, (vt_tensor_iter)NULL);
  EXPECT_STREQ(get_last_error(), "");

  TriAddCtx ctx{};
  set_last_error("preset");
  st = vt_tensor_iter_for_each_cpu(iter, &tri_add_loop, &ctx);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_STREQ(get_last_error(), "");
  EXPECT_GE(ctx.tiles, 1);

  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(out_data[i], a_data[i] + b_data[i]);
  }

  vt_tensor_iter_destroy(iter);

  // Zero-numel iterator: callback must not be invoked.
  auto out_zero = make_contiguous_tensor({2, 0, 3});
  auto a_zero   = make_contiguous_tensor({2, 0, 3});
  auto b_zero   = make_contiguous_tensor({2, 0, 3});

  vt_tensor out_zero_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out_zero);
  vt_tensor a_zero_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(a_zero);
  vt_tensor b_zero_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(b_zero);

  vt_tensor tensors_zero[3] = {out_zero_h, a_zero_h, b_zero_h};
  iter = NULL;
  ctx.tiles = 0;

  set_last_error("preset");
  st = vt_tensor_iter_build_elementwise(&cfg,
                                        /*ntensors=*/3,
                                        tensors_zero,
                                        &iter);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_NE(iter, (vt_tensor_iter)NULL);
  EXPECT_STREQ(get_last_error(), "");

  set_last_error("preset");
  st = vt_tensor_iter_for_each_cpu(iter, &tri_add_loop, &ctx);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_STREQ(get_last_error(), "");
  EXPECT_EQ(ctx.tiles, 0);

  vt_tensor_iter_destroy(iter);

  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(a_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(b_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_zero_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(a_zero_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(b_zero_h);
}

TEST(TensorIterPluginHandleTest, ElementwiseBadArgsAndTls) {
  using vbt::dispatch::plugin::get_last_error;
  using vbt::dispatch::plugin::set_last_error;

  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});

  vt_tensor out_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out);
  vt_tensor a_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(a);

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;
  vt_tensor_iter iter = NULL;

  // ntensors < 2 should yield INVALID_ARG.
  set_last_error("");
  vt_status st = vt_tensor_iter_build_elementwise(&cfg,
                                                  /*ntensors=*/1,
                                                  &out_h,
                                                  &iter);
  EXPECT_EQ(st, VT_STATUS_INVALID_ARG);
  std::string err = get_last_error();
  EXPECT_FALSE(err.empty());

  // max_rank out of range should yield INVALID_ARG.
  vt_iter_config bad_cfg = cfg;
  bad_cfg.max_rank = -1;
  set_last_error("");
  vt_tensor tensors[2] = {out_h, a_h};
  st = vt_tensor_iter_build_elementwise(&bad_cfg,
                                        /*ntensors=*/2,
                                        tensors,
                                        &iter);
  EXPECT_EQ(st, VT_STATUS_INVALID_ARG);
  err = get_last_error();
  EXPECT_FALSE(err.empty());

  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(a_h);
}

#if VBT_WITH_CUDA
TEST(TensorIterPluginHandleTest, CudaExportCudaDescBasic) {
  using vbt::dispatch::plugin::get_last_error;
  using vbt::dispatch::plugin::set_last_error;

  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;
  const std::int64_t N = 16;
  const std::size_t nbytes = static_cast<std::size_t>(N) * sizeof(float);

  auto st_out = vbt::cuda::new_cuda_storage(nbytes, dev);
  auto st_a   = vbt::cuda::new_cuda_storage(nbytes, dev);
  auto st_b   = vbt::cuda::new_cuda_storage(nbytes, dev);

  TensorImpl out(st_out, {N}, {1}, 0, ScalarType::Float32, Device::cuda(dev));
  TensorImpl a(st_a, {N}, {1}, 0, ScalarType::Float32, Device::cuda(dev));
  TensorImpl b(st_b, {N}, {1}, 0, ScalarType::Float32, Device::cuda(dev));

  vt_tensor out_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out);
  vt_tensor a_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(a);
  vt_tensor b_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(b);

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;
  vt_tensor_iter iter = NULL;

  vt_tensor tensors[3] = {out_h, a_h, b_h};
  set_last_error("");
  vt_status st = vt_tensor_iter_build_elementwise(&cfg,
                                                  /*ntensors=*/3,
                                                  tensors,
                                                  &iter);
  ASSERT_EQ(st, VT_STATUS_OK);
  ASSERT_NE(iter, (vt_tensor_iter)NULL);

  vt_tensor_iter_cuda_desc desc_out{};
  vt_tensor_iter_cuda_desc desc_a{};
  vt_tensor_iter_cuda_desc desc_b{};

  set_last_error("");
  st = vt_tensor_iter_export_cuda_desc(iter,
                                       /*operand_index=*/0,
                                       VT_TENSOR_ITER_CUDA_MAX_NDIM,
                                       &desc_out);
  EXPECT_EQ(st, VT_STATUS_OK);

  st = vt_tensor_iter_export_cuda_desc(iter,
                                       /*operand_index=*/1,
                                       VT_TENSOR_ITER_CUDA_MAX_NDIM,
                                       &desc_a);
  EXPECT_EQ(st, VT_STATUS_OK);

  st = vt_tensor_iter_export_cuda_desc(iter,
                                       /*operand_index=*/2,
                                       VT_TENSOR_ITER_CUDA_MAX_NDIM,
                                       &desc_b);
  EXPECT_EQ(st, VT_STATUS_OK);

  EXPECT_EQ(desc_out.ndim, 1);
  EXPECT_EQ(desc_a.ndim, 1);
  EXPECT_EQ(desc_b.ndim, 1);

  EXPECT_EQ(desc_out.sizes[0], N);
  EXPECT_EQ(desc_a.sizes[0], N);
  EXPECT_EQ(desc_b.sizes[0], N);

  EXPECT_EQ(desc_out.strides[0], 1);
  EXPECT_EQ(desc_a.strides[0], 1);
  EXPECT_EQ(desc_b.strides[0], 1);

  vt_tensor_iter_destroy(iter);

  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(a_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(b_h);
}

TEST(TensorIterPluginHandleTest, CudaExportCudaDesc25DAnd26DBehavior) {
  using vbt::dispatch::plugin::get_last_error;
  using vbt::dispatch::plugin::set_last_error;

  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }

  const int dev = 0;
  const int rank25 = VT_TENSOR_ITER_CUDA_MAX_NDIM;

  // 25D: succeed and report ndim == rank25.
  // Use a shape where all but the leading dimension are 1 so that
  // a 26D output with an extra leading dimension remains
  // broadcastable under right-aligned rules.
  std::vector<int64_t> sizes25(static_cast<std::size_t>(rank25), 1);
  sizes25[0] = 3;

  TensorImpl out25 = make_contiguous_tensor_cuda(sizes25, dev);
  TensorImpl a25   = make_contiguous_tensor_cuda(sizes25, dev);

  vt_tensor out25_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out25);
  vt_tensor a25_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(a25);

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;
  vt_tensor_iter iter25 = NULL;

  vt_tensor tensors25[2] = {out25_h, a25_h};
  set_last_error("");
  vt_status st = vt_tensor_iter_build_elementwise(&cfg,
                                                  /*ntensors=*/2,
                                                  tensors25,
                                                  &iter25);
  ASSERT_EQ(st, VT_STATUS_OK);
  ASSERT_NE(iter25, (vt_tensor_iter)NULL);

  vt_tensor_iter_cuda_desc desc25{};
  std::memset(&desc25, 0, sizeof(desc25));
  set_last_error("");
  st = vt_tensor_iter_export_cuda_desc(iter25,
                                       /*operand_index=*/0,
                                       VT_TENSOR_ITER_CUDA_MAX_NDIM,
                                       &desc25);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_STREQ(get_last_error(), "");
  EXPECT_EQ(desc25.ndim, rank25);
  EXPECT_EQ(desc25.sizes[0], sizes25[0]);
  EXPECT_EQ(desc25.sizes[1], sizes25[1]);

  // 26D: UNSUPPORTED and out_desc must be left unchanged.
  std::vector<int64_t> sizes26(static_cast<std::size_t>(rank25 + 1), 1);
  // Insert an extra leading dimension so that sizes26 broadcasts with
  // sizes25 under PyTorch-style right-aligned rules: sizes25 is exactly
  // sizes26 with the leading dimension dropped.
  sizes26[0] = 2;
  for (int i = 0; i < rank25; ++i) {
    sizes26[i + 1] = sizes25[static_cast<std::size_t>(i)];
  }

  TensorImpl out26 = make_contiguous_tensor_cuda(sizes26, dev);
  vt_tensor out26_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out26);

  vt_tensor tensors26[2] = {out26_h, a25_h};
  vt_tensor_iter iter26 = NULL;
  set_last_error("");
  st = vt_tensor_iter_build_elementwise(&cfg,
                                        /*ntensors=*/2,
                                        tensors26,
                                        &iter26);
  ASSERT_EQ(st, VT_STATUS_OK);
  ASSERT_NE(iter26, (vt_tensor_iter)NULL);

  vt_tensor_iter_cuda_desc desc26{};
  desc26.ndim = 123;  // sentinel to check preservation
  set_last_error("");
  st = vt_tensor_iter_export_cuda_desc(iter26,
                                       /*operand_index=*/0,
                                       VT_TENSOR_ITER_CUDA_MAX_NDIM,
                                       &desc26);
  EXPECT_EQ(st, VT_STATUS_UNSUPPORTED);
  std::string err = get_last_error();
  EXPECT_NE(err.find("iteration rank exceeds max_ndim"), std::string::npos);
  EXPECT_EQ(desc26.ndim, 123);

  vt_tensor_iter_destroy(iter25);
  vt_tensor_iter_destroy(iter26);

  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out25_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(a25_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out26_h);
}
#endif  // VBT_WITH_CUDA
