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

struct UnaryLoopCtx {
  int64_t tiles{0};
};

static void unary_add_one_loop_ti(char** data,
                                  const int64_t* strides,
                                  int64_t size,
                                  void* ctx_void) {
  auto* ctx = static_cast<UnaryLoopCtx*>(ctx_void);
  ++ctx->tiles;
  char* out_base = data[0];
  char* a_base   = data[1];
  const int64_t out_stride = strides[0];
  const int64_t a_stride   = strides[1];
  for (int64_t i = 0; i < size; ++i) {
    auto* out = reinterpret_cast<float*>(out_base + i * out_stride);
    const auto* a = reinterpret_cast<const float*>(a_base + i * a_stride);
    *out = *a + 1.0f;
  }
}

struct BinaryLoopCtx {
  int64_t tiles{0};
};

static void binary_add_loop_ti(char** data,
                               const int64_t* strides,
                               int64_t size,
                               void* ctx_void) {
  auto* ctx = static_cast<BinaryLoopCtx*>(ctx_void);
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

static void throwing_loop(char** /*data*/, const int64_t* /*strides*/, int64_t /*size*/, void* /*ctx_void*/) {
  throw std::runtime_error("unary callback error");
}

}  // namespace

TEST(TensorIterPluginHelperTest, UnarySuccessAndZeroNumel) {
  using vbt::dispatch::plugin::get_last_error;
  using vbt::dispatch::plugin::set_last_error;

  // Basic unary success on 1D contiguous tensor.
  auto out = make_contiguous_tensor({8});
  auto in  = make_contiguous_tensor({8});

  auto* out_data = static_cast<float*>(out.data());
  auto* in_data  = static_cast<float*>(in.data());
  for (int i = 0; i < 8; ++i) {
    in_data[i] = static_cast<float>(i * 2);
    out_data[i] = 0.0f;
  }

  vt_tensor out_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out);
  vt_tensor in_h  = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(in);

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;
  UnaryLoopCtx ctx{};

  set_last_error("preset");
  vt_status st = vt_tensor_iter_unary_cpu(&cfg, out_h, in_h,
                                          &unary_add_one_loop_ti,
                                          &ctx);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_STREQ(get_last_error(), "");
  EXPECT_EQ(ctx.tiles, 1);
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(out_data[i], in_data[i] + 1.0f);
  }

  // Zero-numel iterator: callback must not be invoked.
  auto out_zero = make_contiguous_tensor({2, 0, 3});
  auto in_zero  = make_contiguous_tensor({2, 0, 3});
  vt_tensor out_zero_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out_zero);
  vt_tensor in_zero_h  = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(in_zero);

  ctx.tiles = 0;
  set_last_error("preset");
  st = vt_tensor_iter_unary_cpu(&cfg, out_zero_h, in_zero_h,
                                &unary_add_one_loop_ti,
                                &ctx);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_STREQ(get_last_error(), "");
  EXPECT_EQ(ctx.tiles, 0);

  // Clean up borrowed handles.
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(in_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_zero_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(in_zero_h);
}

TEST(TensorIterPluginHelperTest, BinarySuccessAndConfigValidation) {
  using vbt::dispatch::plugin::get_last_error;
  using vbt::dispatch::plugin::set_last_error;

  // Basic binary success on 2D contiguous tensors.
  auto out = make_contiguous_tensor({2, 3});
  auto a   = make_contiguous_tensor({2, 3});
  auto b   = make_contiguous_tensor({2, 3});

  auto* out_data = static_cast<float*>(out.data());
  auto* a_data   = static_cast<float*>(a.data());
  auto* b_data   = static_cast<float*>(b.data());

  int idx = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      a_data[idx] = static_cast<float>(idx);
      b_data[idx] = static_cast<float>(2 * idx);
      out_data[idx] = 0.0f;
      ++idx;
    }
  }

  vt_tensor out_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out);
  vt_tensor a_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(a);
  vt_tensor b_h   = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(b);

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;
  BinaryLoopCtx ctx{};

  set_last_error("preset");
  vt_status st = vt_tensor_iter_binary_cpu(&cfg, out_h, a_h, b_h,
                                           &binary_add_loop_ti,
                                           &ctx);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_STREQ(get_last_error(), "");
  EXPECT_GE(ctx.tiles, 1);

  idx = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(out_data[idx], a_data[idx] + b_data[idx]);
      ++idx;
    }
  }

  // Invalid max_rank should yield INVALID_ARG and set a non-empty error.
  vt_iter_config bad_max_rank_cfg;
  bad_max_rank_cfg.max_rank = static_cast<int64_t>(::vbt::core::kTensorIterMaxRank) + 1;
  bad_max_rank_cfg.check_mem_overlap = VT_ITER_OVERLAP_ENABLE;

  set_last_error("");
  st = vt_tensor_iter_binary_cpu(&bad_max_rank_cfg, out_h, a_h, b_h,
                                 &binary_add_loop_ti,
                                 &ctx);
  EXPECT_EQ(st, VT_STATUS_INVALID_ARG);
  std::string err = get_last_error();
  EXPECT_FALSE(err.empty());

  // Invalid overlap mode should also yield INVALID_ARG.
  vt_iter_config bad_overlap_cfg;
  bad_overlap_cfg.max_rank = 0;
  bad_overlap_cfg.check_mem_overlap = static_cast<vt_iter_overlap_mode>(42);

  set_last_error("");
  st = vt_tensor_iter_binary_cpu(&bad_overlap_cfg, out_h, a_h, b_h,
                                 &binary_add_loop_ti,
                                 &ctx);
  EXPECT_EQ(st, VT_STATUS_INVALID_ARG);
  err = get_last_error();
  EXPECT_FALSE(err.empty());

  // Clean up borrowed handles.
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(a_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(b_h);
}

TEST(TensorIterPluginHelperTest, ErrorMappingAndInvalidArgs) {
  using vbt::dispatch::plugin::get_last_error;
  using vbt::dispatch::plugin::set_last_error;

  auto out = make_contiguous_tensor({4});
  auto in  = make_contiguous_tensor({4});

  vt_tensor out_h = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(out);
  vt_tensor in_h  = vbt::dispatch::plugin::detail::make_borrowed_handle_for_tests(in);

  vt_iter_config cfg = VT_ITER_CONFIG_DEFAULT_INIT;

  // Null callback → INVALID_ARG and non-empty error.
  set_last_error("");
  vt_status st = vt_tensor_iter_unary_cpu(&cfg, out_h, in_h,
                                          /*loop=*/nullptr,
                                          /*ctx=*/nullptr);
  EXPECT_EQ(st, VT_STATUS_INVALID_ARG);
  std::string err = get_last_error();
  EXPECT_FALSE(err.empty());

  // out and in sharing the same handle is rejected.
  set_last_error("");
  st = vt_tensor_iter_unary_cpu(&cfg, out_h, out_h,
                                &unary_add_one_loop_ti,
                                /*ctx=*/nullptr);
  EXPECT_EQ(st, VT_STATUS_INVALID_ARG);
  err = get_last_error();
  EXPECT_FALSE(err.empty());

  // Callback throwing std::exception should map to RUNTIME_ERROR and propagate message.
  set_last_error("");
  st = vt_tensor_iter_unary_cpu(&cfg, out_h, in_h,
                                &throwing_loop,
                                /*ctx=*/nullptr);
  EXPECT_EQ(st, VT_STATUS_RUNTIME_ERROR);
  err = get_last_error();
  EXPECT_FALSE(err.empty());
  EXPECT_NE(err.find("unary callback error"), std::string::npos);

  // A subsequent successful call must clear TLS.
  UnaryLoopCtx ctx{};
  set_last_error("preset");
  st = vt_tensor_iter_unary_cpu(&cfg, out_h, in_h,
                                &unary_add_one_loop_ti,
                                &ctx);
  EXPECT_EQ(st, VT_STATUS_OK);
  EXPECT_STREQ(get_last_error(), "");

  // Clean up borrowed handles.
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(out_h);
  vbt::dispatch::plugin::detail::destroy_borrowed_handle_for_tests(in_h);
}
