// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/plugin_loader.h"
#include "vbt/plugin/vbt_plugin.h"

extern "C" const struct vbt_host_api* vbt_reference_ti_add_get_host_api(void);

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::dispatch::Dispatcher;
using vbt::dispatch::BoxedStack;

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

static void ensure_plugin_loaded(const char* path) {
  using vbt::dispatch::plugin::load_library;
  using vbt::dispatch::plugin::get_last_error;

  vt_status st = load_library(path);
  if (st == VT_STATUS_OK) {
    return;
  }
  std::string msg = get_last_error();
  if (st == VT_STATUS_INVALID_ARG &&
      msg.find("plugin already loaded") != std::string::npos) {
    return;  // benign duplicate
  }
  FAIL() << "load_library failed: status=" << static_cast<int>(st)
         << " msg=" << msg;
}

}  // namespace

TEST(TensorIterPluginTiReferenceTest, HostApiAndHelpersAvailable) {
#ifndef VBT_TI_REF_PLUGIN_PATH
  GTEST_SKIP() << "VBT_TI_REF_PLUGIN_PATH not defined";
#else
  ensure_plugin_loaded(VBT_TI_REF_PLUGIN_PATH);

  const struct vbt_host_api* host = vbt_reference_ti_add_get_host_api();
  ASSERT_NE(host, nullptr);

  EXPECT_EQ(host->host_abi_major, VBT_PLUGIN_ABI_VERSION_MAJOR);
  EXPECT_GE(host->host_abi_minor, (uint32_t)2);

  ASSERT_NE(host->vt_tensor_iter_unary_cpu, nullptr);
  ASSERT_NE(host->vt_tensor_iter_binary_cpu, nullptr);
#endif
}

TEST(TensorIterPluginTiReferenceTest, PluginAddMatchesCoreAddEqualShape) {
#ifndef VBT_TI_REF_PLUGIN_PATH
  GTEST_SKIP() << "VBT_TI_REF_PLUGIN_PATH not defined";
#else
  ensure_plugin_loaded(VBT_TI_REF_PLUGIN_PATH);

  auto& D = Dispatcher::instance();
  if (!D.has("vt::ti_add")) {
    GTEST_SKIP() << "vt::ti_add not registered";
  }

  auto a = make_contiguous_tensor({2, 3});
  auto b = make_contiguous_tensor({2, 3});

  auto* a_data = static_cast<float*>(a.data());
  auto* b_data = static_cast<float*>(b.data());
  int idx = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      a_data[idx] = static_cast<float>(idx);
      b_data[idx] = static_cast<float>(2 * idx + 1);
      ++idx;
    }
  }

  const std::size_t ne = static_cast<std::size_t>(a.numel());
  std::vector<float> expected(ne);
  for (std::size_t i = 0; i < ne; ++i) {
    expected[i] = a_data[i] + b_data[i];
  }

  // Plugin vt::ti_add result
  BoxedStack s_plugin{a, b};
  D.callBoxed("vt::ti_add", s_plugin);
  ASSERT_EQ(s_plugin.size(), 1u);
  const TensorImpl& out_plugin = s_plugin.back();

  EXPECT_EQ(out_plugin.sizes(), a.sizes());
  EXPECT_EQ(out_plugin.dtype(), a.dtype());
  EXPECT_EQ(out_plugin.device(), a.device());

  const float* plugin_data = static_cast<const float*>(out_plugin.data());
  for (std::size_t i = 0; i < ne; ++i) {
    EXPECT_FLOAT_EQ(plugin_data[i], expected[i]);
  }
#endif
}

TEST(TensorIterPluginTiReferenceTest, PluginErrorPropagatesBroadcastFailure) {
#ifndef VBT_TI_REF_PLUGIN_PATH
  GTEST_SKIP() << "VBT_TI_REF_PLUGIN_PATH not defined";
#else
  using vbt::dispatch::plugin::get_last_error;

  ensure_plugin_loaded(VBT_TI_REF_PLUGIN_PATH);

  auto& D = Dispatcher::instance();
  if (!D.has("vt::ti_add")) {
    GTEST_SKIP() << "vt::ti_add not registered";
  }

  // Shape mismatch: {4} vs {2} should cause a TI broadcast error inside the
  // plugin, which is mapped through throw_from_status and surfaced both as a
  // C++ exception and via the plugin TLS error string.
  auto a = make_contiguous_tensor({4});
  auto b = make_contiguous_tensor({2});

  BoxedStack s{a, b};
  try {
    D.callBoxed("vt::ti_add", s);
    FAIL() << "Expected std::invalid_argument from vt::ti_add";
  } catch (const std::invalid_argument& e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("broadcast"), std::string::npos) << msg;
  } catch (...) {
    FAIL() << "Unexpected exception type from vt::ti_add";
  }

  std::string tls_err = get_last_error();
  EXPECT_FALSE(tls_err.empty());
  EXPECT_NE(tls_err.find("broadcast"), std::string::npos) << tls_err;
#endif
}
