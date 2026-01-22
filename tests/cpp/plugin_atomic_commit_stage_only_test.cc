// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#include "vbt/dispatch/plugin_loader.h"

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;

#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

static TensorImpl cpu_tensor_f32_1d(const std::vector<float>& values) {
  auto st = make_storage_bytes(values.size() * sizeof(float));
  float* p = static_cast<float*>(st->data());
  for (std::size_t i = 0; i < values.size(); ++i) {
    p[i] = values[i];
  }
  TensorImpl t(st,
               /*sizes=*/{static_cast<std::int64_t>(values.size())},
               /*strides=*/{1},
               /*storage_offset=*/0,
               ScalarType::Float32,
               Device::cpu());
  return t;
}

static TensorImpl p3_hot_identity_impl(const TensorImpl& a,
                                      const TensorImpl& /*b*/) {
  return a;
}

}  // namespace

TEST(PluginAtomicCommitStageOnlyTest, StagesDefsAndKernelsUntilInitReturns) {
#ifndef PLUGIN_P3_STAGE_SLEEP_PATH
  GTEST_SKIP() << "PLUGIN_P3_STAGE_SLEEP_PATH not provided";
#else
#if !(defined(__linux__) || defined(__APPLE__))
  GTEST_SKIP() << "dlopen/RTLD_NOLOAD not supported on this platform";
#else
  const char* so = PLUGIN_P3_STAGE_SLEEP_PATH;

  // Ensure the plugin is not already loaded in this process.
  (void)dlerror();
  void* pre = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre) {
    dlclose(pre);
    GTEST_SKIP() << "p3_stage_sleep plugin already loaded";
  }

  // Enable dispatcher v2 for this test binary.
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  // Enable atomic commit mode for plugin loader.
  setenv("VBT_PLUGIN_ATOMIC_COMMIT", "1", 1);

  auto& D = Dispatcher::instance();

  // Define a tiny op to stress v2 dispatch while the plugin loads/commits.
  if (!D.has("vt::p3_hot")) {
    (void)D.def("vt::p3_hot(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::p3_hot", &p3_hot_identity_impl);
  }
  const auto hot = D.find("vt::p3_hot");

  std::atomic<bool> stop{false};
  std::atomic<bool> ok{true};
  std::mutex err_mu;
  std::string err_msg;
  std::vector<std::thread> stress_threads;

  auto stress_worker = [&] {
    try {
      TensorImpl a = cpu_tensor_f32_1d({1.0f, 2.0f, 3.0f, 4.0f});
      TensorImpl b = cpu_tensor_f32_1d({10.0f, 20.0f, 30.0f, 40.0f});
      BoxedStack stack;
      stack.reserve(2);

      while (!stop.load(std::memory_order_acquire)) {
        stack.clear();
        stack.push_back(a);
        stack.push_back(b);
        D.callBoxed(hot, stack);

        if (stack.size() != 1u) {
          throw std::runtime_error("vt::p3_hot: wrong output stack size");
        }
        const float* p = static_cast<const float*>(stack[0].data());
        if (!p || p[0] != 1.0f || p[1] != 2.0f || p[2] != 3.0f ||
            p[3] != 4.0f) {
          throw std::runtime_error("vt::p3_hot: wrong result");
        }
      }
    } catch (const std::exception& e) {
      if (ok.exchange(false, std::memory_order_acq_rel)) {
        std::lock_guard<std::mutex> lg(err_mu);
        err_msg = e.what();
      }
      stop.store(true, std::memory_order_release);
    } catch (...) {
      if (ok.exchange(false, std::memory_order_acq_rel)) {
        std::lock_guard<std::mutex> lg(err_mu);
        err_msg = "unknown exception in vt::p3_hot stress thread";
      }
      stop.store(true, std::memory_order_release);
    }
  };

  for (int i = 0; i < 4; ++i) {
    stress_threads.emplace_back(stress_worker);
  }

  struct StressJoinGuard {
    std::atomic<bool>& stop;
    std::vector<std::thread>& threads;

    ~StressJoinGuard() {
      stop.store(true, std::memory_order_release);
      for (auto& th : threads) {
        if (th.joinable()) th.join();
      }
    }
  } stress_guard{stop, stress_threads};

  std::atomic<vt_status> st{VT_STATUS_INTERNAL};
  std::thread t([&] {
    st.store(vbt::dispatch::plugin::load_library(so),
             std::memory_order_release);
  });

  struct LoadJoinGuard {
    std::thread& t;
    ~LoadJoinGuard() {
      if (t.joinable()) t.join();
    }
  } load_guard{t};

  // Wait until the library is dlopen'd so we can query the progress marker.
  void* h = nullptr;
  for (int i = 0; i < 200; ++i) {
    (void)dlerror();
    h = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
    if (h) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_NE(h, nullptr) << "plugin was not dlopen'd";

  using get_state_fn = int (*)();
  auto* get_state =
      reinterpret_cast<get_state_fn>(dlsym(h, "vbt_p3_stage_sleep_get_state"));
  ASSERT_NE(get_state, nullptr) << "missing state symbol";

  int state = 0;
  for (int i = 0; i < 200; ++i) {
    state = get_state();
    if (state >= 2) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_GE(state, 2) << "plugin init did not reach staged state";

  // While the plugin is sleeping inside vbt_plugin_init, the op must not be
  // visible in the dispatcher registry.
  EXPECT_FALSE(D.has("vt::p3_stage_sleep"));

  dlclose(h);

  t.join();

  stop.store(true, std::memory_order_release);
  for (auto& th : stress_threads) {
    if (th.joinable()) th.join();
  }
  EXPECT_TRUE(ok.load(std::memory_order_acquire)) << err_msg;

  ASSERT_EQ(st.load(std::memory_order_acquire), VT_STATUS_OK)
      << vbt::dispatch::plugin::get_last_error();

  ASSERT_TRUE(D.has("vt::p3_stage_sleep"));

  // Kernel must be installed and callable.
  TensorImpl a = cpu_tensor_f32_1d({1.0f, 2.0f, 3.0f, 4.0f});
  TensorImpl b = cpu_tensor_f32_1d({10.0f, 20.0f, 30.0f, 40.0f});
  BoxedStack stack{a, b};
  D.callBoxed("vt::p3_stage_sleep", stack);

  ASSERT_EQ(stack.size(), 1u);
  const TensorImpl& out = stack[0];
  ASSERT_EQ(out.device(), Device::cpu());
  ASSERT_EQ(out.dtype(), ScalarType::Float32);

  const float* pout = static_cast<const float*>(out.data());
  ASSERT_NE(pout, nullptr);
  EXPECT_FLOAT_EQ(pout[0], 11.0f);
  EXPECT_FLOAT_EQ(pout[1], 22.0f);
  EXPECT_FLOAT_EQ(pout[2], 33.0f);
  EXPECT_FLOAT_EQ(pout[3], 44.0f);
#endif
#endif
}

#endif  // VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
