// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
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

static std::filesystem::path make_temp_dir(const std::string& prefix) {
  namespace fs = std::filesystem;
  const fs::path base = fs::temp_directory_path();
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();

  for (int i = 0; i < 100; ++i) {
    fs::path p = base / (prefix + std::to_string(now) + "_" + std::to_string(i));
    std::error_code ec;
    if (fs::create_directory(p, ec)) {
      return p;
    }
  }
  throw std::runtime_error("failed to create temp dir");
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

static TensorImpl p3_hot_add_impl(const TensorImpl& a, const TensorImpl& b) {
  if (a.device() != Device::cpu() || b.device() != Device::cpu()) {
    throw std::runtime_error("p3_hot_add: device mismatch");
  }
  if (a.dtype() != ScalarType::Float32 || b.dtype() != ScalarType::Float32) {
    throw std::runtime_error("p3_hot_add: dtype mismatch");
  }
  if (a.sizes() != b.sizes()) {
    throw std::runtime_error("p3_hot_add: size mismatch");
  }

  const int64_t n = a.numel();
  auto st = make_storage_bytes(static_cast<std::size_t>(n) * sizeof(float));
  const float* pa = static_cast<const float*>(a.data());
  const float* pb = static_cast<const float*>(b.data());
  float* pc = static_cast<float*>(st->data());
  for (int64_t i = 0; i < n; ++i) pc[i] = pa[i] + pb[i];

  return TensorImpl(st,
                    /*sizes=*/a.sizes(),
                    /*strides=*/a.strides(),
                    /*storage_offset=*/0,
                    ScalarType::Float32,
                    Device::cpu());
}

}  // namespace

TEST(PluginAtomicCommitDeterminismConcurrencyTest,
     ConcurrentDuplicateDefIsDeterministic) {
#ifndef PLUGIN_P3_DUP_DEF_A_PATH
  GTEST_SKIP() << "PLUGIN_P3_DUP_DEF_A_PATH not provided";
#elif !defined(PLUGIN_P3_DUP_DEF_B_PATH)
  GTEST_SKIP() << "PLUGIN_P3_DUP_DEF_B_PATH not provided";
#else
#if !(defined(__linux__) || defined(__APPLE__))
  GTEST_SKIP() << "dlopen not supported on this platform";
#else
  const char* so_a = PLUGIN_P3_DUP_DEF_A_PATH;
  const char* so_b = PLUGIN_P3_DUP_DEF_B_PATH;

  // Ensure the plugins are not already loaded in this process.
  (void)dlerror();
  void* pre_a = dlopen(so_a, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre_a) {
    dlclose(pre_a);
    GTEST_SKIP() << "p3_dup_def_a plugin already loaded";
  }
  (void)dlerror();
  void* pre_b = dlopen(so_b, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre_b) {
    dlclose(pre_b);
    GTEST_SKIP() << "p3_dup_def_b plugin already loaded";
  }

  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  ASSERT_FALSE(D.has("vt::p3_dup_def"))
      << "vt::p3_dup_def already exists; test requires a fresh process";

  // Ensure atomic commit mode is enabled (cached on first use in the loader).
  setenv("VBT_PLUGIN_ATOMIC_COMMIT", "1", 1);

  const std::filesystem::path marker_dir = make_temp_dir("vbt_p3_dup_def_");
  const std::string marker_dir_str = marker_dir.string();
  setenv("VBT_P3_DUP_DEF_MARKER_DIR", marker_dir_str.c_str(), 1);

  struct MarkerDirGuard {
    std::filesystem::path dir;
    explicit MarkerDirGuard(std::filesystem::path p) : dir(std::move(p)) {}
    ~MarkerDirGuard() {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
#if defined(__unix__) || defined(__APPLE__)
      unsetenv("VBT_P3_DUP_DEF_MARKER_DIR");
#endif
    }
  } marker_dir_guard(marker_dir);

  std::atomic<int> ready{0};
  std::atomic<bool> go{false};

  vt_status st_a = VT_STATUS_INTERNAL;
  vt_status st_b = VT_STATUS_INTERNAL;
  std::string err_a;
  std::string err_b;

  auto worker = [&](const char* so, vt_status* out_st, std::string* out_err) {
    ready.fetch_add(1, std::memory_order_acq_rel);
    while (!go.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    *out_st = vbt::dispatch::plugin::load_library(so);
    if (*out_st != VT_STATUS_OK) {
      *out_err = vbt::dispatch::plugin::get_last_error();
    }
  };

  std::thread t1(worker, so_a, &st_a, &err_a);
  std::thread t2(worker, so_b, &st_b, &err_b);

  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  go.store(true, std::memory_order_release);

  t1.join();
  t2.join();

  const bool ok_a = st_a == VT_STATUS_OK;
  const bool ok_b = st_b == VT_STATUS_OK;

  // Prove that the duplicate-def failure occurred during vbt_plugin_init (stage
  // time), not during commit: each plugin creates a marker file only if it is
  // about to return VT_STATUS_OK from vbt_plugin_init.
  const bool marker_a =
      std::filesystem::exists(marker_dir / "p3_dup_def_a.ok");
  const bool marker_b =
      std::filesystem::exists(marker_dir / "p3_dup_def_b.ok");
  EXPECT_EQ(marker_a, ok_a);
  EXPECT_EQ(marker_b, ok_b);

  ASSERT_NE(ok_a, ok_b)
      << "expected exactly one plugin load to succeed; got st_a="
      << static_cast<int>(st_a) << " err_a=" << err_a
      << " st_b=" << static_cast<int>(st_b) << " err_b=" << err_b;

  const vt_status fail_st = ok_a ? st_b : st_a;
  const std::string& fail_err = ok_a ? err_b : err_a;

  EXPECT_EQ(fail_st, VT_STATUS_INVALID_ARG)
      << "err=" << fail_err;
  EXPECT_NE(fail_err.find("duplicate def: vt::p3_dup_def"), std::string::npos)
      << fail_err;

  ASSERT_TRUE(D.has("vt::p3_dup_def"));

  // Kernel must be installed and callable.
  TensorImpl a = cpu_tensor_f32_1d({1.0f, 2.0f, 3.0f, 4.0f});
  TensorImpl b = cpu_tensor_f32_1d({10.0f, 20.0f, 30.0f, 40.0f});
  BoxedStack stack{a, b};
  D.callBoxed("vt::p3_dup_def", stack);

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

TEST(PluginAtomicCommitDeterminismConcurrencyTest,
     StressDispatchWhilePluginLoads) {
#ifndef PLUGIN_P3_STAGE_SLEEP_PATH
  GTEST_SKIP() << "PLUGIN_P3_STAGE_SLEEP_PATH not provided";
#else
#if !(defined(__linux__) || defined(__APPLE__))
  GTEST_SKIP() << "dlopen not supported on this platform";
#else
  const char* so = PLUGIN_P3_STAGE_SLEEP_PATH;

  // Ensure the plugin is not already loaded in this process.
  (void)dlerror();
  void* pre = dlopen(so, RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  if (pre) {
    dlclose(pre);
    GTEST_SKIP() << "p3_stage_sleep plugin already loaded";
  }

  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();

  // Define a tiny op to exercise v2 dispatch while the plugin loads/commits.
  if (!D.has("vt::p3_hot_add")) {
    (void)D.def("vt::p3_hot_add(Tensor, Tensor) -> Tensor");
    D.registerCpuKernel("vt::p3_hot_add", &p3_hot_add_impl);
  }
  const auto hot = D.find("vt::p3_hot_add");

  // Ensure atomic commit mode is enabled (cached on first use in the loader).
  setenv("VBT_PLUGIN_ATOMIC_COMMIT", "1", 1);

  std::atomic<bool> stop{false};
  std::atomic<bool> ok{true};
  std::mutex err_mu;
  std::string err_msg;
  std::vector<std::thread> stress_threads;

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
          throw std::runtime_error("vt::p3_hot_add: wrong output stack size");
        }
        const float* p = static_cast<const float*>(stack[0].data());
        if (!p || p[0] != 11.0f || p[1] != 22.0f || p[2] != 33.0f ||
            p[3] != 44.0f) {
          throw std::runtime_error("vt::p3_hot_add: wrong result");
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
        err_msg = "unknown exception in vt::p3_hot_add stress thread";
      }
      stop.store(true, std::memory_order_release);
    }
  };

  for (int i = 0; i < 4; ++i) {
    stress_threads.emplace_back(stress_worker);
  }

  struct LoadResult {
    std::atomic<bool> done{false};
    vt_status st{VT_STATUS_INTERNAL};
    std::string err;
  };

  auto lr = std::make_shared<LoadResult>();

  std::thread load_thread([lr, so] {
    lr->st = vbt::dispatch::plugin::load_library(so);
    if (lr->st != VT_STATUS_OK) {
      lr->err = vbt::dispatch::plugin::get_last_error();
    }
    lr->done.store(true, std::memory_order_release);
  });

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!lr->done.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() > deadline) {
      // stop + join stress threads before failing
      stop.store(true, std::memory_order_release);
      for (auto& th : stress_threads) {
        if (th.joinable()) th.join();
      }
      // Avoid blocking forever if load_library deadlocks.
      load_thread.detach();
      FAIL() << "load_library timed out (possible deadlock)";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  load_thread.join();

  stop.store(true, std::memory_order_release);
  for (auto& th : stress_threads) {
    if (th.joinable()) th.join();
  }

  EXPECT_TRUE(ok.load(std::memory_order_acquire)) << err_msg;

  ASSERT_EQ(lr->st, VT_STATUS_OK) << lr->err;

  ASSERT_TRUE(D.has("vt::p3_stage_sleep"));

  // Kernel must be installed and callable after the load.
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

#else

TEST(PluginAtomicCommitDeterminismConcurrencyTest, Skipped) {
  GTEST_SKIP() << "requires VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS";
}

#endif  // VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
