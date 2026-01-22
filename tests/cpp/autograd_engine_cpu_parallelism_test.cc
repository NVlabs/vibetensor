// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/function.h"
#include "vbt/autograd/meta.h"
#include "vbt/autograd/accumulate_grad.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::autograd::AccumulateGrad;
using vbt::autograd::FunctionNode;
using vbt::autograd::InputMeta;
using vbt::autograd::OptionalTensor;
using vbt::autograd::ensure_next_edges_sized;
using vbt::autograd::run_backward;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

namespace {

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  const std::size_t nbytes = ne * sizeof(float);

  void* buf = nullptr;
  if (nbytes > 0) {
    buf = ::operator new(nbytes);
  }
  DataPtr dp(buf, [](void* p) noexcept { ::operator delete(p); });
  StoragePtr st = vbt::core::make_intrusive<Storage>(std::move(dp), nbytes);

  std::vector<int64_t> strides(sizes.size());
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(sizes.size()) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    acc *= (sizes[idx] == 0 ? 1 : sizes[idx]);
  }

  TensorImpl t(st, sizes, strides, /*offset=*/0, ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) p[i] = fill;
  return t;
}

struct MtToggleRestore {
  bool prev;
  MtToggleRestore() : prev(vbt::autograd::is_multithreading_enabled()) {
    vbt::autograd::set_multithreading_enabled(true);
  }
  ~MtToggleRestore() {
    vbt::autograd::set_multithreading_enabled(prev);
  }
};

} // namespace

TEST(AutogradCpuParallelismTest, IndependentBranchesUseMultipleThreadsWhenAvailable) {
  MtToggleRestore mt_guard;

  if (std::thread::hardware_concurrency() < 2) {
    GTEST_SKIP() << "hardware_concurrency < 2";
  }

  constexpr int kBranches = 4;
  const auto kSleep = std::chrono::milliseconds(150);

  // Leaf and its AccumulateGrad sink.
  TensorImpl leaf = make_cpu_dense_f32({4}, 0.0f);
  auto* meta = vbt::autograd::get_autograd_meta(leaf, /*create_if_missing=*/true);
  ASSERT_NE(meta, nullptr);
  auto acc = vbt::core::make_intrusive<AccumulateGrad>(meta);

  std::mutex mu;
  std::unordered_set<std::thread::id> tids;

  // Deterministic parallelism check: ensure at least one branch enters apply()
  // while another branch is still inside apply().
  std::mutex overlap_mu;
  std::condition_variable overlap_cv;
  bool gate_started = false;
  bool first_waiting = false;
  bool overlapped = false;

  std::vector<InputMeta> meta1 = {
      InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true}};

  // Branch nodes: sleep to amplify overlap and record thread id.
  std::vector<intrusive_ptr<FunctionNode>> branches;
  branches.reserve(kBranches);
  for (int i = 0; i < kBranches; ++i) {
    auto bi = vbt::core::make_intrusive<FunctionNode>(
        "B" + std::to_string(i),
        meta1,
        [&](std::vector<OptionalTensor>&& gin) {
          {
            std::lock_guard<std::mutex> lock(mu);
            tids.insert(std::this_thread::get_id());
          }
          {
            std::unique_lock<std::mutex> lock(overlap_mu);
            if (!gate_started) {
              gate_started = true;
              first_waiting = true;
              overlap_cv.notify_all();
              overlap_cv.wait_for(
                  lock, std::chrono::seconds(5), [&]() { return overlapped; });
              first_waiting = false;
            } else if (first_waiting && !overlapped) {
              overlapped = true;
              overlap_cv.notify_all();
            }
          }
          std::this_thread::sleep_for(kSleep);
          std::vector<OptionalTensor> out(1);
          if (!gin.empty()) out[0] = std::move(gin[0]);
          return out;
        });
    ensure_next_edges_sized(*bi);
    bi->next_edges[0] = vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(acc.get()), 0};
    branches.emplace_back(std::move(bi));
  }

  // Root routes one grad to each independent branch.
  std::vector<InputMeta> root_meta;
  root_meta.reserve(kBranches);
  for (int i = 0; i < kBranches; ++i) {
    root_meta.push_back(InputMeta{ScalarType::Float32, Device::cpu(), {4}, /*is_strided_dense=*/true});
  }

  auto Root = vbt::core::make_intrusive<FunctionNode>(
      "Root",
      root_meta,
      [&](std::vector<OptionalTensor>&& gin) {
        std::vector<OptionalTensor> out(static_cast<std::size_t>(kBranches));
        for (int i = 0; i < kBranches; ++i) {
          if (static_cast<std::size_t>(i) < gin.size()) out[static_cast<std::size_t>(i)] = std::move(gin[static_cast<std::size_t>(i)]);
        }
        return out;
      });
  ensure_next_edges_sized(*Root);
  for (int i = 0; i < kBranches; ++i) {
    Root->next_edges[static_cast<std::size_t>(i)] =
        vbt::autograd::Edge{intrusive_ptr<vbt::autograd::Node>(branches[static_cast<std::size_t>(i)].get()), 0};
  }

  std::vector<OptionalTensor> seed(static_cast<std::size_t>(kBranches));
  for (int i = 0; i < kBranches; ++i) {
    seed[static_cast<std::size_t>(i)] = make_cpu_dense_f32({4}, 1.0f);
  }

  EXPECT_NO_THROW(run_backward(intrusive_ptr<vbt::autograd::Node>(Root.get()), seed, {}));

  // With a working CPU pool, we should see at least 2 distinct worker thread ids.
  EXPECT_GE(tids.size(), 2u);

  {
    std::lock_guard<std::mutex> lock(overlap_mu);
    EXPECT_TRUE(overlapped) << "did not observe overlapping branch execution";
  }

  // Grad correctness: leaf accumulates kBranches contributions of 1.0.
  ASSERT_TRUE(meta->grad_ptr != nullptr && meta->grad_has);
  const TensorImpl& g = *meta->grad_ptr;
  const float* p = static_cast<const float*>(g.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(p[i], static_cast<float>(kBranches)) << "index " << i;
  }
}
