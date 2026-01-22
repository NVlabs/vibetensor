// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vbt/autograd/engine.h"
#include "vbt/autograd/engine_toggles.h"
#include "vbt/autograd/node.h"
#include "vbt/autograd/types.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"

using vbt::autograd::AutogradDeviceMode;
using vbt::autograd::Node;
using vbt::autograd::OptionalTensor;
using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::core::intrusive_ptr;

#if VBT_AUTOGRAD_TESTING
namespace vbt { namespace autograd {
void _test_clear_last_backward_snapshot() noexcept;
AutogradDeviceMode _test_last_backward_device_mode_snapshot() noexcept;
}} // namespace vbt::autograd
#endif

namespace {

struct DeviceModeRestore {
  AutogradDeviceMode prev;
  DeviceModeRestore() : prev(vbt::autograd::get_device_mode()) {}
  ~DeviceModeRestore() { vbt::autograd::set_device_mode(prev); }
};

static TensorImpl make_cpu_dense_f32(const std::vector<int64_t>& sizes, float fill) {
  std::size_t ne = 1;
  for (auto s : sizes) {
    ne *= static_cast<std::size_t>(s == 0 ? 1 : s);
  }
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
    strides[static_cast<std::size_t>(i)] = acc;
    acc *= (sizes[static_cast<std::size_t>(i)] == 0 ? 1 : sizes[static_cast<std::size_t>(i)]);
  }

  TensorImpl t(st, sizes, strides, /*offset=*/0, ScalarType::Float32, Device::cpu());
  float* p = static_cast<float*>(t.data());
  for (std::size_t i = 0; i < ne; ++i) {
    p[i] = fill;
  }
  return t;
}

struct ToggleFlippingRoot final : Node {
  explicit ToggleFlippingRoot(AutogradDeviceMode target) : target_(target) {
    name = "ToggleFlippingRoot";
  }

  uint32_t num_inputs() const noexcept override { return 1; }

  std::vector<OptionalTensor> apply(std::vector<OptionalTensor>&& grads_in) override {
    (void)grads_in;
    vbt::autograd::set_device_mode(target_);
    return {};  // sink
  }

 private:
  AutogradDeviceMode target_;
};

} // namespace

TEST(AutogradDeviceModeToggleTest, ToggleRoundtrip) {
  DeviceModeRestore restore;

  vbt::autograd::set_device_mode(AutogradDeviceMode::SingleDevice);
  EXPECT_EQ(vbt::autograd::get_device_mode(), AutogradDeviceMode::SingleDevice);

  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);
  EXPECT_EQ(vbt::autograd::get_device_mode(), AutogradDeviceMode::MultiDeviceExperimental);

  vbt::autograd::set_device_mode(AutogradDeviceMode::SingleDevice);
  EXPECT_EQ(vbt::autograd::get_device_mode(), AutogradDeviceMode::SingleDevice);
}

#if VBT_AUTOGRAD_TESTING
TEST(AutogradDeviceModeToggleTest, SnapshotSemanticsNotAffectedByMidRunToggleFlip) {
  DeviceModeRestore restore;

  // Run 1: snapshot SingleDevice, flip global to MultiDeviceExperimental mid-run.
  vbt::autograd::set_device_mode(AutogradDeviceMode::SingleDevice);
  vbt::autograd::_test_clear_last_backward_snapshot();

  auto root1 = vbt::core::make_intrusive<ToggleFlippingRoot>(
      AutogradDeviceMode::MultiDeviceExperimental);
  std::vector<OptionalTensor> seed1(1);
  seed1[0] = make_cpu_dense_f32({1}, 1.0f);

  ASSERT_NO_THROW(vbt::autograd::run_backward(intrusive_ptr<Node>(root1.get()), seed1, {}));
  EXPECT_EQ(vbt::autograd::get_device_mode(), AutogradDeviceMode::MultiDeviceExperimental);
  EXPECT_EQ(vbt::autograd::_test_last_backward_device_mode_snapshot(),
            AutogradDeviceMode::SingleDevice);

  // Run 2: snapshot MultiDeviceExperimental, flip global back to SingleDevice mid-run.
  vbt::autograd::set_device_mode(AutogradDeviceMode::MultiDeviceExperimental);
  vbt::autograd::_test_clear_last_backward_snapshot();

  auto root2 = vbt::core::make_intrusive<ToggleFlippingRoot>(AutogradDeviceMode::SingleDevice);
  std::vector<OptionalTensor> seed2(1);
  seed2[0] = make_cpu_dense_f32({1}, 1.0f);

  ASSERT_NO_THROW(vbt::autograd::run_backward(intrusive_ptr<Node>(root2.get()), seed2, {}));
  EXPECT_EQ(vbt::autograd::get_device_mode(), AutogradDeviceMode::SingleDevice);
  EXPECT_EQ(vbt::autograd::_test_last_backward_device_mode_snapshot(),
            AutogradDeviceMode::MultiDeviceExperimental);
}
#else
TEST(AutogradDeviceModeToggleTest, SnapshotSemanticsNotAffectedByMidRunToggleFlip) {
  GTEST_SKIP() << "VBT_AUTOGRAD_TESTING disabled";
}
#endif
