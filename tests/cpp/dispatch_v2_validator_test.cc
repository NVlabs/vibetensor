// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vbt/core/data_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/core/storage.h"
#include "vbt/core/tensor.h"
#include "vbt/dispatch/boxed.h"
#include "vbt/dispatch/dispatcher.h"
#if VBT_WITH_CUDA
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"
#endif

using vbt::core::DataPtr;
using vbt::core::Device;
using vbt::core::ScalarType;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::TensorImpl;
using vbt::dispatch::BoxedStack;
using vbt::dispatch::Dispatcher;

#if VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS

extern "C" void vbt_register_indexing_kernels();

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(
      DataPtr(::operator new(nbytes),
              [](void* p) noexcept { ::operator delete(p); }),
      nbytes);
}

static TensorImpl cpu_scalar_i64(std::int64_t v) {
  auto st = make_storage_bytes(sizeof(std::int64_t));
  TensorImpl t(st, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0,
               ScalarType::Int64, Device::cpu());
  auto* p = static_cast<std::int64_t*>(t.data());
  if (p) *p = v;
  return t;
}

static TensorImpl cpu_scalar_bool(bool v) {
  auto st = make_storage_bytes(sizeof(std::uint8_t));
  TensorImpl t(st, /*sizes=*/{}, /*strides=*/{}, /*storage_offset=*/0,
               ScalarType::Bool, Device::cpu());
  auto* p = static_cast<std::uint8_t*>(t.data());
  if (p) *p = v ? 1u : 0u;
  return t;
}

static TensorImpl cpu_tensor_f32_1d(std::size_t n) {
  auto st = make_storage_bytes(n * sizeof(float));
  TensorImpl t(st, /*sizes=*/{static_cast<std::int64_t>(n)}, /*strides=*/{1},
               /*storage_offset=*/0, ScalarType::Float32, Device::cpu());
  return t;
}

#if VBT_WITH_CUDA
static TensorImpl cuda_tensor_f32_1d(std::size_t n, int dev) {
  auto st = vbt::cuda::new_cuda_storage(
      n * vbt::core::itemsize(ScalarType::Float32), dev);
  TensorImpl t(st, /*sizes=*/{static_cast<std::int64_t>(n)}, /*strides=*/{1},
               /*storage_offset=*/0, ScalarType::Float32, Device::cuda(dev));
  return t;
}
#endif  // VBT_WITH_CUDA

// Undefined but shaped tensor (storage==nullptr, sizes non-empty).
static TensorImpl undefined_but_shaped(ScalarType dtype,
                                      Device dev,
                                      std::vector<std::int64_t> sizes) {
  std::vector<std::int64_t> strides;
  if (!sizes.empty()) {
    strides.resize(sizes.size());
    // Contiguous row-major.
    std::int64_t st = 1;
    for (std::size_t i = sizes.size(); i-- > 0;) {
      strides[i] = st;
      st *= sizes[i] == 0 ? 1 : sizes[i];
    }
  }
  return TensorImpl(StoragePtr{}, std::move(sizes), std::move(strides),
                    /*storage_offset=*/0, dtype, dev);
}

static std::atomic<int> g_called{0};

static TensorImpl passthrough(const TensorImpl& a, const TensorImpl& /*b*/) {
  g_called.fetch_add(1, std::memory_order_relaxed);
  return a;
}

}  // namespace

TEST(DispatchV2ValidatorTest, CannotDetermineDispatchDeviceAllUndefined) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  const std::string fqname = "test_v2_validator::all_undefined";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_v2_validator");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");
  D.registerCpuKernel(fqname, &passthrough);

  BoxedStack stack;
  stack.emplace_back(TensorImpl{});
  stack.emplace_back(TensorImpl{});

  try {
    D.callBoxed(fqname, stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find(fqname), std::string::npos) << msg;
    EXPECT_NE(msg.find("cannot determine dispatch device"), std::string::npos)
        << msg;
  }
}

TEST(DispatchV2ValidatorTest, CannotDetermineDispatchDeviceMaskedArg0Undefined) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  const std::string fqname = "test_v2_validator::masked_arg0_undefined";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_v2_validator");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");
  D.registerCpuKernel(fqname, &passthrough);

  const std::array<vbt::dispatch::DeviceConstraint, 1> cs = {
      vbt::dispatch::DeviceConstraint{1,
                                      vbt::dispatch::ConstraintKind::MustBeCPUScalarInt64_0d}};

  D.set_device_policy(
      fqname,
      vbt::dispatch::DevicePolicy::MaskedSameDevice,
      /*dispatch_arg_mask=*/1,
      std::span<const vbt::dispatch::DeviceConstraint>{cs},
      /*allow_undefined_mask=*/0);

  BoxedStack stack;
  stack.emplace_back(TensorImpl{});  // arg0 undefined
  stack.emplace_back(cpu_scalar_i64(0));

  try {
    D.callBoxed(fqname, stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find(fqname), std::string::npos) << msg;
    EXPECT_NE(msg.find("cannot determine dispatch device"), std::string::npos)
        << msg;
  }
}

TEST(DispatchV2ValidatorTest, ScalarInt64MismatchHasStableSubstrings) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  const std::string fqname = "test_v2_validator::scalar_i64_mismatch";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_v2_validator");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");
  D.registerCpuKernel(fqname, &passthrough);

  const std::array<vbt::dispatch::DeviceConstraint, 1> cs = {
      vbt::dispatch::DeviceConstraint{1,
                                      vbt::dispatch::ConstraintKind::MustBeCPUScalarInt64_0d}};

  D.set_device_policy(
      fqname,
      vbt::dispatch::DevicePolicy::MaskedSameDevice,
      /*dispatch_arg_mask=*/1,
      std::span<const vbt::dispatch::DeviceConstraint>{cs},
      /*allow_undefined_mask=*/0);

  g_called.store(0, std::memory_order_relaxed);

  BoxedStack stack;
  stack.emplace_back(cpu_tensor_f32_1d(4));
  stack.emplace_back(cpu_scalar_bool(true));  // wrong dtype

  try {
    D.callBoxed(fqname, stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find(fqname), std::string::npos) << msg;
    EXPECT_NE(msg.find("arg[1]"), std::string::npos) << msg;
    EXPECT_NE(msg.find("CPU int64 scalar (0-d)"), std::string::npos) << msg;
  }

  EXPECT_EQ(g_called.load(std::memory_order_relaxed), 0);
}

TEST(DispatchV2ValidatorTest, ScalarBoolMismatchHasStableSubstrings) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  const std::string fqname = "test_v2_validator::scalar_bool_mismatch";
  ASSERT_FALSE(D.has(fqname)) << "test op already registered: " << fqname;

  D.registerLibrary("test_v2_validator");
  D.def(fqname + "(Tensor, Tensor) -> Tensor");
  D.registerCpuKernel(fqname, &passthrough);

  const std::array<vbt::dispatch::DeviceConstraint, 1> cs = {
      vbt::dispatch::DeviceConstraint{1,
                                      vbt::dispatch::ConstraintKind::MustBeCPUScalarBool_0d}};

  D.set_device_policy(
      fqname,
      vbt::dispatch::DevicePolicy::MaskedSameDevice,
      /*dispatch_arg_mask=*/1,
      std::span<const vbt::dispatch::DeviceConstraint>{cs},
      /*allow_undefined_mask=*/0);

  g_called.store(0, std::memory_order_relaxed);

  BoxedStack stack;
  stack.emplace_back(cpu_tensor_f32_1d(4));
  stack.emplace_back(cpu_scalar_i64(1));  // wrong dtype

  try {
    D.callBoxed(fqname, stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find(fqname), std::string::npos) << msg;
    EXPECT_NE(msg.find("arg[1]"), std::string::npos) << msg;
    EXPECT_NE(msg.find("CPU bool scalar (0-d)"), std::string::npos) << msg;
  }

  EXPECT_EQ(g_called.load(std::memory_order_relaxed), 0);
}

TEST(DispatchV2ValidatorTest, DeferToKernelUndefinedMetaPreventsIndexCrash) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  vbt_register_indexing_kernels();

  // vt::index: special-op policy assignment (design/dispatcher/p2 §6.2).
  {
    auto h = D.find("vt::index");
    const auto* st = h.get().state_v2.load(std::memory_order_acquire);
    ASSERT_NE(st, nullptr);
    EXPECT_EQ(st->device_policy, vbt::dispatch::DevicePolicy::MaskedSameDevice);
    EXPECT_EQ(st->dispatch_arg_mask, 0b001u);
    EXPECT_EQ(st->allow_undefined_mask, 0b010u);
    EXPECT_EQ(st->constraint_kind_by_index[1],
              vbt::dispatch::ConstraintKind::DeferToKernel);
    EXPECT_EQ(st->constraint_kind_by_index[2],
              vbt::dispatch::ConstraintKind::DeferToKernel);
  }

  // self: CPU Float32 [2]
  TensorImpl self = cpu_tensor_f32_1d(2);

  // index: CPU Int64 [1]
  {
    auto st = make_storage_bytes(sizeof(std::int64_t));
    TensorImpl idx(st, /*sizes=*/{1}, /*strides=*/{1}, /*storage_offset=*/0,
                   ScalarType::Int64, Device::cpu());
    auto* p = static_cast<std::int64_t*>(idx.data());
    if (p) *p = 0;

    // meta: undefined-but-shaped Int64 [4]
    TensorImpl meta = undefined_but_shaped(ScalarType::Int64, Device::cpu(), {4});

    BoxedStack stack;
    stack.push_back(self);
    stack.push_back(idx);
    stack.push_back(meta);

    try {
      D.callBoxed("vt::index", stack);
      FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
      const std::string msg(e.what());
      EXPECT_NE(msg.find("vt::index"), std::string::npos) << msg;
      EXPECT_NE(msg.find("arg[2]"), std::string::npos) << msg;
    }
  }
}

TEST(DispatchV2ValidatorTest, DeferToKernelUndefinedAccPreventsIndexPutCrash) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  vbt_register_indexing_kernels();

  // vt::index_put: special-op policy assignment (design/dispatcher/p2 §6.3).
  {
    auto h = D.find("vt::index_put");
    const auto* st = h.get().state_v2.load(std::memory_order_acquire);
    ASSERT_NE(st, nullptr);
    EXPECT_EQ(st->device_policy, vbt::dispatch::DevicePolicy::MaskedSameDevice);
    EXPECT_EQ(st->dispatch_arg_mask, 0b00001u);
    EXPECT_EQ(st->allow_undefined_mask, 0u);
    EXPECT_EQ(st->constraint_kind_by_index[1],
              vbt::dispatch::ConstraintKind::DeferToKernel);
    EXPECT_EQ(st->constraint_kind_by_index[2],
              vbt::dispatch::ConstraintKind::DeferToKernel);
    EXPECT_EQ(st->constraint_kind_by_index[3],
              vbt::dispatch::ConstraintKind::DeferToKernel);
    EXPECT_EQ(st->constraint_kind_by_index[4],
              vbt::dispatch::ConstraintKind::DeferToKernel);
  }

  // self: CPU Float32 [2]
  TensorImpl self = cpu_tensor_f32_1d(2);

  // index: CPU Int64 [1]
  auto idx_storage = make_storage_bytes(sizeof(std::int64_t));
  TensorImpl idx(idx_storage, /*sizes=*/{1}, /*strides=*/{1}, /*storage_offset=*/0,
                 ScalarType::Int64, Device::cpu());
  auto* idx_p = static_cast<std::int64_t*>(idx.data());
  if (idx_p) *idx_p = 0;

  // value: CPU Float32 [1]
  TensorImpl value = cpu_tensor_f32_1d(1);

  // meta: CPU Int64 [4]
  TensorImpl meta;
  {
    auto meta_storage = make_storage_bytes(4 * sizeof(std::int64_t));
    meta = TensorImpl(meta_storage, /*sizes=*/{4}, /*strides=*/{1}, /*storage_offset=*/0,
                      ScalarType::Int64, Device::cpu());
    auto* m = static_cast<std::int64_t*>(meta.data());
    if (m) {
      m[0] = 0;
      m[1] = 1;
      m[2] = 0;
      m[3] = 0;
    }
  }

  // accumulate: undefined 0-d Bool
  TensorImpl acc = undefined_but_shaped(ScalarType::Bool, Device::cpu(), /*sizes=*/{});

  BoxedStack stack;
  stack.push_back(self);
  stack.push_back(idx);
  stack.push_back(value);
  stack.push_back(meta);
  stack.push_back(acc);

  try {
    D.callBoxed("vt::index_put", stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("vt::index_put"), std::string::npos) << msg;
    EXPECT_NE(msg.find("arg[4]"), std::string::npos) << msg;
  }
}

TEST(DispatchV2ValidatorTest, IndexDeviceMismatchErrorOwnedByKernel) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  vbt_register_indexing_kernels();

  auto h = D.find("vt::index");
  const auto* st = h.get().state_v2.load(std::memory_order_acquire);
  ASSERT_NE(st, nullptr);
  ASSERT_EQ(st->device_policy, vbt::dispatch::DevicePolicy::MaskedSameDevice);

#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  TensorImpl self = cuda_tensor_f32_1d(/*n=*/4, /*dev=*/0);

  auto idx_storage = make_storage_bytes(sizeof(std::int64_t));
  // Deliberate mismatch: idx is on a different CUDA device than self.
  TensorImpl idx(idx_storage, /*sizes=*/{1}, /*strides=*/{1}, /*storage_offset=*/0,
                 ScalarType::Int64, Device::cuda(/*idx=*/1));
  auto* idx_p = static_cast<std::int64_t*>(idx.data());
  if (idx_p) *idx_p = 0;

  TensorImpl meta;
  {
    auto meta_storage = make_storage_bytes(4 * sizeof(std::int64_t));
    meta = TensorImpl(meta_storage, /*sizes=*/{4}, /*strides=*/{1},
                      /*storage_offset=*/0, ScalarType::Int64, Device::cpu());
    auto* m = static_cast<std::int64_t*>(meta.data());
    if (m) {
      m[0] = 0;
      m[1] = 1;
      m[2] = 0;
      m[3] = 0;
    }
  }

  BoxedStack stack{self, idx, meta};
  try {
    D.callBoxed("vt::index", stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("vt::index: index tensor must be on CPU or the same device as self"),
              std::string::npos)
        << msg;
    EXPECT_EQ(msg.find("Expected all tensors to be on the same device"),
              std::string::npos)
        << msg;
  }
#endif
}

TEST(DispatchV2ValidatorTest, IndexPutDeviceMismatchErrorOwnedByKernel) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  vbt_register_indexing_kernels();

  auto h = D.find("vt::index_put");
  const auto* st = h.get().state_v2.load(std::memory_order_acquire);
  ASSERT_NE(st, nullptr);
  ASSERT_EQ(st->device_policy, vbt::dispatch::DevicePolicy::MaskedSameDevice);

#if !VBT_WITH_CUDA
  GTEST_SKIP() << "CUDA not built";
#else
  if (vbt::cuda::device_count() == 0) GTEST_SKIP() << "No CUDA device";

  TensorImpl self = cuda_tensor_f32_1d(/*n=*/2, /*dev=*/0);
  TensorImpl value = cuda_tensor_f32_1d(/*n=*/1, /*dev=*/0);

  auto idx_storage = make_storage_bytes(sizeof(std::int64_t));
  // Deliberate mismatch: idx is on a different CUDA device than self.
  TensorImpl idx(idx_storage, /*sizes=*/{1}, /*strides=*/{1}, /*storage_offset=*/0,
                 ScalarType::Int64, Device::cuda(/*idx=*/1));
  auto* idx_p = static_cast<std::int64_t*>(idx.data());
  if (idx_p) *idx_p = 0;

  TensorImpl meta;
  {
    auto meta_storage = make_storage_bytes(4 * sizeof(std::int64_t));
    meta = TensorImpl(meta_storage, /*sizes=*/{4}, /*strides=*/{1},
                      /*storage_offset=*/0, ScalarType::Int64, Device::cpu());
    auto* m = static_cast<std::int64_t*>(meta.data());
    if (m) {
      m[0] = 0;
      m[1] = 1;
      m[2] = 0;
      m[3] = 0;
    }
  }

  TensorImpl acc = cpu_scalar_bool(false);

  BoxedStack stack{self, idx, value, meta, acc};
  try {
    D.callBoxed("vt::index_put", stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("vt::index_put: index tensor must be on CPU or the same device as self"),
              std::string::npos)
        << msg;
    EXPECT_EQ(msg.find("Expected all tensors to be on the same device"),
              std::string::npos)
        << msg;
  }
#endif
}

TEST(DispatchV2ValidatorTest, CheckStreamTemplateValidationOwnedByDispatcher) {
  vbt::dispatch::DispatchV2ModeGuard v2_guard(/*enabled=*/true);

  auto& D = Dispatcher::instance();
  const std::string fqname = "vt::check_stream";

  if (!D.has(fqname)) {
    D.registerLibrary("vt");
    D.def(fqname + "(Tensor, Tensor) -> Tensor");
  }

  auto h = D.find(fqname);
  const auto* st = h.get().state_v2.load(std::memory_order_acquire);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(st->device_policy, vbt::dispatch::DevicePolicy::MaskedSameDevice);
  EXPECT_EQ(st->dispatch_arg_mask, 0b01u);
  EXPECT_EQ(st->allow_undefined_mask, 0u);
  EXPECT_EQ(st->constraint_kind_by_index[1],
            vbt::dispatch::ConstraintKind::MustBeCPUScalarInt64_0d);

  BoxedStack stack;
  stack.push_back(cpu_tensor_f32_1d(4));
  stack.push_back(cpu_scalar_bool(true));

  try {
    D.callBoxed(fqname, stack);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    const std::string msg(e.what());
    EXPECT_NE(msg.find("vt::check_stream"), std::string::npos) << msg;
    EXPECT_NE(msg.find("arg[1]"), std::string::npos) << msg;
    EXPECT_NE(msg.find("CPU int64 scalar (0-d)"), std::string::npos) << msg;
  }
}

#else

TEST(DispatchV2ValidatorTest, SkippedWhenDispatchV2Disabled) {
  GTEST_SKIP() << "dispatch v2 or internal tests disabled";
}

#endif  // VBT_WITH_DISPATCH_V2 && VBT_INTERNAL_TESTS
