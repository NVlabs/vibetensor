// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <dlpack/dlpack.h>

#include "vbt/core/tensor.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/interop/dlpack.h"
#include "vbt/core/intrusive_ptr.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::interop::to_dlpack;

static vbt::core::StoragePtr mk_storage(void* base, std::size_t nbytes) {
  return vbt::core::make_intrusive<Storage>(DataPtr(base, [](void*) noexcept {}), nbytes);
}

TEST(DLPackExporterTest, ExporterZeroSize) {
  auto base = reinterpret_cast<void*>(0xABCD0000);
  auto st = mk_storage(base, 4096);
  TensorImpl t(st, {0, 3}, {3, 1}, 0, ScalarType::Float32, Device::cpu());
  auto up = to_dlpack(t);
  DLManagedTensor* mt = up.get();
  ASSERT_NE(mt, nullptr);
  const DLTensor& dl = mt->dl_tensor;
  EXPECT_EQ(dl.device.device_type, kDLCPU);
  EXPECT_EQ(dl.device.device_id, 0);
  EXPECT_EQ(dl.data, nullptr);
  EXPECT_EQ(dl.byte_offset, 0u);
  ASSERT_EQ(dl.ndim, 2);
  ASSERT_NE(dl.shape, nullptr);
  EXPECT_EQ(dl.shape[0], 0);
  EXPECT_EQ(dl.shape[1], 3);
  // strides policy: contiguous => NULL for legacy
  EXPECT_EQ(dl.strides, nullptr);
}

TEST(DLPackExporterTest, ExporterViewOffset) {
  auto base = reinterpret_cast<void*>(0xDEAD0000);
  auto st = mk_storage(base, 4096);
  TensorImpl base_t(st, {10, 10}, {10, 1}, 0, ScalarType::Float32, Device::cpu());
  // take a view with offset k elements
  const int64_t k = 7;
  TensorImpl v = base_t.as_strided({5}, {2}, k);
  auto up = to_dlpack(v);
  const DLTensor& dl = up->dl_tensor;
  EXPECT_EQ(dl.device.device_type, kDLCPU);
  EXPECT_EQ(dl.device.device_id, 0);
  // data should point to base allocation
  EXPECT_EQ(dl.data, base);
  // byte_offset = k * itemsize (4)
  EXPECT_EQ(dl.byte_offset, static_cast<uint64_t>(k * 4));
}

TEST(DLPackExporterTest, ExporterCapsuleFields) {
  // Scalar (ndim==0)
  auto st0 = mk_storage(reinterpret_cast<void*>(0x1000), 16);
  TensorImpl s(st0, {}, {}, 0, ScalarType::Int64, Device::cpu());
  auto up0 = to_dlpack(s);
  const DLTensor& dl0 = up0->dl_tensor;
  EXPECT_EQ(dl0.ndim, 0);
  EXPECT_EQ(dl0.shape, nullptr);
  EXPECT_EQ(dl0.strides, nullptr);

  // 1D contiguous
  auto st1 = mk_storage(reinterpret_cast<void*>(0x2000), 64);
  TensorImpl a(st1, {4}, {1}, 0, ScalarType::Int64, Device::cpu());
  auto up1 = to_dlpack(a);
  const DLTensor& dl1 = up1->dl_tensor;
  EXPECT_EQ(dl1.ndim, 1);
  ASSERT_NE(dl1.shape, nullptr);
  EXPECT_EQ(dl1.shape[0], 4);
  EXPECT_EQ(dl1.strides, nullptr);

  // 2D non-contiguous
  auto st2 = mk_storage(reinterpret_cast<void*>(0x3000), 1024);
  TensorImpl b(st2, {2, 3}, {5, 2}, 1, ScalarType::Float32, Device::cpu());
  auto up2 = to_dlpack(b);
  const DLTensor& dl2 = up2->dl_tensor;
  EXPECT_EQ(dl2.ndim, 2);
  ASSERT_NE(dl2.shape, nullptr);
  EXPECT_EQ(dl2.shape[0], 2);
  EXPECT_EQ(dl2.shape[1], 3);
  ASSERT_NE(dl2.strides, nullptr);
  EXPECT_EQ(dl2.strides[0], 5);
  EXPECT_EQ(dl2.strides[1], 2);
}
