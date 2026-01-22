// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <dlpack/dlpack.h>
#include <cstdint>

#include "vbt/core/tensor.h"
#include "vbt/core/dtype.h"
#include "vbt/core/device.h"
#include "vbt/interop/dlpack.h"

using vbt::core::TensorImpl;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::interop::from_dlpack;

static DLManagedTensor* make_dlmt(void* base, int ndim, const int64_t* shape, const int64_t* strides, uint64_t byte_offset, DLDataType dtype) {
  auto* mt = new DLManagedTensor{};
  mt->dl_tensor.data = base;
  mt->dl_tensor.device = DLDevice{.device_type=kDLCPU, .device_id=0};
  mt->dl_tensor.dtype = dtype;
  mt->dl_tensor.ndim = ndim;
  if (ndim > 0) {
    auto* sh = new int64_t[ndim];
    for (int i=0;i<ndim;++i) sh[i] = shape[i];
    mt->dl_tensor.shape = sh;
    if (strides) {
      auto* st = new int64_t[ndim];
      for (int i=0;i<ndim;++i) st[i] = strides[i];
      mt->dl_tensor.strides = st;
    } else {
      mt->dl_tensor.strides = nullptr;
    }
  } else {
    mt->dl_tensor.shape = nullptr;
    mt->dl_tensor.strides = nullptr;
  }
  mt->dl_tensor.byte_offset = byte_offset;
  mt->manager_ctx = nullptr;
  mt->deleter = [](DLManagedTensor* p){ if (!p) return; delete[] p->dl_tensor.shape; delete[] p->dl_tensor.strides; delete p; };
  return mt;
}

TEST(DLPackCpuImporterTest, ContiguousNullStridesImport) {
  int64_t shape[2] = {2,3};
  DLDataType dt = vbt::core::to_dlpack_dtype(ScalarType::Float32);
  void* base = reinterpret_cast<void*>(0xBEEF0000);
  auto* mt = make_dlmt(base, 2, shape, /*strides*/nullptr, /*byte_offset*/0, dt);
  TensorImpl t = from_dlpack(mt);
  EXPECT_EQ(t.sizes(), std::vector<int64_t>({2,3}));
  // Synthesized contiguous strides
  EXPECT_EQ(t.strides(), std::vector<int64_t>({3,1}));
  EXPECT_EQ(t.storage_offset(), 0);
  // Effective data equals base for contiguous with offset 0
  EXPECT_EQ(t.data(), base);
}

TEST(DLPackCpuImporterTest, MixedSignStridesSpanComputesStorageOffset) {
  int64_t shape[2] = {2,3};
  int64_t strides[2] = {-3, 1};
  DLDataType dt = vbt::core::to_dlpack_dtype(ScalarType::Float32);
  void* base = reinterpret_cast<void*>(0xBAAD0000);
  // byte_offset=0 → p_eff=base
  auto* mt = make_dlmt(base, 2, shape, strides, /*byte_offset*/0, dt);
  TensorImpl t = from_dlpack(mt);
  // storage_offset = -min_elem_off = -(-3)=3
  EXPECT_EQ(t.storage_offset(), 3);
}

TEST(DLPackCpuImporterTest, OneShotConsumptionIsEnforced) {
  int64_t shape[1] = {4};
  DLDataType dt = vbt::core::to_dlpack_dtype(ScalarType::Float32);
  void* base = reinterpret_cast<void*>(0xDEAD1000);
  auto* mt = make_dlmt(base, 1, shape, /*strides*/nullptr, /*byte_offset*/0, dt);
  TensorImpl t = from_dlpack(mt);
  // Second import of the same DLManagedTensor must fail
  EXPECT_THROW({
    auto t2 = from_dlpack(mt);
    (void)t2;
  }, std::runtime_error);
}

TEST(DLPackCpuImporterTest, ZeroSizeStrictPolicy) {
  int64_t shape[2] = {0, 3};
  DLDataType dt = vbt::core::to_dlpack_dtype(ScalarType::Float32);
  void* base = reinterpret_cast<void*>(0xDEAD2000);
  // Non-null data with zero-size should be rejected
  auto* mt1 = make_dlmt(base, 2, shape, /*strides*/nullptr, /*byte_offset*/0, dt);
  EXPECT_THROW({ auto t = from_dlpack(mt1); (void)t; }, std::runtime_error);
  // Non-zero byte_offset also rejected even if data null
  auto* mt2 = make_dlmt(nullptr, 2, shape, /*strides*/nullptr, /*byte_offset*/4, dt);
  EXPECT_THROW({ auto t = from_dlpack(mt2); (void)t; }, std::runtime_error);
}

TEST(DLPackCpuImporterTest, AlignmentGating) {
  int64_t shape[1] = {1};
  DLDataType dt = vbt::core::to_dlpack_dtype(ScalarType::Float32);
  // Misaligned base for float32 (4-byte alignment)
  void* misaligned = reinterpret_cast<void*>(0x1002);
  auto* mt = make_dlmt(misaligned, 1, shape, /*strides*/nullptr, /*byte_offset*/0, dt);
#if VBT_REQUIRE_DLPACK_ALIGNMENT
  EXPECT_THROW({ auto t = from_dlpack(mt); (void)t; }, std::runtime_error);
#else
  EXPECT_NO_THROW({ auto t = from_dlpack(mt); (void)t; });
#endif
}
