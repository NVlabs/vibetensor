// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "vbt/core/tensor_iter.h"
#include "vbt/core/storage.h"
#include "vbt/core/data_ptr.h"
#include "vbt/core/intrusive_ptr.h"
#include "vbt/core/device.h"
#include "vbt/core/dtype.h"
#include "vbt/cuda/device.h"
#include "vbt/cuda/storage.h"

using vbt::core::TensorImpl;
using vbt::core::Storage;
using vbt::core::StoragePtr;
using vbt::core::DataPtr;
using vbt::core::ScalarType;
using vbt::core::Device;
using vbt::core::TensorIter;
using vbt::core::TensorIterConfig;
using vbt::core::DeviceStrideMeta;

namespace vbt::cuda_detail {
bool should_use_int32_index(std::int64_t N) noexcept;
}  // namespace vbt::cuda_detail

namespace {

static StoragePtr make_storage_bytes(std::size_t nbytes) {
  void* base = nullptr;
  if (nbytes > 0) {
    base = ::operator new(nbytes);
  }
  return vbt::core::make_intrusive<Storage>(
      DataPtr(base, [](void* p) noexcept { ::operator delete(p); }), nbytes);
}

static TensorImpl make_contiguous_tensor(const std::vector<int64_t>& sizes,
                                         ScalarType dtype = ScalarType::Float32) {
  const std::size_t nd = sizes.size();
  std::vector<int64_t> strides(nd, 0);
  int64_t acc = 1;
  for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 1; i >= 0; --i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    strides[idx] = acc;
    const auto sz = sizes[idx];
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

  const std::size_t item_b = static_cast<std::size_t>(vbt::core::itemsize(dtype));
  const std::size_t nbytes = static_cast<std::size_t>(ne) * item_b;
  auto storage = make_storage_bytes(nbytes);
  return TensorImpl(storage, sizes, strides, /*storage_offset=*/0, dtype,
                    Device::cpu());
}

static std::int64_t compute_offset_elems_host(std::int64_t li,
                                              const DeviceStrideMeta& m) {
  if (m.ndim == 0) {
    return 0;
  }
  std::int64_t off = 0;
  for (std::int64_t d = 0; d < m.ndim; ++d) {
    std::int64_t size_d = (m.sizes[d] == 0 ? 1 : m.sizes[d]);
    std::int64_t idx_d  = (size_d == 1) ? 0 : (li % size_d);
    li = (size_d == 1) ? li : (li / size_d);
    off += idx_d * m.strides[d];
  }
  return off;
}

static void noop_loop(char** /*data*/, const std::int64_t* /*strides*/,
                      std::int64_t /*size*/, void* /*ctx_void*/) {}

}  // namespace

TEST(TensorIterCudaMetaTest, CpuContiguous2DExportMatchesShapeAndStrides) {
  auto out = make_contiguous_tensor({2, 3});
  auto a   = make_contiguous_tensor({2, 3});
  auto b   = make_contiguous_tensor({2, 3});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.add_input(b);
  TensorIter iter = cfg.build();

  ASSERT_EQ(iter.ndim(), 2);
  ASSERT_EQ(iter.shape().size(), 2u);

  DeviceStrideMeta mo{};
  DeviceStrideMeta ma{};
  DeviceStrideMeta mb{};

  iter.export_device_meta(0, &mo);
  iter.export_device_meta(1, &ma);
  iter.export_device_meta(2, &mb);

  const int R = iter.ndim();
  const auto& shape = iter.shape();
  ASSERT_EQ(mo.ndim, R);
  ASSERT_EQ(ma.ndim, R);
  ASSERT_EQ(mb.ndim, R);

  for (int d = 0; d < R; ++d) {
    ASSERT_EQ(mo.sizes[d], shape[static_cast<std::size_t>(d)]);
    ASSERT_EQ(ma.sizes[d], shape[static_cast<std::size_t>(d)]);
    ASSERT_EQ(mb.sizes[d], shape[static_cast<std::size_t>(d)]);
  }

  auto check_operand = [&](int idx, const DeviceStrideMeta& m) {
    const auto& op = iter.operand(idx);
    const auto item_b = static_cast<std::int64_t>(vbt::core::itemsize(op.dtype));
    for (int d = 0; d < R; ++d) {
      const auto stride_bytes =
          op.dim_stride_bytes[static_cast<std::size_t>(d)];
      if (stride_bytes == 0) {
        EXPECT_EQ(m.strides[d], 0);
      } else {
        EXPECT_EQ(m.strides[d], stride_bytes / item_b);
      }
    }
  };

  check_operand(0, mo);
  check_operand(1, ma);
  check_operand(2, mb);
}

TEST(TensorIterCudaMetaTest, CpuBroadcastScalarHasZeroStrides) {
  auto out    = make_contiguous_tensor({4, 5});
  auto scalar = make_contiguous_tensor({});
  auto vec    = make_contiguous_tensor({4, 5});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(scalar);
  cfg.add_input(vec);
  TensorIter iter = cfg.build();

  ASSERT_EQ(iter.ndim(), 2);

  DeviceStrideMeta meta_scalar{};
  iter.export_device_meta(1, &meta_scalar);

  ASSERT_EQ(meta_scalar.ndim, iter.ndim());
  for (int d = 0; d < meta_scalar.ndim; ++d) {
    EXPECT_EQ(meta_scalar.sizes[d], iter.shape()[static_cast<std::size_t>(d)]);
    EXPECT_EQ(meta_scalar.strides[d], 0);
  }
}

TEST(TensorIterCudaMetaTest, ActiveDimsSizeOneStrideIgnoredInOffsets) {
  // TI's device index decomposition treats size==1 dims as a no-op.
  // This underpins the CUDA reduction "active dims" policy (stride checks must
  // ignore size-1 dims).
  DeviceStrideMeta m_pos{};
  m_pos.ndim = 3;
  m_pos.sizes[0] = 2;
  m_pos.sizes[1] = 1;
  m_pos.sizes[2] = 3;
  m_pos.strides[0] = 3;
  m_pos.strides[1] = 7;
  m_pos.strides[2] = 1;

  DeviceStrideMeta m_neg = m_pos;
  m_neg.strides[1] = -7;

  const std::int64_t N = 2 * 1 * 3;
  for (std::int64_t li = 0; li < N; ++li) {
    EXPECT_EQ(compute_offset_elems_host(li, m_pos),
              compute_offset_elems_host(li, m_neg))
        << "li=" << li;
  }
}

TEST(TensorIterCudaMetaTest, ExportDeviceMetaMaxNdimViolation) {
  // 9D non-degenerate shape so that iteration rank == 9.
  auto out = make_contiguous_tensor({2, 2, 2, 2, 2, 2, 2, 2, 2});
  auto a   = make_contiguous_tensor({2, 2, 2, 2, 2, 2, 2, 2, 2});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  TensorIter iter = cfg.build();

  ASSERT_EQ(iter.ndim(), 9);

  DeviceStrideMeta meta{};
  bool threw = false;
  try {
    iter.export_device_meta(0, &meta, /*max_ndim=*/8);
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("iteration rank exceeds max_ndim"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}

TEST(TensorIterCudaMetaTest, ExportDeviceMetaNullOutMetaThrows) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  TensorIter iter = cfg.build();

  bool threw = false;
  try {
    iter.export_device_meta(0, nullptr);
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("out_meta must not be null"), std::string::npos);
  }
  EXPECT_TRUE(threw);
}

TEST(TensorIterCudaMetaTest, ExportDeviceMetaOperandIndexOutOfRangeThrows) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  TensorIter iter = cfg.build();

  DeviceStrideMeta meta{};

  bool threw_low = false;
  try {
    iter.export_device_meta(-1, &meta);
  } catch (const std::out_of_range& e) {
    threw_low = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("operand index out of range"), std::string::npos);
  }
  EXPECT_TRUE(threw_low);

  bool threw_high = false;
  try {
    iter.export_device_meta(iter.ntensors(), &meta);
  } catch (const std::out_of_range& e) {
    threw_high = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("operand index out of range"), std::string::npos);
  }
  EXPECT_TRUE(threw_high);
}

TEST(TensorIterCudaMetaTest, ExportDeviceMetaMaxNdimOutOfRangeThrows) {
  auto out = make_contiguous_tensor({4});
  auto a   = make_contiguous_tensor({4});

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  TensorIter iter = cfg.build();

  DeviceStrideMeta meta{};

  bool threw_low = false;
  try {
    iter.export_device_meta(0, &meta, /*max_ndim=*/0);
  } catch (const std::invalid_argument& e) {
    threw_low = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("max_ndim out of range"), std::string::npos);
  }
  EXPECT_TRUE(threw_low);

  bool threw_high = false;
  try {
    iter.export_device_meta(0, &meta,
                            static_cast<std::int64_t>(vbt::core::kTensorIterMaxRank) + 1);
  } catch (const std::invalid_argument& e) {
    threw_high = true;
    std::string msg = e.what();
    EXPECT_NE(msg.find("max_ndim out of range"), std::string::npos);
  }
  EXPECT_TRUE(threw_high);
}

TEST(TensorIterCudaMetaTest, CudaContiguous1DExportMatchesSizesAndStrides) {
#if VBT_WITH_CUDA
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

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.add_input(b);
  TensorIter iter = cfg.build();

  ASSERT_EQ(iter.ndim(), 1);

  DeviceStrideMeta mo{};
  DeviceStrideMeta ma{};
  DeviceStrideMeta mb{};
  iter.export_device_meta(0, &mo, vbt::core::kTensorIterMaxRank);
  iter.export_device_meta(1, &ma, vbt::core::kTensorIterMaxRank);
  iter.export_device_meta(2, &mb, vbt::core::kTensorIterMaxRank);

  EXPECT_EQ(mo.ndim, 1);
  EXPECT_EQ(ma.ndim, 1);
  EXPECT_EQ(mb.ndim, 1);
  EXPECT_EQ(mo.sizes[0], N);
  EXPECT_EQ(ma.sizes[0], N);
  EXPECT_EQ(mb.sizes[0], N);
  EXPECT_EQ(mo.strides[0], 1);
  EXPECT_EQ(ma.strides[0], 1);
  EXPECT_EQ(mb.strides[0], 1);
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

TEST(TensorIterCudaMetaTest, CudaBroadcast2DFrom1DOffsetsMatchExpectation) {
#if VBT_WITH_CUDA
  if (vbt::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device";
  }
  const int dev = 0;
  const int rows = 2;
  const int cols = 256;
  const std::int64_t N = static_cast<std::int64_t>(rows) * cols;

  const std::size_t nbytes_a = static_cast<std::size_t>(N) * sizeof(float);
  const std::size_t nbytes_b = static_cast<std::size_t>(cols) * sizeof(float);

  auto st_out = vbt::cuda::new_cuda_storage(nbytes_a, dev);
  auto st_a   = vbt::cuda::new_cuda_storage(nbytes_a, dev);
  auto st_b   = vbt::cuda::new_cuda_storage(nbytes_b, dev);

  TensorImpl out(st_out, {rows, cols}, {cols, 1}, 0,
                 ScalarType::Float32, Device::cuda(dev));
  TensorImpl a(st_a, {rows, cols}, {cols, 1}, 0,
               ScalarType::Float32, Device::cuda(dev));
  TensorImpl b(st_b, {cols}, {1}, 0,
               ScalarType::Float32, Device::cuda(dev));

  TensorIterConfig cfg;
  cfg.add_output(vbt::core::OptionalTensorImplRef(&out, /*defined=*/true));
  cfg.add_input(a);
  cfg.add_input(b);
  TensorIter iter = cfg.build();

  EXPECT_EQ(iter.numel(), N);

  DeviceStrideMeta ma{};
  DeviceStrideMeta mb{};
  iter.export_device_meta(1, &ma, vbt::core::kTensorIterMaxRank);
  iter.export_device_meta(2, &mb, vbt::core::kTensorIterMaxRank);

  ASSERT_EQ(ma.ndim, mb.ndim);
  ASSERT_GT(ma.ndim, 0);

  for (std::int64_t li = 0; li < N; ++li) {
    const std::int64_t off_a = compute_offset_elems_host(li, ma);
    const std::int64_t off_b = compute_offset_elems_host(li, mb);

    const std::int64_t r = li / cols;
    const std::int64_t c = li % cols;

    EXPECT_EQ(off_a, r * cols + c) << "li=" << li;
    EXPECT_EQ(off_b, c) << "li=" << li;
  }
#else
  GTEST_SKIP() << "Built without CUDA";
#endif
}

