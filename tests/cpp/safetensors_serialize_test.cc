// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vbt/interop/safetensors.h"

using vbt::interop::safetensors::DType;
using vbt::interop::safetensors::ErrorCode;
using vbt::interop::safetensors::SafeTensorsError;
using vbt::interop::safetensors::SafeTensorsView;
using vbt::interop::safetensors::SerializeOptions;
using vbt::interop::safetensors::TensorEntry;
using vbt::interop::safetensors::serialize;

namespace {
std::vector<std::byte> bytes(std::initializer_list<unsigned char> xs) {
  std::vector<std::byte> out;
  out.reserve(xs.size());
  for (auto x : xs) out.push_back(static_cast<std::byte>(x));
  return out;
}

std::uint64_t read_u64_le(const std::byte* p) {
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= (static_cast<std::uint64_t>(static_cast<unsigned char>(p[i])) << (8 * i));
  }
  return v;
}

TensorEntry make_tensor(std::string name,
                        DType dtype,
                        std::vector<std::size_t> shape,
                        const std::vector<std::byte>& data) {
  TensorEntry t;
  t.name = std::move(name);
  t.info.dtype = dtype;
  t.info.shape = std::move(shape);
  t.info.data_offsets = {0, 0};
  t.data = std::span<const std::byte>(data.data(), data.size());
  return t;
}
} // namespace

TEST(SafeTensorsSerializeTest, RoundTripIsValidAndDeterministic) {
  const std::vector<std::byte> a_data = bytes({1, 2, 3, 4});
  const std::vector<std::byte> b_data = bytes({5, 6, 7, 8});
  const std::vector<std::byte> z_data = bytes({9, 10, 11, 12});
  const std::vector<std::byte> c_data = bytes({13});

  std::vector<TensorEntry> tensors;
  tensors.push_back(make_tensor("z", DType::F16, {2}, z_data));
  tensors.push_back(make_tensor("b", DType::I32, {1}, b_data));
  tensors.push_back(make_tensor("a", DType::I32, {1}, a_data));
  tensors.push_back(make_tensor("c", DType::F4, {2}, c_data));

  const auto buf =
      serialize(std::span<const TensorEntry>(tensors.data(), tensors.size()), std::nullopt);

  // Header padding: N is a multiple of 8 and any extra bytes are ASCII spaces.
  ASSERT_GE(buf.size(), 8u);
  const std::uint64_t n = read_u64_le(buf.data());
  EXPECT_EQ(n % 8u, 0u);
  ASSERT_GE(buf.size(), 8u + static_cast<std::size_t>(n));
  const std::string header(reinterpret_cast<const char*>(buf.data() + 8),
                           static_cast<std::size_t>(n));
  ASSERT_FALSE(header.empty());
  EXPECT_EQ(header.front(), '{');
  const std::size_t last_non_space = header.find_last_not_of(' ');
  ASSERT_NE(last_non_space, std::string::npos);
  EXPECT_EQ(header[last_non_space], '}');
  for (std::size_t i = last_non_space + 1; i < header.size(); ++i) {
    EXPECT_EQ(header[i], ' ');
  }

  SafeTensorsView view =
      SafeTensorsView::deserialize(std::span<const std::byte>(buf.data(), buf.size()));

  ASSERT_EQ(view.metadata().tensors_by_offset.size(), 4u);
  // Deterministic order: dtype alignment (descending enum) then name.
  EXPECT_EQ(view.metadata().tensors_by_offset[0].name, "a");
  EXPECT_EQ(view.metadata().tensors_by_offset[1].name, "b");
  EXPECT_EQ(view.metadata().tensors_by_offset[2].name, "z");
  EXPECT_EQ(view.metadata().tensors_by_offset[3].name, "c");

  const auto ta = view.tensor("a");
  EXPECT_EQ(ta.dtype, DType::I32);
  EXPECT_EQ(ta.shape, (std::vector<std::size_t>{1u}));
  ASSERT_EQ(ta.data.size(), a_data.size());
  EXPECT_EQ(std::vector<std::byte>(ta.data.begin(), ta.data.end()), a_data);

  const auto tb = view.tensor("b");
  EXPECT_EQ(tb.dtype, DType::I32);
  EXPECT_EQ(tb.shape, (std::vector<std::size_t>{1u}));
  ASSERT_EQ(tb.data.size(), b_data.size());
  EXPECT_EQ(std::vector<std::byte>(tb.data.begin(), tb.data.end()), b_data);

  const auto tz = view.tensor("z");
  EXPECT_EQ(tz.dtype, DType::F16);
  EXPECT_EQ(tz.shape, (std::vector<std::size_t>{2u}));
  ASSERT_EQ(tz.data.size(), z_data.size());
  EXPECT_EQ(std::vector<std::byte>(tz.data.begin(), tz.data.end()), z_data);

  const auto tc = view.tensor("c");
  EXPECT_EQ(tc.dtype, DType::F4);
  EXPECT_EQ(tc.shape, (std::vector<std::size_t>{2u}));
  ASSERT_EQ(tc.data.size(), c_data.size());
  EXPECT_EQ(std::vector<std::byte>(tc.data.begin(), tc.data.end()), c_data);

  // Determinism: input order should not matter when sorting is enabled.
  std::vector<TensorEntry> tensors2;
  tensors2.push_back(make_tensor("c", DType::F4, {2}, c_data));
  tensors2.push_back(make_tensor("a", DType::I32, {1}, a_data));
  tensors2.push_back(make_tensor("z", DType::F16, {2}, z_data));
  tensors2.push_back(make_tensor("b", DType::I32, {1}, b_data));

  const auto buf2 =
      serialize(std::span<const TensorEntry>(tensors2.data(), tensors2.size()), std::nullopt);
  EXPECT_EQ(buf2, buf);
}

TEST(SafeTensorsSerializeTest, TensorInvalidInfoSizeMismatchThrows) {
  const std::vector<std::byte> empty;

  TensorEntry t;
  t.name = "t";
  t.info.dtype = DType::I32;
  t.info.shape = {1};
  t.info.data_offsets = {0, 0};
  t.data = std::span<const std::byte>(empty.data(), empty.size());

  try {
    (void)serialize(std::span<const TensorEntry>(&t, 1));
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::TensorInvalidInfo);
    EXPECT_EQ(e.tensor_name(), "t");
  }
}

TEST(SafeTensorsSerializeTest, MisalignedSliceThrows) {
  const std::vector<std::byte> empty;

  TensorEntry t;
  t.name = "t";
  t.info.dtype = DType::F4;
  t.info.shape = {1};
  t.info.data_offsets = {0, 0};
  t.data = std::span<const std::byte>(empty.data(), empty.size());

  try {
    (void)serialize(std::span<const TensorEntry>(&t, 1));
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::MisalignedSlice);
    EXPECT_EQ(e.tensor_name(), std::string_view{});
  }
}

TEST(SafeTensorsSerializeTest, HeaderTooLargeRespectsSerializeOptions) {
  const std::vector<std::byte> empty;

  TensorEntry t;
  t.name = "t";
  t.info.dtype = DType::I32;
  t.info.shape = {0};
  t.info.data_offsets = {0, 0};
  t.data = std::span<const std::byte>(empty.data(), empty.size());

  SerializeOptions opts;
  opts.max_header_size_bytes = 1;

  try {
    (void)serialize(std::span<const TensorEntry>(&t, 1), std::nullopt, opts);
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::HeaderTooLarge);
  }
}

TEST(SafeTensorsSerializeTest, ValidationOverflowThrows) {
  const std::vector<std::byte> empty;

  TensorEntry t;
  t.name = "t";
  t.info.dtype = DType::U8;
  t.info.shape = {std::numeric_limits<std::size_t>::max(), 2};
  t.info.data_offsets = {0, 0};
  t.data = std::span<const std::byte>(empty.data(), empty.size());

  try {
    (void)serialize(std::span<const TensorEntry>(&t, 1));
    FAIL() << "expected SafeTensorsError";
  } catch (const SafeTensorsError& e) {
    EXPECT_EQ(e.code(), ErrorCode::ValidationOverflow);
    EXPECT_EQ(e.tensor_name(), "t");
  }
}
